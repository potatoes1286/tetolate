#!/usr/bin/env python3
"""Private HTTP worker for tetolate's optional PaddleOCR runtime."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import tempfile
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

import paddle_ocr_image


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8090
DEFAULT_MAX_IMAGE_BYTES = 128 * 1024 * 1024
MAX_CACHED_ENGINES = 4
EXTRACTION_OPTION_KEYS = frozenset(
    {
        "min_score",
        "tile_enabled",
        "tile_width",
        "tile_height",
        "tile_overlap",
        "tile_include_full_image",
        "tile_dedupe_iou",
        "tile_dedupe_containment",
    }
)


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    image_base64: str = Field(alias="imageBase64")
    page: int = Field(ge=0)
    engine: str
    options: dict[str, Any] = Field(default_factory=dict)


_engine_lock = threading.RLock()
_engine_cache: OrderedDict[str, Any] = OrderedDict()


def max_image_bytes() -> int:
    value = os.environ.get("TETOLATE_OCR_MAX_IMAGE_BYTES", "").strip()
    if not value:
        return DEFAULT_MAX_IMAGE_BYTES
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "TETOLATE_OCR_MAX_IMAGE_BYTES must be an integer."
        ) from exc
    if parsed <= 0:
        raise RuntimeError("TETOLATE_OCR_MAX_IMAGE_BYTES must be positive.")
    return parsed


def decode_image(value: str) -> bytes:
    estimated_size = (len(value) * 3) // 4
    if estimated_size > max_image_bytes():
        raise HTTPException(status_code=413, detail="OCR image exceeds the worker size limit.")
    try:
        image_data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="imageBase64 is not valid base64.") from exc
    if not image_data:
        raise HTTPException(status_code=400, detail="OCR image is empty.")
    if len(image_data) > max_image_bytes():
        raise HTTPException(status_code=413, detail="OCR image exceeds the worker size limit.")
    return image_data


def image_suffix(image_data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            image_format = str(image.format or "").upper()
    except (OSError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded OCR data is not a supported image.") from exc
    suffixes = {
        "BMP": ".bmp",
        "JPEG": ".jpg",
        "PNG": ".png",
        "PPM": ".ppm",
        "TIFF": ".tiff",
        "WEBP": ".webp",
    }
    suffix = suffixes.get(image_format)
    if suffix is None:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded OCR image format {image_format or 'unknown'} is not supported.",
        )
    return suffix


def engine_options(options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in options.items() if key not in EXTRACTION_OPTION_KEYS}


def engine_cache_key(engine: str, options: dict[str, Any]) -> str:
    try:
        serialized = json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise paddle_ocr_image.InputError(f"OCR engine options are not valid JSON: {exc}") from exc
    return f"{engine}:{serialized}"


def create_local_engine(engine: str, options: dict[str, Any]) -> Any:
    device = str(options.get("device") or paddle_ocr_image.DEFAULT_DEVICE)
    paddle_ocr_image.configure_paddlex_runtime(device)
    try:
        if engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            from paddleocr import PaddleOCRVL

            return PaddleOCRVL(**options)
        from paddleocr import PaddleOCR

        return PaddleOCR(**options)
    except (TypeError, ValueError) as exc:
        raise paddle_ocr_image.InputError(f"PaddleOCR rejected the worker options: {exc}") from exc


def cached_engine(engine: str, options: dict[str, Any]) -> Any:
    key = engine_cache_key(engine, options)
    cached = _engine_cache.get(key)
    if cached is not None:
        _engine_cache.move_to_end(key)
        return cached

    created = create_local_engine(engine, options)
    _engine_cache[key] = created
    while len(_engine_cache) > MAX_CACHED_ENGINES:
        _, evicted = _engine_cache.popitem(last=False)
        paddle_ocr_image.close_ocr_engine(evicted)
    return created


def option_float(options: dict[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise paddle_ocr_image.InputError(f"OCR option {key} must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise paddle_ocr_image.InputError(f"OCR option {key} must be a number.") from exc


def option_int(options: dict[str, Any], key: str, default: int) -> int:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise paddle_ocr_image.InputError(f"OCR option {key} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise paddle_ocr_image.InputError(f"OCR option {key} must be an integer.") from exc


def run_request(request: OCRRequest, image_path: Path) -> list[dict[str, Any]]:
    engine_name = paddle_ocr_image.normalize_ocr_engine(request.engine)
    creation_options = engine_options(request.options)
    with _engine_lock:
        ocr = cached_engine(engine_name, creation_options)
        min_score = option_float(
            request.options,
            "min_score",
            paddle_ocr_image.DEFAULT_MIN_SCORE,
        )
        if engine_name == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            return paddle_ocr_image.extract_paddleocr_vl_image_records(
                ocr,
                image_path,
                request.page,
                min_score,
            )
        return paddle_ocr_image.extract_image_records(
            ocr,
            image_path,
            request.page,
            min_score,
            tile_enabled=bool(request.options.get("tile_enabled", False)),
            tile_width=option_int(
                request.options,
                "tile_width",
                paddle_ocr_image.DEFAULT_TILE_WIDTH,
            ),
            tile_height=option_int(
                request.options,
                "tile_height",
                paddle_ocr_image.DEFAULT_TILE_HEIGHT,
            ),
            tile_overlap=option_int(
                request.options,
                "tile_overlap",
                paddle_ocr_image.DEFAULT_TILE_OVERLAP,
            ),
            tile_include_full_image=bool(
                request.options.get(
                    "tile_include_full_image",
                    paddle_ocr_image.DEFAULT_TILE_INCLUDE_FULL_IMAGE,
                )
            ),
            tile_dedupe_iou=option_float(
                request.options,
                "tile_dedupe_iou",
                paddle_ocr_image.DEFAULT_TILE_DEDUPE_IOU,
            ),
            tile_dedupe_containment=option_float(
                request.options,
                "tile_dedupe_containment",
                paddle_ocr_image.DEFAULT_TILE_DEDUPE_CONTAINMENT,
            ),
        )


def close_engines() -> None:
    with _engine_lock:
        while _engine_cache:
            _, engine = _engine_cache.popitem()
            paddle_ocr_image.close_ocr_engine(engine)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    close_engines()


app = FastAPI(
    title="tetolate PaddleOCR Worker",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
def ocr(request: OCRRequest) -> dict[str, list[dict[str, Any]]]:
    image_data = decode_image(request.image_base64)
    suffix = image_suffix(image_data)
    try:
        with tempfile.TemporaryDirectory(
            prefix="tetolate_ocr_worker_"
        ) as temp_dir_name:
            image_path = Path(temp_dir_name) / f"input{suffix}"
            image_path.write_bytes(image_data)
            records = run_request(request, image_path)
    except paddle_ocr_image.InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"OCR worker file error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PaddleOCR worker failed: {type(exc).__name__}: {exc}",
        ) from exc
    return {"records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve tetolate PaddleOCR over a private HTTP API."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.port <= 0 or args.port > 65535:
        raise SystemExit("error: --port must be between 1 and 65535")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
