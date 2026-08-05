#!/usr/bin/env python3
"""Run PaddleOCR on one image and optionally draw detected bounding boxes."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_COLOR = "#ff2d55"
DEFAULT_WIDTH = 3
DEFAULT_FONT_SIZE = 18
DEFAULT_LANG = "japan"
DEFAULT_DEVICE = "cpu"
DEFAULT_PADDLEX_CACHE = Path(__file__).resolve().parent / ".paddlex-cache"
OCR_ENGINE_PADDLE = "paddle"
OCR_ENGINE_PADDLEOCR_VL = "paddleocr_vl"
OCR_ENGINE_ALIASES = {
    "paddle": OCR_ENGINE_PADDLE,
    "paddleocr": OCR_ENGINE_PADDLE,
    "ppocr": OCR_ENGINE_PADDLE,
    "paddleocr_vl": OCR_ENGINE_PADDLEOCR_VL,
    "paddleocr-vl": OCR_ENGINE_PADDLEOCR_VL,
    "paddleocr_vl_1_6": OCR_ENGINE_PADDLEOCR_VL,
    "paddleocr-vl-1.6": OCR_ENGINE_PADDLEOCR_VL,
    "vl": OCR_ENGINE_PADDLEOCR_VL,
}
OCR_ENGINES = frozenset((OCR_ENGINE_PADDLE, OCR_ENGINE_PADDLEOCR_VL))
DEFAULT_OCR_ENGINE = OCR_ENGINE_PADDLEOCR_VL
DEFAULT_PADDLEOCR_VL_PIPELINE_VERSION = "v1.6"
DEFAULT_PADDLEOCR_VL_BACKEND = "llama-cpp-server"
DEFAULT_OCR_SERVICE_URL = os.environ.get("TETOLATE_OCR_SERVICE_URL") or None
DEFAULT_PADDLEOCR_VL_SERVER_URL = os.environ.get(
    "TETOLATE_PADDLEOCR_VL_SERVER_URL",
    "http://127.0.0.1:8081/v1",
)
DEFAULT_PADDLEOCR_VL_MODEL = os.environ.get(
    "TETOLATE_PADDLEOCR_VL_MODEL",
    "PaddlePaddle/PaddleOCR-VL-1.6",
)
DEFAULT_PADDLEOCR_VL_API_KEY = (
    os.environ.get("TETOLATE_PADDLEOCR_VL_API_KEY") or None
)
DEFAULT_OCR_SERVICE_TIMEOUT = 3600.0
DEFAULT_TEXT_DET_LIMIT_SIDE_LEN = None
DEFAULT_TEXT_DET_LIMIT_TYPE = None
DEFAULT_MIN_SCORE = 0.75
DEFAULT_OCR_VERSION = None
DEFAULT_TEXT_DET_THRESH = None
DEFAULT_TEXT_DET_BOX_THRESH = None
DEFAULT_TEXT_DET_UNCLIP_RATIO = None
DEFAULT_TEXT_REC_SCORE_THRESH = None
DEFAULT_TILE_ENABLED = False
DEFAULT_TILE_WIDTH = 750
DEFAULT_TILE_HEIGHT = 750
DEFAULT_TILE_OVERLAP = 256
DEFAULT_TILE_INCLUDE_FULL_IMAGE = True
DEFAULT_TILE_DEDUPE_IOU = 0.55
DEFAULT_TILE_DEDUPE_CONTAINMENT = 0.88
DEFAULT_TILE_EDGE_MARGIN = 8
PADDLEOCR_VL_SKIP_LABELS = frozenset(
    (
        "image",
        "header_image",
        "footer_image",
    )
)


class InputError(ValueError):
    """Raised when inputs or OCR results cannot be used."""


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class OCRServiceClient:
    base_url: str
    engine: str
    timeout: float
    options: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on an image and write OCR JSON records.",
    )
    parser.add_argument("input_image", type=Path, help="Image to OCR")
    parser.add_argument("output_json", type=Path, help="OCR JSON file to write")
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="Page number to assign to OCR records. Default: 0",
    )
    parser.add_argument(
        "--engine",
        choices=tuple(sorted(OCR_ENGINES)),
        default=DEFAULT_OCR_ENGINE,
        help=(
            "OCR engine to use. 'paddle' uses the normal PaddleOCR text detector; "
            "'paddleocr_vl' uses PaddleOCR-VL 1.6 through a llama.cpp server. "
            f"Default: {DEFAULT_OCR_ENGINE}"
        ),
    )
    parser.add_argument(
        "--paddleocr-vl-server-url",
        default=DEFAULT_PADDLEOCR_VL_SERVER_URL,
        help=(
            "OpenAI-compatible llama.cpp URL for PaddleOCR-VL recognition. "
            f"Default: {DEFAULT_PADDLEOCR_VL_SERVER_URL}"
        ),
    )
    parser.add_argument(
        "--paddleocr-vl-model",
        default=DEFAULT_PADDLEOCR_VL_MODEL,
        help=f"PaddleOCR-VL model name sent to the llama.cpp server. Default: {DEFAULT_PADDLEOCR_VL_MODEL}",
    )
    parser.add_argument(
        "--paddleocr-vl-api-key",
        default="",
        help="Optional API key for the PaddleOCR-VL llama.cpp/OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--paddleocr-vl-max-concurrency",
        type=int,
        default=None,
        help="Optional maximum concurrent PaddleOCR-VL recognition requests.",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"PaddleOCR language code. Default: {DEFAULT_LANG}",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=(
            "PaddleOCR device, such as cpu, gpu, or gpu:0. "
            f"Default: {DEFAULT_DEVICE}"
        ),
    )
    parser.add_argument(
        "--text-det-limit-side-len",
        type=int,
        default=DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
        help="PaddleOCR text detection side-length limit. Default: PaddleOCR default",
    )
    parser.add_argument(
        "--text-det-limit-type",
        choices=("min", "max"),
        default=DEFAULT_TEXT_DET_LIMIT_TYPE,
        help="PaddleOCR text detection limit type. Default: PaddleOCR default",
    )
    parser.add_argument(
        "--use-doc-preprocessor",
        action="store_true",
        help="Reserved for future coordinate remapping; currently rejected for box output.",
    )
    parser.add_argument(
        "--use-textline-orientation",
        action="store_true",
        help="Enable PaddleOCR textline orientation classification.",
    )
    parser.add_argument(
        "--ocr-version",
        default=DEFAULT_OCR_VERSION,
        help="PaddleOCR model version, such as PP-OCRv6 or PP-OCRv5.",
    )
    parser.add_argument(
        "--text-detection-model-name",
        default=None,
        help="PaddleOCR 3.x text detection model name, such as PP-OCRv6_medium_det.",
    )
    parser.add_argument(
        "--text-recognition-model-name",
        default=None,
        help="PaddleOCR 3.x text recognition model name, such as PP-OCRv6_medium_rec.",
    )
    parser.add_argument(
        "--text-detection-model-dir",
        default=None,
        help="Local directory for a PaddleOCR text detection model.",
    )
    parser.add_argument(
        "--text-recognition-model-dir",
        default=None,
        help="Local directory for a PaddleOCR text recognition model.",
    )
    parser.add_argument(
        "--text-det-thresh",
        type=float,
        default=DEFAULT_TEXT_DET_THRESH,
        help="PaddleOCR text detector pixel threshold. Lower can improve recall.",
    )
    parser.add_argument(
        "--text-det-box-thresh",
        type=float,
        default=DEFAULT_TEXT_DET_BOX_THRESH,
        help="PaddleOCR text detector box threshold. Lower can improve recall.",
    )
    parser.add_argument(
        "--text-det-unclip-ratio",
        type=float,
        default=DEFAULT_TEXT_DET_UNCLIP_RATIO,
        help="PaddleOCR text detector box expansion ratio.",
    )
    parser.add_argument(
        "--text-rec-score-thresh",
        type=float,
        default=DEFAULT_TEXT_REC_SCORE_THRESH,
        help="PaddleOCR recognition score threshold before this script's --min-score filter.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Drop OCR records below this recognition score. Default: {DEFAULT_MIN_SCORE}",
    )
    parser.add_argument(
        "--tile-enabled",
        action="store_true",
        help="Run OCR on overlapping tiles, optionally with a full-image pass.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=DEFAULT_TILE_WIDTH,
        help=f"Tiled OCR crop width in pixels. Default: {DEFAULT_TILE_WIDTH}",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=DEFAULT_TILE_HEIGHT,
        help=f"Tiled OCR crop height in pixels. Default: {DEFAULT_TILE_HEIGHT}",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=DEFAULT_TILE_OVERLAP,
        help=f"Overlap between adjacent OCR tiles in pixels. Default: {DEFAULT_TILE_OVERLAP}",
    )
    parser.add_argument(
        "--tile-include-full-image",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TILE_INCLUDE_FULL_IMAGE,
        help="Include one full-image OCR pass in addition to tiles. Default: enabled.",
    )
    parser.add_argument(
        "--tile-dedupe-iou",
        type=float,
        default=DEFAULT_TILE_DEDUPE_IOU,
        help=f"IoU threshold for deduplicating tiled OCR boxes. Default: {DEFAULT_TILE_DEDUPE_IOU}",
    )
    parser.add_argument(
        "--tile-dedupe-containment",
        type=float,
        default=DEFAULT_TILE_DEDUPE_CONTAINMENT,
        help=(
            "Intersection-over-smaller-box threshold for deduplicating tiled OCR boxes. "
            f"Default: {DEFAULT_TILE_DEDUPE_CONTAINMENT}"
        ),
    )
    parser.add_argument(
        "--annotated-image",
        type=Path,
        help="Optional image output with PaddleOCR boxes and boxno labels drawn.",
    )
    parser.add_argument(
        "--color",
        default=DEFAULT_COLOR,
        help=f"Box and label color for --annotated-image. Default: {DEFAULT_COLOR}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Bounding box stroke width in pixels. Default: {DEFAULT_WIDTH}",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_FONT_SIZE,
        help=f"Label font size in pixels. Default: {DEFAULT_FONT_SIZE}",
    )
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.page < 0:
        raise InputError("--page must be zero or greater.")
    if not args.input_image.exists():
        raise InputError(f"Input image not found: {args.input_image}")
    if args.input_image.resolve() == args.output_json.resolve():
        raise InputError("Output JSON must be different from the input image.")
    if args.annotated_image and args.input_image.resolve() == args.annotated_image.resolve():
        raise InputError("Annotated image must be different from the input image.")
    if args.annotated_image and args.output_json.resolve() == args.annotated_image.resolve():
        raise InputError("Annotated image must be different from the output JSON.")
    if args.width <= 0:
        raise InputError("--width must be greater than zero.")
    if args.font_size <= 0:
        raise InputError("--font-size must be greater than zero.")
    if args.text_det_limit_side_len is not None and args.text_det_limit_side_len <= 0:
        raise InputError("--text-det-limit-side-len must be greater than zero.")
    if not math.isfinite(args.min_score) or args.min_score < 0 or args.min_score > 1:
        raise InputError("--min-score must be between 0 and 1.")
    if args.tile_width <= 0:
        raise InputError("--tile-width must be greater than zero.")
    if args.tile_height <= 0:
        raise InputError("--tile-height must be greater than zero.")
    if args.tile_overlap < 0:
        raise InputError("--tile-overlap must be zero or greater.")
    if args.tile_overlap >= min(args.tile_width, args.tile_height):
        raise InputError("--tile-overlap must be smaller than both tile dimensions.")
    if not math.isfinite(args.tile_dedupe_iou) or args.tile_dedupe_iou <= 0 or args.tile_dedupe_iou > 1:
        raise InputError("--tile-dedupe-iou must be greater than 0 and at most 1.")
    if (
        not math.isfinite(args.tile_dedupe_containment)
        or args.tile_dedupe_containment <= 0
        or args.tile_dedupe_containment > 1
    ):
        raise InputError("--tile-dedupe-containment must be greater than 0 and at most 1.")
    for option in (
        "text_det_thresh",
        "text_det_box_thresh",
        "text_det_unclip_ratio",
    ):
        value = getattr(args, option)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise InputError(f"--{option.replace('_', '-')} must be greater than zero.")
    if args.text_rec_score_thresh is not None and (
        not math.isfinite(args.text_rec_score_thresh) or args.text_rec_score_thresh < 0
    ):
        raise InputError("--text-rec-score-thresh must be zero or greater.")
    if args.paddleocr_vl_max_concurrency is not None and args.paddleocr_vl_max_concurrency <= 0:
        raise InputError("--paddleocr-vl-max-concurrency must be positive when provided.")


def normalize_ocr_engine(value: Any) -> str:
    engine = str(value or DEFAULT_OCR_ENGINE).strip().lower()
    normalized = OCR_ENGINE_ALIASES.get(engine)
    if normalized is None:
        raise InputError(
            "OCR engine must be one of: "
            + ", ".join(sorted(OCR_ENGINES))
            + "."
        )
    return normalized


def normalize_ocr_service_url(value: Any) -> str | None:
    service_url = str(value or "").strip().rstrip("/")
    if not service_url:
        return None
    parsed = urllib.parse.urlsplit(service_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InputError("OCR service URL must be an absolute HTTP or HTTPS URL.")
    if parsed.query or parsed.fragment:
        raise InputError("OCR service URL must not contain a query string or fragment.")
    return service_url


def create_ocr_service_client(
    service_url: str,
    engine: str,
    timeout: float,
    options: dict[str, Any],
) -> OCRServiceClient:
    normalized_url = normalize_ocr_service_url(service_url)
    if normalized_url is None:
        raise InputError("OCR service URL is required for remote OCR.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise InputError("OCR service timeout must be greater than zero.")
    return OCRServiceClient(
        base_url=normalized_url,
        engine=normalize_ocr_engine(engine),
        timeout=float(timeout),
        options=dict(options),
    )


def service_ocr_records(
    client: OCRServiceClient,
    input_image: Path,
    page: int,
    min_score: float,
    extraction_options: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        image_data = base64.b64encode(input_image.read_bytes()).decode("ascii")
    except OSError as exc:
        raise InputError(f"Failed to read OCR input image {input_image}: {exc}") from exc

    payload = {
        "imageBase64": image_data,
        "page": page,
        "engine": client.engine,
        "options": {**client.options, **extraction_options, "min_score": min_score},
    }
    request = urllib.request.Request(
        f"{client.base_url}/ocr",
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace").strip()
        raise InputError(
            f"OCR service returned HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise InputError(f"OCR service request failed: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise InputError(f"OCR service request failed: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("OCR service returned invalid JSON.") from exc

    records = response_data.get("records") if isinstance(response_data, dict) else None
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise InputError("OCR service response must contain a records array.")
    return records


def ensure_paddlex_cache_home() -> None:
    cache_home = os.environ.get("PADDLE_PDX_CACHE_HOME")
    if cache_home:
        cache_path = Path(cache_home)
    else:
        cache_path = DEFAULT_PADDLEX_CACHE
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        probe_path = cache_path / ".write_test"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
    except OSError:
        cache_path = Path(tempfile.gettempdir()) / "paddlex_cache"
        cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_path)
    os.environ.setdefault("PADDLEX_HOME", str(cache_path))
    temp_dir = Path(os.environ.get("PADDLEX_TEMP_DIR") or cache_path / "temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLEX_TEMP_DIR", str(temp_dir))


def configure_paddlex_runtime(device: str) -> None:
    ensure_paddlex_cache_home()
    if not device or device.lower().split(":", 1)[0] == "cpu":
        # PaddlePaddle 3.3.1 + PP-OCRv6 currently fails on some CPU paths when
        # PaddleX enables oneDNN/MKLDNN by default. Force this off for tetolate CPU
        # OCR; a running web process may already have imported PaddleX, so patch
        # its cached flag too when present.
        os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
        try:
            flags = importlib.import_module("paddlex.utils.flags")
        except Exception:
            pass
        else:
            try:
                setattr(flags, "ENABLE_MKLDNN_BYDEFAULT", False)
            except Exception:
                pass
        try:
            pp_option = importlib.import_module(
                "paddlex.inference.models.runners.paddle_static.config.pp_option"
            )
        except Exception:
            return
        try:
            setattr(pp_option, "ENABLE_MKLDNN_BYDEFAULT", False)
        except Exception:
            pass


def add_optional(kwargs: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        kwargs[key] = value


def create_paddle_ocr(
    lang: str,
    device: str,
    text_det_limit_side_len: int | None,
    text_det_limit_type: str | None,
    use_doc_preprocessor: bool,
    use_textline_orientation: bool,
    *,
    ocr_version: str | None = None,
    text_detection_model_name: str | None = None,
    text_recognition_model_name: str | None = None,
    text_detection_model_dir: str | None = None,
    text_recognition_model_dir: str | None = None,
    text_det_thresh: float | None = None,
    text_det_box_thresh: float | None = None,
    text_det_unclip_ratio: float | None = None,
    text_rec_score_thresh: float | None = None,
    service_url: str | None = None,
    service_timeout: float = DEFAULT_OCR_SERVICE_TIMEOUT,
) -> Any:
    if use_doc_preprocessor:
        raise InputError(
            "Document orientation/unwarping preprocessing is not supported for region output "
            "because PaddleOCR returns boxes in transformed image coordinates."
        )
    kwargs: dict[str, Any] = {
        "lang": lang,
        "engine": "paddle",
        "use_doc_orientation_classify": use_doc_preprocessor,
        "use_doc_unwarping": use_doc_preprocessor,
        "use_textline_orientation": use_textline_orientation,
    }
    if device:
        kwargs["device"] = device

    add_optional(kwargs, "text_det_limit_side_len", text_det_limit_side_len)
    add_optional(kwargs, "text_det_limit_type", text_det_limit_type)
    add_optional(kwargs, "ocr_version", ocr_version)
    add_optional(kwargs, "text_detection_model_name", text_detection_model_name)
    add_optional(kwargs, "text_recognition_model_name", text_recognition_model_name)
    add_optional(kwargs, "text_detection_model_dir", text_detection_model_dir)
    add_optional(kwargs, "text_recognition_model_dir", text_recognition_model_dir)
    add_optional(kwargs, "text_det_thresh", text_det_thresh)
    add_optional(kwargs, "text_det_box_thresh", text_det_box_thresh)
    add_optional(kwargs, "text_det_unclip_ratio", text_det_unclip_ratio)
    add_optional(kwargs, "text_rec_score_thresh", text_rec_score_thresh)

    if normalize_ocr_service_url(service_url) is not None:
        return create_ocr_service_client(
            str(service_url),
            OCR_ENGINE_PADDLE,
            service_timeout,
            kwargs,
        )

    configure_paddlex_runtime(device)

    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise InputError(
            "PaddleOCR is not installed for this Python. Install the project's 'ocr' extra "
            "or use --ocr-service-url/TETOLATE_OCR_SERVICE_URL with a PaddleOCR worker."
        ) from exc

    try:
        return PaddleOCR(**kwargs)
    except ValueError as exc:
        raise InputError(f"PaddleOCR rejected the configured OCR options: {exc}") from exc
    except TypeError as exc:
        raise InputError(
            "The installed PaddleOCR does not support the configured 3.x API options. "
            "Install the project versions from uv.lock."
        ) from exc


def create_paddleocr_vl(
    device: str,
    server_url: str = DEFAULT_PADDLEOCR_VL_SERVER_URL,
    model_name: str = DEFAULT_PADDLEOCR_VL_MODEL,
    *,
    api_key: str | None = None,
    max_concurrency: int | None = None,
    service_url: str | None = None,
    service_timeout: float = DEFAULT_OCR_SERVICE_TIMEOUT,
) -> Any:
    kwargs: dict[str, Any] = {
        "pipeline_version": DEFAULT_PADDLEOCR_VL_PIPELINE_VERSION,
        "vl_rec_backend": DEFAULT_PADDLEOCR_VL_BACKEND,
        "vl_rec_server_url": server_url,
        "vl_rec_api_model_name": model_name,
    }
    if device:
        kwargs["device"] = device
    add_optional(kwargs, "vl_rec_api_key", api_key)
    add_optional(kwargs, "vl_rec_max_concurrency", max_concurrency)

    if normalize_ocr_service_url(service_url) is not None:
        return create_ocr_service_client(
            str(service_url),
            OCR_ENGINE_PADDLEOCR_VL,
            service_timeout,
            kwargs,
        )

    configure_paddlex_runtime(device)

    try:
        from paddleocr import PaddleOCRVL
    except ModuleNotFoundError as exc:
        raise InputError(
            "PaddleOCR-VL is not installed for this Python. Install the project's 'ocr' "
            "extra or use --ocr-service-url/TETOLATE_OCR_SERVICE_URL with a "
            "PaddleOCR worker."
        ) from exc
    except ImportError as exc:
        raise InputError(
            "The installed PaddleOCR does not expose PaddleOCRVL. Install a PaddleOCR "
            "release with doc_parser/PaddleOCR-VL support."
        ) from exc

    try:
        return PaddleOCRVL(**kwargs)
    except ValueError as exc:
        raise InputError(f"PaddleOCR-VL rejected the configured options: {exc}") from exc
    except TypeError as exc:
        raise InputError(
            "The installed PaddleOCR rejected the PaddleOCR-VL options. "
            "Check that paddleocr[doc-parser] is installed and up to date."
        ) from exc


def run_paddle_ocr(ocr: Any, input_image: Path) -> Any:
    if not hasattr(ocr, "predict"):
        raise InputError("Unsupported PaddleOCR object: no predict() method found.")
    return ocr.predict(str(input_image))


def run_paddleocr_vl(ocr: Any, input_image: Path) -> Any:
    if not hasattr(ocr, "predict"):
        raise InputError("Unsupported PaddleOCR-VL object: no predict() method found.")
    return ocr.predict(str(input_image))


def close_ocr_engine(ocr: Any) -> None:
    close = getattr(ocr, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        print(f"warning: failed to close OCR engine: {exc}", file=sys.stderr)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return value


def result_to_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result
    else:
        data: dict[str, Any] = {}
        if hasattr(result, "to_dict"):
            try:
                maybe_data = result.to_dict()
                if isinstance(maybe_data, dict):
                    data = maybe_data
            except Exception:
                data = {}
        if not data and hasattr(result, "json"):
            try:
                maybe_json = result.json() if callable(result.json) else result.json
                if isinstance(maybe_json, dict):
                    data = maybe_json
            except Exception:
                data = {}
        if not data:
            try:
                maybe_data = dict(result)
                if isinstance(maybe_data, dict):
                    data = maybe_data
            except Exception:
                data = {}
        if not data and hasattr(result, "res") and isinstance(result.res, dict):
            data = result.res
        if not data and hasattr(result, "__dict__"):
            data = vars(result)

    if "res" in data and isinstance(data["res"], dict):
        return data["res"]
    return data


def first_present(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    value = json_safe(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def polygon_from_box(box: list[Any]) -> list[list[float]] | None:
    if len(box) != 4:
        return None
    try:
        left, top, right, bottom = [float(item) for item in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def normalize_polygon(value: Any) -> list[list[float]] | None:
    value = json_safe(value)
    if isinstance(value, list) and len(value) == 4 and all(
        isinstance(item, (int, float)) for item in value
    ):
        return polygon_from_box(value)
    if not isinstance(value, list):
        return None

    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append([x, y])
    return points if len(points) >= 2 else None


def numeric_size_value(value: Any) -> int | None:
    value = json_safe(value)
    if isinstance(value, list) and value:
        value = value[0]
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def scale_polygon(
    polygon: list[list[float]],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[list[float]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0 or source_size == target_size:
        return polygon
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [[point[0] * scale_x, point[1] * scale_y] for point in polygon]


def bbox_from_polygon(polygon: list[list[float]], image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = max(0, min(width, math.floor(min(xs))))
    top = max(0, min(height, math.floor(min(ys))))
    right = max(0, min(width, math.ceil(max(xs))))
    bottom = max(0, min(height, math.ceil(max(ys))))
    return [left, top, right, bottom]


def keep_score(score: float | None, min_score: float) -> bool:
    return score is None or score >= min_score


def score_at(scores: list[Any], index: int) -> float | None:
    if index >= len(scores):
        return None
    try:
        score = float(scores[index])
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def ocr_record(
    page: int,
    text: str,
    score: float | None,
    polygon: list[list[float]],
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    region = bbox_from_polygon(polygon, image_size)
    if region[2] <= region[0] or region[3] <= region[1]:
        return None

    record: dict[str, Any] = {
        "page": page,
        "boxno": 0,
        "region": region,
        "points": polygon,
        "sfx": False,
        "openLettering": False,
        "text": text,
    }
    if score is not None:
        record["score"] = score
    return record


def normalize_predict_results(
    raw_results: Any,
    page: int,
    min_score: float,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    result_items = raw_results if isinstance(raw_results, (list, tuple)) else [raw_results]
    records: list[dict[str, Any]] = []

    for result_item in result_items:
        result_data = result_to_mapping(result_item)
        if not result_data:
            continue

        texts = as_list(result_data.get("rec_texts"))
        scores = as_list(result_data.get("rec_scores"))
        polygons = as_list(result_data.get("rec_polys"))
        boxes = as_list(result_data.get("rec_boxes"))
        detection_count = max(len(texts), len(polygons), len(boxes), len(scores))

        for local_index in range(detection_count):
            text = str(texts[local_index]) if local_index < len(texts) else ""
            score = score_at(scores, local_index)
            if not keep_score(score, min_score):
                continue

            polygon = (
                normalize_polygon(polygons[local_index])
                if local_index < len(polygons)
                else None
            )
            if polygon is None and local_index < len(boxes):
                polygon = normalize_polygon(boxes[local_index])
            if polygon is None:
                continue

            record = ocr_record(page, text, score, polygon, image_size)
            if record is not None:
                records.append(record)

    return renumber_boxnos(records)


def extract_records(
    result: Any,
    page: int,
    min_score: float,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    records = normalize_predict_results(result, page, min_score, image_size)
    if records:
        return records
    if recognized_empty_or_filtered_result(result):
        return []
    raise InputError(
        "PaddleOCR returned an unsupported result structure; refusing to treat it as empty OCR."
    )


def recognized_empty_or_filtered_result(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, (list, tuple)) and not result:
        return True
    if isinstance(result, (list, tuple)) and all(
        item is None or (isinstance(item, (list, tuple)) and not item)
        for item in result
    ):
        return True

    items = result if isinstance(result, (list, tuple)) else [result]
    known_keys = {
        "rec_texts",
        "rec_scores",
        "rec_polys",
        "rec_boxes",
    }
    for item in items:
        mapping = result_to_mapping(item)
        if any(key in mapping for key in known_keys):
            return True

    return False


def scale_ocr_record(
    record: dict[str, Any],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> dict[str, Any]:
    if source_size == target_size:
        return dict(record)
    scaled = dict(record)
    points = record.get("points")
    if isinstance(points, list):
        scaled_points = scale_polygon(points, source_size, target_size)
        scaled["points"] = scaled_points
        scaled["region"] = bbox_from_polygon(scaled_points, target_size)
    else:
        left, top, right, bottom = record["region"]
        source_width, source_height = source_size
        target_width, target_height = target_size
        scale_x = target_width / source_width if source_width > 0 else 1.0
        scale_y = target_height / source_height if source_height > 0 else 1.0
        scaled["region"] = [
            round(left * scale_x),
            round(top * scale_y),
            round(right * scale_x),
            round(bottom * scale_y),
        ]
    return scaled


def paddleocr_vl_block_to_dict(block: Any) -> dict[str, Any] | None:
    block = json_safe(block)
    if isinstance(block, dict):
        return block
    if not hasattr(block, "__dict__"):
        return None

    data = vars(block)
    converted: dict[str, Any] = {
        "block_label": data.get("label"),
        "block_content": data.get("content"),
        "block_bbox": data.get("bbox"),
    }
    polygon_points = data.get("polygon_points")
    if polygon_points is not None:
        converted["block_polygon_points"] = polygon_points
    group_id = data.get("group_id")
    if group_id is not None:
        converted["group_id"] = group_id
    global_block_id = data.get("global_block_id")
    if global_block_id is not None:
        converted["global_block_id"] = global_block_id
    global_group_id = data.get("global_group_id")
    if global_group_id is not None:
        converted["global_group_id"] = global_group_id
    return json_safe(converted)


def block_text(block: dict[str, Any]) -> str:
    value = first_present(
        block,
        (
            "block_content",
            "content",
            "text",
            "rec_text",
            "markdown",
        ),
    )
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value if item is not None)
    return str(value).strip()


def block_score(block: dict[str, Any]) -> float | None:
    value = first_present(block, ("score", "block_score", "confidence", "rec_score"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def block_polygon(block: dict[str, Any]) -> list[list[float]] | None:
    polygon = normalize_polygon(
        first_present(
            block,
            (
                "block_polygon_points",
                "polygon_points",
                "points",
                "poly",
                "polygon",
            ),
        )
    )
    if polygon is not None:
        return polygon
    return normalize_polygon(
        first_present(
            block,
            (
                "block_bbox",
                "bbox",
                "box",
                "coordinate",
                "region",
            ),
        )
    )


def normalize_paddleocr_vl_results(
    raw_results: Any,
    page: int,
    min_score: float,
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    result_items = raw_results if isinstance(raw_results, (list, tuple)) else [raw_results]
    records: list[dict[str, Any]] = []

    for result_item in result_items:
        result_data = result_to_mapping(result_item)
        if not result_data:
            continue

        result_width = numeric_size_value(result_data.get("width")) or image_size[0]
        result_height = numeric_size_value(result_data.get("height")) or image_size[1]
        result_size = (result_width, result_height)
        spotting_res = result_data.get("spotting_res")
        if spotting_res:
            spotting_items = spotting_res if isinstance(spotting_res, list) else [spotting_res]
            for spotting_item in spotting_items:
                spotting_records = normalize_predict_results(
                    spotting_item,
                    page,
                    min_score,
                    result_size,
                )
                for record in spotting_records:
                    scaled = scale_ocr_record(record, result_size, image_size)
                    scaled["ocrSource"] = OCR_ENGINE_PADDLEOCR_VL
                    scaled["ocrBlockLabel"] = "spotting"
                    records.append(scaled)

        blocks = result_data.get("parsing_res_list")
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            block = paddleocr_vl_block_to_dict(block)
            if block is None:
                continue

            label = str(block.get("block_label") or block.get("label") or "").strip()
            if label.lower() in PADDLEOCR_VL_SKIP_LABELS:
                continue

            text = block_text(block)
            if not text:
                continue

            score = block_score(block)
            if not keep_score(score, min_score):
                continue

            polygon = block_polygon(block)
            if polygon is None:
                continue
            polygon = scale_polygon(polygon, result_size, image_size)
            record = ocr_record(page, text, score, polygon, image_size)
            if record is None:
                continue
            record["ocrSource"] = OCR_ENGINE_PADDLEOCR_VL
            if label:
                record["ocrBlockLabel"] = label
            block_id = block.get("block_id")
            if isinstance(block_id, int):
                record["ocrBlockId"] = block_id
            records.append(record)

    return dedupe_ocr_records(records, DEFAULT_TILE_DEDUPE_IOU, DEFAULT_TILE_DEDUPE_CONTAINMENT)


def extract_paddleocr_vl_image_records(
    ocr: Any,
    input_image: Path,
    page: int,
    min_score: float,
) -> list[dict[str, Any]]:
    if isinstance(ocr, OCRServiceClient):
        return service_ocr_records(ocr, input_image, page, min_score, {})
    with Image.open(input_image) as image:
        image_size = image.size
    return normalize_paddleocr_vl_results(
        run_paddleocr_vl(ocr, input_image),
        page,
        min_score,
        image_size,
    )


def renumber_boxnos(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for boxno, record in enumerate(records):
        record["boxno"] = boxno
    return records


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    if stride <= 0:
        raise InputError("Tile overlap must be smaller than tile size.")

    starts = [0]
    while starts[-1] + tile_size < length:
        next_start = min(starts[-1] + stride, length - tile_size)
        if next_start == starts[-1]:
            break
        starts.append(next_start)
    return starts


def image_tiles(width: int, height: int, tile_width: int, tile_height: int, overlap: int) -> list[Tile]:
    x_starts = tile_starts(width, tile_width, overlap)
    y_starts = tile_starts(height, tile_height, overlap)
    return [
        Tile(
            x=x,
            y=y,
            width=min(tile_width, width - x),
            height=min(tile_height, height - y),
        )
        for y in y_starts
        for x in x_starts
    ]


def tile_edge_touch(
    region: list[int],
    tile: Tile,
    image_width: int,
    image_height: int,
    margin: int = DEFAULT_TILE_EDGE_MARGIN,
) -> bool:
    left, top, right, bottom = region
    return (
        (tile.x > 0 and left <= tile.x + margin)
        or (tile.y > 0 and top <= tile.y + margin)
        or (tile.right < image_width and right >= tile.right - margin)
        or (tile.bottom < image_height and bottom >= tile.bottom - margin)
    )


def mark_source(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["ocrSource"] = source
        item["tileEdgeTouch"] = False
        marked.append(item)
    return marked


def offset_tile_record(
    record: dict[str, Any],
    tile: Tile,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    left, top, right, bottom = record["region"]
    shifted_region = [
        max(0, min(image_width, left + tile.x)),
        max(0, min(image_height, top + tile.y)),
        max(0, min(image_width, right + tile.x)),
        max(0, min(image_height, bottom + tile.y)),
    ]
    if shifted_region[2] <= shifted_region[0] or shifted_region[3] <= shifted_region[1]:
        return None

    shifted = dict(record)
    shifted["region"] = shifted_region
    shifted["points"] = [
        [point[0] + tile.x, point[1] + tile.y]
        for point in record.get("points", [])
    ]
    shifted["ocrSource"] = "tile"
    shifted["tile"] = {
        "x": tile.x,
        "y": tile.y,
        "width": tile.width,
        "height": tile.height,
    }
    shifted["tileEdgeTouch"] = tile_edge_touch(shifted_region, tile, image_width, image_height)
    return shifted


def region_area(region: list[int]) -> int:
    return max(0, region[2] - region[0]) * max(0, region[3] - region[1])


def intersection_area(a: list[int], b: list[int]) -> int:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def text_agrees(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_text = normalized_text(a.get("text"))
    b_text = normalized_text(b.get("text"))
    if not a_text or not b_text:
        return True
    return a_text == b_text or a_text in b_text or b_text in a_text


def duplicate_ocr_records(
    a: dict[str, Any],
    b: dict[str, Any],
    iou_threshold: float,
    containment_threshold: float,
) -> bool:
    a_region = a["region"]
    b_region = b["region"]
    overlap = intersection_area(a_region, b_region)
    if overlap <= 0:
        return False
    a_area = region_area(a_region)
    b_area = region_area(b_region)
    if min(a_area, b_area) <= 0:
        return False

    union = a_area + b_area - overlap
    iou = overlap / union if union else 0.0
    if iou >= iou_threshold:
        return True

    containment = overlap / min(a_area, b_area)
    return containment >= containment_threshold and text_agrees(a, b)


def score_value(record: dict[str, Any]) -> float:
    score = record.get("score")
    return float(score) if isinstance(score, (int, float)) else -1.0


def should_replace_record(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    existing_text = normalized_text(existing.get("text"))
    candidate_text = normalized_text(candidate.get("text"))
    if not existing_text and candidate_text:
        return True
    if existing_text and not candidate_text:
        return False
    existing_edge = bool(existing.get("tileEdgeTouch"))
    candidate_edge = bool(candidate.get("tileEdgeTouch"))
    if existing_edge != candidate_edge:
        return existing_edge and not candidate_edge
    existing_score = score_value(existing)
    candidate_score = score_value(candidate)
    if candidate_score > existing_score + 0.05:
        return True
    if candidate_score < existing_score:
        return False
    return len(candidate_text) > len(existing_text)


def dedupe_ocr_records(
    records: list[dict[str, Any]],
    iou_threshold: float,
    containment_threshold: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for record in records:
        duplicate_index = None
        for index, kept_record in enumerate(kept):
            if duplicate_ocr_records(record, kept_record, iou_threshold, containment_threshold):
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(record)
        elif should_replace_record(kept[duplicate_index], record):
            kept[duplicate_index] = record
    return renumber_boxnos(kept)


def extract_image_records(
    ocr: Any,
    input_image: Path,
    page: int,
    min_score: float,
    *,
    tile_enabled: bool = DEFAULT_TILE_ENABLED,
    tile_width: int = DEFAULT_TILE_WIDTH,
    tile_height: int = DEFAULT_TILE_HEIGHT,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
    tile_include_full_image: bool = DEFAULT_TILE_INCLUDE_FULL_IMAGE,
    tile_dedupe_iou: float = DEFAULT_TILE_DEDUPE_IOU,
    tile_dedupe_containment: float = DEFAULT_TILE_DEDUPE_CONTAINMENT,
) -> list[dict[str, Any]]:
    if isinstance(ocr, OCRServiceClient):
        return service_ocr_records(
            ocr,
            input_image,
            page,
            min_score,
            {
                "tile_enabled": tile_enabled,
                "tile_width": tile_width,
                "tile_height": tile_height,
                "tile_overlap": tile_overlap,
                "tile_include_full_image": tile_include_full_image,
                "tile_dedupe_iou": tile_dedupe_iou,
                "tile_dedupe_containment": tile_dedupe_containment,
            },
        )
    with Image.open(input_image) as image:
        source_image = image.convert("RGB")
    image_width, image_height = source_image.size

    full_records: list[dict[str, Any]] = []
    if not tile_enabled or tile_include_full_image:
        full_records = mark_source(
            extract_records(
                run_paddle_ocr(ocr, input_image),
                page,
                min_score,
                source_image.size,
            ),
            "full_image",
        )
    if not tile_enabled:
        return renumber_boxnos(full_records)

    tiles = image_tiles(image_width, image_height, tile_width, tile_height, tile_overlap)
    if full_records and len(tiles) == 1 and tiles[0].x == 0 and tiles[0].y == 0:
        return renumber_boxnos(full_records)

    records: list[dict[str, Any]] = list(full_records)
    with tempfile.TemporaryDirectory(prefix="paddle_ocr_tiles_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for tile_index, tile in enumerate(tiles):
            tile_path = temp_dir / f"tile_{tile_index:04d}_{tile.x}_{tile.y}.png"
            crop = source_image.crop((tile.x, tile.y, tile.right, tile.bottom))
            crop.save(tile_path, format="PNG")
            tile_records = extract_records(
                run_paddle_ocr(ocr, tile_path),
                page,
                min_score,
                crop.size,
            )
            for record in tile_records:
                shifted = offset_tile_record(record, tile, image_width, image_height)
                if shifted is not None:
                    records.append(shifted)

    return dedupe_ocr_records(records, tile_dedupe_iou, tile_dedupe_containment)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(records, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.chmod(temp_path, output_mode)
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise InputError(f"Failed to write OCR JSON {path}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    boxno: int,
    left: int,
    top: int,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    text = str(boxno)
    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text, font=font)
    text_width = text_right - text_left
    text_height = text_bottom - text_top
    padding = max(4, round(text_height * 0.2))
    label_width = text_width + padding * 2
    label_height = text_height + padding * 2
    label_top = top - label_height if top >= label_height else top

    draw.rectangle(
        (left, label_top, left + label_width, label_top + label_height),
        fill=color,
    )
    draw.text(
        (left + padding, label_top + padding - text_top),
        text,
        fill="white",
        font=font,
    )


def draw_boxes(
    records: list[dict[str, Any]],
    input_image: Path,
    output_image: Path,
    color: str,
    width: int,
    font_size: int,
) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    font = load_font(font_size)

    with Image.open(input_image) as image:
        annotated = image.convert("RGBA")

    draw = ImageDraw.Draw(annotated)
    for record in records:
        left, top, right, bottom = record["region"]
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=width)
        draw_label(draw, record["boxno"], left, top, color, font)

    if output_image.suffix.lower() in {".jpg", ".jpeg"}:
        annotated = annotated.convert("RGB")
    annotated.save(output_image)


def main() -> int:
    args = parse_args()
    ocr: Any = None

    try:
        check_args(args)
        engine = normalize_ocr_engine(args.engine)
        if engine == OCR_ENGINE_PADDLEOCR_VL:
            ocr = create_paddleocr_vl(
                args.device,
                args.paddleocr_vl_server_url,
                args.paddleocr_vl_model,
                api_key=args.paddleocr_vl_api_key,
                max_concurrency=args.paddleocr_vl_max_concurrency,
            )
            records = extract_paddleocr_vl_image_records(
                ocr,
                args.input_image,
                args.page,
                args.min_score,
            )
        else:
            ocr = create_paddle_ocr(
                args.lang,
                args.device,
                args.text_det_limit_side_len,
                args.text_det_limit_type,
                args.use_doc_preprocessor,
                args.use_textline_orientation,
                ocr_version=args.ocr_version,
                text_detection_model_name=args.text_detection_model_name,
                text_recognition_model_name=args.text_recognition_model_name,
                text_detection_model_dir=args.text_detection_model_dir,
                text_recognition_model_dir=args.text_recognition_model_dir,
                text_det_thresh=args.text_det_thresh,
                text_det_box_thresh=args.text_det_box_thresh,
                text_det_unclip_ratio=args.text_det_unclip_ratio,
                text_rec_score_thresh=args.text_rec_score_thresh,
            )
            records = extract_image_records(
                ocr,
                args.input_image,
                args.page,
                args.min_score,
                tile_enabled=args.tile_enabled,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                tile_overlap=args.tile_overlap,
                tile_include_full_image=args.tile_include_full_image,
                tile_dedupe_iou=args.tile_dedupe_iou,
                tile_dedupe_containment=args.tile_dedupe_containment,
            )
        write_json(args.output_json, records)
        if args.annotated_image:
            draw_boxes(
                records,
                args.input_image,
                args.annotated_image,
                args.color,
                args.width,
                args.font_size,
            )
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if ocr is not None:
            close_ocr_engine(ocr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
