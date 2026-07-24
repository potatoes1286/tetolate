#!/usr/bin/env python3
"""Minimal TorchScript LaMa inference used by tetolate's text cleanup stage."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from filelock import FileLock
from PIL import Image


DEFAULT_MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
DEFAULT_MODEL_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
DEFAULT_CROP_TRIGGER_SIZE = 800
DEFAULT_CROP_MARGIN = 128
PAD_MODULO = 8
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

Box = tuple[int, int, int, int]
_DOWNLOAD_LOCK = threading.Lock()


class LaMaError(RuntimeError):
    """Raised when the model cannot be loaded or an image cannot be inpainted."""


def default_model_path() -> Path:
    override = os.environ.get("TETOLATE_LAMA_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    return torch_home / "hub" / "checkpoints" / "big-lama.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def ensure_model(model_path: Path | None = None) -> Path:
    custom_path = model_path is not None or bool(
        os.environ.get("TETOLATE_LAMA_MODEL_PATH")
    )
    path = (model_path or default_model_path()).expanduser().resolve()
    if path.exists() and file_sha256(path) == DEFAULT_MODEL_SHA256:
        return path
    if custom_path:
        raise LaMaError(
            f"Configured LaMa model is missing or failed its SHA-256 check: {path}"
        )

    with _DOWNLOAD_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with FileLock(path.with_suffix(f"{path.suffix}.lock"), timeout=300):
                if path.exists() and file_sha256(path) == DEFAULT_MODEL_SHA256:
                    return path
                if path.exists():
                    path.unlink()

                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".download",
                    delete=False,
                ) as destination:
                    temp_path = Path(destination.name)
                    request = urllib.request.Request(
                        DEFAULT_MODEL_URL,
                        headers={"User-Agent": "tetolate/0.1 LaMa model downloader"},
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                            destination.write(chunk)
                if file_sha256(temp_path) != DEFAULT_MODEL_SHA256:
                    raise LaMaError("Downloaded LaMa model failed its SHA-256 check.")
                os.replace(temp_path, path)
                temp_path = None
        except (OSError, urllib.error.URLError) as exc:
            raise LaMaError(f"Could not download the LaMa model: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise LaMaError("LaMa device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise LaMaError("LaMa device mps was requested, but MPS is unavailable.")
        return torch.device("mps")
    if device_name != "cpu":
        raise LaMaError(f"Unsupported LaMa device: {device_name}")
    return torch.device("cpu")


@lru_cache(maxsize=3)
def load_model(model_path: str, device_name: str) -> torch.jit.ScriptModule:
    device = resolve_device(device_name)
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except (OSError, RuntimeError) as exc:
        raise LaMaError(f"Could not load LaMa model {model_path}: {exc}") from exc
    model.eval()
    return model


def ceil_modulo(value: int, modulo: int = PAD_MODULO) -> int:
    return ((value + modulo - 1) // modulo) * modulo


def pad_to_modulo(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    pad_height = ceil_modulo(height) - height
    pad_width = ceil_modulo(width) - width
    if array.ndim == 2:
        padding = ((0, pad_height), (0, pad_width))
    else:
        padding = ((0, pad_height), (0, pad_width), (0, 0))
    return np.pad(array, padding, mode="symmetric")


def boxes_connected(first: Box, second: Box) -> bool:
    return not (
        first[2] < second[0]
        or second[2] < first[0]
        or first[3] < second[1]
        or second[3] < first[1]
    )


def merge_connected_boxes(boxes: Sequence[Box]) -> list[Box]:
    groups = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    changed = True
    while changed:
        changed = False
        merged: list[Box] = []
        while groups:
            current = groups.pop()
            index = 0
            while index < len(groups):
                candidate = groups[index]
                if boxes_connected(current, candidate):
                    current = (
                        min(current[0], candidate[0]),
                        min(current[1], candidate[1]),
                        max(current[2], candidate[2]),
                        max(current[3], candidate[3]),
                    )
                    groups.pop(index)
                    changed = True
                    index = 0
                else:
                    index += 1
            merged.append(current)
        groups = merged
    return groups


def crop_with_margin(box: Box, width: int, height: int, margin: int) -> Box:
    left, top, right, bottom = box
    target_width = right - left + margin * 2
    target_height = bottom - top + margin * 2
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    raw_left = center_x - target_width // 2
    raw_right = center_x + target_width // 2
    raw_top = center_y - target_height // 2
    raw_bottom = center_y + target_height // 2

    crop_left = max(raw_left, 0)
    crop_right = min(raw_right, width)
    crop_top = max(raw_top, 0)
    crop_bottom = min(raw_bottom, height)

    if raw_left < 0:
        crop_right += abs(raw_left)
    if raw_right > width:
        crop_left -= raw_right - width
    if raw_top < 0:
        crop_bottom += abs(raw_top)
    if raw_bottom > height:
        crop_top -= raw_bottom - height

    return (
        max(crop_left, 0),
        max(crop_top, 0),
        min(crop_right, width),
        min(crop_bottom, height),
    )


def infer_crop(
    model: torch.jit.ScriptModule,
    device: torch.device,
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    original_height, original_width = image.shape[:2]
    padded_image = pad_to_modulo(image)
    padded_mask = pad_to_modulo(mask)

    image_data = np.transpose(padded_image, (2, 0, 1)).astype("float32") / 255.0
    mask_data = padded_mask[np.newaxis, :, :].astype("float32") / 255.0
    image_tensor = torch.from_numpy(image_data).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy((mask_data > 0).astype("int64")).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor, mask_tensor)
    result = prediction[0].permute(1, 2, 0).detach().cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype("uint8")
    result = result[:original_height, :original_width]

    keep = mask[:, :, np.newaxis] < 127
    return np.where(keep, image, result)


def inpaint_image(
    input_image: Path,
    mask_path: Path,
    output_image: Path,
    *,
    device_name: str = "cpu",
    model_path: Path | None = None,
    crop_boxes: Sequence[Box] = (),
    crop_trigger_size: int = DEFAULT_CROP_TRIGGER_SIZE,
    crop_margin: int = DEFAULT_CROP_MARGIN,
) -> None:
    if crop_trigger_size < 1:
        raise LaMaError("LaMa crop trigger size must be positive.")
    if crop_margin < 0:
        raise LaMaError("LaMa crop margin cannot be negative.")

    with Image.open(input_image) as source:
        image = np.asarray(source.convert("RGB")).copy()
    with Image.open(mask_path) as source_mask:
        mask_image = source_mask.convert("L")
        if mask_image.size != (image.shape[1], image.shape[0]):
            mask_image = mask_image.resize(
                (image.shape[1], image.shape[0]),
                Image.Resampling.NEAREST,
            )
        mask = np.asarray(mask_image).copy()
    mask = np.where(mask >= 127, 255, 0).astype("uint8")

    verified_model_path = ensure_model(model_path)
    device = resolve_device(device_name)
    model = load_model(str(verified_model_path), device_name)

    height, width = image.shape[:2]
    result = image.copy()
    if max(height, width) > crop_trigger_size and crop_boxes:
        for component in merge_connected_boxes(crop_boxes):
            left, top, right, bottom = crop_with_margin(
                component, width, height, crop_margin
            )
            result[top:bottom, left:right] = infer_crop(
                model,
                device,
                result[top:bottom, left:right],
                mask[top:bottom, left:right],
            )
    else:
        result = infer_crop(model, device, result, mask)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_image, format="PNG")
