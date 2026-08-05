#!/usr/bin/env python3
"""Clean text regions from an image using the bundled minimal LaMa runner."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

import lama_inpaint


DEFAULT_DEVICE = "cpu"
DEFAULT_PADDING = 4
DEFAULT_CROP_TRIGGER_SIZE = lama_inpaint.DEFAULT_CROP_TRIGGER_SIZE
DEFAULT_CROP_MARGIN = lama_inpaint.DEFAULT_CROP_MARGIN


class InputError(ValueError):
    """Raised when inputs cannot be used for inpainting."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use LaMa to clean text-box regions from an image.",
    )
    parser.add_argument("text_json", type=Path, help="JSON file describing text regions")
    parser.add_argument("input_image", type=Path, help="Source image")
    parser.add_argument("output_image", type=Path, help="Cleaned PNG file to write")
    parser.add_argument(
        "--page",
        type=int,
        help="Only clean entries for this page number.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=DEFAULT_PADDING,
        help=f"Pixels to expand each region before inpainting. Default: {DEFAULT_PADDING}",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=DEFAULT_DEVICE,
        help=f"Device for LaMa. Default: {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional local big-lama.pt path. The verified model is downloaded when omitted.",
    )
    parser.add_argument(
        "--crop-trigger-size",
        type=int,
        default=DEFAULT_CROP_TRIGGER_SIZE,
        help=f"Use per-mask crops above this image size. Default: {DEFAULT_CROP_TRIGGER_SIZE}",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=DEFAULT_CROP_MARGIN,
        help=f"Context pixels around each mask crop. Default: {DEFAULT_CROP_MARGIN}",
    )
    parser.add_argument(
        "--keep-mask",
        type=Path,
        help="Optional path to save the generated black/white mask for inspection.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as exc:
        raise InputError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc.msg} at line {exc.lineno}") from exc

    if not isinstance(data, list):
        raise InputError("The JSON root must be a list of text entries.")

    entries: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputError(f"Entry {index} must be an object.")
        entries.append(item)

    return entries


def selected_entries(entries: list[dict[str, Any]], page: int | None) -> list[dict[str, Any]]:
    if page is None:
        for index, item in enumerate(entries):
            validate_entry(index, item)
        return entries

    if page < 0:
        raise InputError("--page must be zero or greater.")

    selected: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        entry_page = item.get("page")
        if not isinstance(entry_page, int) or isinstance(entry_page, bool) or entry_page < 0:
            raise InputError(
                f"Entry {index} must have page as a non-negative integer when --page is used."
            )
        if entry_page == page:
            validate_entry(index, item)
            selected.append(item)
    return selected


def validate_entry(index: int, item: dict[str, Any]) -> None:
    region = item.get("region")
    if not isinstance(region, list) or len(region) != 4:
        raise InputError(f"Entry {index} must have region as a list of four numbers.")

    for coordinate in region:
        if (
            not isinstance(coordinate, (int, float))
            or isinstance(coordinate, bool)
            or not math.isfinite(coordinate)
        ):
            raise InputError(f"Entry {index} region values must be finite numbers.")

    left, top, right, bottom = region
    if right <= left or bottom <= top:
        raise InputError(f"Entry {index} region must have positive width and height.")


def check_paths(
    text_json: Path,
    input_image: Path,
    output_image: Path,
    model_path: Path | None,
    keep_mask: Path | None,
    require_lama: bool,
) -> None:
    if not input_image.exists():
        raise InputError(f"Input image not found: {input_image}")
    protected_paths = {
        text_json.resolve(): "the text JSON",
        input_image.resolve(): "the input image",
    }
    if model_path is not None:
        protected_paths[model_path.resolve()] = "the LaMa model"
    output_resolved = output_image.resolve()
    if output_resolved in protected_paths:
        raise InputError(f"Output image must be different from {protected_paths[output_resolved]}.")
    if keep_mask is not None:
        keep_mask_resolved = keep_mask.resolve()
        if keep_mask_resolved == output_resolved:
            raise InputError("Retained mask must be different from the output image.")
        if keep_mask_resolved in protected_paths:
            raise InputError(
                f"Retained mask must be different from {protected_paths[keep_mask_resolved]}."
            )
    if require_lama and model_path is not None and not model_path.is_file():
        raise InputError(f"LaMa model file not found: {model_path}")


def rounded_region(region: list[int | float]) -> tuple[int, int, int, int]:
    left, top, right, bottom = (round(value) for value in region)
    return left, top, right, bottom


def padded_clip_region(
    region: list[int | float],
    padding: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = rounded_region(region)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image_width, right + padding)
    bottom = min(image_height, bottom + padding)

    if right <= left or bottom <= top:
        raise InputError(f"Region does not overlap image bounds: {region}")

    return left, top, right, bottom


def build_mask(
    entries: list[dict[str, Any]],
    input_image: Path,
    mask_path: Path,
    padding: int,
) -> tuple[int, int]:
    if padding < 0:
        raise InputError("--padding must be zero or greater.")

    with Image.open(input_image) as image:
        image_width, image_height = image.size

    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    for entry in entries:
        left, top, right, bottom = padded_clip_region(
            entry["region"],
            padding,
            image_width,
            image_height,
        )
        draw.rectangle((left, top, right - 1, bottom - 1), fill=255)

    mask.save(mask_path)
    return image_width, image_height


def copy_as_png(input_image: Path, output_image: Path) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_mode = output_image.stat().st_mode & 0o777 if output_image.exists() else 0o644
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_image.parent,
            prefix=f".{output_image.name}.",
            suffix=".png",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
        with Image.open(input_image) as image:
            image.save(temp_path, format="PNG")
        os.chmod(temp_path, output_mode)
        os.replace(temp_path, output_image)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
        shutil.copyfile(source, temp_path)
        os.chmod(temp_path, output_mode)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def run_lama(
    device: str,
    input_image: Path,
    mask_path: Path,
    output_image: Path,
    model_path: Path | None,
    crop_boxes: list[lama_inpaint.Box],
    crop_trigger_size: int,
    crop_margin: int,
    session: lama_inpaint.LaMaSession | None = None,
) -> None:
    try:
        if session is None:
            lama_inpaint.inpaint_image(
                input_image,
                mask_path,
                output_image,
                device_name=device,
                model_path=model_path,
                crop_boxes=crop_boxes,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
            )
        else:
            session.inpaint_image(
                input_image,
                mask_path,
                output_image,
                crop_boxes=crop_boxes,
                crop_trigger_size=crop_trigger_size,
                crop_margin=crop_margin,
            )
    except lama_inpaint.LaMaError as exc:
        raise InputError(f"LaMa failed while cleaning text regions: {exc}") from exc


def clean_text_regions(
    entries: list[dict[str, Any]],
    input_image: Path,
    output_image: Path,
    padding: int,
    device: str,
    model_path: Path | None,
    crop_trigger_size: int,
    crop_margin: int,
    keep_mask: Path | None,
    lama_session: lama_inpaint.LaMaSession | None = None,
) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)

    if not entries:
        copy_as_png(input_image, output_image)
        if keep_mask is not None:
            with tempfile.TemporaryDirectory(prefix="clean-text-mask-") as temp_dir_name:
                mask_path = Path(temp_dir_name) / "mask.png"
                build_mask([], input_image, mask_path, padding)
                copy_file_atomic(mask_path, keep_mask)
        return

    with tempfile.TemporaryDirectory(prefix="clean-text-regions-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        mask_path = temp_dir / "mask.png"
        lama_output = temp_dir / "lama-output.png"

        image_width, image_height = build_mask(entries, input_image, mask_path, padding)
        crop_boxes = [
            padded_clip_region(entry["region"], padding, image_width, image_height)
            for entry in entries
        ]

        if keep_mask is not None:
            copy_file_atomic(mask_path, keep_mask)

        run_lama(
            device,
            input_image,
            mask_path,
            lama_output,
            model_path,
            crop_boxes,
            crop_trigger_size,
            crop_margin,
            lama_session,
        )

        if not lama_output.exists():
            raise InputError(f"LaMa did not produce expected output: {lama_output}")

        copy_file_atomic(lama_output, output_image)


def main() -> int:
    args = parse_args()

    try:
        entries = selected_entries(load_entries(args.text_json), args.page)
        check_paths(
            args.text_json,
            args.input_image,
            args.output_image,
            args.model_path,
            args.keep_mask,
            require_lama=bool(entries),
        )
        clean_text_regions(
            entries,
            args.input_image,
            args.output_image,
            args.padding,
            args.device,
            args.model_path,
            args.crop_trigger_size,
            args.crop_margin,
            args.keep_mask,
        )
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
