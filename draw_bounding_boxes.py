#!/usr/bin/env python3
"""Draw page-specific OCR bounding boxes and box numbers on an image."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_COLOR = "#ff2d55"
DEFAULT_WIDTH = 3
DEFAULT_FONT_SIZE = 18


class InputError(ValueError):
    """Raised when inputs cannot be drawn."""


def reject_json_constant(value: str) -> Any:
    raise InputError(f"JSON contains non-finite number: {value}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw bounding boxes and boxno tags for one page of a master JSON file.",
    )
    parser.add_argument("master_json", type=Path, help="Master JSON file")
    parser.add_argument("page", type=int, help="Page number to draw")
    parser.add_argument("input_image", type=Path, help="Source page image")
    parser.add_argument("output_image", type=Path, help="Annotated image to write")
    parser.add_argument(
        "--color",
        default=DEFAULT_COLOR,
        help=f"Box and label color. Default: {DEFAULT_COLOR}",
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
    parser.add_argument(
        "--testnorm",
        action="store_true",
        help="Read normalized box_2d values as [y_min, x_min, y_max, x_max] on a 0..1000 scale.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file, parse_constant=reject_json_constant)
    except FileNotFoundError as exc:
        raise InputError(f"Master JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid JSON in {path}: {exc.msg} at line {exc.lineno}") from exc

    if not isinstance(data, list):
        raise InputError("The master JSON root must be a list.")

    entries: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise InputError(f"Entry {index} must be an object.")
        entries.append(item)

    return entries


def require_non_negative_int(record: dict[str, Any], field: str, label: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(f"{label} must have {field} as a non-negative integer.")
    return value


def require_region(record: dict[str, Any], label: str) -> tuple[int, int, int, int]:
    region = record.get("region")
    if not isinstance(region, list) or len(region) != 4:
        raise InputError(f"{label} must have region as a list of four numbers.")

    for coordinate in region:
        if (
            not isinstance(coordinate, (int, float))
            or isinstance(coordinate, bool)
            or not math.isfinite(coordinate)
        ):
            raise InputError(f"{label} region values must be finite numbers.")

    left, top, right, bottom = (round(value) for value in region)
    if right <= left or bottom <= top:
        raise InputError(f"{label} region must have positive width and height.")

    return left, top, right, bottom


def require_box_2d(record: dict[str, Any], label: str) -> tuple[float, float, float, float]:
    box = record.get("box_2d")
    if not isinstance(box, list) or len(box) != 4:
        raise InputError(f"{label} must have box_2d as a list of four numbers.")

    for coordinate in box:
        if (
            not isinstance(coordinate, (int, float))
            or isinstance(coordinate, bool)
            or not math.isfinite(coordinate)
        ):
            raise InputError(f"{label} box_2d values must be finite numbers.")
        if coordinate < 0 or coordinate > 1000:
            raise InputError(f"{label} box_2d values must be between 0 and 1000.")

    y_min, x_min, y_max, x_max = box
    if x_max <= x_min or y_max <= y_min:
        raise InputError(f"{label} box_2d must have positive width and height.")

    return y_min, x_min, y_max, x_max


def normalized_box_to_region(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    y_min, x_min, y_max, x_max = box
    left = round(x_min / 1000 * image_width)
    top = round(y_min / 1000 * image_height)
    right = round(x_max / 1000 * image_width)
    bottom = round(y_max / 1000 * image_height)
    return left, top, right, bottom


def selected_entries(entries: list[dict[str, Any]], page: int) -> list[dict[str, Any]]:
    if page < 0:
        raise InputError("page must be zero or greater.")

    selected: list[dict[str, Any]] = []
    for index, item in enumerate(entries):
        label = f"Entry {index}"
        entry_page = require_non_negative_int(item, "page", label)
        if entry_page != page:
            continue
        require_non_negative_int(item, "boxno", label)
        selected.append(item)

    return selected


def check_paths(input_image: Path, output_image: Path) -> None:
    if not input_image.exists():
        raise InputError(f"Input image not found: {input_image}")
    if input_image.resolve() == output_image.resolve():
        raise InputError("Output image must be different from the input image.")


def load_font(font_size: int) -> ImageFont.ImageFont:
    if font_size <= 0:
        raise InputError("--font-size must be greater than zero.")

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def clip_region(
    region: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = region
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def draw_label(
    draw: ImageDraw.ImageDraw,
    boxno: int,
    left: int,
    top: int,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    text = str(boxno)
    padding = 4
    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text, font=font)
    text_width = text_right - text_left
    text_height = text_bottom - text_top
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
    entries: list[dict[str, Any]],
    input_image: Path,
    output_image: Path,
    color: str,
    width: int,
    font_size: int,
    testnorm: bool,
) -> None:
    if width <= 0:
        raise InputError("--width must be greater than zero.")

    output_image.parent.mkdir(parents=True, exist_ok=True)
    font = load_font(font_size)

    with Image.open(input_image) as image:
        annotated = image.convert("RGBA")

    image_width, image_height = annotated.size
    draw = ImageDraw.Draw(annotated)

    for entry in entries:
        boxno = require_non_negative_int(entry, "boxno", "Selected entry")
        label = f"Entry page {entry['page']} boxno {boxno}"
        if testnorm:
            box = require_box_2d(entry, label)
            original_region = normalized_box_to_region(box, image_width, image_height)
        else:
            original_region = require_region(entry, label)

        clipped_region = clip_region(original_region, image_width, image_height)
        if clipped_region is None:
            print(
                f"warning: skipping page {entry['page']} boxno {boxno}; "
                f"region does not overlap image bounds: {list(original_region)}",
                file=sys.stderr,
            )
            continue

        left, top, right, bottom = clipped_region
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=width)
        draw_label(draw, boxno, left, top, color, font)

    if output_image.suffix.lower() in {".jpg", ".jpeg"}:
        annotated = annotated.convert("RGB")
    annotated.save(output_image)


def main() -> int:
    args = parse_args()

    try:
        check_paths(args.input_image, args.output_image)
        entries = selected_entries(load_entries(args.master_json), args.page)
        draw_boxes(
            entries,
            args.input_image,
            args.output_image,
            args.color,
            args.width,
            args.font_size,
            args.testnorm,
        )
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
