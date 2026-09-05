#!/usr/bin/env python3
"""Overlay text from JSON regions onto an image using ImageMagick."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import ImageFont


DEFAULT_FONT = str(
    Path(__file__).resolve().parent / "data" / "bundled_fonts" / "ComicNeue-Bold.ttf"
)
DEFAULT_FILL = "white"
DEFAULT_GRAVITY = "center"
DEFAULT_STROKE_WIDTH = 3
DEFAULT_MIN_POINT_SIZE = 14
DEFAULT_RENDER_BLEED = 8
PAGE_HEIGHT_MIN_POINT_RATIO = 0.008
PAGE_HEIGHT_MAX_POINT_RATIO = 0.024
PAGE_HEIGHT_STROKE_WIDTH_RATIO = 0.0012
MIN_HYPHENATED_WORD_LENGTH = 7
MIN_HYPHEN_SEGMENT_LENGTH = 3
APPROX_FONT_WIDTH_RATIO = 0.56
APPROX_LINE_HEIGHT_RATIO = 1.20
AESTHETIC_WIDTH_POINT_RATIO = 0.24
AESTHETIC_HEIGHT_POINT_RATIO = 0.30
TARGET_HEIGHT_FILL_RATIO = 0.62
TARGET_WIDTH_FILL_RATIO = 0.78
MAX_COMFORTABLE_HEIGHT_FILL_RATIO = 0.82
MAX_COMFORTABLE_WIDTH_FILL_RATIO = 0.92
TALL_CAPTION_ASPECT_RATIO = 1.75
TALL_CAPTION_TARGET_HEIGHT_FILL_RATIO = 0.62
TALL_CAPTION_TARGET_WIDTH_FILL_RATIO = 0.80
TALL_CAPTION_MAX_COMFORTABLE_HEIGHT_FILL_RATIO = 0.86
TALL_CAPTION_MAX_COMFORTABLE_WIDTH_FILL_RATIO = 0.94
TALL_CAPTION_WRAP_RATIOS = (0.75, 0.85, 0.95, 1.0)
LINE_COUNT_PENALTY = 0.08
POINT_SIZE_REDUCTION_PENALTY = 3.0
TRAILING_PUNCTUATION = frozenset(",.;:!?%~)]}\u00bb\u201d\u2019\u2026")
VOWELS = set("aeiouyAEIOUY")


class InputError(ValueError):
    """Raised when the input JSON cannot be used."""


def reject_json_constant(value: str) -> Any:
    raise InputError(f"JSON contains non-finite number: {value}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay englishText entries from text.json onto image.png.",
    )
    parser.add_argument("text_json", type=Path, help="JSON file describing text regions")
    parser.add_argument("input_image", type=Path, help="Source image, usually image.png")
    parser.add_argument("output_image", type=Path, help="PNG file to write")
    parser.add_argument(
        "--page",
        type=int,
        help="Only overlay entries for this page number.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file, parse_constant=reject_json_constant)
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

    if not isinstance(item.get("englishText"), str):
        raise InputError(f"Entry {index} must have englishText as a string.")

    for optional_key in ("font", "fill", "gravity", "stroke"):
        if optional_key in item and not isinstance(item[optional_key], str):
            raise InputError(f"Entry {index} {optional_key} must be a string.")

    if "strokeWidth" in item:
        stroke_width = item["strokeWidth"]
        if (
            not isinstance(stroke_width, (int, float))
            or isinstance(stroke_width, bool)
            or not math.isfinite(stroke_width)
            or stroke_width < 0
        ):
            raise InputError(f"Entry {index} strokeWidth must be a non-negative number.")

    if "minPointSize" in item:
        min_point_size = item["minPointSize"]
        if (
            not isinstance(min_point_size, (int, float))
            or isinstance(min_point_size, bool)
            or not math.isfinite(min_point_size)
            or min_point_size <= 0
        ):
            raise InputError(f"Entry {index} minPointSize must be a positive number.")

    if "maxPointSize" in item:
        max_point_size = item["maxPointSize"]
        if (
            not isinstance(max_point_size, (int, float))
            or isinstance(max_point_size, bool)
            or not math.isfinite(max_point_size)
            or max_point_size <= 0
        ):
            raise InputError(f"Entry {index} maxPointSize must be a positive number.")

    if "fontSizeWidthPercent" in item:
        font_size_percent = item["fontSizeWidthPercent"]
        if (
            not isinstance(font_size_percent, (int, float))
            or isinstance(font_size_percent, bool)
            or not math.isfinite(font_size_percent)
            or font_size_percent <= 0
            or font_size_percent > 100
        ):
            raise InputError(
                f"Entry {index} fontSizeWidthPercent must be between 0 and 100."
            )

    if "renderBleed" in item:
        render_bleed = item["renderBleed"]
        if (
            not isinstance(render_bleed, (int, float))
            or isinstance(render_bleed, bool)
            or not math.isfinite(render_bleed)
            or render_bleed < 0
        ):
            raise InputError(f"Entry {index} renderBleed must be a non-negative number.")


def rounded_region(region: list[int | float]) -> tuple[int, int, int, int]:
    left, top, right, bottom = (round(value) for value in region)
    if right <= left or bottom <= top:
        raise InputError(f"Region collapses after pixel rounding: {region}")
    return left, top, right - left, bottom - top


def outline_for_fill(fill: str) -> str:
    normalized = fill.strip().lower()
    if normalized in {"black", "#000", "#000000", "gray0", "grey0"}:
        return "white"
    return "black"


def scaled_page_stroke_width(image_height: int | None) -> float:
    if image_height is None or image_height <= 0:
        return float(DEFAULT_STROKE_WIDTH)
    return max(float(DEFAULT_STROKE_WIDTH), round(image_height * PAGE_HEIGHT_STROKE_WIDTH_RATIO))


def stroke_width(entry: dict[str, Any], image_height: int | None = None) -> float:
    value = entry.get("strokeWidth", DEFAULT_STROKE_WIDTH)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        configured = max(0, float(value))
    else:
        configured = float(DEFAULT_STROKE_WIDTH)
    return max(configured, scaled_page_stroke_width(image_height))


def scaled_page_min_point_size(image_height: int | None) -> float:
    if image_height is None or image_height <= 0:
        return float(DEFAULT_MIN_POINT_SIZE)
    return max(float(DEFAULT_MIN_POINT_SIZE), image_height * PAGE_HEIGHT_MIN_POINT_RATIO)


def min_point_size(entry: dict[str, Any], image_height: int | None = None) -> float:
    value = entry.get("minPointSize", DEFAULT_MIN_POINT_SIZE)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        configured = max(1, float(value))
    else:
        configured = float(DEFAULT_MIN_POINT_SIZE)
    return max(configured, scaled_page_min_point_size(image_height))


def render_bleed(entry: dict[str, Any]) -> int:
    value = entry.get("renderBleed", DEFAULT_RENDER_BLEED)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, round(float(value)))
    return DEFAULT_RENDER_BLEED


def configured_max_point_size(entry: dict[str, Any]) -> float | None:
    value = entry.get("maxPointSize")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(1, float(value))
    return None


def page_relative_max_point_size(
    image_height: int | None,
    width: int,
    height: int,
) -> float:
    if image_height is not None and image_height > 0:
        return max(1.0, image_height * PAGE_HEIGHT_MAX_POINT_RATIO)
    return max(1.0, max(width * AESTHETIC_WIDTH_POINT_RATIO, height * AESTHETIC_HEIGHT_POINT_RATIO))


def aesthetic_max_point_size(
    entry: dict[str, Any],
    width: int,
    height: int,
    image_height: int | None = None,
) -> float:
    minimum = min_point_size(entry, image_height)
    configured = configured_max_point_size(entry)
    page_cap = page_relative_max_point_size(image_height, width, height)
    region_cap = min(
        max(minimum, page_cap),
        max(minimum, width * AESTHETIC_WIDTH_POINT_RATIO),
        max(minimum, height * AESTHETIC_HEIGHT_POINT_RATIO),
    )
    if configured is not None:
        region_cap = min(region_cap, configured)
    return max(minimum, region_cap)


def is_tall_caption_region(width: int, height: int) -> bool:
    return height / max(1, width) >= TALL_CAPTION_ASPECT_RATIO


def wrap_ratios_for_region(width: int, height: int) -> tuple[float, ...]:
    if is_tall_caption_region(width, height):
        return TALL_CAPTION_WRAP_RATIOS
    return (1.0,)


def best_hyphen_break(word: str, segment_limit: int) -> int:
    max_break = min(len(word) - MIN_HYPHEN_SEGMENT_LENGTH, segment_limit - 1)
    if max_break <= MIN_HYPHEN_SEGMENT_LENGTH:
        return max_break

    for index in range(max_break, MIN_HYPHEN_SEGMENT_LENGTH - 1, -1):
        if word[index - 1] not in VOWELS and word[index] not in VOWELS:
            return index

    for index in range(max_break, MIN_HYPHEN_SEGMENT_LENGTH - 1, -1):
        if word[index - 1] in VOWELS and word[index] not in VOWELS:
            return index

    return max_break


def normalize_caption_text(text: str) -> str:
    lines = [
        re.sub(
            r"\s+([,.;:!?%~)\]}\u00bb\u201d\u2019\u2026]+)",
            r"\1",
            " ".join(line.split()),
        )
        for line in text.strip().splitlines()
    ]
    return "\n".join(line for line in lines if line)


def punctuation_only(value: str) -> bool:
    return bool(value) and all(character in TRAILING_PUNCTUATION for character in value)


def split_trailing_punctuation(word: str) -> tuple[str, str]:
    split_at = len(word)
    while split_at > 0 and word[split_at - 1] in TRAILING_PUNCTUATION:
        split_at -= 1
    return word[:split_at], word[split_at:]


def split_word_for_capacity(word: str, capacity: int) -> list[str]:
    if len(word) <= capacity:
        return [word]
    core, punctuation = split_trailing_punctuation(word)
    if not core:
        return [word]
    if capacity <= 1:
        segments = list(core)
        segments[-1] += punctuation
        return segments

    segments: list[str] = []
    remaining = core
    while len(remaining) > capacity:
        existing_hyphen = remaining.rfind("-", 1, capacity + 1)
        if existing_hyphen > 0:
            segments.append(remaining[: existing_hyphen + 1])
            remaining = remaining[existing_hyphen + 1 :]
            continue

        if (
            len(remaining) >= MIN_HYPHENATED_WORD_LENGTH
            and capacity > MIN_HYPHEN_SEGMENT_LENGTH
        ):
            break_at = best_hyphen_break(remaining, capacity)
            if break_at >= MIN_HYPHEN_SEGMENT_LENGTH:
                segments.append(f"{remaining[:break_at]}-")
                remaining = remaining[break_at:]
                continue

        segments.append(remaining[:capacity])
        remaining = remaining[capacity:]

    if remaining:
        segments.append(remaining)
    segments[-1] += punctuation
    return segments


def wrap_line_for_capacity(line: str, capacity: int) -> list[str]:
    if not line:
        return [""]

    wrapped: list[str] = []
    current = ""
    for raw_word in line.split():
        if punctuation_only(raw_word):
            if current:
                current += raw_word
            elif wrapped:
                wrapped[-1] += raw_word
            else:
                current = raw_word
            continue

        word_parts = split_word_for_capacity(raw_word, capacity)
        for word in word_parts:
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= capacity:
                current = f"{current} {word}"
            else:
                wrapped.append(current)
                current = word

    if current:
        wrapped.append(current)
    return wrapped or [line]


@functools.lru_cache(maxsize=512)
def loaded_layout_font(font: str, point_size: int) -> ImageFont.FreeTypeFont | None:
    try:
        return ImageFont.truetype(font, max(1, point_size))
    except (OSError, ValueError):
        return None


def lines_fit(
    lines: list[str],
    point_size: float,
    width: int,
    height: int,
    font: str | None = None,
) -> bool:
    estimated_width, estimated_height = estimated_text_extent(lines, point_size, font)
    return estimated_width <= width and estimated_height <= height


def estimated_text_extent(
    lines: list[str],
    point_size: float,
    font: str | None = None,
) -> tuple[float, float]:
    layout_font = loaded_layout_font(font, round(point_size)) if font else None
    if layout_font is not None:
        estimated_width = max(
            (float(layout_font.getlength(line)) for line in lines),
            default=0.0,
        )
    else:
        longest_line = max((len(line) for line in lines), default=0)
        estimated_width = longest_line * point_size * APPROX_FONT_WIDTH_RATIO
    estimated_height = len(lines) * point_size * APPROX_LINE_HEIGHT_RATIO
    return estimated_width, estimated_height


def layout_lines_for_point_size(
    text: str,
    point_size: float,
    width: int,
    wrap_width_ratio: float = 1.0,
) -> list[str]:
    wrap_width = max(1, round(width * max(0.1, min(1.0, wrap_width_ratio))))
    capacity = max(1, math.floor(wrap_width / max(1, point_size * APPROX_FONT_WIDTH_RATIO)))
    lines: list[str] = []
    for line in text.splitlines() or [""]:
        lines.extend(wrap_line_for_capacity(line, capacity))
    return lines


def line_visible_length(line: str) -> int:
    return sum(1 for character in line if character.isalnum())


def layout_quality_score(
    lines: list[str],
    point_size: float,
    width: int,
    height: int,
    point_size_cap: float,
    tall_caption: bool,
    wrap_width_ratio: float,
    font: str | None = None,
) -> float:
    estimated_width, estimated_height = estimated_text_extent(lines, point_size, font)
    width_ratio = estimated_width / max(1, width)
    height_ratio = estimated_height / max(1, height)

    if tall_caption:
        score = abs(height_ratio - TALL_CAPTION_TARGET_HEIGHT_FILL_RATIO) * 5.2
        score += abs(width_ratio - TALL_CAPTION_TARGET_WIDTH_FILL_RATIO) * 2.2
        if height_ratio > TALL_CAPTION_MAX_COMFORTABLE_HEIGHT_FILL_RATIO:
            score += (height_ratio - TALL_CAPTION_MAX_COMFORTABLE_HEIGHT_FILL_RATIO) * 14.0
        if width_ratio > TALL_CAPTION_MAX_COMFORTABLE_WIDTH_FILL_RATIO:
            score += (width_ratio - TALL_CAPTION_MAX_COMFORTABLE_WIDTH_FILL_RATIO) * 8.0
        score += abs(wrap_width_ratio - TALL_CAPTION_TARGET_WIDTH_FILL_RATIO) * 0.45
    else:
        score = abs(height_ratio - TARGET_HEIGHT_FILL_RATIO) * 4.0
        score += abs(width_ratio - TARGET_WIDTH_FILL_RATIO) * 1.4

        if height_ratio > MAX_COMFORTABLE_HEIGHT_FILL_RATIO:
            score += (height_ratio - MAX_COMFORTABLE_HEIGHT_FILL_RATIO) * 12.0
        if width_ratio > MAX_COMFORTABLE_WIDTH_FILL_RATIO:
            score += (width_ratio - MAX_COMFORTABLE_WIDTH_FILL_RATIO) * 6.0

    nonblank_lines = [line for line in lines if line.strip()]
    visible_lengths = [line_visible_length(line) for line in nonblank_lines]
    punctuation_only_lines = sum(
        1 for line in nonblank_lines if punctuation_only(line.strip())
    )
    one_character_lines = sum(1 for length in visible_lengths if length == 1)
    two_character_lines = sum(1 for length in visible_lengths if length == 2)
    hyphenated_lines = sum(1 for line in lines if line.rstrip().endswith("-"))
    score += punctuation_only_lines * 8.0
    score += one_character_lines * 3.0
    score += two_character_lines * 0.35
    score += hyphenated_lines * 1.35
    score += max(0, len(nonblank_lines) - 2) * LINE_COUNT_PENALTY

    if visible_lengths:
        longest = max(visible_lengths)
        shortest = min(visible_lengths)
        if longest > 0:
            score += ((longest - shortest) / longest) * 0.35

    size_ratio = min(1.0, point_size / max(1.0, point_size_cap))
    score += (1.0 - size_ratio) * POINT_SIZE_REDUCTION_PENALTY
    # Prefer larger text only as a mild tie-breaker; the fill targets carry the layout.
    score -= min(point_size, point_size_cap) * 0.008
    return score


def technical_fitting_point_size(
    text: str,
    width: int,
    height: int,
    image_height: int | None = None,
    font: str | None = None,
    maximum_point_size: float | None = None,
) -> float:
    wrap_ratios = wrap_ratios_for_region(width, height)
    search_maximum = (
        maximum_point_size
        if maximum_point_size is not None
        else page_relative_max_point_size(image_height, width, height)
    )
    start_size = max(1, math.ceil(search_maximum))
    for point_size in range(start_size, 0, -1):
        for wrap_width_ratio in wrap_ratios:
            lines = layout_lines_for_point_size(text, float(point_size), width, wrap_width_ratio)
            if lines_fit(lines, float(point_size), width, height, font):
                return float(point_size)
    return 1.0


def explicit_caption_layout(
    entry: dict[str, Any],
    text: str,
    width: int,
    height: int,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[str, float, float]:
    font = entry.get("font") if isinstance(entry.get("font"), str) else None
    fixed_percent = entry.get("fontSizeWidthPercent")
    if (
        isinstance(fixed_percent, (int, float))
        and not isinstance(fixed_percent, bool)
        and image_width is not None
        and image_width > 0
    ):
        requested_size = max(1.0, image_width * float(fixed_percent) / 100.0)
        technical_size = technical_fitting_point_size(
            text,
            width,
            height,
            image_height,
            font,
            requested_size,
        )
        point_size = min(requested_size, technical_size)
        candidates = [
            layout_lines_for_point_size(text, point_size, width, ratio)
            for ratio in wrap_ratios_for_region(width, height)
        ]
        fitting = [
            lines for lines in candidates if lines_fit(lines, point_size, width, height, font)
        ]
        selected = min(fitting or candidates, key=lambda lines: len(lines))
        return "\n".join(selected), point_size, technical_size

    minimum = min_point_size(entry, image_height)
    point_size_cap = aesthetic_max_point_size(entry, width, height, image_height)
    technical_best_size = technical_fitting_point_size(
        text, width, height, image_height, font
    )
    tall_caption = is_tall_caption_region(width, height)
    wrap_ratios = wrap_ratios_for_region(width, height)
    best_candidate: tuple[float, list[str], float] | None = None
    start_size = max(1, math.floor(point_size_cap))
    stop_size = math.ceil(minimum)
    candidate_sizes = list(range(start_size, stop_size - 1, -1))
    if minimum <= point_size_cap and all(abs(size - minimum) > 0.001 for size in candidate_sizes):
        candidate_sizes.append(minimum)

    for point_size in candidate_sizes:
        for wrap_width_ratio in wrap_ratios:
            lines = layout_lines_for_point_size(text, float(point_size), width, wrap_width_ratio)
            if not lines_fit(lines, float(point_size), width, height, font):
                continue
            score = layout_quality_score(
                lines,
                float(point_size),
                width,
                height,
                point_size_cap,
                tall_caption,
                wrap_width_ratio,
                font,
            )
            if best_candidate is None or score < best_candidate[0]:
                best_candidate = (score, lines, float(point_size))

    if best_candidate is not None:
        _, best_lines, best_size = best_candidate
        return "\n".join(best_lines), best_size, technical_best_size

    forced_lines = layout_lines_for_point_size(text, minimum, width, wrap_ratios[-1])
    return "\n".join(forced_lines), minimum, technical_best_size


def explicit_pointsize_args(
    entry: dict[str, Any],
    point_size: float,
    estimated_fit_size: float,
    image_height: int | None = None,
) -> list[str]:
    minimum = min_point_size(entry, image_height)
    if estimated_fit_size < minimum:
        label = entry.get("boxno", "?")
        print(
            (
                f"warning: text for box {label} in region {entry['region']} would fit at "
                f"approximately {estimated_fit_size:g}px; enforcing minimum {minimum:g}px. "
                "Text may clip or overflow."
            ),
            file=sys.stderr,
        )

    return ["-pointsize", f"{point_size:g}"]


def render_text_layer(
    magick: str,
    entry: dict[str, Any],
    layer_path: Path,
    text_path: Path,
    image_width: int,
    image_height: int,
) -> None:
    left, top, width, height = rounded_region(entry["region"])
    del left, top

    fill = entry.get("fill", DEFAULT_FILL)
    stroke = entry.get("stroke", outline_for_fill(fill))
    outline_width = stroke_width(entry, image_height)
    bleed = render_bleed(entry)
    padding = max(0, round(outline_width * 2))
    inner_width = max(1, width - padding * 2)
    inner_height = max(1, height - padding * 2)
    layer_width = width + bleed * 2
    layer_height = height + bleed * 2
    layer_inner_width = inner_width + bleed * 2
    layer_inner_height = inner_height + bleed * 2
    gravity = entry.get("gravity", DEFAULT_GRAVITY)
    layout_text = normalize_caption_text(entry["englishText"])
    rendered_text, point_size, estimated_fit_size = explicit_caption_layout(
        entry,
        layout_text,
        inner_width,
        inner_height,
        image_width,
        image_height,
    )
    point_size_args = explicit_pointsize_args(entry, point_size, estimated_fit_size, image_height)
    text_path.write_text(rendered_text, encoding="utf-8")

    command = [
        magick,
        "(",
            "-size",
            f"{layer_inner_width}x{layer_inner_height}",
            "-background",
            "none",
            "-font",
            entry.get("font", DEFAULT_FONT),
            *point_size_args,
            "-gravity",
            gravity,
            "-fill",
            stroke,
            "-stroke",
            stroke,
            "-strokewidth",
            f"{outline_width:g}",
            f"caption:@{text_path}",
            "-background",
            "none",
            "-gravity",
            gravity,
            "-extent",
            f"{layer_width}x{layer_height}",
        ")",
        "(",
            "-size",
            f"{layer_inner_width}x{layer_inner_height}",
            "-background",
            "none",
            "-font",
            entry.get("font", DEFAULT_FONT),
            *point_size_args,
            "-gravity",
            gravity,
            "-fill",
            fill,
            "-stroke",
            "none",
            f"caption:@{text_path}",
            "-background",
            "none",
            "-gravity",
            gravity,
            "-extent",
            f"{layer_width}x{layer_height}",
        ")",
        "-composite",
        str(layer_path),
    ]
    run_magick(command, f"rendering text layer for region {entry['region']}")


def composite_layer(
    magick: str,
    base_path: Path,
    layer_path: Path,
    output_path: Path,
    entry: dict[str, Any],
) -> None:
    left, top, width, height = rounded_region(entry["region"])
    del width, height
    bleed = render_bleed(entry)

    command = [
        magick,
        str(base_path),
        str(layer_path),
        "-geometry",
        f"+{left - bleed}+{top - bleed}",
        "-composite",
        str(output_path),
    ]
    run_magick(command, f"compositing text layer at ({left - bleed}, {top - bleed})")


def run_magick(command: list[str], action: str) -> None:
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"ImageMagick failed while {action}."
        if details:
            message = f"{message}\n{details}"
        raise InputError(message) from exc


def image_dimensions(magick: str, image_path: Path) -> tuple[int, int]:
    command = [magick, "identify", "-format", "%w %h", str(image_path)]
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = f"ImageMagick failed while reading image dimensions: {image_path}"
        if details:
            message = f"{message}\n{details}"
        raise InputError(message) from exc

    parts = completed.stdout.strip().split()
    if len(parts) != 2:
        raise InputError(f"Could not parse image dimensions for {image_path}: {completed.stdout!r}")
    try:
        width, height = (int(value) for value in parts)
    except ValueError as exc:
        raise InputError(f"Could not parse image dimensions for {image_path}: {completed.stdout!r}") from exc
    if width <= 0 or height <= 0:
        raise InputError(f"Image dimensions must be positive for {image_path}: {width}x{height}")
    return width, height


def ensure_magick() -> str:
    magick = shutil.which("magick")
    if magick is None:
        raise InputError("ImageMagick command not found: magick")
    return magick


def check_paths(text_json: Path, input_image: Path, output_image: Path) -> None:
    if not input_image.exists():
        raise InputError(f"Input image not found: {input_image}")
    if input_image.resolve() == output_image.resolve():
        raise InputError("Output image must be different from the input image.")
    if text_json.resolve() == output_image.resolve():
        raise InputError("Output image must be different from the text JSON.")


def clamp_entry_to_image(
    entry: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    left, top, right, bottom = entry["region"]
    clamped_region = [
        max(0.0, min(float(image_width), float(left))),
        max(0.0, min(float(image_height), float(top))),
        max(0.0, min(float(image_width), float(right))),
        max(0.0, min(float(image_height), float(bottom))),
    ]
    if clamped_region[2] <= clamped_region[0] or clamped_region[3] <= clamped_region[1]:
        raise InputError(f"Region does not overlap image bounds: {entry['region']}")
    if render_bleed(entry) > max(image_width, image_height):
        raise InputError(
            f"renderBleed for box {entry.get('boxno', '?')} is larger than the page."
        )
    if clamped_region != [float(value) for value in entry["region"]]:
        print(
            f"warning: clamped box {entry.get('boxno', '?')} region to image bounds: {clamped_region}",
            file=sys.stderr,
        )
    clamped = dict(entry)
    clamped["region"] = clamped_region
    return clamped


def overlay_text(entries: list[dict[str, Any]], input_image: Path, output_image: Path) -> None:
    magick = ensure_magick()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="overlay-text-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        current_path = temp_dir / "base.png"
        run_magick([magick, str(input_image), str(current_path)], "preparing the base image")
        image_width, image_height = image_dimensions(magick, current_path)

        for index, original_entry in enumerate(entries):
            entry = clamp_entry_to_image(original_entry, image_width, image_height)
            text_path = temp_dir / f"text-{index}.txt"
            layer_path = temp_dir / f"layer-{index}.png"
            next_path = temp_dir / f"composited-{index}.png"

            render_text_layer(
                magick,
                entry,
                layer_path,
                text_path,
                image_width,
                image_height,
            )
            composite_layer(magick, current_path, layer_path, next_path, entry)
            current_path = next_path

        output_mode = output_image.stat().st_mode & 0o777 if output_image.exists() else 0o644
        final_temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_image.parent,
                prefix=f".{output_image.name}.",
                suffix=".png",
                delete=False,
            ) as file:
                final_temp_path = Path(file.name)
            run_magick([magick, str(current_path), str(final_temp_path)], "writing the output PNG")
            os.chmod(final_temp_path, output_mode)
            os.replace(final_temp_path, output_image)
        finally:
            if final_temp_path is not None:
                final_temp_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()

    try:
        check_paths(args.text_json, args.input_image, args.output_image)
        entries = selected_entries(load_entries(args.text_json), args.page)
        overlay_text(entries, args.input_image, args.output_image)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
