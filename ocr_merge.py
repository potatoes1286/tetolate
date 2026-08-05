"""Deterministic grouping of neighboring OCR detections into text regions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from pipeline_types import Page, PipelineError


OCR_MERGE_SIDE_OVERLAP_RATIO = 0.1
OCR_MERGE_STACK_OVERLAP_RATIO = 0.1
OCR_MERGE_SIDE_GAP_MIN = 8
OCR_MERGE_SIDE_GAP_MAX = 12
OCR_MERGE_STACK_GAP_MIN = 4
OCR_MERGE_STACK_GAP_MAX = 8
OCR_MERGE_SIDE_HEIGHT_RATIO_MAX = 3.0
OCR_MERGE_SIDE_CENTER_RATIO = 0.55
OCR_MERGE_SIDE_CENTER_MIN = 12
OCR_MERGE_SIDE_CENTER_MAX = 96
OCR_MERGE_VERTICAL_SIDE_OVERLAP_RATIO = 0.25
OCR_MERGE_VERTICAL_SIDE_GAP_MAX = 24
OCR_MERGE_VERTICAL_ASPECT_RATIO = 2.2
OCR_MERGE_VERTICAL_MAX_WIDTH = 48
OCR_MERGE_VERTICAL_COMBINED_WIDTH_MAX = 120
OCR_MERGE_VERTICAL_CENTER_RATIO = 1.05
OCR_MERGE_VERTICAL_TOP_ALIGNMENT_MAX = 32


def region_width(region: list[int | float]) -> float:
    return max(0.0, float(region[2]) - float(region[0]))


def region_height(region: list[int | float]) -> float:
    return max(0.0, float(region[3]) - float(region[1]))


def region_center_x(region: list[int | float]) -> float:
    return (float(region[0]) + float(region[2])) / 2.0


def region_center_y(region: list[int | float]) -> float:
    return (float(region[1]) + float(region[3])) / 2.0


def axis_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def axis_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, max(a_min, b_min) - min(a_max, b_max))


def merge_side_gap_limit(a: list[int | float], b: list[int | float]) -> float:
    min_width = min(region_width(a), region_width(b))
    min_height = min(region_height(a), region_height(b))
    return max(
        OCR_MERGE_SIDE_GAP_MIN,
        min(OCR_MERGE_SIDE_GAP_MAX, max(min_width * 1.2, min_height * 0.06)),
    )


def merge_stack_gap_limit(a: list[int | float], b: list[int | float]) -> float:
    min_width = min(region_width(a), region_width(b))
    min_height = min(region_height(a), region_height(b))
    return max(
        OCR_MERGE_STACK_GAP_MIN,
        min(OCR_MERGE_STACK_GAP_MAX, max(min_width * 0.8, min_height * 0.05)),
    )


def merge_side_center_limit(a: list[int | float], b: list[int | float]) -> float:
    min_height = min(region_height(a), region_height(b))
    return max(
        OCR_MERGE_SIDE_CENTER_MIN,
        min(OCR_MERGE_SIDE_CENTER_MAX, min_height * OCR_MERGE_SIDE_CENTER_RATIO),
    )


def can_side_merge_ocr_records(
    a_region: list[int | float],
    b_region: list[int | float],
    a_height: float,
    b_height: float,
) -> bool:
    height_ratio = max(a_height, b_height) / min(a_height, b_height)
    if height_ratio > OCR_MERGE_SIDE_HEIGHT_RATIO_MAX:
        return False

    center_delta = abs(region_center_y(a_region) - region_center_y(b_region))
    return center_delta <= merge_side_center_limit(a_region, b_region)


def is_narrow_vertical_text_region(width: float, height: float) -> bool:
    return width <= OCR_MERGE_VERTICAL_MAX_WIDTH and height / max(1.0, width) >= OCR_MERGE_VERTICAL_ASPECT_RATIO


def can_vertical_side_merge_ocr_records(
    a_region: list[int | float],
    b_region: list[int | float],
    a_width: float,
    b_width: float,
    a_height: float,
    b_height: float,
    x_gap: float,
    y_overlap_ratio: float,
) -> bool:
    if not (
        is_narrow_vertical_text_region(a_width, a_height)
        and is_narrow_vertical_text_region(b_width, b_height)
    ):
        return False
    if y_overlap_ratio < OCR_MERGE_VERTICAL_SIDE_OVERLAP_RATIO:
        return False
    if x_gap > OCR_MERGE_VERTICAL_SIDE_GAP_MAX:
        return False

    combined_width = max(a_region[2], b_region[2]) - min(a_region[0], b_region[0])
    if combined_width > OCR_MERGE_VERTICAL_COMBINED_WIDTH_MAX:
        return False

    top_delta = abs(float(a_region[1]) - float(b_region[1]))
    if top_delta > OCR_MERGE_VERTICAL_TOP_ALIGNMENT_MAX:
        return False

    center_delta = abs(region_center_y(a_region) - region_center_y(b_region))
    center_limit = max(
        OCR_MERGE_SIDE_CENTER_MIN,
        min(region_height(a_region), region_height(b_region)) * OCR_MERGE_VERTICAL_CENTER_RATIO,
    )
    return center_delta <= center_limit


def should_merge_ocr_records(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_region = a["region"]
    b_region = b["region"]
    a_width = region_width(a_region)
    b_width = region_width(b_region)
    a_height = region_height(a_region)
    b_height = region_height(b_region)
    if min(a_width, b_width, a_height, b_height) <= 0:
        return False

    x_overlap = axis_overlap(a_region[0], a_region[2], b_region[0], b_region[2])
    y_overlap = axis_overlap(a_region[1], a_region[3], b_region[1], b_region[3])
    x_gap = axis_gap(a_region[0], a_region[2], b_region[0], b_region[2])
    y_gap = axis_gap(a_region[1], a_region[3], b_region[1], b_region[3])
    x_overlap_ratio = x_overlap / min(a_width, b_width)
    y_overlap_ratio = y_overlap / min(a_height, b_height)

    if (
        y_overlap_ratio >= OCR_MERGE_SIDE_OVERLAP_RATIO
        and x_gap <= merge_side_gap_limit(a_region, b_region)
        and can_side_merge_ocr_records(a_region, b_region, a_height, b_height)
    ):
        return True

    if can_vertical_side_merge_ocr_records(
        a_region,
        b_region,
        a_width,
        b_width,
        a_height,
        b_height,
        x_gap,
        y_overlap_ratio,
    ):
        return True

    if (
        x_overlap_ratio >= OCR_MERGE_STACK_OVERLAP_RATIO
        and y_gap <= merge_stack_gap_limit(a_region, b_region)
    ):
        return True

    return False


def find_parent(parents: dict[int, int], item: int) -> int:
    parent = parents[item]
    if parent != item:
        parents[item] = find_parent(parents, parent)
    return parents[item]


def union_parent(parents: dict[int, int], a: int, b: int) -> None:
    a_parent = find_parent(parents, a)
    b_parent = find_parent(parents, b)
    if a_parent != b_parent:
        parents[b_parent] = a_parent


def raw_record_source_key(
    record: dict[str, Any],
    right_to_left: bool,
) -> tuple[float, float, int]:
    region = record["region"]
    center_x = region_center_x(region)
    return (
        -center_x if right_to_left else center_x,
        region_center_y(region),
        int(record["boxno"]),
    )


def merged_record_page_key(
    record: dict[str, Any],
    right_to_left: bool = True,
) -> tuple[int, float, float, int]:
    region = record["region"]
    center_x = region_center_x(region)
    # A coarse top band keeps upper panels before lower panels while preserving source order inside a band.
    return (
        round(float(region[1]) / 80.0),
        -center_x if right_to_left else center_x,
        region_center_y(region),
        record["boxno"],
    )


def merge_ocr_records_for_page(
    page: Page,
    raw_records: list[dict[str, Any]],
    right_to_left: bool = True,
) -> list[dict[str, Any]]:
    if not raw_records:
        return []

    parents = {int(record["boxno"]): int(record["boxno"]) for record in raw_records}
    raw_by_boxno = {int(record["boxno"]): record for record in raw_records}
    for index, record in enumerate(raw_records):
        for other in raw_records[index + 1 :]:
            if should_merge_ocr_records(record, other):
                union_parent(parents, int(record["boxno"]), int(other["boxno"]))

    grouped: dict[int, list[dict[str, Any]]] = {}
    for boxno, record in raw_by_boxno.items():
        grouped.setdefault(find_parent(parents, boxno), []).append(record)

    merged: list[dict[str, Any]] = []
    for group in grouped.values():
        ordered_sources = sorted(
            group,
            key=lambda record: raw_record_source_key(record, right_to_left),
        )
        source_regions = [record["region"] for record in ordered_sources]
        source_texts = [str(record.get("text", "")) for record in ordered_sources]
        text_separator = "" if right_to_left else "\n"
        merged.append(
            {
                "page": page.index,
                "boxno": len(merged),
                "sourceBoxnos": [int(record["boxno"]) for record in ordered_sources],
                "sourceTexts": source_texts,
                "region": union_regions(source_regions, page.image_path),
                "text": text_separator.join(source_texts),
            }
        )

    merged.sort(key=lambda record: merged_record_page_key(record, right_to_left))
    for boxno, record in enumerate(merged):
        record["boxno"] = boxno
    return merged


def union_regions(regions: list[list[int | float]], image_path: Path) -> list[int]:
    with Image.open(image_path) as image:
        width, height = image.size

    left = round(min(region[0] for region in regions))
    top = round(min(region[1] for region in regions))
    right = round(max(region[2] for region in regions))
    bottom = round(max(region[3] for region in regions))

    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right <= left or bottom <= top:
        raise PipelineError(f"Merged region does not overlap image bounds: {[left, top, right, bottom]}")
    return [left, top, right, bottom]

