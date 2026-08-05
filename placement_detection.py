"""Geometry and ray-based text-container detection for placement regions."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline_types import Page, PipelineError


PLACEMENT_DETECT_TOLERANCES = (28, 42)
PLACEMENT_DETECT_MIN_SEARCH_PAD = 80
PLACEMENT_DETECT_SEARCH_SCALE = 3.0
PLACEMENT_DETECT_MAX_IMAGE_AREA_RATIO = 0.50
PLACEMENT_DETECT_INSET_MIN = 4
PLACEMENT_DETECT_INSET_RATIO = 0.01
PLACEMENT_DETECT_FALLBACK_WIDENING = 1.0
PLACEMENT_DETECT_FALLBACK_HEIGHT_INCREASE = 0.25
PLACEMENT_RAY_DIRECTIONS = (
    ("left", -1.0, 0.0),
    ("right", 1.0, 0.0),
    ("up", 0.0, -1.0),
    ("down", 0.0, 1.0),
    ("up_left", -math.sqrt(0.5), -math.sqrt(0.5)),
    ("up_right", math.sqrt(0.5), -math.sqrt(0.5)),
    ("down_left", -math.sqrt(0.5), math.sqrt(0.5)),
    ("down_right", math.sqrt(0.5), math.sqrt(0.5)),
)
PLACEMENT_RAY_THICKNESS = 3
PLACEMENT_RAY_IGNORE_MARGIN = 4
PLACEMENT_RAY_BOUNDARY_RUN = 3
PLACEMENT_RAY_DARK_BOUNDARY = 110
PLACEMENT_RAY_LIGHT_BOUNDARY = 145
PLACEMENT_RAY_MID_CONTRAST = 65
PLACEMENT_RAY_GRADIENT = 45
PLACEMENT_RAY_MIN_HIT_COUNT = 4
PLACEMENT_RAY_CARDINAL_DIRECTIONS = {"left", "right", "up", "down"}
PLACEMENT_RAY_MIN_CARDINAL_HIT_COUNT = 3
PLACEMENT_BLOCKER_PAD_X_MIN = 2
PLACEMENT_BLOCKER_PAD_X_MAX = 6
PLACEMENT_BLOCKER_PAD_Y_MIN = 4
PLACEMENT_BLOCKER_PAD_Y_MAX = 8
PLACEMENT_BLOCKER_PAD_X_RATIO = 0.05
PLACEMENT_BLOCKER_PAD_Y_RATIO = 0.04
PLACEMENT_BLOCKER_SHADOW_PAD_Y_MIN = 24
PLACEMENT_BLOCKER_SHADOW_PAD_Y_MAX = 48
PLACEMENT_BLOCKER_SHADOW_PAD_Y_RATIO = 0.22
PLACEMENT_BLOCKER_SHADOW_WIDTH_MIN = 18
PLACEMENT_BLOCKER_SHADOW_WIDTH_RATIO = 0.55


def normalized_box_to_region(box: list[int | float], image_path: Path) -> list[int]:
    if len(box) != 4:
        raise PipelineError("box_2d must have four values.")
    values: list[float] = []
    for value in box:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PipelineError(f"box_2d values must be numbers: {box}")
        values.append(float(value))

    y_min, x_min, y_max, x_max = values
    if x_max <= x_min or y_max <= y_min:
        raise PipelineError(f"box_2d must have positive width and height before clamping: {box}")

    adjusted = list(values)
    y_min, x_min, y_max, x_max = adjusted
    box_width = x_max - x_min
    box_height = y_max - y_min
    if x_min < 0 and x_max <= 1000:
        adjusted[3] = min(1000, x_max - x_min)
        adjusted[1] = 0
    elif x_max > 1000 and x_min >= 0:
        adjusted[1] = max(0, 1000 - box_width)
        adjusted[3] = 1000
    if y_min < 0 and y_max <= 1000:
        adjusted[2] = min(1000, y_max - y_min)
        adjusted[0] = 0
    elif y_max > 1000 and y_min >= 0:
        adjusted[0] = max(0, 1000 - box_height)
        adjusted[2] = 1000

    clamped = [min(1000, max(0, value)) for value in adjusted]
    if clamped != values:
        printable = [
            round(float(value)) if float(value).is_integer() else value
            for value in clamped
        ]
        print(f"warning: adjusted box_2d from {box} to {printable}", file=sys.stderr)

    y_min, x_min, y_max, x_max = clamped
    if x_max <= x_min or y_max <= y_min:
        raise PipelineError(f"box_2d must have positive width and height after clamping: {box}")
    with Image.open(image_path) as image:
        width, height = image.size
    return [
        round(x_min / 1000 * width),
        round(y_min / 1000 * height),
        round(x_max / 1000 * width),
        round(y_max / 1000 * height),
    ]


def region_to_normalized_box(region: list[int | float], image_path: Path) -> list[int]:
    left, top, right, bottom = region
    with Image.open(image_path) as image:
        width, height = image.size
    return [
        round(top / height * 1000),
        round(left / width * 1000),
        round(bottom / height * 1000),
        round(right / width * 1000),
    ]


def parse_expansion_value(value: Any, label: str, key: str) -> float:
    if isinstance(value, bool):
        raise PipelineError(f"{label} {key} must be a non-negative number.")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        raw_value = value.strip()
        try:
            result = float(raw_value[:-1]) / 100 if raw_value.endswith("%") else float(raw_value)
        except ValueError as exc:
            raise PipelineError(f"{label} {key} must be a non-negative number.") from exc
    else:
        raise PipelineError(f"{label} {key} must be a non-negative number.")

    if not math.isfinite(result) or result < 0:
        raise PipelineError(f"{label} {key} must be a non-negative finite number.")
    return result


def parse_fraction_value(value: Any, label: str, key: str) -> float:
    if isinstance(value, bool):
        raise PipelineError(f"{label} {key} must be a number.")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        raw_value = value.strip()
        try:
            result = float(raw_value[:-1]) / 100 if raw_value.endswith("%") else float(raw_value)
        except ValueError as exc:
            raise PipelineError(f"{label} {key} must be a number.") from exc
    else:
        raise PipelineError(f"{label} {key} must be a number.")

    if not math.isfinite(result):
        raise PipelineError(f"{label} {key} must be a finite number.")

    clamped = min(1.0, max(0.0, result))
    if clamped != result:
        print(f"warning: clamped {label} {key} from {result} to {clamped}", file=sys.stderr)
    return clamped


def expand_region(
    region: list[int | float],
    box_widening: float,
    height_increase: float,
    image_path: Path,
) -> list[int]:
    left, top, right, bottom = (float(value) for value in region)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise PipelineError(f"Cannot expand non-positive region: {region}")

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    expanded_width = width * (1 + box_widening)
    expanded_height = height * (1 + height_increase)

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    expanded = [
        round(center_x - expanded_width / 2),
        round(center_y - expanded_height / 2),
        round(center_x + expanded_width / 2),
        round(center_y + expanded_height / 2),
    ]
    expanded[0] = max(0, expanded[0])
    expanded[1] = max(0, expanded[1])
    expanded[2] = min(image_width, expanded[2])
    expanded[3] = min(image_height, expanded[3])
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        raise PipelineError(f"Expanded region does not overlap image bounds: {expanded}")
    return expanded


def expand_region_to_container_width(
    region: list[int | float],
    target_width_ratio: float,
    container_x_min: float,
    container_x_max: float,
    height_increase: float,
    image_path: Path,
) -> list[int]:
    left, top, right, bottom = (float(value) for value in region)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise PipelineError(f"Cannot expand non-positive region: {region}")
    if target_width_ratio <= 0:
        raise PipelineError(f"target_width_ratio must be greater than 0: {target_width_ratio}")
    if container_x_max <= container_x_min:
        raise PipelineError(
            f"container_x_max must be greater than container_x_min: "
            f"{container_x_min}, {container_x_max}"
        )

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    container_left = container_x_min * image_width
    container_right = container_x_max * image_width
    container_width = container_right - container_left
    target_width = min(container_width, max(width, container_width * target_width_ratio))
    if target_width <= 0:
        raise PipelineError(
            f"Expanded target width must be positive for container: "
            f"{container_x_min}, {container_x_max}"
        )

    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    expanded_height = height * (1 + height_increase)

    expanded_left = center_x - target_width / 2
    expanded_right = center_x + target_width / 2
    if expanded_left < container_left:
        expanded_right += container_left - expanded_left
        expanded_left = container_left
    if expanded_right > container_right:
        expanded_left -= expanded_right - container_right
        expanded_right = container_right

    expanded = [
        round(max(0, expanded_left)),
        round(max(0, center_y - expanded_height / 2)),
        round(min(image_width, expanded_right)),
        round(min(image_height, center_y + expanded_height / 2)),
    ]
    if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
        raise PipelineError(f"Expanded region does not overlap image bounds: {expanded}")
    return expanded


def clip_region_values(
    region: Any,
    image_width: int,
    image_height: int,
    label: str,
) -> list[int]:
    if not isinstance(region, list) or len(region) != 4:
        raise PipelineError(f"{label} must be a region array of four numbers.")
    values: list[int] = []
    for coordinate in region:
        if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
            raise PipelineError(f"{label} region values must be numbers.")
        values.append(round(coordinate))
    left, top, right, bottom = values
    clipped = [
        max(0, min(image_width, left)),
        max(0, min(image_height, top)),
        max(0, min(image_width, right)),
        max(0, min(image_height, bottom)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise PipelineError(f"{label} region does not overlap image bounds: {region}")
    return clipped


def expanded_region_bounds(
    region: list[int],
    image_width: int,
    image_height: int,
    pad_x: int,
    pad_y: int,
) -> list[int]:
    left, top, right, bottom = region
    return [
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_width, right + pad_x),
        min(image_height, bottom + pad_y),
    ]


def sample_stride(bounds: list[int], max_samples: int = 12000) -> int:
    left, top, right, bottom = bounds
    area = max(1, (right - left) * (bottom - top))
    return max(1, math.ceil(math.sqrt(area / max_samples)))


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        raise PipelineError("Cannot calculate percentile for an empty sample.")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[max(0, min(len(ordered) - 1, index))]


def sampled_pixel_values(pixels: Any, bounds: list[int], max_samples: int = 12000) -> list[int]:
    stride = sample_stride(bounds, max_samples)
    values: list[int] = []
    for y in range(bounds[1], bounds[3], stride):
        for x in range(bounds[0], bounds[2], stride):
            values.append(int(pixels[x, y]))
    return values


def dominant_text_container_brightness(values: list[int]) -> int | None:
    if not values:
        return None
    bright_values = [value for value in values if value >= 180]
    dark_values = [value for value in values if value <= 90]
    bright_ratio = len(bright_values) / len(values)
    dark_ratio = len(dark_values) / len(values)

    if bright_ratio >= 0.20 and bright_ratio >= dark_ratio * 0.75:
        return percentile(bright_values, 0.70)
    if dark_ratio >= 0.20 and dark_ratio > bright_ratio:
        return percentile(dark_values, 0.30)
    return None


def estimate_background_brightness(
    pixels: Any,
    region: list[int],
    image_width: int,
    image_height: int,
) -> int:
    left, top, right, bottom = region
    region_width = right - left
    region_height = bottom - top
    max_dimension = max(region_width, region_height)
    inner_bounds = expanded_region_bounds(region, image_width, image_height, 8, 8)
    inner_background = dominant_text_container_brightness(
        sampled_pixel_values(pixels, inner_bounds, 5000)
    )
    if inner_background is not None:
        return inner_background

    pad_x = max(16, round(min(max(region_width * 0.75, max_dimension * 0.20), 120)))
    pad_y = max(16, round(min(max(region_height * 0.50, max_dimension * 0.15), 80)))
    bounds = expanded_region_bounds(region, image_width, image_height, pad_x, pad_y)
    values = sampled_pixel_values(pixels, bounds)

    if not values:
        return 255

    bright_count = sum(1 for value in values if value >= 180)
    dark_count = sum(1 for value in values if value <= 90)
    if bright_count >= dark_count and bright_count >= len(values) * 0.25:
        return percentile(values, 0.85)
    if dark_count >= len(values) * 0.25:
        return percentile(values, 0.15)
    return percentile(values, 0.50)


def background_pixel_matches(value: int, background: int, tolerance: int) -> bool:
    if background >= 170:
        return value >= max(0, background - tolerance)
    if background <= 85:
        return value <= min(255, background + tolerance)
    return abs(value - background) <= tolerance


def placement_detection_search_bounds(
    region: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    left, top, right, bottom = region
    region_width = right - left
    region_height = bottom - top
    max_dimension = max(region_width, region_height)
    pad_x = max(
        PLACEMENT_DETECT_MIN_SEARCH_PAD,
        round(region_width * PLACEMENT_DETECT_SEARCH_SCALE),
        round(max_dimension * 1.50),
    )
    pad_y = max(
        PLACEMENT_DETECT_MIN_SEARCH_PAD,
        round(region_height * PLACEMENT_DETECT_SEARCH_SCALE),
        round(max_dimension * 1.50),
    )
    return expanded_region_bounds(region, image_width, image_height, pad_x, pad_y)


def find_background_seed(
    pixels: Any,
    region: list[int],
    search_bounds: list[int],
    background: int,
    tolerance: int,
) -> tuple[int, int] | None:
    left, top, right, bottom = region
    center_x = (left + right - 1) / 2
    center_y = (top + bottom - 1) / 2
    radius = max(
        PLACEMENT_DETECT_MIN_SEARCH_PAD,
        round(max(right - left, bottom - top) * 1.5),
    )
    seed_bounds = [
        max(search_bounds[0], math.floor(center_x - radius)),
        max(search_bounds[1], math.floor(center_y - radius)),
        min(search_bounds[2], math.ceil(center_x + radius) + 1),
        min(search_bounds[3], math.ceil(center_y + radius) + 1),
    ]
    best_seed: tuple[int, int] | None = None
    best_distance: float | None = None
    for y in range(seed_bounds[1], seed_bounds[3]):
        for x in range(seed_bounds[0], seed_bounds[2]):
            if not background_pixel_matches(int(pixels[x, y]), background, tolerance):
                continue
            distance = (x - center_x) ** 2 + (y - center_y) ** 2
            if best_distance is None or distance < best_distance:
                best_seed = (x, y)
                best_distance = distance
    return best_seed


def point_in_bounds(x: float, y: float, bounds: list[int]) -> bool:
    return bounds[0] <= x < bounds[2] and bounds[1] <= y < bounds[3]


def point_in_region_with_margin(x: float, y: float, region: list[int], margin: int) -> bool:
    return (
        region[0] - margin <= x < region[2] + margin
        and region[1] - margin <= y < region[3] + margin
    )


def point_in_any_region_with_margin(
    x: float,
    y: float,
    regions: list[list[int]],
    margin: int,
) -> bool:
    return any(point_in_region_with_margin(x, y, region, margin) for region in regions)


def opposite_boundary_pixel(value: int, background: int) -> bool:
    if background >= 170:
        return value <= PLACEMENT_RAY_DARK_BOUNDARY
    if background <= 85:
        return value >= PLACEMENT_RAY_LIGHT_BOUNDARY
    return abs(value - background) >= PLACEMENT_RAY_MID_CONTRAST


def local_gradient(
    pixels: Any,
    x: int,
    y: int,
    image_width: int,
    image_height: int,
) -> int:
    value = int(pixels[x, y])
    gradients: list[int] = []
    if x > 0:
        gradients.append(abs(value - int(pixels[x - 1, y])))
    if x < image_width - 1:
        gradients.append(abs(value - int(pixels[x + 1, y])))
    if y > 0:
        gradients.append(abs(value - int(pixels[x, y - 1])))
    if y < image_height - 1:
        gradients.append(abs(value - int(pixels[x, y + 1])))
    return max(gradients) if gradients else 0


def ray_sample_is_boundary(
    pixels: Any,
    x: float,
    y: float,
    perpendicular_x: float,
    perpendicular_y: float,
    image_width: int,
    image_height: int,
    search_bounds: list[int],
    background: int,
) -> bool:
    half_thickness = PLACEMENT_RAY_THICKNESS // 2
    boundary_votes = 0
    sample_count = 0
    for offset in range(-half_thickness, half_thickness + 1):
        sample_x = round(x + perpendicular_x * offset)
        sample_y = round(y + perpendicular_y * offset)
        if sample_x < search_bounds[0] or sample_x >= search_bounds[2]:
            continue
        if sample_y < search_bounds[1] or sample_y >= search_bounds[3]:
            continue
        if sample_x < 0 or sample_x >= image_width or sample_y < 0 or sample_y >= image_height:
            continue
        sample_count += 1
        value = int(pixels[sample_x, sample_y])
        if opposite_boundary_pixel(value, background):
            boundary_votes += 1
            continue
        if local_gradient(pixels, sample_x, sample_y, image_width, image_height) >= PLACEMENT_RAY_GRADIENT:
            boundary_votes += 1
    return sample_count == 0 or boundary_votes > 0


def cast_placement_ray(
    pixels: Any,
    seed: tuple[int, int],
    direction_name: str,
    direction_x: float,
    direction_y: float,
    original_region: list[int],
    blocker_regions: list[list[int]],
    search_bounds: list[int],
    background: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    perpendicular_x = -direction_y
    perpendicular_y = direction_x
    start_x, start_y = seed
    last_clear = (float(start_x), float(start_y))
    boundary_start = last_clear
    boundary_run = 0
    max_steps = math.ceil(
        math.hypot(search_bounds[2] - search_bounds[0], search_bounds[3] - search_bounds[1])
    ) + 2

    for step in range(1, max_steps + 1):
        x = start_x + direction_x * step
        y = start_y + direction_y * step
        if not point_in_bounds(x, y, search_bounds):
            reached_image_edge = (
                (x < search_bounds[0] and search_bounds[0] == 0)
                or (x >= search_bounds[2] and search_bounds[2] == image_width)
                or (y < search_bounds[1] and search_bounds[1] == 0)
                or (y >= search_bounds[3] and search_bounds[3] == image_height)
            )
            return {
                "direction": direction_name,
                "x": round(last_clear[0]),
                "y": round(last_clear[1]),
                "hitBoundary": reached_image_edge,
                "stopReason": "image_bounds" if reached_image_edge else "search_bounds",
            }
        if point_in_region_with_margin(x, y, original_region, PLACEMENT_RAY_IGNORE_MARGIN):
            last_clear = (x, y)
            boundary_run = 0
            continue
        if point_in_any_region_with_margin(x, y, blocker_regions, PLACEMENT_RAY_IGNORE_MARGIN):
            return {
                "direction": direction_name,
                "x": round(last_clear[0]),
                "y": round(last_clear[1]),
                "hitBoundary": True,
                "stopReason": "nearby_text_region",
            }

        if ray_sample_is_boundary(
            pixels,
            x,
            y,
            perpendicular_x,
            perpendicular_y,
            image_width,
            image_height,
            search_bounds,
            background,
        ):
            if boundary_run == 0:
                boundary_start = last_clear
            boundary_run += 1
            if boundary_run >= PLACEMENT_RAY_BOUNDARY_RUN:
                return {
                    "direction": direction_name,
                    "x": round(boundary_start[0]),
                    "y": round(boundary_start[1]),
                    "hitBoundary": True,
                    "stopReason": "boundary",
                }
        else:
            last_clear = (x, y)
            boundary_run = 0

    return {
        "direction": direction_name,
        "x": round(last_clear[0]),
        "y": round(last_clear[1]),
        "hitBoundary": False,
        "stopReason": "max_steps",
    }


def ray_cast_component(
    pixels: Any,
    seed: tuple[int, int],
    original_region: list[int],
    blocker_regions: list[list[int]],
    search_bounds: list[int],
    background: int,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    ray_endpoints = [
        cast_placement_ray(
            pixels,
            seed,
            direction_name,
            direction_x,
            direction_y,
            original_region,
            blocker_regions,
            search_bounds,
            background,
            image_width,
            image_height,
        )
        for direction_name, direction_x, direction_y in PLACEMENT_RAY_DIRECTIONS
    ]
    by_direction = {endpoint["direction"]: endpoint for endpoint in ray_endpoints}
    up_endpoint = by_direction["up"]
    down_endpoint = by_direction["down"]

    left_endpoints = [
        by_direction[direction]
        for direction in ("left", "up_left", "down_left")
        if by_direction[direction]["hitBoundary"]
    ]
    right_endpoints = [
        by_direction[direction]
        for direction in ("right", "up_right", "down_right")
        if by_direction[direction]["hitBoundary"]
    ]

    # A center ray can pass through touching balloons. The most inward result on
    # each side gives a rectangle supported by all three rays on that side.
    left = max(
        (endpoint["x"] for endpoint in left_endpoints),
        default=original_region[0],
    )
    right = min(
        (endpoint["x"] + 1 for endpoint in right_endpoints),
        default=original_region[2],
    )
    top = up_endpoint["y"] if up_endpoint["hitBoundary"] else original_region[1]
    bottom = down_endpoint["y"] + 1 if down_endpoint["hitBoundary"] else original_region[3]

    for endpoint in ray_endpoints:
        endpoint["usedForRegion"] = bool(endpoint["hitBoundary"])

    return {
        "region": [
            max(0, min(image_width, min(left, original_region[0]))),
            max(0, min(image_height, min(top, original_region[1]))),
            max(0, min(image_width, max(right, original_region[2]))),
            max(0, min(image_height, max(bottom, original_region[3]))),
        ],
        "rayEndpoints": ray_endpoints,
        "hitRayCount": sum(1 for endpoint in ray_endpoints if endpoint["hitBoundary"]),
        "cardinalHitRayCount": sum(
            1
            for endpoint in ray_endpoints
            if endpoint["hitBoundary"] and endpoint["direction"] in PLACEMENT_RAY_CARDINAL_DIRECTIONS
        ),
        "seed": [seed[0], seed[1]],
    }


def ray_rejection_reason(
    component: dict[str, Any],
    original_region: list[int],
    image_width: int,
    image_height: int,
) -> str | None:
    left, top, right, bottom = component["region"]
    width = right - left
    height = bottom - top
    original_width = original_region[2] - original_region[0]
    original_height = original_region[3] - original_region[1]
    area = width * height
    image_area = image_width * image_height

    if component["cardinalHitRayCount"] < PLACEMENT_RAY_MIN_CARDINAL_HIT_COUNT:
        return "insufficient_cardinal_ray_boundaries"
    if component["hitRayCount"] < PLACEMENT_RAY_MIN_HIT_COUNT:
        return "insufficient_total_ray_boundaries"
    if width < 8 or height < 8:
        return "ray_region_too_small"
    if area > image_area * PLACEMENT_DETECT_MAX_IMAGE_AREA_RATIO:
        return "ray_region_too_large"
    if (
        width <= max(original_width + 4, round(original_width * 1.15))
        and height <= max(original_height + 4, round(original_height * 1.05))
    ):
        return "ray_region_not_larger_than_ocr_box"
    return None


def inset_detected_region(region: list[int]) -> tuple[list[int], int]:
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    inset = max(PLACEMENT_DETECT_INSET_MIN, round(min(width, height) * PLACEMENT_DETECT_INSET_RATIO))
    if width <= inset * 2 + 4 or height <= inset * 2 + 4:
        return region, 0
    return [left + inset, top + inset, right - inset, bottom - inset], inset


def fallback_detected_region(
    region: list[int],
    image_width: int,
    image_height: int,
) -> list[int]:
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    expanded_width = width * (1 + PLACEMENT_DETECT_FALLBACK_WIDENING)
    expanded_height = height * (1 + PLACEMENT_DETECT_FALLBACK_HEIGHT_INCREASE)
    expanded = [
        round(center_x - expanded_width / 2),
        round(center_y - expanded_height / 2),
        round(center_x + expanded_width / 2),
        round(center_y + expanded_height / 2),
    ]
    return [
        max(0, expanded[0]),
        max(0, expanded[1]),
        min(image_width, expanded[2]),
        min(image_height, expanded[3]),
    ]


def padded_region(
    region: list[int],
    image_width: int,
    image_height: int,
    pad_x: int,
    pad_y: int,
) -> list[int]:
    return [
        max(0, region[0] - pad_x),
        max(0, region[1] - pad_y),
        min(image_width, region[2] + pad_x),
        min(image_height, region[3] + pad_y),
    ]


def placement_blocker_regions(
    region: list[int],
    image_width: int,
    image_height: int,
) -> list[list[int]]:
    width = region[2] - region[0]
    height = region[3] - region[1]
    hard_pad_x = round(
        max(
            PLACEMENT_BLOCKER_PAD_X_MIN,
            min(PLACEMENT_BLOCKER_PAD_X_MAX, width * PLACEMENT_BLOCKER_PAD_X_RATIO),
        )
    )
    hard_pad_y = round(
        max(
            PLACEMENT_BLOCKER_PAD_Y_MIN,
            min(PLACEMENT_BLOCKER_PAD_Y_MAX, height * PLACEMENT_BLOCKER_PAD_Y_RATIO),
        )
    )
    hard_region = padded_region(region, image_width, image_height, hard_pad_x, hard_pad_y)

    shadow_pad_y = round(
        max(
            PLACEMENT_BLOCKER_SHADOW_PAD_Y_MIN,
            min(
                PLACEMENT_BLOCKER_SHADOW_PAD_Y_MAX,
                height * PLACEMENT_BLOCKER_SHADOW_PAD_Y_RATIO,
            ),
        )
    )
    shadow_width = min(
        width,
        max(PLACEMENT_BLOCKER_SHADOW_WIDTH_MIN, round(width * PLACEMENT_BLOCKER_SHADOW_WIDTH_RATIO)),
    )
    center_x = (region[0] + region[2]) / 2
    shadow_region = [
        max(0, round(center_x - shadow_width / 2)),
        max(0, region[1] - shadow_pad_y),
        min(image_width, round(center_x + shadow_width / 2)),
        min(image_height, region[3] + shadow_pad_y),
    ]

    return [hard_region, shadow_region]


def detect_record_placement_region(
    pixels: Any,
    record: dict[str, Any],
    blocker_regions: list[list[int]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    label = f"page {record['page']} boxno {record['boxno']}"
    original_region = clip_region_values(record["region"], image_width, image_height, label)
    search_bounds = placement_detection_search_bounds(original_region, image_width, image_height)
    background = estimate_background_brightness(pixels, original_region, image_width, image_height)
    fallback_reason = "no_background_seed_found"

    for tolerance in PLACEMENT_DETECT_TOLERANCES:
        seed = find_background_seed(pixels, original_region, search_bounds, background, tolerance)
        if seed is None:
            continue
        component = ray_cast_component(
            pixels,
            seed,
            original_region,
            blocker_regions,
            search_bounds,
            background,
            image_width,
            image_height,
        )
        rejection_reason = ray_rejection_reason(
            component,
            original_region,
            image_width,
            image_height,
        )
        if rejection_reason is not None:
            fallback_reason = rejection_reason
            continue

        placement_region, inset = inset_detected_region(component["region"])
        return {
            "label": record["boxno"],
            "boxno": record["boxno"],
            "placementRegion": placement_region,
            "placementMethod": "ray_cast",
            "detectedBackground": background,
            "tolerance": tolerance,
            "raySeed": component["seed"],
            "rayEndpoints": component["rayEndpoints"],
            "hitRayCount": component["hitRayCount"],
            "cardinalHitRayCount": component["cardinalHitRayCount"],
            "searchRegion": search_bounds,
            "blockerRegions": blocker_regions,
            "inset": inset,
        }

    return {
        "label": record["boxno"],
        "boxno": record["boxno"],
        "placementRegion": fallback_detected_region(original_region, image_width, image_height),
        "placementMethod": "fallback_expand_region",
        "fallbackReason": fallback_reason,
        "detectedBackground": background,
        "searchRegion": search_bounds,
        "blockerRegions": blocker_regions,
        "box_widening": PLACEMENT_DETECT_FALLBACK_WIDENING,
        "height_increase": PLACEMENT_DETECT_FALLBACK_HEIGHT_INCREASE,
    }


def separated_source_axis(
    first: list[int],
    second: list[int],
) -> tuple[str, int] | None:
    horizontal_gap = max(first[0], second[0]) - min(first[2], second[2])
    vertical_gap = max(first[1], second[1]) - min(first[3], second[3])
    if horizontal_gap <= 0 and vertical_gap <= 0:
        return None
    if vertical_gap > horizontal_gap:
        if first[1] <= second[1]:
            return "vertical", round((first[3] + second[1]) / 2)
        return "vertical", round((second[3] + first[1]) / 2)
    if first[0] <= second[0]:
        return "horizontal", round((first[2] + second[0]) / 2)
    return "horizontal", round((second[2] + first[0]) / 2)


def resolve_overlapping_expansions(
    expansions: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Split overlapping placement regions between separate source boxes."""
    source_by_boxno = {record["boxno"]: record["region"] for record in records}
    for first_index, first in enumerate(expansions):
        for second in expansions[first_index + 1 :]:
            first_region = first["placementRegion"]
            second_region = second["placementRegion"]
            if (
                min(first_region[2], second_region[2]) <= max(first_region[0], second_region[0])
                or min(first_region[3], second_region[3]) <= max(first_region[1], second_region[1])
            ):
                continue

            first_source = source_by_boxno[first["boxno"]]
            second_source = source_by_boxno[second["boxno"]]
            separation = separated_source_axis(first_source, second_source)
            if separation is None:
                continue
            axis, boundary = separation
            if axis == "vertical":
                upper, lower = (
                    (first, second)
                    if first_source[1] <= second_source[1]
                    else (second, first)
                )
                upper["placementRegion"][3] = min(upper["placementRegion"][3], boundary)
                lower["placementRegion"][1] = max(lower["placementRegion"][1], boundary)
            else:
                left, right = (
                    (first, second)
                    if first_source[0] <= second_source[0]
                    else (second, first)
                )
                left["placementRegion"][2] = min(left["placementRegion"][2], boundary)
                right["placementRegion"][0] = max(right["placementRegion"][0], boundary)
            first["overlapAdjusted"] = True
            second["overlapAdjusted"] = True


def detect_expansions_page(
    page: Page,
    structured_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expansions: list[dict[str, Any]] = []
    records_to_expand = [
        record for record in structured_records if not record["openLettering"]
    ]
    with Image.open(page.image_path) as image:
        grayscale = image.convert("L")
        pixels = grayscale.load()
        image_width, image_height = grayscale.size
        clipped_by_boxno = {
            record["boxno"]: clip_region_values(
                record["region"],
                image_width,
                image_height,
                f"page {record['page']} boxno {record['boxno']}",
            )
            for record in structured_records
        }
        for record in records_to_expand:
            blocker_regions = [
                blocker_region
                for boxno, region in clipped_by_boxno.items()
                if boxno != record["boxno"]
                for blocker_region in placement_blocker_regions(region, image_width, image_height)
            ]
            expansion = detect_record_placement_region(
                pixels,
                record,
                blocker_regions,
                image_width,
                image_height,
            )
            expansions.append(expansion)
        resolve_overlapping_expansions(expansions, records_to_expand)
        for expansion in expansions:
            expansion["box_2d"] = [
                round(expansion["placementRegion"][1] / image_height * 1000),
                round(expansion["placementRegion"][0] / image_width * 1000),
                round(expansion["placementRegion"][3] / image_height * 1000),
                round(expansion["placementRegion"][2] / image_width * 1000),
            ]
    return sorted(expansions, key=lambda item: item["label"])
