#!/usr/bin/env python3
"""End-to-end manga/manhwa/manhua CBZ translation pipeline."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sys
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from PIL import Image

import clean_text_regions
import overlay_text
import paddle_ocr_image


# File suffixes treated as CBZ page images during extraction and resume loading.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Maximum number of entries accepted in an input CBZ, including metadata files.
MAX_ARCHIVE_MEMBERS = 10_000

# Maximum number of image pages accepted in one input CBZ.
MAX_ARCHIVE_IMAGE_PAGES = 2_000

# Maximum declared uncompressed size of one archive member.
MAX_ARCHIVE_MEMBER_BYTES = 512 * 1024 * 1024

# Maximum combined declared uncompressed size of all input archive members.
MAX_ARCHIVE_TOTAL_BYTES = 20 * 1024 * 1024 * 1024

# Maximum size of one non-image file copied from the source to translated archives.
MAX_PRESERVED_MEMBER_BYTES = 64 * 1024 * 1024

# Maximum combined size of non-image files copied into translated archives.
MAX_PRESERVED_TOTAL_BYTES = 256 * 1024 * 1024

# Maximum decoded pixel count of one page image.
MAX_PAGE_PIXELS = 120_000_000

# Maximum combined decoded pixel count of all page images.
MAX_TOTAL_PAGE_PIXELS = 5_000_000_000

# Binds resumable intermediate files to the exact source archive that produced them.
INPUT_MANIFEST_NAME = "input_manifest.json"

# Main translated archive using PNG pages; this is the canonical/default CBZ.
TRANSLATED_PNG_CBZ_NAME = "translated.cbz"

# Alternate translated archive using lossy WebP pages for smaller output size.
TRANSLATED_WEBP_CBZ_NAME = "translated_webp.cbz"

# Alternate translated archive using lossy JPEG XL pages for smaller output size.
TRANSLATED_JXL_CBZ_NAME = "translated_jxl.cbz"

# Cover image suffix used inside WebP/JXL CBZ variants for readers that cannot
# use WebP/JXL files as archive thumbnails.
TRANSLATED_ALT_COVER_SUFFIX = ".jpg"

# ImageMagick quality setting used when creating the alternate WebP archive.
TRANSLATED_WEBP_QUALITY = 70

# ImageMagick quality setting used when creating the alternate JPEG XL archive.
TRANSLATED_JXL_QUALITY = 65

# Local font file suffixes accepted from the fonts directory and VLM font choices.
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}

# Root directory for user-managed configuration, fonts, jobs, models, and caches.
DATA_DIR = Path(
    os.environ.get("TETOLATE_DATA_DIR", Path(__file__).resolve().parent / "data")
).expanduser()

# Directory containing user-provided fonts that ImageMagick can render directly.
FONT_DIR = DATA_DIR / "fonts"

# User file describing intended uses for installed fonts; included in placement prompts.
FONT_USE_PATH = FONT_DIR / "font_use.txt"

# Fallback font passed to ImageMagick when a placement does not specify a local font.
DEFAULT_RENDER_FONT = "DejaVu-Sans"

# Fallback text fill color for rendered translations when placement styling omits fill.
DEFAULT_RENDER_FILL = "black"

# Default ImageMagick caption gravity, used to center text inside placement regions.
DEFAULT_RENDER_GRAVITY = "center"

# Supported source-language profiles for the first multilingual pass.
DEFAULT_SOURCE_LANGUAGE = "jp"
DEFAULT_TARGET_LANGUAGE = "en"

# Default VLM response budget when max_tokens is omitted from the config.
DEFAULT_VLM_MAX_TOKENS = 32768

# Default VLM sampling temperature when temperature is omitted from the config.
DEFAULT_VLM_TEMPERATURE = 0.7

# Default request timeout in seconds for each VLM API call.
DEFAULT_VLM_TIMEOUT = 1000.0

# Default for the VLM erase-safety pass that decides whether text needs alternate placement.
DEFAULT_ALT_PLACEMENT_ENABLED = True

# Default llama.cpp/Gemma thinking budget. Negative means omit the field and
# leave reasoning unlimited/server-defined.
DEFAULT_VLM_THINKING_BUDGET_TOKENS = 2048

# Number of times to retry a VLM call when JSON parsing or validation fails.
DEFAULT_VLM_RETRIES = 3

# Number of extra request-level retries for cold-start/model-loading HTTP 503s.
DEFAULT_VLM_MODEL_LOADING_RETRIES = 6

# Wait schedule in seconds for model-loading retries before the request is retried.
VLM_MODEL_LOADING_BACKOFF_SECONDS = (10, 20, 40, 60, 60, 60)

# Default outline width around rendered text; overlay_text may scale this upward on high-res pages.
DEFAULT_RENDER_STROKE_WIDTH = 3

# Default filename for generated notes intended to help translate related CBZs.
GENERATED_TRANSLATION_NOTES_NAME = "translation_notes.txt"

# Debug box color for VLM-structured OCR groups after reject/order/classification.
VLM_STRUCTURED_DEBUG_COLOR = "#2f80ed"

# Debug box color for script-merged OCR groups before VLM structure classification.
OCR_MERGED_DEBUG_COLOR = "#af52de"

# Debug box color for final placement regions used for rendering translated text.
VLM_PLACEMENT_DEBUG_COLOR = "#00a86b"

# Debug box color for detected/expanded non-open-lettering container regions.
PLACEMENT_EXPAND_DEBUG_COLOR = "#ff9500"

# Stroke width for debug bounding boxes drawn over page images.
VLM_DEBUG_BOX_WIDTH = 3

# Font size for numeric debug labels drawn on bounding boxes.
VLM_DEBUG_FONT_SIZE = 18

# Minimum vertical overlap ratio required before neighboring OCR boxes can side-merge.
OCR_MERGE_SIDE_OVERLAP_RATIO = 0.1

# Minimum horizontal overlap ratio required before stacked OCR boxes can merge.
OCR_MERGE_STACK_OVERLAP_RATIO = 0.1

# Lower bound in pixels for the allowed horizontal gap between side-merged OCR boxes.
OCR_MERGE_SIDE_GAP_MIN = 8

# Upper bound in pixels for the allowed horizontal gap between side-merged OCR boxes.
OCR_MERGE_SIDE_GAP_MAX = 12

# Lower bound in pixels for the allowed vertical gap between stacked OCR boxes.
OCR_MERGE_STACK_GAP_MIN = 4

# Upper bound in pixels for the allowed vertical gap between stacked OCR boxes.
OCR_MERGE_STACK_GAP_MAX = 8

# Largest allowed height ratio between side-merged OCR boxes; prevents merging mismatched lines.
OCR_MERGE_SIDE_HEIGHT_RATIO_MAX = 3.0

# Fraction of the shorter box height used to limit side-merge centerline drift.
OCR_MERGE_SIDE_CENTER_RATIO = 0.55

# Minimum centerline drift tolerance in pixels for side-merge checks.
OCR_MERGE_SIDE_CENTER_MIN = 12

# Maximum centerline drift tolerance in pixels for side-merge checks.
OCR_MERGE_SIDE_CENTER_MAX = 96

# Vertical overlap ratio used by the special narrow-vertical-text side-merge path.
OCR_MERGE_VERTICAL_SIDE_OVERLAP_RATIO = 0.25

# Maximum horizontal gap for merging adjacent narrow vertical text columns.
OCR_MERGE_VERTICAL_SIDE_GAP_MAX = 24

# Minimum height/width ratio for a box to be treated as narrow vertical text.
OCR_MERGE_VERTICAL_ASPECT_RATIO = 2.2

# Maximum width in pixels for a box to qualify as narrow vertical text.
OCR_MERGE_VERTICAL_MAX_WIDTH = 48

# Maximum combined width after merging adjacent narrow vertical text boxes.
OCR_MERGE_VERTICAL_COMBINED_WIDTH_MAX = 120

# Centerline tolerance multiplier for adjacent narrow vertical text merges.
OCR_MERGE_VERTICAL_CENTER_RATIO = 1.05

# Maximum top-edge offset allowed when merging adjacent vertical text columns.
OCR_MERGE_VERTICAL_TOP_ALIGNMENT_MAX = 32

# Pixel color tolerances tried when detecting the containing white/blank text region.
PLACEMENT_DETECT_TOLERANCES = (28, 42)

# Minimum search padding around an OCR box when looking for a containing region.
PLACEMENT_DETECT_MIN_SEARCH_PAD = 80

# Multiplier used to grow the search area relative to the original OCR box size.
PLACEMENT_DETECT_SEARCH_SCALE = 3.0

# Reject detected regions larger than this fraction of the whole image.
PLACEMENT_DETECT_MAX_IMAGE_AREA_RATIO = 0.50

# Minimum inward trim applied to detected container regions before rendering text.
PLACEMENT_DETECT_INSET_MIN = 4

# Image-size-relative inward trim applied to detected container regions.
PLACEMENT_DETECT_INSET_RATIO = 0.01

# Fallback width expansion factor when container detection cannot find a better region.
PLACEMENT_DETECT_FALLBACK_WIDENING = 1.0

# Fallback height expansion factor when container detection cannot find a better region.
PLACEMENT_DETECT_FALLBACK_HEIGHT_INCREASE = 0.25

# Rays cast from an OCR box center to estimate container boundaries in eight directions.
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

# Thickness in pixels for sampling each placement boundary ray.
PLACEMENT_RAY_THICKNESS = 3

# Pixels near the original OCR box ignored so rays do not immediately hit the text itself.
PLACEMENT_RAY_IGNORE_MARGIN = 4

# Consecutive boundary-like pixels required before a ray treats an edge as a hit.
PLACEMENT_RAY_BOUNDARY_RUN = 3

# Grayscale value below which a sampled pixel counts as a dark boundary candidate.
PLACEMENT_RAY_DARK_BOUNDARY = 110

# Grayscale value above which a sampled pixel counts as a light background candidate.
PLACEMENT_RAY_LIGHT_BOUNDARY = 145

# Contrast threshold for mid-tone boundary detection in textured manga panels.
PLACEMENT_RAY_MID_CONTRAST = 65

# Per-step grayscale gradient threshold for detecting strong drawn edges.
PLACEMENT_RAY_GRADIENT = 45

# Minimum total ray hits required before ray-based placement expansion is trusted.
PLACEMENT_RAY_MIN_HIT_COUNT = 4

# Cardinal ray names used to require left/right/up/down evidence separately.
PLACEMENT_RAY_CARDINAL_DIRECTIONS = {"left", "right", "up", "down"}

# Minimum cardinal direction hits required before ray-based expansion is trusted.
PLACEMENT_RAY_MIN_CARDINAL_HIT_COUNT = 3

# Minimum horizontal padding around text boxes treated as blockers during expansion.
PLACEMENT_BLOCKER_PAD_X_MIN = 2

# Maximum horizontal padding around text boxes treated as blockers during expansion.
PLACEMENT_BLOCKER_PAD_X_MAX = 6

# Minimum vertical padding around text boxes treated as blockers during expansion.
PLACEMENT_BLOCKER_PAD_Y_MIN = 4

# Maximum vertical padding around text boxes treated as blockers during expansion.
PLACEMENT_BLOCKER_PAD_Y_MAX = 8

# Horizontal blocker padding as a fraction of the blocker text box width.
PLACEMENT_BLOCKER_PAD_X_RATIO = 0.05

# Vertical blocker padding as a fraction of the blocker text box height.
PLACEMENT_BLOCKER_PAD_Y_RATIO = 0.04

# Minimum downward shadow padding used to keep expansion away from nearby lower text.
PLACEMENT_BLOCKER_SHADOW_PAD_Y_MIN = 24

# Maximum downward shadow padding used to keep expansion away from nearby lower text.
PLACEMENT_BLOCKER_SHADOW_PAD_Y_MAX = 48

# Downward shadow padding as a fraction of blocker height.
PLACEMENT_BLOCKER_SHADOW_PAD_Y_RATIO = 0.22

# Minimum width for the downward blocker shadow region.
PLACEMENT_BLOCKER_SHADOW_WIDTH_MIN = 18

# Downward blocker shadow width as a fraction of blocker width.
PLACEMENT_BLOCKER_SHADOW_WIDTH_RATIO = 0.55

# Ordered phase names accepted by --resume-from.
RESUME_PHASES = (
    "extract",
    "ocr_raw",
    "ocr_structured",
    "alt_placement",
    "translations",
    "proofreading",
    "translation_notes",
    "placements",
    "render",
    "package",
)

# Resume phases that can safely regenerate exactly one page and reuse existing later files.
SINGLE_PAGE_PHASES = {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements", "render"}

# Possible streaming delta fields used by OpenAI-compatible endpoints for hidden reasoning text.
REASONING_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "reasoning_delta",
    "thoughts",
    "thinking",
    "analysis",
)


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot continue safely."""


class PipelineCancelled(PipelineError):
    """Raised when the pipeline receives a termination signal."""


def handle_cancel_signal(signum: int, _frame: Any) -> None:
    raise PipelineCancelled(f"received signal {signum}; cancelling pipeline")


@dataclass(frozen=True)
class Page:
    index: int
    image_path: Path


@dataclass(frozen=True)
class VLMConfig:
    base_url: str
    api_key: str
    model: str | None
    temperature: float
    max_tokens: int
    thinking_budget_tokens: int
    timeout: float
    provider: dict[str, Any] | None


@dataclass(frozen=True)
class VLMStreamResult:
    output: str
    reasoning: str
    elapsed_seconds: float
    generated_tokens: int
    completion_tokens: int | None
    reasoning_chunks: int
    output_chunks: int


@dataclass(frozen=True)
class OCRConfig:
    engine: str
    service_url: str | None
    service_timeout: float
    lang: str
    device: str
    min_score: float
    text_det_limit_side_len: int | None
    text_det_limit_type: str | None
    use_doc_preprocessor: bool
    use_textline_orientation: bool
    ocr_version: str | None
    text_detection_model_name: str | None
    text_recognition_model_name: str | None
    text_detection_model_dir: str | None
    text_recognition_model_dir: str | None
    text_det_thresh: float | None
    text_det_box_thresh: float | None
    text_det_unclip_ratio: float | None
    text_rec_score_thresh: float | None
    tile_enabled: bool
    tile_width: int
    tile_height: int
    tile_overlap: int
    tile_include_full_image: bool
    tile_dedupe_iou: float
    tile_dedupe_containment: float
    paddleocr_vl_server_url: str
    paddleocr_vl_model: str
    paddleocr_vl_api_key: str | None
    paddleocr_vl_max_concurrency: int | None


@dataclass(frozen=True)
class SourceLanguageProfile:
    code: str
    name: str
    ocr_lang: str
    reading_order: str
    structure_context: str


@dataclass(frozen=True)
class TargetLanguageProfile:
    code: str
    name: str


@dataclass(frozen=True)
class LanguageConfig:
    source: SourceLanguageProfile
    target: TargetLanguageProfile


@dataclass(frozen=True)
class PostprocessConfig:
    proofread_translations: bool
    write_translation_notes: bool


@dataclass(frozen=True)
class PipelineConfig:
    vlm: VLMConfig | None
    language: LanguageConfig
    render_font: str
    render_fill: str
    render_gravity: str
    webp_quality: int
    jxl_quality: int
    alt_placement_enabled: bool
    ocr: OCRConfig
    postprocess: PostprocessConfig


SOURCE_LANGUAGE_PROFILES: dict[str, SourceLanguageProfile] = {
    "jp": SourceLanguageProfile(
        code="jp",
        name="Japanese",
        ocr_lang="japan",
        reading_order="right-to-left manga reading order",
        structure_context="The page is manga-style and is usually read right-to-left.",
    ),
    "kr": SourceLanguageProfile(
        code="kr",
        name="Korean",
        ocr_lang="korean",
        reading_order="left-to-right/top-to-bottom manhwa reading order unless the page layout indicates otherwise",
        structure_context=(
            "The page is Korean manhwa/comic-style; use normal left-to-right/top-to-bottom "
            "reading order unless the panel layout clearly indicates otherwise."
        ),
    ),
    "cn": SourceLanguageProfile(
        code="cn",
        name="Chinese",
        ocr_lang="ch",
        reading_order="right-to-left manga/manhua reading order by default unless the page layout indicates otherwise",
        structure_context=(
            "The page is Chinese manhua/comic-style; default to right-to-left manga-style "
            "ordering for this pipeline unless the panel layout clearly indicates otherwise."
        ),
    ),
}
SOURCE_LANGUAGE_ALIASES = {
    "japanese": "jp",
    "japan": "jp",
    "ja": "jp",
    "jp": "jp",
    "korean": "kr",
    "korea": "kr",
    "ko": "kr",
    "kr": "kr",
    "chinese": "cn",
    "china": "cn",
    "zh": "cn",
    "cn": "cn",
    "ch": "cn",
}
TARGET_LANGUAGE_PROFILES: dict[str, TargetLanguageProfile] = {
    "en": TargetLanguageProfile(code="en", name="English"),
}
TARGET_LANGUAGE_ALIASES = {
    "english": "en",
    "en": "en",
}


def normalize_source_language(value: Any) -> str:
    code = str(value if value is not None else DEFAULT_SOURCE_LANGUAGE).strip().lower()
    normalized = SOURCE_LANGUAGE_ALIASES.get(code, code)
    if normalized not in SOURCE_LANGUAGE_PROFILES:
        allowed = ", ".join(sorted(SOURCE_LANGUAGE_PROFILES))
        raise PipelineError(f"Unsupported source language {value!r}; expected one of: {allowed}.")
    return normalized


def normalize_target_language(value: Any) -> str:
    code = str(value if value is not None else DEFAULT_TARGET_LANGUAGE).strip().lower()
    normalized = TARGET_LANGUAGE_ALIASES.get(code, code)
    if normalized not in TARGET_LANGUAGE_PROFILES:
        allowed = ", ".join(sorted(TARGET_LANGUAGE_PROFILES))
        raise PipelineError(f"Unsupported target language {value!r}; expected one of: {allowed}.")
    return normalized


def quality_arg(value: str) -> int:
    try:
        quality = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("quality must be an integer from 1 to 100") from exc
    if quality < 1 or quality > 100:
        raise argparse.ArgumentTypeError("quality must be an integer from 1 to 100")
    return quality


def normalize_vlm_base_url(value: Any, label: str = "VLM base URL") -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{label} must be a non-empty HTTP(S) URL.")
    endpoint = value.strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise PipelineError(f"{label} must be a valid HTTP(S) URL.") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PipelineError(f"{label} must be a valid HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise PipelineError(f"{label} must not contain URL credentials.")
    if port is not None and (port <= 0 or port > 65535):
        raise PipelineError(f"{label} port must be between 1 and 65535.")
    return endpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a manga/manhwa/manhua CBZ using OCR, a VLM, LaMa, and ImageMagick.",
    )
    parser.add_argument("input_cbz", type=Path, help="Input CBZ file")
    parser.add_argument("output_dir", type=Path, help="Directory for all pipeline outputs")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config for the OpenAI-compatible VLM endpoint and pipeline defaults.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help=(
            "Optional directory with ocr_structured/page_XXXX.json, "
            "alt_placement/page_XXXX.json, translations/page_XXXX.json, "
            "and placement_*/page_XXXX.json fixtures."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir before running if it already exists.",
    )
    parser.add_argument(
        "--resume-from",
        choices=RESUME_PHASES,
        help="Resume an existing output directory from this phase.",
    )
    parser.add_argument(
        "--resume-page",
        type=int,
        default=0,
        help="Zero-based page to restart within --resume-from. Defaults to 0.",
    )
    parser.add_argument(
        "--single-page",
        action="store_true",
        help=(
            "With --resume-from ocr_raw, ocr_structured, alt_placement, translations, placements, or render, "
            "regenerate only --resume-page through render and then rebuild translated.cbz."
        ),
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help=(
            "Run the requested work without rebuilding translated CBZ archives. "
            "Intended for batched web reruns that package once at the end."
        ),
    )
    parser.add_argument(
        "--webp-quality",
        type=quality_arg,
        help="Override output.webp_quality for generated WebP CBZ output. Range: 1-100.",
    )
    parser.add_argument(
        "--jxl-quality",
        type=quality_arg,
        help="Override output.jxl_quality for generated JPEG XL CBZ output. Range: 1-100.",
    )
    parser.add_argument(
        "--translation-notes-json",
        type=Path,
        help="Optional JSON file with job/page translation notes for VLM translation prompts.",
    )
    parser.add_argument(
        "--thinking-budget-tokens",
        type=int,
        help=(
            "Override VLM thinking budget tokens. 0 disables thinking when supported; "
            "negative omits the budget for unlimited/server-defined thinking."
        ),
    )
    parser.add_argument(
        "--vlm-base-url",
        help="Override the OpenAI-compatible VLM base URL from the config.",
    )
    parser.add_argument(
        "--source-language",
        choices=tuple(sorted(SOURCE_LANGUAGE_PROFILES)),
        help="Override default_language.source. Supported: jp, kr, cn. Default: jp.",
    )
    parser.add_argument(
        "--target-language",
        choices=tuple(sorted(TARGET_LANGUAGE_PROFILES)),
        help="Override default_language.target. Only en is supported in this phase.",
    )
    parser.add_argument(
        "--proofread-translations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the whole-book proofreading postprocess pass.",
    )
    parser.add_argument(
        "--write-translation-notes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable generated whole-book translation notes.",
    )
    parser.add_argument(
        "--alt-placement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the VLM alternate-placement erase-safety pass.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=tuple(sorted(paddle_ocr_image.OCR_ENGINES)),
        help=(
            "Override ocr.engine. 'paddle' uses regular PaddleOCR; "
            "'paddleocr_vl' uses PaddleOCR-VL 1.6 through a llama.cpp server."
        ),
    )
    parser.add_argument(
        "--ocr-service-url",
        help="Override the hidden PaddleOCR worker URL. An empty value uses local PaddleOCR.",
    )
    parser.add_argument(
        "--paddleocr-vl-server-url",
        help=(
            "Override the hidden PaddleOCR-VL server URL. Defaults to "
            f"{paddle_ocr_image.DEFAULT_PADDLEOCR_VL_SERVER_URL} when unset."
        ),
    )
    parser.add_argument(
        "--paddleocr-vl-model",
        help="Override the hidden PaddleOCR-VL model identifier.",
    )
    parser.add_argument(
        "--paddleocr-vl-api-key",
        help="Override the hidden PaddleOCR-VL API key.",
    )
    parser.add_argument(
        "--paddleocr-vl-max-concurrency",
        type=int,
        help="Override the hidden PaddleOCR-VL request concurrency.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("ocr_raw",),
        help="Stop successfully after the named phase instead of running the full pipeline.",
    )
    return parser.parse_args()


def reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"non-finite number {value}")


def load_json(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file, parse_constant=reject_nonfinite_json)
    except FileNotFoundError as exc:
        raise PipelineError(f"{label} file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON in {label} file {path}: {exc.msg}") from exc
    except ValueError as exc:
        raise PipelineError(f"Invalid JSON in {label} file {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
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
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.chmod(temp_path, output_mode)
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineError(f"Failed to write JSON file {path}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def config_float(data: dict[str, Any], key: str, default: float, section: str) -> float:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise PipelineError(f"Config field {section}.{key} must be a number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Config field {section}.{key} must be a number.") from exc
    if not math.isfinite(parsed):
        raise PipelineError(f"Config field {section}.{key} must be finite.")
    return parsed


def config_int(data: dict[str, Any], key: str, default: int, section: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise PipelineError(f"Config field {section}.{key} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise PipelineError(f"Config field {section}.{key} must be an integer.")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise PipelineError(f"Config field {section}.{key} must be an integer.") from exc
    raise PipelineError(f"Config field {section}.{key} must be an integer.")


def config_section_bool(data: dict[str, Any], key: str, default: bool, section: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise PipelineError(f"Config field {section}.{key} must be true or false when provided.")
    return value


def config_quality(data: dict[str, Any], key: str, default: int, label: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise PipelineError(f"Config field {label}.{key} must be an integer from 1 to 100.")
    try:
        quality = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Config field {label}.{key} must be an integer from 1 to 100.") from exc
    if quality < 1 or quality > 100:
        raise PipelineError(f"Config field {label}.{key} must be an integer from 1 to 100.")
    return quality


def load_language_config(data: dict[str, Any]) -> LanguageConfig:
    source_code = normalize_source_language(data.get("source", DEFAULT_SOURCE_LANGUAGE))
    target_code = normalize_target_language(data.get("target", DEFAULT_TARGET_LANGUAGE))
    return language_config_from_codes(source_code, target_code)


def language_config_from_codes(source_code: str, target_code: str) -> LanguageConfig:
    return LanguageConfig(
        source=SOURCE_LANGUAGE_PROFILES[normalize_source_language(source_code)],
        target=TARGET_LANGUAGE_PROFILES[normalize_target_language(target_code)],
    )


def load_ocr_config(data: dict[str, Any], language: LanguageConfig) -> OCRConfig:
    try:
        service_url = paddle_ocr_image.normalize_ocr_service_url(
            paddle_ocr_image.DEFAULT_OCR_SERVICE_URL
        )
    except paddle_ocr_image.InputError as exc:
        raise PipelineError(f"Invalid hidden PaddleOCR service URL default: {exc}") from exc

    min_score = config_float(
        data,
        "min_score",
        paddle_ocr_image.DEFAULT_MIN_SCORE,
        "ocr",
    )
    if min_score < 0 or min_score > 1:
        raise PipelineError("Config field ocr.min_score must be between 0 and 1.")

    return OCRConfig(
        engine=paddle_ocr_image.DEFAULT_OCR_ENGINE,
        service_url=service_url,
        service_timeout=paddle_ocr_image.DEFAULT_OCR_SERVICE_TIMEOUT,
        lang=language.source.ocr_lang,
        device=paddle_ocr_image.DEFAULT_DEVICE,
        min_score=min_score,
        text_det_limit_side_len=paddle_ocr_image.DEFAULT_TEXT_DET_LIMIT_SIDE_LEN,
        text_det_limit_type=paddle_ocr_image.DEFAULT_TEXT_DET_LIMIT_TYPE,
        use_doc_preprocessor=False,
        use_textline_orientation=False,
        ocr_version=paddle_ocr_image.DEFAULT_OCR_VERSION,
        text_detection_model_name=None,
        text_recognition_model_name=None,
        text_detection_model_dir=None,
        text_recognition_model_dir=None,
        text_det_thresh=paddle_ocr_image.DEFAULT_TEXT_DET_THRESH,
        text_det_box_thresh=paddle_ocr_image.DEFAULT_TEXT_DET_BOX_THRESH,
        text_det_unclip_ratio=paddle_ocr_image.DEFAULT_TEXT_DET_UNCLIP_RATIO,
        text_rec_score_thresh=paddle_ocr_image.DEFAULT_TEXT_REC_SCORE_THRESH,
        tile_enabled=paddle_ocr_image.DEFAULT_TILE_ENABLED,
        tile_width=paddle_ocr_image.DEFAULT_TILE_WIDTH,
        tile_height=paddle_ocr_image.DEFAULT_TILE_HEIGHT,
        tile_overlap=paddle_ocr_image.DEFAULT_TILE_OVERLAP,
        tile_include_full_image=paddle_ocr_image.DEFAULT_TILE_INCLUDE_FULL_IMAGE,
        tile_dedupe_iou=paddle_ocr_image.DEFAULT_TILE_DEDUPE_IOU,
        tile_dedupe_containment=paddle_ocr_image.DEFAULT_TILE_DEDUPE_CONTAINMENT,
        paddleocr_vl_server_url=paddle_ocr_image.DEFAULT_PADDLEOCR_VL_SERVER_URL,
        paddleocr_vl_model=paddle_ocr_image.DEFAULT_PADDLEOCR_VL_MODEL,
        paddleocr_vl_api_key=paddle_ocr_image.DEFAULT_PADDLEOCR_VL_API_KEY,
        paddleocr_vl_max_concurrency=None,
    )


def load_postprocess_config(data: dict[str, Any], default_enabled: bool) -> PostprocessConfig:
    return PostprocessConfig(
        proofread_translations=config_section_bool(
            data,
            "proofread_translations",
            default_enabled,
            "postprocess",
        ),
        write_translation_notes=config_section_bool(
            data,
            "write_translation_notes",
            default_enabled,
            "postprocess",
        ),
    )


def load_config(path: Path | None, fixture_dir: Path | None) -> PipelineConfig:
    data: dict[str, Any] = {}
    if path is not None:
        loaded = load_json(path, "config")
        if not isinstance(loaded, dict):
            raise PipelineError("Config JSON root must be an object.")
        data = loaded
    elif fixture_dir is None:
        raise PipelineError("--config is required unless --fixture-dir is used.")

    vlm = None
    if fixture_dir is None:
        base_url = normalize_vlm_base_url(data.get("base_url"), "Config field base_url")
        model_value = data.get("model")
        if model_value is not None and not isinstance(model_value, str):
            raise PipelineError("Config field model must be a string when provided.")
        max_tokens = config_int(data, "max_tokens", DEFAULT_VLM_MAX_TOKENS, "root")
        if max_tokens <= 0:
            raise PipelineError("Config field max_tokens must be a positive integer.")
        thinking_budget_tokens = config_int(
            data,
            "thinking_budget_tokens",
            DEFAULT_VLM_THINKING_BUDGET_TOKENS,
            "root",
        )
        temperature = config_float(
            data,
            "temperature",
            DEFAULT_VLM_TEMPERATURE,
            "root",
        )
        if temperature < 0:
            raise PipelineError("Config field temperature must be zero or greater.")
        timeout = config_float(data, "timeout", DEFAULT_VLM_TIMEOUT, "root")
        if timeout <= 0:
            raise PipelineError("Config field timeout must be greater than zero.")
        provider = data.get("provider")
        if provider is not None and not isinstance(provider, dict):
            raise PipelineError("Config field provider must be an object when provided.")
        vlm = VLMConfig(
            base_url=base_url,
            api_key=str(data.get("api_key", "not-needed")),
            model=model_value or None,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            timeout=timeout,
            provider=provider,
        )

    backup_font = data.get("backup_font", {})
    if backup_font is None:
        backup_font = {}
    if not isinstance(backup_font, dict):
        raise PipelineError("Config field backup_font must be an object when provided.")

    language_data = data.get("default_language", {})
    if language_data is None:
        language_data = {}
    if not isinstance(language_data, dict):
        raise PipelineError("Config field default_language must be an object when provided.")
    language = load_language_config(language_data)

    ocr = data.get("ocr", {})
    if ocr is None:
        ocr = {}
    if not isinstance(ocr, dict):
        raise PipelineError("Config field ocr must be an object when provided.")

    output = data.get("output", {})
    if output is None:
        output = {}
    if not isinstance(output, dict):
        raise PipelineError("Config field output must be an object when provided.")

    postprocess = data.get("postprocess", {})
    if postprocess is None:
        postprocess = {}
    if not isinstance(postprocess, dict):
        raise PipelineError("Config field postprocess must be an object when provided.")

    alt_placement = data.get("alt_placement", {})
    if alt_placement is None:
        alt_placement = {}
    if not isinstance(alt_placement, dict):
        raise PipelineError("Config field alt_placement must be an object when provided.")

    return PipelineConfig(
        vlm=vlm,
        language=language,
        render_font=str(backup_font.get("font", DEFAULT_RENDER_FONT)),
        render_fill=str(backup_font.get("fill", DEFAULT_RENDER_FILL)),
        render_gravity=str(backup_font.get("gravity", DEFAULT_RENDER_GRAVITY)),
        webp_quality=config_quality(
            output,
            "webp_quality",
            TRANSLATED_WEBP_QUALITY,
            "output",
        ),
        jxl_quality=config_quality(
            output,
            "jxl_quality",
            TRANSLATED_JXL_QUALITY,
            "output",
        ),
        alt_placement_enabled=config_section_bool(
            alt_placement,
            "enabled",
            DEFAULT_ALT_PLACEMENT_ENABLED if fixture_dir is None else False,
            "alt_placement",
        ),
        ocr=load_ocr_config(ocr, language),
        postprocess=load_postprocess_config(postprocess, fixture_dir is None),
    )


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def is_ignored_cbz_member(name: str) -> bool:
    path = Path(name)
    if path.name.startswith(".") or path.name.startswith("._"):
        return True
    if "__MACOSX" in path.parts:
        return True
    return False


def is_image_member(name: str) -> bool:
    if is_ignored_cbz_member(name):
        return False
    path = Path(name)
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_preserved_non_image_member(info: zipfile.ZipInfo) -> bool:
    return (
        not info.is_dir()
        and not is_ignored_cbz_member(info.filename)
        and not is_image_member(info.filename)
    )


def validate_cbz_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise PipelineError(
            f"Input CBZ contains {len(members)} entries; limit is {MAX_ARCHIVE_MEMBERS}."
        )

    total_bytes = 0
    preserved_bytes = 0
    image_pages = 0
    for info in members:
        normalized_name = info.filename.replace("\\", "/")
        path_parts = normalized_name.split("/")
        if (
            not normalized_name
            or normalized_name.startswith("/")
            or "\0" in normalized_name
            or ".." in path_parts
        ):
            raise PipelineError(f"Input CBZ contains an unsafe entry name: {info.filename!r}")
        if info.is_dir():
            continue
        if info.file_size < 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise PipelineError(
                f"Input CBZ entry is too large: {info.filename} "
                f"({info.file_size} bytes; limit is {MAX_ARCHIVE_MEMBER_BYTES})."
            )
        total_bytes += info.file_size
        if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
            raise PipelineError(
                "Input CBZ expands beyond the "
                f"{MAX_ARCHIVE_TOTAL_BYTES}-byte archive limit."
            )
        if is_image_member(info.filename):
            image_pages += 1
            if image_pages > MAX_ARCHIVE_IMAGE_PAGES:
                raise PipelineError(
                    f"Input CBZ contains more than {MAX_ARCHIVE_IMAGE_PAGES} image pages."
                )
        elif is_preserved_non_image_member(info):
            if info.file_size > MAX_PRESERVED_MEMBER_BYTES:
                raise PipelineError(
                    f"Non-image CBZ entry is too large to preserve: {info.filename} "
                    f"({info.file_size} bytes; limit is {MAX_PRESERVED_MEMBER_BYTES})."
                )
            preserved_bytes += info.file_size
            if preserved_bytes > MAX_PRESERVED_TOTAL_BYTES:
                raise PipelineError(
                    "Input CBZ contains more than "
                    f"{MAX_PRESERVED_TOTAL_BYTES} bytes of preserved non-image data."
                )
    return members


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise PipelineError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)


def extract_cbz(input_cbz: Path, output_dir: Path) -> list[Page]:
    if not input_cbz.exists():
        raise PipelineError(f"Input CBZ not found: {input_cbz}")

    pages_dir = original_pages_dir(output_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(input_cbz) as archive:
            archive_members = validate_cbz_members(archive)
            members = sorted(
                (
                    info
                    for info in archive_members
                    if not info.is_dir() and is_image_member(info.filename)
                ),
                key=lambda info: natural_key(info.filename),
            )
            if not members:
                raise PipelineError(f"No image files found in CBZ: {input_cbz}")

            pages: list[Page] = []
            total_pixels = 0
            width = max(4, len(str(len(members) - 1)))
            for index, info in enumerate(members):
                image_path = pages_dir / f"{index:0{width}d}.png"
                with archive.open(info) as source:
                    with Image.open(source) as image:
                        page_pixels = image.width * image.height
                        if page_pixels > MAX_PAGE_PIXELS:
                            raise PipelineError(
                                f"CBZ page {info.filename} decodes to {page_pixels} pixels; "
                                f"limit is {MAX_PAGE_PIXELS}."
                            )
                        total_pixels += page_pixels
                        if total_pixels > MAX_TOTAL_PAGE_PIXELS:
                            raise PipelineError(
                                "Input CBZ pages exceed the combined "
                                f"{MAX_TOTAL_PAGE_PIXELS}-pixel limit."
                            )
                        image.convert("RGB").save(image_path, format="PNG")
                pages.append(Page(index=index, image_path=image_path))
            write_json(
                input_manifest_path(output_dir),
                input_archive_manifest(input_cbz, len(pages)),
            )
            return pages
    except zipfile.BadZipFile as exc:
        raise PipelineError(f"Input file is not a valid CBZ/zip: {input_cbz}") from exc
    except (OSError, RuntimeError, Image.DecompressionBombError) as exc:
        raise PipelineError(f"Could not decode input CBZ {input_cbz}: {exc}") from exc


def load_extracted_pages(output_dir: Path) -> list[Page]:
    pages_dir = original_pages_dir(output_dir)
    if not pages_dir.is_dir():
        raise PipelineError(f"Cannot resume; extracted pages directory is missing: {pages_dir}")

    image_paths = sorted(
        (
            path
            for path in pages_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )
    if not image_paths:
        raise PipelineError(f"Cannot resume; no extracted page images found in: {pages_dir}")
    return [Page(index=index, image_path=image_path) for index, image_path in enumerate(image_paths)]


def validate_resume_page(pages: list[Page], resume_page: int, phase: str) -> None:
    if resume_page < 0:
        raise PipelineError("--resume-page must be a non-negative integer.")
    if phase != "package" and resume_page >= len(pages):
        raise PipelineError(
            f"--resume-page {resume_page} is outside the page range 0..{len(pages) - 1}."
        )


def page_name(page: int) -> str:
    return f"page_{page:04d}"


def pages_dir(output_dir: Path) -> Path:
    return output_dir / "pages"


def original_pages_dir(output_dir: Path) -> Path:
    return pages_dir(output_dir) / "original"


def cleaned_pages_dir(output_dir: Path) -> Path:
    return pages_dir(output_dir) / "cleaned"


def final_pages_dir(output_dir: Path) -> Path:
    return pages_dir(output_dir) / "final"


def data_dir(output_dir: Path) -> Path:
    return output_dir / "data"


def input_manifest_path(output_dir: Path) -> Path:
    return data_dir(output_dir) / INPUT_MANIFEST_NAME


def input_archive_manifest(input_cbz: Path, page_count: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        with input_cbz.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        size = input_cbz.stat().st_size
    except OSError as exc:
        raise PipelineError(f"Could not fingerprint input CBZ {input_cbz}: {exc}") from exc
    return {
        "version": 1,
        "sha256": digest.hexdigest(),
        "sizeBytes": size,
        "pageCount": page_count,
    }


def input_archive_page_count(input_cbz: Path) -> int:
    try:
        with zipfile.ZipFile(input_cbz) as archive:
            members = validate_cbz_members(archive)
            return sum(
                1
                for info in members
                if not info.is_dir() and is_image_member(info.filename)
            )
    except zipfile.BadZipFile as exc:
        raise PipelineError(f"Input file is not a valid CBZ/zip: {input_cbz}") from exc


def verify_resume_input(input_cbz: Path, output_dir: Path) -> None:
    extracted_page_count = count_output_pages(output_dir)
    archive_page_count = input_archive_page_count(input_cbz)
    if archive_page_count != extracted_page_count:
        raise PipelineError(
            f"Cannot resume because the input CBZ has {archive_page_count} pages but the output "
            f"directory has {extracted_page_count} extracted pages."
        )
    current = input_archive_manifest(input_cbz, archive_page_count)
    path = input_manifest_path(output_dir)
    if not path.exists():
        print(
            "warning: this output predates input manifests; binding it to the current CBZ for future resumes.",
            file=sys.stderr,
        )
        write_json(path, current)
        return

    stored = load_json(path, "input manifest")
    if not isinstance(stored, dict):
        raise PipelineError(f"Input manifest must be a JSON object: {path}")
    if stored.get("sha256") != current["sha256"] or stored.get("sizeBytes") != current["sizeBytes"]:
        raise PipelineError(
            "Cannot resume with a different input CBZ. Use the archive that created this output "
            "directory or start a new output directory."
        )
    stored_page_count = stored.get("pageCount")
    if isinstance(stored_page_count, int) and stored_page_count != extracted_page_count:
        raise PipelineError(
            f"Cannot resume because the extracted page count ({extracted_page_count}) does not "
            f"match the input manifest ({stored_page_count})."
        )


def data_phase_dir(output_dir: Path, phase: str) -> Path:
    return data_dir(output_dir) / phase


def data_page_index_path(output_dir: Path, phase: str, page_index: int) -> Path:
    return data_phase_dir(output_dir, phase) / f"{page_name(page_index)}.json"


def data_page_path(output_dir: Path, phase: str, page: Page) -> Path:
    return data_page_index_path(output_dir, phase, page.index)


def master_json_path(output_dir: Path) -> Path:
    return data_dir(output_dir) / "master.json"


def debug_dir(output_dir: Path) -> Path:
    return output_dir / "debug"


def debug_phase_dir(output_dir: Path, phase: str) -> Path:
    return debug_dir(output_dir) / phase


def debug_image_path(output_dir: Path, phase: str, page: Page) -> Path:
    return debug_phase_dir(output_dir, phase) / f"{page.image_path.stem}.png"


def traces_dir(output_dir: Path) -> Path:
    return output_dir / "traces"


def trace_phase_dir(output_dir: Path, phase: str) -> Path:
    return traces_dir(output_dir) / phase


def trace_page_path(output_dir: Path, phase: str, page: Page) -> Path:
    return trace_phase_dir(output_dir, phase) / f"{page_name(page.index)}.json"


def translated_cbz_path(output_dir: Path) -> Path:
    return output_dir / TRANSLATED_PNG_CBZ_NAME


def translated_webp_cbz_path(output_dir: Path) -> Path:
    return output_dir / TRANSLATED_WEBP_CBZ_NAME


def translated_jxl_cbz_path(output_dir: Path) -> Path:
    return output_dir / TRANSLATED_JXL_CBZ_NAME


def load_image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chat_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def object_field(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    if hasattr(value, key):
        return getattr(value, key)
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict) and key in model_extra:
        return model_extra[key]
    pydantic_extra = getattr(value, "__pydantic_extra__", None)
    if isinstance(pydantic_extra, dict) and key in pydantic_extra:
        return pydantic_extra[key]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except TypeError:
            dumped = {}
        if isinstance(dumped, dict):
            return dumped.get(key, default)
    return default


def text_from_delta_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return chat_content_to_text(value)
    return str(value)


def answer_delta_text(delta: Any) -> str:
    return text_from_delta_field(object_field(delta, "content"))


def reasoning_delta_text(delta: Any) -> str:
    for field_name in REASONING_FIELD_NAMES:
        text = text_from_delta_field(object_field(delta, field_name))
        if text:
            return text
    return ""


def chunk_choices(chunk: Any) -> list[Any]:
    choices = object_field(chunk, "choices", [])
    if choices is None:
        return []
    return list(choices)


def usage_completion_tokens(chunk: Any) -> int | None:
    usage = object_field(chunk, "usage")
    if usage is None:
        return None
    completion_tokens = object_field(usage, "completion_tokens")
    if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
        return completion_tokens
    return None


def vlm_status_live_updates_enabled() -> bool:
    return sys.stderr.isatty()


def write_vlm_status_line(text: str, previous_length: int) -> int:
    if not vlm_status_live_updates_enabled():
        return previous_length
    clear = " " * max(0, previous_length - len(text))
    print(f"\r{text}{clear}", end="", file=sys.stderr, flush=True)
    return len(text)


def clear_vlm_status_line(previous_length: int) -> None:
    if previous_length and vlm_status_live_updates_enabled():
        print(f"\r{' ' * previous_length}\r", end="", file=sys.stderr, flush=True)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds - minutes * 60
    return f"{minutes}m{remaining_seconds:04.1f}s"


def vlm_status_text(label: str, state: str, token_count: int, elapsed: float) -> str:
    return (
        f"{label}: {state} | {format_elapsed(elapsed)} elapsed | "
        f"{token_count} tokens | {token_count / max(elapsed, 0.001):.1f} tok/s"
    )


def finish_vlm_status_line(
    label: str,
    state: str,
    generated_tokens: int,
    elapsed: float,
    previous_length: int,
    completion_tokens: int | None,
) -> None:
    token_count = completion_tokens if completion_tokens is not None else generated_tokens
    status = vlm_status_text(label, f"done {state}", token_count, elapsed)
    if vlm_status_live_updates_enabled():
        write_vlm_status_line(status, previous_length)
        print(file=sys.stderr, flush=True)
    else:
        print(status, file=sys.stderr, flush=True)


def close_vlm_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def stream_vlm_response(stream: Any, label: str) -> VLMStreamResult:
    started_at = time.monotonic()
    generated_tokens = 0
    completion_tokens: int | None = None
    reasoning_chunks = 0
    output_chunks = 0
    state = "waiting"
    previous_length = write_vlm_status_line(vlm_status_text(label, state, 0, 0.0), 0)
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []

    try:
        for chunk in stream:
            maybe_completion_tokens = usage_completion_tokens(chunk)
            if maybe_completion_tokens is not None:
                completion_tokens = maybe_completion_tokens

            chunk_changed = maybe_completion_tokens is not None
            for choice in chunk_choices(chunk):
                delta = object_field(choice, "delta", {})
                reasoning_text = reasoning_delta_text(delta)
                answer_text = answer_delta_text(delta)
                if reasoning_text:
                    state = "reasoning"
                    generated_tokens += 1
                    reasoning_chunks += 1
                    chunk_changed = True
                    reasoning_parts.append(reasoning_text)
                if answer_text:
                    state = "answering"
                    generated_tokens += 1
                    output_chunks += 1
                    chunk_changed = True
                    answer_parts.append(answer_text)

            if chunk_changed:
                elapsed = max(time.monotonic() - started_at, 0.001)
                token_count = completion_tokens if completion_tokens is not None else generated_tokens
                previous_length = write_vlm_status_line(
                    vlm_status_text(label, state, token_count, elapsed),
                    previous_length,
                )
    except Exception:
        clear_vlm_status_line(previous_length)
        raise
    finally:
        close_vlm_stream(stream)

    elapsed = max(time.monotonic() - started_at, 0.001)
    finish_vlm_status_line(
        label,
        state,
        generated_tokens,
        elapsed,
        previous_length,
        completion_tokens,
    )
    return VLMStreamResult(
        output="".join(answer_parts),
        reasoning="".join(reasoning_parts),
        elapsed_seconds=elapsed,
        generated_tokens=generated_tokens,
        completion_tokens=completion_tokens,
        reasoning_chunks=reasoning_chunks,
        output_chunks=output_chunks,
    )


def stream_options_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "stream_options" in message or "include_usage" in message


def vlm_exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and not isinstance(response_status, bool):
        return response_status
    return None


def vlm_exception_text(exc: Exception) -> str:
    parts = [str(exc)]
    for attr in ("body", "response", "message"):
        value = getattr(exc, attr, None)
        if value is not None:
            parts.append(str(value))
    return "\n".join(parts).lower()


def vlm_model_loading_error(exc: Exception) -> bool:
    status_code = vlm_exception_status_code(exc)
    text = vlm_exception_text(exc)
    has_503 = status_code == 503 or "error code: 503" in text or "'code': 503" in text
    if not has_503:
        return False
    return (
        "loading model" in text
        or "model loading" in text
        or "unavailable_error" in text
        or "unavailable" in text
    )


def create_vlm_stream(client: Any, request_args: dict[str, Any], include_usage: bool) -> Any:
    if include_usage:
        return client.chat.completions.create(
            **request_args,
            stream_options={"include_usage": True},
        )
    return client.chat.completions.create(**request_args)


def call_vlm_once(client: Any, request_args: dict[str, Any], label: str) -> VLMStreamResult:
    try:
        return stream_vlm_response(create_vlm_stream(client, request_args, True), label)
    except PipelineCancelled:
        raise
    except Exception as exc:
        if not stream_options_unsupported(exc):
            raise
        print(
            f"{label}: endpoint rejected stream_options; retrying without usage stats",
            file=sys.stderr,
        )

    return stream_vlm_response(create_vlm_stream(client, request_args, False), label)


def openai_client(config: VLMConfig) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise PipelineError(
            "The OpenAI Python client is not installed. Run `uv add openai` in this project."
        ) from exc

    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def close_openai_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        print(f"warning: failed to close VLM client: {exc}", file=sys.stderr)


def vlm_extra_body(config: VLMConfig) -> dict[str, Any]:
    extra_body: dict[str, Any] = {}
    if config.provider is not None:
        extra_body["provider"] = config.provider
    if config.thinking_budget_tokens >= 0:
        extra_body["thinking_budget_tokens"] = config.thinking_budget_tokens
        if config.thinking_budget_tokens == 0:
            chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
            if isinstance(chat_template_kwargs, dict):
                chat_template_kwargs["enable_thinking"] = False
    return extra_body


def finalize_vlm_config(config: PipelineConfig) -> PipelineConfig:
    if config.vlm is None or config.vlm.model:
        return config

    client = openai_client(config.vlm)
    try:
        model = resolve_vlm_model(client, config.vlm)
    finally:
        close_openai_client(client)
    return replace(config, vlm=replace(config.vlm, model=model))


def call_vlm(
    config: VLMConfig,
    prompt: str,
    image_path: Path | None,
    label: str,
    system_prompt: str = "Return only valid JSON. Do not include markdown or explanations.",
) -> VLMStreamResult:
    client = openai_client(config)
    if not config.model:
        raise PipelineError("VLM model was not resolved before calling the VLM.")
    user_content: Any = prompt
    if image_path is not None:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": load_image_data_url(image_path)}},
        ]
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    request_args = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": messages,
        "stream": True,
    }
    extra_body = vlm_extra_body(config)
    if extra_body:
        request_args["extra_body"] = extra_body

    try:
        for loading_attempt in range(1, DEFAULT_VLM_MODEL_LOADING_RETRIES + 2):
            try:
                return call_vlm_once(client, request_args, label)
            except PipelineCancelled:
                raise
            except Exception as exc:
                if (
                    loading_attempt > DEFAULT_VLM_MODEL_LOADING_RETRIES
                    or not vlm_model_loading_error(exc)
                ):
                    raise PipelineError(f"VLM request failed for {label}: {exc}") from exc
                wait_seconds = VLM_MODEL_LOADING_BACKOFF_SECONDS[
                    min(loading_attempt - 1, len(VLM_MODEL_LOADING_BACKOFF_SECONDS) - 1)
                ]
                print(
                    (
                        f"{label}: model is loading/unavailable; waiting {wait_seconds}s "
                        f"before request retry {loading_attempt}/{DEFAULT_VLM_MODEL_LOADING_RETRIES}"
                    ),
                    file=sys.stderr,
                )
                time.sleep(wait_seconds)
    finally:
        close_openai_client(client)

    raise PipelineError(f"VLM request failed for {label}: exhausted model-loading retries")


def resolve_vlm_model(client: Any, config: VLMConfig) -> str:
    if config.model:
        return config.model

    try:
        models = client.models.list()
    except PipelineCancelled:
        raise
    except Exception as exc:
        raise PipelineError(
            f"Config model is empty and model discovery failed at {config.base_url}: {exc}"
        ) from exc

    data = getattr(models, "data", None)
    if not data:
        raise PipelineError(
            f"Config model is empty and no models were returned by {config.base_url}."
        )

    first = data[0]
    model_id = getattr(first, "id", None)
    if not isinstance(model_id, str) or not model_id:
        raise PipelineError(
            f"Config model is empty and the first discovered model has no string id: {first!r}"
        )
    print(f"Using discovered VLM model: {model_id}", file=sys.stderr)
    return model_id


def strip_markdown_code_fence(raw_text: str) -> str:
    lines = raw_text.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
    return "\n".join(lines).strip()


def strip_json_trailing_commas(raw_text: str) -> str:
    cleaned: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(raw_text):
        character = raw_text[index]
        if in_string:
            cleaned.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            cleaned.append(character)
            index += 1
            continue

        if character == ",":
            lookahead = index + 1
            while lookahead < len(raw_text) and raw_text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < len(raw_text) and raw_text[lookahead] in "]}":
                index += 1
                continue

        cleaned.append(character)
        index += 1
    return "".join(cleaned)


def parse_top_level_json_row_sequence(raw_text: str) -> list[Any] | None:
    """Recover rows when a VLM omits only the enclosing JSON array."""
    decoder = json.JSONDecoder(parse_constant=reject_nonfinite_json)
    values: list[Any] = []
    index = 0
    length = len(raw_text)

    while True:
        while index < length and raw_text[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            value, index = decoder.raw_decode(raw_text, index)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(value, (list, dict)):
            return None
        values.append(value)

        while index < length and raw_text[index].isspace():
            index += 1
        if index >= length:
            break
        if raw_text[index] != ",":
            return None
        index += 1

    return values if len(values) >= 2 else None


def parse_json_array(raw_text: str, raw_path: Path) -> list[Any]:
    json_text = strip_json_trailing_commas(strip_markdown_code_fence(raw_text))
    try:
        parsed = json.loads(json_text, parse_constant=reject_nonfinite_json)
    except (json.JSONDecodeError, ValueError) as exc:
        parsed = parse_top_level_json_row_sequence(json_text)
        if (
            parsed is None
            and json_text.startswith("[")
            and not json_text.startswith("[[")
            and json_text.endswith("]]")
        ):
            parsed = parse_top_level_json_row_sequence(json_text[:-1].rstrip())
        if parsed is None:
            raise PipelineError(
                f"VLM response was not valid JSON at {raw_path}: "
                f"{exc.msg if isinstance(exc, json.JSONDecodeError) else exc}"
            ) from exc
        print(
            f"warning: recovered VLM JSON rows missing their outer array at {raw_path}",
            file=sys.stderr,
        )
    if not isinstance(parsed, list):
        if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
            return parsed["rows"]
        raise PipelineError(f"VLM response must be a JSON array or object with rows: {raw_path}")
    return parsed


def vlm_attempt_stem(page: Page, attempt: int | None = None) -> str:
    stem = page_name(page.index)
    if attempt is None:
        return stem
    return f"{stem}.attempt_{attempt}"


def vlm_named_attempt_stem(stem: str, attempt: int | None = None) -> str:
    if attempt is None:
        return stem
    return f"{stem}.attempt_{attempt}"


def vlm_raw_named_path(output_dir: Path, phase: str, stem: str, attempt: int | None = None) -> Path:
    return traces_dir(output_dir) / "vlm_raw" / phase / f"{vlm_named_attempt_stem(stem, attempt)}.txt"


def vlm_debug_named_path(output_dir: Path, phase: str, stem: str, attempt: int | None = None) -> Path:
    return traces_dir(output_dir) / "vlm_debug" / phase / f"{vlm_named_attempt_stem(stem, attempt)}.json"


def vlm_raw_path(output_dir: Path, phase: str, page: Page, attempt: int | None = None) -> Path:
    return vlm_raw_named_path(output_dir, phase, page_name(page.index), attempt)


def vlm_debug_path(output_dir: Path, phase: str, page: Page, attempt: int | None = None) -> Path:
    return vlm_debug_named_path(output_dir, phase, page_name(page.index), attempt)


def call_vlm_named(
    phase: str,
    stem: str,
    label: str,
    prompt: str,
    output_dir: Path,
    config: VLMConfig,
    image_path: Path | None = None,
    attempt: int | None = None,
    system_prompt: str = "Return only valid JSON. Do not include markdown or explanations.",
) -> str:
    raw_path = vlm_raw_named_path(output_dir, phase, stem, attempt)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if attempt is not None:
        label = f"{label} attempt {attempt}/{DEFAULT_VLM_RETRIES}"
    response = call_vlm(config, prompt, image_path, label, system_prompt)
    raw_path.write_text(response.output, encoding="utf-8")
    debug_record = {
        "phase": phase,
        "scope": stem,
        "model": config.model,
        "thinkingBudgetTokens": config.thinking_budget_tokens,
        "elapsedSeconds": round(response.elapsed_seconds, 3),
        "generatedTokens": response.generated_tokens,
        "completionTokens": response.completion_tokens,
        "reasoningChunks": response.reasoning_chunks,
        "outputChunks": response.output_chunks,
        "imagePath": str(image_path) if image_path is not None else None,
        "rawOutputPath": str(raw_path),
        "reasoning": response.reasoning,
        "output": response.output,
    }
    if attempt is not None:
        debug_record["attempt"] = attempt
        debug_record["maxAttempts"] = DEFAULT_VLM_RETRIES
    write_json(vlm_debug_named_path(output_dir, phase, stem, attempt), debug_record)
    return response.output


def get_vlm_array(
    phase: str,
    page: Page,
    prompt: str,
    output_dir: Path,
    config: VLMConfig | None,
    fixture_dir: Path | None,
    image_path: Path | None = None,
    attempt: int | None = None,
) -> list[Any]:
    if fixture_dir is not None:
        fixture_path = fixture_dir / phase / f"{page_name(page.index)}.json"
        data = load_json(fixture_path, f"{phase} fixture")
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data["rows"]
        if not isinstance(data, list):
            raise PipelineError(f"Fixture must be a JSON array or object with rows: {fixture_path}")
        return data

    if config is None:
        raise PipelineError("VLM config is missing.")

    request_image_path = image_path or page.image_path
    label = f"VLM {phase} page {page.index}"
    raw_text = call_vlm_named(
        phase,
        page_name(page.index),
        label,
        prompt,
        output_dir,
        config,
        request_image_path,
        attempt,
    )
    raw_path = vlm_raw_path(output_dir, phase, page, attempt)
    return parse_json_array(raw_text, raw_path)


def mark_vlm_attempt_failed(
    phase: str,
    page: Page,
    output_dir: Path,
    config: VLMConfig | None,
    image_path: Path | None,
    attempt: int,
    exc: Exception,
) -> None:
    debug_path = vlm_debug_path(output_dir, phase, page, attempt)
    debug_record: dict[str, Any] = {}
    if debug_path.exists():
        try:
            loaded = json.loads(debug_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                debug_record = loaded
        except json.JSONDecodeError:
            debug_record = {}

    raw_path = vlm_raw_path(output_dir, phase, page, attempt)
    debug_record.update(
        {
            "phase": phase,
            "page": page.index,
            "attempt": attempt,
            "maxAttempts": DEFAULT_VLM_RETRIES,
            "failed": True,
            "error": str(exc),
            "model": config.model if config is not None else None,
            "imagePath": str(image_path or page.image_path),
        }
    )
    if raw_path.exists():
        debug_record["rawOutputPath"] = str(raw_path)
    write_json(debug_path, debug_record)


def mark_vlm_named_attempt_failed(
    phase: str,
    stem: str,
    output_dir: Path,
    config: VLMConfig | None,
    image_path: Path | None,
    attempt: int,
    exc: Exception,
) -> None:
    debug_path = vlm_debug_named_path(output_dir, phase, stem, attempt)
    debug_record: dict[str, Any] = {}
    if debug_path.exists():
        try:
            loaded = json.loads(debug_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                debug_record = loaded
        except json.JSONDecodeError:
            debug_record = {}

    raw_path = vlm_raw_named_path(output_dir, phase, stem, attempt)
    debug_record.update(
        {
            "phase": phase,
            "scope": stem,
            "attempt": attempt,
            "maxAttempts": DEFAULT_VLM_RETRIES,
            "failed": True,
            "error": str(exc),
            "model": config.model if config is not None else None,
            "imagePath": str(image_path) if image_path is not None else None,
        }
    )
    if raw_path.exists():
        debug_record["rawOutputPath"] = str(raw_path)
    write_json(debug_path, debug_record)


def promote_vlm_attempt_files(output_dir: Path, phase: str, page: Page, attempt: int) -> None:
    for path_fn in (vlm_raw_path, vlm_debug_path):
        attempt_path = path_fn(output_dir, phase, page, attempt)
        final_path = path_fn(output_dir, phase, page)
        if attempt_path.exists():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_path.replace(final_path)
    final_debug_path = vlm_debug_path(output_dir, phase, page)
    if final_debug_path.exists():
        try:
            debug_record = json.loads(final_debug_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            debug_record = None
        if isinstance(debug_record, dict):
            debug_record["rawOutputPath"] = str(vlm_raw_path(output_dir, phase, page))
            write_json(final_debug_path, debug_record)


def promote_vlm_named_attempt_files(output_dir: Path, phase: str, stem: str, attempt: int) -> None:
    for path_fn in (vlm_raw_named_path, vlm_debug_named_path):
        attempt_path = path_fn(output_dir, phase, stem, attempt)
        final_path = path_fn(output_dir, phase, stem)
        if attempt_path.exists():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            attempt_path.replace(final_path)
    final_debug_path = vlm_debug_named_path(output_dir, phase, stem)
    if final_debug_path.exists():
        try:
            debug_record = json.loads(final_debug_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            debug_record = None
        if isinstance(debug_record, dict):
            debug_record["rawOutputPath"] = str(vlm_raw_named_path(output_dir, phase, stem))
            write_json(final_debug_path, debug_record)


def prompt_with_validation_feedback(
    prompt: str,
    last_error: Exception | None,
) -> str:
    if last_error is None:
        return prompt
    return (
        prompt
        + "\n\nVALIDATION RETRY:\n"
        + f"The previous response was rejected: {last_error}\n"
        + "Return the complete response again, correcting that error and following "
        + "the requested schema exactly. Do not explain the correction."
    )


def get_validated_vlm_array(
    phase: str,
    page: Page,
    prompt: str,
    output_dir: Path,
    config: VLMConfig | None,
    fixture_dir: Path | None,
    validator: Callable[[list[Any]], Any],
    image_path: Path | None = None,
) -> Any:
    if fixture_dir is not None:
        return validator(get_vlm_array(phase, page, prompt, output_dir, config, fixture_dir, image_path))
    if config is None:
        raise PipelineError("VLM config is missing.")

    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_VLM_RETRIES + 1):
        try:
            items = get_vlm_array(
                phase,
                page,
                prompt_with_validation_feedback(prompt, last_error),
                output_dir,
                config,
                fixture_dir,
                image_path,
                attempt,
            )
            validated = validator(items)
        except PipelineCancelled:
            raise
        except PipelineError as exc:
            last_error = exc
            mark_vlm_attempt_failed(phase, page, output_dir, config, image_path, attempt, exc)
            print(
                (
                    f"warning: VLM {phase} page {page.index} attempt "
                    f"{attempt}/{DEFAULT_VLM_RETRIES} failed: {exc}"
                ),
                file=sys.stderr,
            )
            if attempt == DEFAULT_VLM_RETRIES:
                raise PipelineError(
                    f"VLM {phase} page {page.index} failed after "
                    f"{DEFAULT_VLM_RETRIES} attempts: {exc}"
                ) from exc
            continue

        promote_vlm_attempt_files(output_dir, phase, page, attempt)
        return validated

    raise PipelineError(
        f"VLM {phase} page {page.index} failed after {DEFAULT_VLM_RETRIES} attempts: {last_error}"
    )


def get_validated_vlm_array_named(
    phase: str,
    stem: str,
    label: str,
    prompt: str,
    output_dir: Path,
    config: VLMConfig | None,
    fixture_dir: Path | None,
    validator: Callable[[list[Any]], Any],
) -> Any:
    if fixture_dir is not None:
        fixture_path = fixture_dir / phase / f"{stem}.json"
        data = load_json(fixture_path, f"{phase} fixture")
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return validator(data["rows"])
        if not isinstance(data, list):
            raise PipelineError(f"Fixture must be a JSON array or object with rows: {fixture_path}")
        return validator(data)
    if config is None:
        raise PipelineError("VLM config is missing.")

    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_VLM_RETRIES + 1):
        try:
            raw_text = call_vlm_named(
                phase,
                stem,
                label,
                prompt_with_validation_feedback(prompt, last_error),
                output_dir,
                config,
                attempt=attempt,
            )
            items = parse_json_array(raw_text, vlm_raw_named_path(output_dir, phase, stem, attempt))
            validated = validator(items)
        except PipelineCancelled:
            raise
        except PipelineError as exc:
            last_error = exc
            mark_vlm_named_attempt_failed(phase, stem, output_dir, config, None, attempt, exc)
            print(
                (
                    f"warning: {label} attempt {attempt}/{DEFAULT_VLM_RETRIES} "
                    f"failed: {exc}"
                ),
                file=sys.stderr,
            )
            if attempt == DEFAULT_VLM_RETRIES:
                raise PipelineError(
                    f"{label} failed after {DEFAULT_VLM_RETRIES} attempts: {exc}"
                ) from exc
            continue

        promote_vlm_named_attempt_files(output_dir, phase, stem, attempt)
        return validated

    raise PipelineError(f"{label} failed after {DEFAULT_VLM_RETRIES} attempts: {last_error}")


def get_vlm_text_named(
    phase: str,
    stem: str,
    label: str,
    prompt: str,
    output_dir: Path,
    config: VLMConfig | None,
    fixture_dir: Path | None,
) -> str:
    if fixture_dir is not None:
        fixture_path = fixture_dir / phase / f"{stem}.txt"
        return fixture_path.read_text(encoding="utf-8")
    if config is None:
        raise PipelineError("VLM config is missing.")
    return call_vlm_named(
        phase,
        stem,
        label,
        prompt,
        output_dir,
        config,
        system_prompt=(
            "Write concise plain text. Do not use JSON, markdown code fences, or explanations "
            "outside the requested notes."
        ),
    )


def get_validated_vlm_text_named(
    phase: str,
    stem: str,
    label: str,
    prompt: str,
    output_dir: Path,
    config: VLMConfig | None,
    fixture_dir: Path | None,
    validator: Callable[[str], Any],
    system_prompt: str = (
        "Return only the requested plain text. Do not use JSON, markdown code fences, "
        "bullets, numbering, or explanations."
    ),
) -> Any:
    if fixture_dir is not None:
        fixture_path = fixture_dir / phase / f"{stem}.txt"
        return validator(fixture_path.read_text(encoding="utf-8"))
    if config is None:
        raise PipelineError("VLM config is missing.")

    last_error: Exception | None = None
    for attempt in range(1, DEFAULT_VLM_RETRIES + 1):
        try:
            raw_text = call_vlm_named(
                phase,
                stem,
                label,
                prompt_with_validation_feedback(prompt, last_error),
                output_dir,
                config,
                attempt=attempt,
                system_prompt=system_prompt,
            )
            validated = validator(raw_text)
        except PipelineCancelled:
            raise
        except PipelineError as exc:
            last_error = exc
            mark_vlm_named_attempt_failed(phase, stem, output_dir, config, None, attempt, exc)
            print(
                (
                    f"warning: {label} attempt {attempt}/{DEFAULT_VLM_RETRIES} "
                    f"failed: {exc}"
                ),
                file=sys.stderr,
            )
            if attempt == DEFAULT_VLM_RETRIES:
                raise PipelineError(
                    f"{label} failed after {DEFAULT_VLM_RETRIES} attempts: {exc}"
                ) from exc
            continue

        promote_vlm_named_attempt_files(output_dir, phase, stem, attempt)
        return validated

    raise PipelineError(f"{label} failed after {DEFAULT_VLM_RETRIES} attempts: {last_error}")


def run_ocr(
    pages: list[Page],
    output_dir: Path,
    config: PipelineConfig,
    start_page: int = 0,
    end_page: int | None = None,
    existing_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
        ocr = paddle_ocr_image.create_paddleocr_vl(
            config.ocr.device,
            config.ocr.paddleocr_vl_server_url,
            config.ocr.paddleocr_vl_model,
            api_key=config.ocr.paddleocr_vl_api_key,
            max_concurrency=config.ocr.paddleocr_vl_max_concurrency,
            service_url=config.ocr.service_url,
            service_timeout=config.ocr.service_timeout,
        )
    else:
        ocr = paddle_ocr_image.create_paddle_ocr(
            config.ocr.lang,
            config.ocr.device,
            config.ocr.text_det_limit_side_len,
            config.ocr.text_det_limit_type,
            use_doc_preprocessor=config.ocr.use_doc_preprocessor,
            use_textline_orientation=config.ocr.use_textline_orientation,
            ocr_version=config.ocr.ocr_version,
            text_detection_model_name=config.ocr.text_detection_model_name,
            text_recognition_model_name=config.ocr.text_recognition_model_name,
            text_detection_model_dir=config.ocr.text_detection_model_dir,
            text_recognition_model_dir=config.ocr.text_recognition_model_dir,
            text_det_thresh=config.ocr.text_det_thresh,
            text_det_box_thresh=config.ocr.text_det_box_thresh,
            text_det_unclip_ratio=config.ocr.text_det_unclip_ratio,
            text_rec_score_thresh=config.ocr.text_rec_score_thresh,
            service_url=config.ocr.service_url,
            service_timeout=config.ocr.service_timeout,
        )

    by_page: dict[int, list[dict[str, Any]]] = dict(existing_by_page or {})
    for page in pages_in_range(pages, start_page, end_page):
        print(f"OCR page {page.index}", file=sys.stderr)
        if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            print(
                (
                    f"OCR page {page.index}: PaddleOCR-VL 1.6 via "
                    f"{config.ocr.paddleocr_vl_server_url}"
                ),
                file=sys.stderr,
            )
            records = paddle_ocr_image.extract_paddleocr_vl_image_records(
                ocr,
                page.image_path,
                page.index,
                config.ocr.min_score,
            )
        else:
            if config.ocr.tile_enabled:
                print(
                    (
                        f"OCR page {page.index}: tiled "
                        f"{config.ocr.tile_width}x{config.ocr.tile_height} "
                        f"overlap {config.ocr.tile_overlap}"
                    ),
                    file=sys.stderr,
                )
            records = paddle_ocr_image.extract_image_records(
                ocr,
                page.image_path,
                page.index,
                config.ocr.min_score,
                tile_enabled=config.ocr.tile_enabled,
                tile_width=config.ocr.tile_width,
                tile_height=config.ocr.tile_height,
                tile_overlap=config.ocr.tile_overlap,
                tile_include_full_image=config.ocr.tile_include_full_image,
                tile_dedupe_iou=config.ocr.tile_dedupe_iou,
                tile_dedupe_containment=config.ocr.tile_dedupe_containment,
            )
        if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL and config.ocr.tile_enabled:
            print(
                f"OCR page {page.index}: warning: tiled OCR is ignored by PaddleOCR-VL.",
                file=sys.stderr,
            )
        by_page[page.index] = records
        write_json(data_page_path(output_dir, "ocr_raw", page), records)
        paddle_ocr_image.draw_boxes(
            records,
            page.image_path,
            debug_image_path(output_dir, "ocr_raw_img", page),
            "#ff2d55",
            3,
            18,
        )
    paddle_ocr_image.close_ocr_engine(ocr)
    ensure_all_pages(by_page, pages, "OCR records")
    return by_page


def flatten_pages(by_page: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in sorted(by_page):
        records.extend(by_page[page])
    return records


def draw_region_debug_image(
    records: list[dict[str, Any]],
    page: Page,
    output_dir: Path,
    directory_name: str,
    color: str,
    region_key: str = "region",
) -> Path:
    output_path = debug_image_path(output_dir, directory_name, page)
    debug_records: list[dict[str, Any]] = []
    for record in records:
        region = record.get(region_key)
        boxno = record.get("boxno", record.get("label"))
        if isinstance(region, list) and len(region) == 4 and isinstance(boxno, int):
            debug_records.append(
                {
                    "boxno": boxno,
                    "region": region,
                }
            )
    paddle_ocr_image.draw_boxes(
        debug_records,
        page.image_path,
        output_path,
        color,
        VLM_DEBUG_BOX_WIDTH,
        VLM_DEBUG_FONT_SIZE,
    )
    return output_path


def ensure_structured_debug_image(
    records: list[dict[str, Any]],
    page: Page,
    output_dir: Path,
) -> Path:
    output_path = debug_image_path(output_dir, "ocr_structured_img", page)
    if not output_path.exists():
        draw_region_debug_image(
            records,
            page,
            output_dir,
            "ocr_structured_img",
            VLM_STRUCTURED_DEBUG_COLOR,
        )
    return output_path


def pages_before(pages: list[Page], page_index: int) -> list[Page]:
    return [page for page in pages if page.index < page_index]


def pages_in_range(
    pages: list[Page],
    start_page: int,
    end_page: int | None = None,
) -> list[Page]:
    return [
        page
        for page in pages
        if page.index >= start_page and (end_page is None or page.index <= end_page)
    ]


def pages_except(pages: list[Page], page_index: int) -> list[Page]:
    return [page for page in pages if page.index != page_index]


def ensure_all_pages(
    by_page: dict[int, list[dict[str, Any]]],
    pages: list[Page],
    label: str,
) -> None:
    missing = [page.index for page in pages if page.index not in by_page]
    if missing:
        raise PipelineError(f"{label} missing pages: {missing}")


def load_json_array(path: Path, label: str) -> list[Any]:
    data = load_json(path, label)
    if not isinstance(data, list):
        raise PipelineError(f"{label} must be a JSON array: {path}")
    return data


def load_phase_records(
    output_dir: Path,
    directory_name: str,
    pages: list[Page],
    label: str,
) -> dict[int, list[dict[str, Any]]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        path = data_page_path(output_dir, directory_name, page)
        records = load_json_array(path, f"{label} page {page.index}")
        by_page[page.index] = [require_object(record, f"{label} page {page.index} item {index}") for index, record in enumerate(records)]
    return by_page


def summarize_records(records: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    compact: list[dict[str, Any]] = []
    for record in records:
        compact.append({field: record[field] for field in fields if field in record})
    return json.dumps(compact, ensure_ascii=False, indent=2)


def compact_prompt_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def compact_records_table(records: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    table = {
        "cols": list(fields),
        "rows": [
            [compact_prompt_value(record.get(field)) for field in fields]
            for record in records
        ],
    }
    return json.dumps(table, ensure_ascii=False, separators=(",", ":"))


def available_font_names() -> set[str]:
    if not FONT_DIR.is_dir():
        return set()
    return {
        path.name
        for path in FONT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
    }


def normalize_font_name(value: Any, backup_font: str | None = None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return backup_font
    normalized = value.strip().strip("[]")
    if backup_font is not None and normalized == backup_font:
        return backup_font
    name = Path(normalized).name
    if name in available_font_names():
        return name
    raise PipelineError(
        f"Placement font is neither the backup font nor a file in {FONT_DIR}: {value}"
    )


def local_font_path(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        name = Path(value.strip().strip("[]")).name
        candidate = FONT_DIR / name
        if candidate.is_file() and candidate.suffix.lower() in FONT_EXTENSIONS:
            return str(candidate)
        return value
    return fallback


def normalize_fill(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PipelineError(f"Placement fill must be black or white: {value!r}")
    normalized = value.strip().lower()
    if normalized not in {"black", "white"}:
        raise PipelineError(f"Placement fill must be black or white: {value!r}")
    return normalized


def outline_for_fill(fill: str) -> str:
    if fill.strip().lower() == "black":
        return "white"
    return "black"


def readable_fill_for_brightness(brightness: int) -> str | None:
    if brightness >= 160:
        return "black"
    if brightness <= 95:
        return "white"
    return None


def placement_region_brightness(
    pixels: Any,
    image_width: int,
    image_height: int,
    region: Any,
) -> int | None:
    if not isinstance(region, list) or len(region) != 4:
        return None
    try:
        left, top, right, bottom = [round(float(value)) for value in region]
    except (TypeError, ValueError):
        return None
    bounds = [
        max(0, min(image_width, left)),
        max(0, min(image_height, top)),
        max(0, min(image_width, right)),
        max(0, min(image_height, bottom)),
    ]
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    values = sampled_pixel_values(pixels, bounds, 8000)
    if not values:
        return None
    dominant = dominant_text_container_brightness(values)
    if dominant is not None:
        return dominant
    return percentile(values, 0.50)


def correct_fill_for_legibility(
    page: Page,
    placement: dict[str, Any],
    fill: str,
    pixels: Any,
    image_width: int,
    image_height: int,
) -> str:
    brightness = placement_region_brightness(
        pixels,
        image_width,
        image_height,
        placement.get("placementRegion", placement.get("region")),
    )
    if brightness is None:
        return fill
    recommended = readable_fill_for_brightness(brightness)
    if recommended is None or recommended == fill:
        return fill
    print(
        "warning: adjusted fill for page "
        f"{page.index} boxno {placement.get('boxno')} from {fill} to {recommended} "
        f"for background brightness {brightness}",
        file=sys.stderr,
    )
    return recommended


def correct_placement_fills_for_legibility(
    page: Page,
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    with Image.open(page.image_path) as image:
        gray = image.convert("L")
        pixels = gray.load()
        image_width, image_height = gray.size
        corrected: list[dict[str, Any]] = []
        for placement in placements:
            item = dict(placement)
            fill = item.get("fill")
            if isinstance(fill, str):
                item["fill"] = correct_fill_for_legibility(
                    page,
                    item,
                    fill,
                    pixels,
                    image_width,
                    image_height,
                )
            corrected.append(item)
    return corrected


def font_use_prompt(backup_font: str) -> str:
    if not FONT_USE_PATH.is_file():
        return f"No local font_use.txt was found. Use backup font {backup_font!r}."

    lines: list[str] = []
    font_names = available_font_names()
    for raw_line in FONT_USE_PATH.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or ":" not in raw_line:
            continue
        raw_name, raw_uses = raw_line.split(":", 1)
        font_name = raw_name.strip().strip("[]")
        if font_names and font_name not in font_names:
            continue
        lines.append(f"- {font_name}: {raw_uses.strip()}")

    if not lines:
        return f"No usable local fonts were listed. Use backup font {backup_font!r}."
    return "\n".join(lines)


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


def write_merged_ocr_outputs(
    pages: list[Page],
    merged_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    for page in pages:
        records = merged_by_page[page.index]
        write_json(data_page_path(output_dir, "ocr_merged", page), records)
        draw_region_debug_image(
            records,
            page,
            output_dir,
            "ocr_merged_img",
            OCR_MERGED_DEBUG_COLOR,
        )


def structure_prompt(
    page: Page,
    merged_by_page: dict[int, list[dict[str, Any]]],
    language: LanguageConfig,
) -> str:
    page_ocr = merged_by_page[page.index]
    source_name = language.source.name
    return f"""
You are classifying and ordering OCR groups for one comic page.
{language.source.structure_context}
Reading order guidance: use {language.source.reading_order}.

You are given the page image and script-merged OCR groups for the current page.
The groups are already merged geometrically. Do not merge groups. Do not split groups. Do not drop groups.
Your job is only to put real {source_name} text groups in reading order, reject only clear OCR false positives,
and classify each kept group.
Return only a compact JSON array of rows for the current page.
Row format:
[mergedBoxno,classification]
or, only when correcting OCR:
[mergedBoxno,classification,textCorrection]
- mergedBoxno: one script-merged OCR group boxno from this page.
- classification: exactly one of "text", "sfx", or "reject".
- textCorrection: optional corrected OCR text string. Omit it when no correction is needed.

The array order itself is the reading order. There is no output-order field.
The first value in every row is always a numbered image label, never an output position.
For example, if labels 2, 0, and 1 are normal text in that reading order, return:
[[2,"text"],[0,"text"],[1,"text"]]
Never return rows such as [0,0,2,...] with a separate order number.

Rejection policy is intentionally conservative:
- Default to reject false.
- Set reject true only when the box is clearly not intentional text or lettering visible in the image.
- Reject only obvious false positives such as screen tone texture, hatching, panel borders, character art,
  background art, dust/noise, random isolated punctuation, cropped fragments with no readable character,
  or OCR boxes that contain only furigana/reading marks separated from their base text.
- Do not reject readable {source_name} text, even if the OCR string is wrong, garbled, incomplete, badly ordered,
  split oddly, mixes furigana with base text, or contains extra punctuation.
- Do not reject text inside speech bubbles, thought bubbles, caption boxes, narration boxes, rectangular
  boxes, signs, labels, UI-like boxes, bordered areas, white/blank text areas, or decorative containers.
- Do not reject text because it is short. One-character or punctuation-only speech/SFX can be intentional.
- If a human comic reader would treat the mark as intentional text or sound effect, reject must be false.
- If unsure whether a group is text/noise/artifact, choose reject false.

The OCR text is already provided. Do not retype it unless correction is needed. If the OCR text is clearly
wrong and you can confidently correct it, put the corrected text in textCorrection. Otherwise omit it.

Return exactly {len(page_ocr)} rows. Every mergedBoxno from this page must appear exactly once, including
rejected boxes. No mergedBoxno may appear twice.

Return kept rows in the requested reading order. Rejected rows may appear where they occur visually.
Do not include page, sourceBoxnos, region, text, markdown, or explanations.

Region coordinates are [left,top,right,bottom] pixels with the origin at the image's top-left.
The numbered labels drawn on the image match mergedBoxno. Use those labels and the image directly;
do not re-derive or debate the coordinate convention. Make one concise classification/order pass and return.

Current page index:
{page.index}

Current page script-merged OCR groups compact table:
{compact_records_table(page_ocr, ("boxno", "sourceBoxnos", "region", "text"))}
""".strip()


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be an object.")
    return value


def require_row(value: Any, label: str, min_length: int) -> list[Any]:
    if not isinstance(value, list):
        raise PipelineError(f"{label} must be an array row.")
    if len(value) < min_length:
        raise PipelineError(f"{label} must have at least {min_length} values.")
    return value


def parse_boolish(value: Any, label: str, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise PipelineError(f"{label} must include boolean `{key}`.")


def require_bool(record: dict[str, Any], key: str, label: str) -> bool:
    return parse_boolish(record.get(key), label, key)


def optional_bool(record: dict[str, Any], key: str, default: bool, label: str) -> bool:
    value = record.get(key, default)
    return parse_boolish(value, label, key)


def require_string(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise PipelineError(f"{label} must include string `{key}`.")
    return value


def optional_non_empty_string(record: dict[str, Any], key: str, label: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PipelineError(f"{label} {key} must be a string when provided.")
    value = value.strip()
    return value or None


def require_non_negative_int(record: dict[str, Any], key: str, label: str) -> int:
    return parse_non_negative_int_value(record.get(key), label, key)


def parse_non_negative_int_value(value: Any, label: str, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PipelineError(f"{label} must include non-negative integer `{key}`.")
    return value


def source_boxnos(record: dict[str, Any], label: str) -> list[int]:
    value = record.get("sourceBoxnos")
    if not isinstance(value, list) or not value:
        raise PipelineError(f"{label} must include non-empty array `sourceBoxnos`.")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise PipelineError(f"{label} sourceBoxnos must be non-negative integers.")
        result.append(item)
    return result


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


def merged_boxno_from_order_record(
    record: dict[str, Any],
    label: str,
    merged_source_lookup: dict[tuple[int, ...], int],
) -> int:
    if "mergedBoxno" in record:
        return require_non_negative_int(record, "mergedBoxno", label)
    if "boxno" in record:
        return require_non_negative_int(record, "boxno", label)
    if "sourceBoxnos" in record:
        refs = tuple(source_boxnos(record, label))
        if refs in merged_source_lookup:
            return merged_source_lookup[refs]
        raise PipelineError(
            f"{label} sourceBoxnos must exactly match one script-merged OCR group."
        )
    raise PipelineError(f"{label} must include `mergedBoxno`.")


def validate_ordered_merged_page(
    page: Page,
    merged_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    merged_by_boxno = {record["boxno"]: record for record in merged_records}
    merged_source_lookup = {
        tuple(record["sourceBoxnos"]): record["boxno"] for record in merged_records
    }
    structured: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, value in enumerate(vlm_items):
        label = f"structured page {page.index} item {index}"
        text_correction: str | None
        open_lettering = False
        if isinstance(value, list):
            row = require_row(value, label, 2)
            merged_boxno = parse_non_negative_int_value(row[0], label, "mergedBoxno")
            classification = (
                row[1].strip().lower() if isinstance(row[1], str) else ""
            )
            if classification in {"text", "sfx", "reject"}:
                if len(row) > 3:
                    raise PipelineError(
                        f"{label} classification row must have two or three values."
                    )
                reject = classification == "reject"
                sfx = classification == "sfx"
                text_correction = None
                if len(row) == 3:
                    if not isinstance(row[2], str):
                        raise PipelineError(
                            f"{label} textCorrection must be a string when provided."
                        )
                    text_correction = row[2].strip() or None
            else:
                # Accept the earlier compact schema for fixtures and manually
                # supplied responses, while new prompts use named classifications.
                reject = parse_boolish(row[1], label, "reject")
                sfx = False
                text_correction = None
                if not reject:
                    if len(row) < 3:
                        raise PipelineError(f"{label} kept row must include sfx.")
                    sfx = parse_boolish(row[2], label, "sfx")
                    if len(row) >= 5:
                        open_lettering = parse_boolish(
                            row[3], label, "openLettering"
                        )
                        if isinstance(row[4], str) and row[4].strip():
                            text_correction = row[4].strip()
                    elif len(row) >= 4:
                        if isinstance(row[3], str):
                            text_correction = row[3].strip() or None
                        else:
                            open_lettering = parse_boolish(
                                row[3], label, "openLettering"
                            )
        else:
            record = require_object(value, label)
            merged_boxno = merged_boxno_from_order_record(record, label, merged_source_lookup)
            reject = optional_bool(record, "reject", False, label)
            sfx = False
            text_correction = None
            if not reject:
                sfx = require_bool(record, "sfx", label)
                open_lettering = optional_bool(record, "openLettering", False, label)
                text_correction = optional_non_empty_string(record, "textCorrection", label)
                if text_correction is None:
                    text_correction = optional_non_empty_string(record, "text", label)
        if merged_boxno not in merged_by_boxno:
            raise PipelineError(f"{label} references missing mergedBoxno {merged_boxno}.")
        if merged_boxno in seen:
            raise PipelineError(f"{label} duplicates mergedBoxno {merged_boxno}.")
        seen.add(merged_boxno)

        merged = merged_by_boxno[merged_boxno]
        if reject:
            continue

        text = text_correction if text_correction is not None else str(merged.get("text", ""))
        structured.append(
            {
                "page": page.index,
                "boxno": len(structured),
                "sourceBoxnos": merged["sourceBoxnos"],
                "sourceTexts": merged["sourceTexts"],
                "region": merged["region"],
                "sfx": sfx,
                "openLettering": open_lettering,
                "text": text,
            }
        )

    missing = sorted(set(merged_by_boxno) - seen)
    if missing:
        raise PipelineError(f"Structured ordering missing page {page.index} mergedBoxnos: {missing}")
    return structured


def run_structure_phase(
    pages: list[Page],
    raw_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    start_page: int = 0,
    end_page: int | None = None,
    existing_by_page: dict[int, list[dict[str, Any]]] | None = None,
    use_existing_merged: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    ensure_all_pages(raw_by_page, pages, "OCR records")
    selected_pages = pages_in_range(pages, start_page, end_page)
    selected_page_indexes = {page.index for page in selected_pages}
    merged_by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        existing_merged_path = data_page_path(output_dir, "ocr_merged", page)
        if use_existing_merged and page.index in selected_page_indexes and existing_merged_path.exists():
            merged_by_page[page.index] = load_json_array(
                existing_merged_path,
                f"script-merged OCR records page {page.index}",
            )
        else:
            merged_by_page[page.index] = merge_ocr_records_for_page(
                page,
                raw_by_page[page.index],
                right_to_left=config.language.source.code != "kr",
            )
    write_merged_ocr_outputs(selected_pages, merged_by_page, output_dir)

    structured_by_page: dict[int, list[dict[str, Any]]] = dict(existing_by_page or {})
    for page in selected_pages:
        if not merged_by_page[page.index]:
            print(f"Skip VLM structure page {page.index}: no merged OCR records", file=sys.stderr)
            structured: list[dict[str, Any]] = []
            structured_by_page[page.index] = structured
            write_json(data_page_path(output_dir, "ocr_structured", page), structured)
            draw_region_debug_image(
                structured,
                page,
                output_dir,
                "ocr_structured_img",
                VLM_STRUCTURED_DEBUG_COLOR,
            )
            continue

        print(f"VLM structure page {page.index}", file=sys.stderr)
        merged_debug_path = draw_region_debug_image(
            merged_by_page[page.index],
            page,
            output_dir,
            "ocr_merged_img",
            OCR_MERGED_DEBUG_COLOR,
        )
        structured = get_validated_vlm_array(
            "ocr_structured",
            page,
            structure_prompt(page, merged_by_page, config.language),
            output_dir,
            config.vlm,
            fixture_dir,
            lambda items, current_page=page: validate_ordered_merged_page(
                current_page,
                merged_by_page[current_page.index],
                items,
            ),
            merged_debug_path,
        )
        structured_by_page[page.index] = structured
        write_json(data_page_path(output_dir, "ocr_structured", page), structured)
        draw_region_debug_image(
            structured,
            page,
            output_dir,
            "ocr_structured_img",
            VLM_STRUCTURED_DEBUG_COLOR,
        )
    ensure_all_pages(structured_by_page, pages, "Structured OCR records")
    return structured_by_page


ALT_PLACEMENT_REASON_CODES = {
    0: "bubble",
    1: "caption_box",
    2: "bordered_box",
    3: "blank_text_area",
    4: "sign_label",
    5: "over_art",
    6: "over_face_body",
    7: "integrated_sfx",
    8: "unclear",
}
ALT_PLACEMENT_REASONS = set(ALT_PLACEMENT_REASON_CODES.values())


def alt_placement_prompt(
    page: Page,
    structured_records: list[dict[str, Any]],
    language: LanguageConfig,
) -> str:
    source_name = language.source.name
    target_name = language.target.name
    return f"""
You are deciding erase safety for one comic page.
{language.source.structure_context}

For each structured {source_name} text box, decide whether the original source text can be safely erased
and redrawn in the same region, or whether the {target_name} translation needs alternate placement.

Return only a compact JSON array of rows. Row format:
[boxno,safeToEraseOriginal,reasonCode]
- boxno: one current-page structured box number.
- safeToEraseOriginal: use 1 when the original text can be cleaned/erased safely, 0 when it needs alternate placement.
- reasonCode: use 0=bubble, 1=caption_box, 2=bordered_box, 3=blank_text_area, 4=sign_label,
  5=over_art, 6=over_face_body, 7=integrated_sfx, or 8=unclear.
- All three values are integers. Do not emit reason names in the JSON.
- The outer array is mandatory, including for one record. A one-record response looks like [[0,1,0]].

Policy:
- Default to safeToEraseOriginal=1.
- Use 1 for text inside speech balloons, thought balloons, caption boxes, narration boxes, rectangular/torn-edge boxes,
  signs, labels, panels reserved for text, or white/blank text areas.
- Use 1 for normal dialogue or narration that has a clear container or intentionally blank text area, even if the
  container is large, irregular, partially cropped, or sits over artwork.
- Use 0 only when erasing the original text would likely damage important page art: faces, bodies, clothing,
  props, action lines, speed lines, detailed backgrounds, or hand-drawn SFX integrated into the artwork.
- Use 0 for text clearly free-floating directly on artwork/background with no enclosing visual container and no
  dedicated blank/white text area.
- SFX can be unsafe only when the lettering itself is integrated into the art with no safe blank/container area.
  SFX inside a bubble, caption box, panel space, or white sound-effect area is safe.
- If uncertain, choose safeToEraseOriginal=1. The purpose is to avoid unnecessary alternate placement.

The page image is annotated with the current box numbers. Return one row for every input record and no others.
Do not include source text, page number, markdown, or explanations.

Current page index:
{page.index}

Structured records compact table:
{compact_records_table(structured_records, ("boxno", "region", "text", "sfx"))}
""".strip()


def normalize_alt_placement_reason(value: Any, safe_to_erase: bool) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        reason = ALT_PLACEMENT_REASON_CODES.get(value)
        if reason is not None:
            return reason
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in ALT_PLACEMENT_REASONS:
            return normalized
    return "blank_text_area" if safe_to_erase else "unclear"


def default_alt_placements_for_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    reason: str = "disabled",
) -> list[dict[str, Any]]:
    return [
        {
            "page": page.index,
            "boxno": record["boxno"],
            "safeToEraseOriginal": True,
            "openLettering": False,
            "altPlacementReason": reason,
        }
        for record in structured_records
    ]


def validate_alt_placement_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    expected_boxnos = {record["boxno"] for record in structured_records}
    if len(expected_boxnos) == 1 and vlm_items and not isinstance(vlm_items[0], (list, dict)):
        # Small models commonly collapse [[row]] to [row] for one-record pages.
        vlm_items = [vlm_items]
    entries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, value in enumerate(vlm_items):
        label = f"alt-placement page {page.index} item {index}"
        if isinstance(value, list):
            row = require_row(value, label, 2)
            boxno = parse_non_negative_int_value(row[0], label, "boxno")
            safe_to_erase = parse_boolish(row[1], label, "safeToEraseOriginal")
            reason_value = row[2] if len(row) >= 3 else None
        else:
            record = require_object(value, label)
            item_page = require_non_negative_int(record, "page", label) if "page" in record else page.index
            if item_page != page.index:
                raise PipelineError(f"{label} has page {item_page}, expected {page.index}.")
            boxno = require_non_negative_int(record, "boxno", label)
            if "safeToEraseOriginal" in record:
                safe_to_erase = require_bool(record, "safeToEraseOriginal", label)
            elif "needsAlternatePlacement" in record:
                safe_to_erase = not require_bool(record, "needsAlternatePlacement", label)
            elif "openLettering" in record:
                safe_to_erase = not require_bool(record, "openLettering", label)
            else:
                raise PipelineError(f"{label} must include safeToEraseOriginal.")
            reason_value = record.get("reason", record.get("altPlacementReason"))
        if boxno not in expected_boxnos:
            raise PipelineError(f"{label} references missing boxno {boxno}.")
        if boxno in seen:
            raise PipelineError(f"{label} duplicates boxno {boxno}.")
        seen.add(boxno)
        entries.append(
            {
                "page": page.index,
                "boxno": boxno,
                "safeToEraseOriginal": safe_to_erase,
                "openLettering": not safe_to_erase,
                "altPlacementReason": normalize_alt_placement_reason(reason_value, safe_to_erase),
            }
        )
    missing = sorted(expected_boxnos - seen)
    if missing:
        raise PipelineError(f"Alt-placement missing page {page.index} boxnos: {missing}")
    return sorted(entries, key=lambda item: item["boxno"])


def apply_alt_placements_to_records(
    structured_records: list[dict[str, Any]],
    alt_placements: list[dict[str, Any]],
) -> None:
    by_boxno = {item["boxno"]: item for item in alt_placements}
    for record in structured_records:
        entry = by_boxno[record["boxno"]]
        record["safeToEraseOriginal"] = entry["safeToEraseOriginal"]
        record["openLettering"] = entry["openLettering"]
        record["altPlacementReason"] = entry["altPlacementReason"]


def run_alt_placement_phase(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    start_page: int = 0,
    end_page: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    ensure_all_pages(structured_by_page, pages, "Structured OCR records")
    by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages_in_range(pages, start_page, end_page):
        structured_records = structured_by_page[page.index]
        if not structured_records:
            print(f"Skip VLM alt-placement page {page.index}: no structured OCR records", file=sys.stderr)
            alt_placements: list[dict[str, Any]] = []
        elif not config.alt_placement_enabled:
            print(f"Skip VLM alt-placement page {page.index}: disabled", file=sys.stderr)
            alt_placements = default_alt_placements_for_page(page, structured_records)
        else:
            print(f"VLM alt-placement page {page.index}", file=sys.stderr)
            structured_debug_path = ensure_structured_debug_image(
                structured_records,
                page,
                output_dir,
            )
            alt_placements = get_validated_vlm_array(
                "alt_placement",
                page,
                alt_placement_prompt(page, structured_records, config.language),
                output_dir,
                config.vlm,
                fixture_dir,
                lambda items, current_page=page: validate_alt_placement_page(
                    current_page,
                    structured_by_page[current_page.index],
                    items,
                ),
                structured_debug_path,
            )
        by_page[page.index] = alt_placements
        write_json(data_page_path(output_dir, "alt_placement", page), alt_placements)
        apply_alt_placements_to_records(structured_records, alt_placements)
        write_json(data_page_path(output_dir, "ocr_structured", page), structured_records)
        draw_region_debug_image(
            structured_records,
            page,
            output_dir,
            "ocr_structured_img",
            VLM_STRUCTURED_DEBUG_COLOR,
        )
    return by_page


def hydrate_alt_placement_fields(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Populate split-out alt-placement fields when loading older structured data."""
    ensure_all_pages(structured_by_page, pages, "Structured OCR records")
    for page in pages:
        structured_records = structured_by_page[page.index]
        alt_path = data_page_path(output_dir, "alt_placement", page)
        if alt_path.exists():
            try:
                alt_data = load_json_array(alt_path, f"alt-placement page {page.index}")
                alt_placements = validate_alt_placement_page(page, structured_records, alt_data)
            except PipelineError as exc:
                print(
                    f"warning: ignoring stale alt-placement page {page.index}: {exc}",
                    file=sys.stderr,
                )
            else:
                apply_alt_placements_to_records(structured_records, alt_placements)
                continue
        for record in structured_records:
            existing_open = record.get("openLettering")
            existing_safe = record.get("safeToEraseOriginal")
            if isinstance(existing_open, bool):
                open_lettering = existing_open
                safe_to_erase = not open_lettering
            elif isinstance(existing_safe, bool):
                safe_to_erase = existing_safe
                open_lettering = not safe_to_erase
            else:
                safe_to_erase = True
                open_lettering = False
            record["safeToEraseOriginal"] = safe_to_erase
            record["openLettering"] = open_lettering
            if not isinstance(record.get("altPlacementReason"), str):
                record["altPlacementReason"] = (
                    "unclear" if open_lettering else "blank_text_area"
                )


def load_structured_records(
    output_dir: Path,
    pages: list[Page],
) -> dict[int, list[dict[str, Any]]]:
    structured_by_page = load_phase_records(
        output_dir,
        "ocr_structured",
        pages,
        "Structured OCR records",
    )
    hydrate_alt_placement_fields(pages, structured_by_page, output_dir)
    return structured_by_page


def previous_translation_records(
    page: Page,
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page_index in sorted(structured_by_page):
        if page_index >= page.index:
            continue
        for record in structured_by_page[page_index]:
            if isinstance(record.get("englishText"), str):
                records.append(record)
    return records


def load_translation_notes(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"job": "", "pages": {}}
    data = load_json(path, "translation notes")
    if isinstance(data, dict) and isinstance(data.get("translationNotes"), dict):
        data = data["translationNotes"]
    if not isinstance(data, dict):
        raise PipelineError("Translation notes JSON root must be an object.")
    job_note = data.get("job", "")
    pages = data.get("pages", {})
    return {
        "job": job_note if isinstance(job_note, str) else "",
        "pages": {
            str(page): note
            for page, note in pages.items()
            if isinstance(note, str)
        }
        if isinstance(pages, dict)
        else {},
    }


def translation_notes_prompt_section(page: Page, translation_notes: dict[str, Any]) -> str:
    job_note = translation_notes.get("job", "")
    pages = translation_notes.get("pages", {})
    page_note = pages.get(str(page.index), "") if isinstance(pages, dict) else ""
    sections: list[str] = []
    if isinstance(job_note, str) and job_note.strip():
        sections.append("Job translation notes:\n" + job_note.strip())
    if isinstance(page_note, str) and page_note.strip():
        sections.append(f"Page {page.index} translation notes:\n" + page_note.strip())
    if not sections:
        return ""
    return "\n\nTranslation notes from the user. Treat these as terminology/style/context guidance:\n" + "\n\n".join(sections)


def translation_prompt(
    page: Page,
    structured_by_page: dict[int, list[dict[str, Any]]],
    translation_notes: dict[str, Any],
    language: LanguageConfig,
) -> str:
    master = flatten_pages(structured_by_page)
    current_records = structured_by_page[page.index]
    previous_translations = previous_translation_records(page, structured_by_page)
    notes_section = translation_notes_prompt_section(page, translation_notes)
    source_name = language.source.name
    target_name = language.target.name
    return f"""
You are given an annotated page image and the structured {source_name} text for the CBZ.
Translate from {source_name} to {target_name}.
Translate only the current page records listed in "Current page records to translate".
Do not translate records from any other page.
Do not invent records, page numbers, or box numbers.
Every returned row must use one boxno from the current page records.
If there are no current page records, return [].
Preserve tone and context.
Use previous completed translations to keep names, terms, pronouns, style, and voice consistent.
Return only a compact JSON array of rows. Row format:
[boxno,englishText]
The complete response must begin with the outer `[` and end with the outer `]`.
Example with two records: [[0,"First translation"],[1,"Second translation"]]
Never emit bare rows such as [0,"First translation"],[1,"Second translation"] without enclosing them.
Do not echo page numbers or source text.
englishText must contain the {target_name} translation.
When translating SFX, if it cannot be meaningfully translated then return an empty translation for that text.

Reading order guidance: use {language.source.reading_order}.

Full structured text compact table:
{compact_records_table(master, ("page", "boxno", "text", "sfx", "openLettering"))}

Previous completed translations compact table:
{compact_records_table(previous_translations, ("page", "boxno", "text", "englishText", "sfx", "openLettering"))}

Current page index:
{page.index}

Current page records to translate compact table:
{compact_records_table(current_records, ("boxno", "text", "sfx", "openLettering"))}
{notes_section}
""".strip()


def validate_translation_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    by_boxno = {record["boxno"]: record for record in structured_records}
    seen: set[int] = set()
    translations: list[dict[str, Any]] = []
    for index, value in enumerate(vlm_items):
        label = f"translation page {page.index} item {index}"
        source_text_echo: str | None = None
        if isinstance(value, list):
            row = require_row(value, label, 2)
            if len(row) >= 3 and isinstance(row[0], int) and not isinstance(row[0], bool) and isinstance(row[1], int) and not isinstance(row[1], bool):
                item_page = parse_non_negative_int_value(row[0], label, "page")
                boxno = parse_non_negative_int_value(row[1], label, "boxno")
                english_value = row[3] if len(row) >= 4 else row[2]
                if len(row) >= 4 and isinstance(row[2], str):
                    source_text_echo = row[2]
            else:
                item_page = page.index
                boxno = parse_non_negative_int_value(row[0], label, "boxno")
                english_value = row[1]
            if not isinstance(english_value, str):
                raise PipelineError(f"{label} must include translation text as a string.")
            english_text = english_value
        else:
            record = require_object(value, label)
            item_page = require_non_negative_int(record, "page", label) if "page" in record else page.index
            boxno = require_non_negative_int(record, "boxno", label)
            source_text_echo = record.get("text") if isinstance(record.get("text"), str) else None
            english_text = require_string(record, "englishText", label)
        if item_page != page.index:
            raise PipelineError(f"{label} has page {item_page}, expected {page.index}.")
        if boxno not in by_boxno:
            raise PipelineError(f"{label} references missing boxno {boxno}.")
        if boxno in seen:
            raise PipelineError(f"{label} duplicates boxno {boxno}.")
        seen.add(boxno)
        if source_text_echo is not None and source_text_echo != by_boxno[boxno]["text"]:
            print(
                f"warning: translation text mismatch page {page.index} boxno {boxno}",
                file=sys.stderr,
            )
        translations.append(
            {
                "page": page.index,
                "boxno": boxno,
                "text": by_boxno[boxno]["text"],
                "englishText": english_text,
            }
        )
    missing = sorted(set(by_boxno) - seen)
    if missing:
        raise PipelineError(f"Translation missing page {page.index} boxnos: {missing}")
    return sorted(translations, key=lambda item: item["boxno"])


def attach_translations(
    pages: list[Page],
    output_dir: Path,
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    translations_by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        path = data_page_path(output_dir, "translations", page)
        data = load_json_array(path, f"translation page {page.index}")
        translations = validate_translation_page(page, structured_by_page[page.index], data)
        translations_by_page[page.index] = translations
        by_boxno = {item["boxno"]: item for item in translations}
        for record in structured_by_page[page.index]:
            record["englishText"] = by_boxno[record["boxno"]]["englishText"]
    return translations_by_page


def ensure_translated_pages(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> None:
    for page in pages:
        for record in structured_by_page[page.index]:
            if not isinstance(record.get("englishText"), str):
                raise PipelineError(
                    f"Missing englishText for page {page.index} boxno {record['boxno']}"
                )


def run_translation_phase(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    translation_notes: dict[str, Any],
    start_page: int = 0,
    end_page: int | None = None,
    existing_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    ensure_all_pages(structured_by_page, pages, "Structured OCR records")
    translations_by_page: dict[int, list[dict[str, Any]]] = dict(existing_by_page or {})
    for page in pages_in_range(pages, start_page, end_page):
        structured_records = structured_by_page[page.index]
        if not structured_records:
            print(f"Skip translation page {page.index}: no structured OCR records", file=sys.stderr)
            translations_by_page[page.index] = []
            write_json(data_page_path(output_dir, "translations", page), [])
            data_page_path(output_dir, "translations_raw", page).unlink(missing_ok=True)
            continue

        print(f"VLM translate page {page.index}", file=sys.stderr)
        structured_debug_path = ensure_structured_debug_image(
            structured_records,
            page,
            output_dir,
        )
        translations = get_validated_vlm_array(
            "translations",
            page,
            translation_prompt(page, structured_by_page, translation_notes, config.language),
            output_dir,
            config.vlm,
            fixture_dir,
            lambda items, current_page=page: validate_translation_page(
                current_page,
                structured_by_page[current_page.index],
                items,
            ),
            structured_debug_path,
        )
        translations_by_page[page.index] = translations
        by_boxno = {item["boxno"]: item for item in translations}
        for record in structured_records:
            record["englishText"] = by_boxno[record["boxno"]]["englishText"]
        write_json(data_page_path(output_dir, "translations", page), translations)
        data_page_path(output_dir, "translations_raw", page).unlink(missing_ok=True)
    ensure_translated_pages(pages, structured_by_page)
    return translations_by_page


def all_translated_records_for_review(
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in flatten_pages(structured_by_page):
        records.append(
            {
                "page": record["page"],
                "boxno": record["boxno"],
                "text": record.get("text", ""),
                "englishText": record.get("englishText", ""),
                "sfx": record.get("sfx", False),
                "openLettering": record.get("openLettering", False),
            }
        )
    return records


def proofread_records_in_order(
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return sorted(
        all_translated_records_for_review(structured_by_page),
        key=lambda record: (record["page"], record["boxno"]),
    )


def compact_proofreading_field(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def compact_proofreading_input(records: list[dict[str, Any]]) -> str:
    lines = ["page\tboxno\tsfx\topenLettering\tsourceText\tenglishText"]
    for record in records:
        lines.append(
            "\t".join(
                (
                    str(record["page"]),
                    str(record["boxno"]),
                    "1" if record.get("sfx") else "0",
                    "1" if record.get("openLettering") else "0",
                    compact_proofreading_field(record.get("text", "")),
                    compact_proofreading_field(record.get("englishText", "")),
                )
            )
        )
    return "\n".join(lines)


def translation_notes_book_prompt_section(translation_notes: dict[str, Any]) -> str:
    job_note = translation_notes.get("job", "")
    pages = translation_notes.get("pages", {})
    sections: list[str] = []
    if isinstance(job_note, str) and job_note.strip():
        sections.append("Job translation notes:\n" + job_note.strip())
    if isinstance(pages, dict):
        page_sections = [
            f"Page {page} notes:\n{note.strip()}"
            for page, note in sorted(pages.items(), key=lambda item: natural_key(str(item[0])))
            if isinstance(note, str) and note.strip()
        ]
        if page_sections:
            sections.append("Page translation notes:\n" + "\n\n".join(page_sections))
    if not sections:
        return "No user translation notes were provided."
    return "\n\n".join(sections)


def proofreading_prompt(
    structured_by_page: dict[int, list[dict[str, Any]]],
    translation_notes: dict[str, Any],
    language: LanguageConfig,
) -> str:
    records = proofread_records_in_order(structured_by_page)
    source_name = language.source.name
    target_name = language.target.name
    return f"""
Proofread the complete comic translation as one whole-book pass.

You are given every translated record with {source_name} source text and current {target_name} text,
plus user translation notes. Input rows are tab-separated and already sorted in correct
page/boxno order. Improve consistency and natural {target_name} flow across the entire CBZ while
preserving meaning.

Goals:
- Keep names, terms, pronouns, titles, attacks, magic, locations, and recurring phrases consistent.
- Improve awkward phrasing, grammar, and dialogue flow.
- Preserve character voice, tone, and scene context.
- Keep translations concise enough for comic lettering.
- Do not rewrite good translations merely to be different.
- Do not invent new records or translate text that is not present.
- Preserve empty englishText for SFX or records intentionally left untranslated unless a concise useful
  translation is clearly warranted.

Output format:
- Return exactly {len(records)} lines.
- Line 1 is the final englishText for input row 1, line 2 for input row 2, and so on.
- Keep the exact input page/boxno order.
- Do not include page numbers, box numbers, labels, bullets, quotes, markdown, JSON, or explanations.
- Each line must contain only the final {target_name} text for that record.
- If the final {target_name} text should be blank, output exactly <EMPTY> on that line.
- Do not add or remove lines.

User translation notes:
{translation_notes_book_prompt_section(translation_notes)}

Input rows:
{compact_proofreading_input(records)}
""".strip()


def validate_proofread_translations(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    raw_text: str,
) -> dict[int, list[dict[str, Any]]]:
    records = proofread_records_in_order(structured_by_page)
    lines = strip_markdown_code_fence(raw_text).splitlines()
    if len(lines) != len(records):
        raise PipelineError(
            f"Proofreading returned {len(lines)} lines, expected {len(records)}."
        )
    by_page: dict[int, list[dict[str, Any]]] = {page.index: [] for page in pages}
    for source, line in zip(records, lines, strict=True):
        english_text = line.strip()
        if english_text == "<EMPTY>":
            english_text = ""
        by_page[source["page"]].append(
            {
                "page": source["page"],
                "boxno": source["boxno"],
                "text": source["text"],
                "englishText": english_text,
            }
        )
    for page_records in by_page.values():
        page_records.sort(key=lambda item: item["boxno"])
    return by_page


def apply_translation_records(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    translations_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    for page in pages:
        translations = translations_by_page.get(page.index, [])
        by_boxno = {item["boxno"]: item for item in translations}
        for record in structured_by_page[page.index]:
            if record["boxno"] in by_boxno:
                record["englishText"] = by_boxno[record["boxno"]]["englishText"]
        write_json(data_page_path(output_dir, "translations", page), translations)


def snapshot_raw_translation_records(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    for page in pages:
        snapshot_path = data_page_path(output_dir, "translations_raw", page)
        if snapshot_path.exists():
            continue
        records = [
            {
                "page": page.index,
                "boxno": record["boxno"],
                "text": record.get("text", ""),
                "englishText": record.get("englishText", ""),
            }
            for record in structured_by_page[page.index]
        ]
        write_json(snapshot_path, records)


def run_proofreading_phase(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    translation_notes: dict[str, Any],
) -> None:
    records = all_translated_records_for_review(structured_by_page)
    if not records:
        print("Skip VLM proofread translations: no translated records", file=sys.stderr)
        return
    print("VLM proofread translations", file=sys.stderr)
    translations_by_page = get_validated_vlm_text_named(
        "proofreading",
        "book",
        "VLM proofread translations",
        proofreading_prompt(structured_by_page, translation_notes, config.language),
        output_dir,
        config.vlm,
        fixture_dir,
        lambda text: validate_proofread_translations(pages, structured_by_page, text),
    )
    snapshot_raw_translation_records(pages, structured_by_page, output_dir)
    apply_translation_records(pages, structured_by_page, translations_by_page, output_dir)


def generated_translation_notes_path(output_dir: Path) -> Path:
    return output_dir / GENERATED_TRANSLATION_NOTES_NAME


def generated_translation_notes_prompt(
    structured_by_page: dict[int, list[dict[str, Any]]],
    translation_notes: dict[str, Any],
    language: LanguageConfig,
) -> str:
    records = all_translated_records_for_review(structured_by_page)
    source_name = language.source.name
    target_name = language.target.name
    return f"""
Create reusable translation notes for translating related CBZs in the same series.

You are given the complete {source_name} source text, final {target_name} translation, and user-provided
translation notes. Write concise plain-text notes that would help keep future related volumes
consistent.

Include useful items such as:
- Character names and consistent romanization/spelling.
- Places, organizations, magic/skill names, titles, honorific choices, and recurring terms.
- Character voice, formality, catchphrases, relationship context, and pronoun choices.
- Ambiguities or unresolved terms a future translator should watch.
- Any user-provided preferences that should carry forward.

Do not write a page-by-page summary unless a page-specific note is genuinely reusable.
Do not include JSON. Do not include markdown code fences.

User translation notes:
{translation_notes_book_prompt_section(translation_notes)}

Final translation records, tab-separated as page, boxno, sfx, openLettering, sourceText, englishText:
{compact_proofreading_input(records)}
""".strip()


def run_generated_translation_notes_phase(
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    translation_notes: dict[str, Any],
) -> None:
    records = all_translated_records_for_review(structured_by_page)
    output_path = generated_translation_notes_path(output_dir)
    if not records:
        output_path.write_text("No translated records were available.\n", encoding="utf-8")
        print(f"Wrote {output_path}", file=sys.stderr)
        return
    print("VLM write translation notes", file=sys.stderr)
    notes = get_vlm_text_named(
        "translation_notes",
        "book",
        "VLM write translation notes",
        generated_translation_notes_prompt(structured_by_page, translation_notes, config.language),
        output_dir,
        config.vlm,
        fixture_dir,
    ).strip()
    output_path.write_text((notes or "No reusable translation notes were generated.") + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", file=sys.stderr)


def run_post_translation_phases(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    translation_notes: dict[str, Any],
    start_phase: str = "proofreading",
) -> None:
    if start_phase not in {"proofreading", "translation_notes"}:
        raise PipelineError(f"Unsupported post-translation start phase: {start_phase}")
    if start_phase == "proofreading" and config.postprocess.proofread_translations:
        run_proofreading_phase(
            pages,
            structured_by_page,
            output_dir,
            config,
            fixture_dir,
            translation_notes,
        )
    if config.postprocess.write_translation_notes:
        run_generated_translation_notes_phase(
            structured_by_page,
            output_dir,
            config,
            fixture_dir,
            translation_notes,
        )


def open_lettering_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["openLettering"]]


def non_open_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if not record["openLettering"]]


def placement_open_prompt(
    page: Page,
    structured_records: list[dict[str, Any]],
    language: LanguageConfig,
) -> str:
    open_records = open_lettering_records(structured_records)
    source_name = language.source.name
    target_name = language.target.name
    return f"""
For open-lettering translations only, choose unobtrusive margin/nearby placement boxes
where the {target_name} text can be readable. Open-lettering text was marked this way because
cleaning the original {source_name} would likely damage important page art, so do not place
the translation over faces, bodies, detailed artwork, action lines, panel borders, or
other important visual content. Do not place text in a tiny sliver directly below or
beside the original if that would force very small {target_name} text.

Coordinates must use Gemma-style normalized box_2d: [y_min, x_min, y_max, x_max],
where each value is an integer from 0 to 1000.
All four box_2d values must be inside 0..1000 inclusive. Never return negative
coordinates and never return coordinates greater than 1000. If a placement would go
outside the page, move or shrink it so y_min, x_min, y_max, and x_max all remain in 0..1000.
Ensure y_min < y_max and x_min < x_max after applying the 0..1000 limit.

Return only a compact JSON array of rows. Row format:
[boxno,box_2d]
Return one row for every input record below and no others.

Reading order guidance: use {language.source.reading_order}.

Current page index:
{page.index}

Open-lettering records compact table:
{compact_records_table(open_records, ("boxno", "region", "text", "englishText", "sfx"))}
""".strip()


def placement_style_records(
    structured_records: list[dict[str, Any]],
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_boxno = {placement["boxno"]: placement for placement in placements}
    records: list[dict[str, Any]] = []
    for record in structured_records:
        item = {
            "page": record["page"],
            "boxno": record["boxno"],
            "text": record["text"],
            "englishText": record["englishText"],
            "openLettering": record["openLettering"],
            "sfx": record["sfx"],
        }
        placement = by_boxno[record["boxno"]]
        item["box_2d"] = placement["box_2d"]
        item["placementRegion"] = placement["placementRegion"]
        records.append(item)
    return records


def placement_style_prompt(
    page: Page,
    structured_records: list[dict[str, Any]],
    placements: list[dict[str, Any]],
    language: LanguageConfig,
    backup_font: str,
) -> str:
    target_name = language.target.name
    return f"""
Choose a font and fill color for each translated text record. The image provided is the
original comic page; use the listed placement boxes to judge background brightness and text
style.

Available fonts; return the exact font filename in the font field:
{font_use_prompt(backup_font)}

Backup font: {backup_font}
Use the backup font when no user-provided font is available or suitable.

Choose fill as either "black" or "white" based on the chosen placement area's background.
The primary purpose of fill is legibility of the main letter shape. Readability is more
important than matching the original style for the {target_name} translation.
- Use black fill on white, pale, light gray, or mostly light backgrounds.
- Use white fill only on dark or mostly black backgrounds.
- If the area is mixed, choose the fill that is readable over most of the placement box.
- If uncertain, choose black.
- Do not choose white fill on a white or pale speech bubble/caption/blank area; that is
  white-on-white and is incorrect even though the renderer adds an outline.
- Do not rely on the outline to make the fill readable; choose the fill as if there were
  no outline. The renderer will automatically add the opposite outline color.

Return only a compact JSON array of rows. Row format:
[boxno,font,fill]
Return one row for every input record below and no others.

Reading order guidance: use {language.source.reading_order}.

Current page index:
{page.index}

Records compact table:
{compact_records_table(placement_style_records(structured_records, placements), ("boxno", "text", "englishText", "openLettering", "sfx", "box_2d", "placementRegion"))}
""".strip()


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


def detect_expansions_page(
    page: Page,
    structured_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expansions: list[dict[str, Any]] = []
    records_to_expand = non_open_records(structured_records)
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
            expansion["box_2d"] = [
                round(expansion["placementRegion"][1] / image_height * 1000),
                round(expansion["placementRegion"][0] / image_width * 1000),
                round(expansion["placementRegion"][3] / image_height * 1000),
                round(expansion["placementRegion"][2] / image_width * 1000),
            ]
            expansions.append(expansion)
    return sorted(expansions, key=lambda item: item["label"])


def validate_open_placements_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    expected_boxnos = {record["boxno"] for record in open_lettering_records(structured_records)}
    placements: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, value in enumerate(vlm_items):
        label = f"open placement page {page.index} item {index}"
        if isinstance(value, list):
            row = require_row(value, label, 2)
            item_page = page.index
            boxno = parse_non_negative_int_value(row[0], label, "boxno")
            box = row[1]
        else:
            record = require_object(value, label)
            item_page = require_non_negative_int(record, "page", label) if "page" in record else page.index
            boxno = require_non_negative_int(record, "boxno", label)
            box = record.get("box_2d")
        if item_page != page.index:
            raise PipelineError(f"{label} has page {item_page}, expected {page.index}.")
        if boxno not in expected_boxnos:
            raise PipelineError(f"{label} references missing boxno {boxno}.")
        if boxno in seen:
            raise PipelineError(f"{label} duplicates boxno {boxno}.")
        if not isinstance(box, list):
            raise PipelineError(f"{label} must include box_2d array.")
        placement = {
            "page": page.index,
            "boxno": boxno,
            "box_2d": box,
            "placementRegion": normalized_box_to_region(box, page.image_path),
        }
        placements.append(placement)
        seen.add(boxno)
    missing = sorted(expected_boxnos - seen)
    if missing:
        raise PipelineError(f"Open placement missing page {page.index} boxnos: {missing}")
    return sorted(placements, key=lambda item: item["boxno"])


def validate_expansions_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    expected_labels = {record["boxno"] for record in non_open_records(structured_records)}
    expansions: list[dict[str, Any]] = []
    seen: set[int] = set()
    with Image.open(page.image_path) as image:
        image_width, image_height = image.size
    for index, value in enumerate(vlm_items):
        label = f"expansion page {page.index} item {index}"
        record = require_object(value, label)
        raw_label = record.get("label", record.get("boxno"))
        if not isinstance(raw_label, int) or isinstance(raw_label, bool) or raw_label < 0:
            raise PipelineError(f"{label} must include non-negative integer `label`.")
        if raw_label not in expected_labels:
            raise PipelineError(f"{label} references non-open or missing label {raw_label}.")
        if raw_label in seen:
            raise PipelineError(f"{label} duplicates label {raw_label}.")
        if "placementRegion" in record:
            expansion = {
                "label": raw_label,
                "boxno": raw_label,
                "placementRegion": clip_region_values(
                    record.get("placementRegion"),
                    image_width,
                    image_height,
                    label,
                ),
            }
            for optional_key in (
                "box_2d",
                "box_widening",
                "target_width_ratio",
                "container_x_min",
                "container_x_max",
                "height_increase",
                "placementMethod",
                "fallbackReason",
                "detectedBackground",
                "tolerance",
                "componentArea",
                "searchRegion",
                "inset",
                "touchesSearchBounds",
                "raySeed",
                "rayEndpoints",
                "hitRayCount",
                "cardinalHitRayCount",
            ):
                if optional_key in record:
                    expansion[optional_key] = record[optional_key]
            expansions.append(expansion)
        elif any(
            key in record
            for key in ("target_width_ratio", "container_x_min", "container_x_max")
        ):
            target_width_ratio = parse_fraction_value(
                record.get("target_width_ratio"),
                label,
                "target_width_ratio",
            )
            if target_width_ratio <= 0:
                raise PipelineError(f"{label} target_width_ratio must be greater than 0.")
            container_x_min = parse_fraction_value(
                record.get("container_x_min"),
                label,
                "container_x_min",
            )
            container_x_max = parse_fraction_value(
                record.get("container_x_max"),
                label,
                "container_x_max",
            )
            if container_x_max <= container_x_min:
                raise PipelineError(f"{label} container_x_max must be greater than container_x_min.")
            expansions.append(
                {
                    "label": raw_label,
                    "target_width_ratio": target_width_ratio,
                    "container_x_min": container_x_min,
                    "container_x_max": container_x_max,
                    "height_increase": parse_expansion_value(
                        record.get("height_increase"),
                        label,
                        "height_increase",
                    ),
                }
            )
        else:
            expansions.append(
                {
                    "label": raw_label,
                    "box_widening": parse_expansion_value(
                        record.get("box_widening"),
                        label,
                        "box_widening",
                    ),
                    "height_increase": parse_expansion_value(
                        record.get("height_increase"),
                        label,
                        "height_increase",
                    ),
                }
            )
        seen.add(raw_label)
    missing = sorted(expected_labels - seen)
    if missing:
        raise PipelineError(f"Expansion missing page {page.index} labels: {missing}")
    return sorted(expansions, key=lambda item: item["label"])


def validate_styles_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
    backup_font: str,
) -> list[dict[str, Any]]:
    expected_boxnos = {record["boxno"] for record in structured_records}
    styles: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, value in enumerate(vlm_items):
        label = f"style page {page.index} item {index}"
        if isinstance(value, list):
            row = require_row(value, label, 3)
            item_page = page.index
            boxno = parse_non_negative_int_value(row[0], label, "boxno")
            font_value = row[1]
            fill_value = row[2]
        else:
            record = require_object(value, label)
            item_page = require_non_negative_int(record, "page", label) if "page" in record else page.index
            boxno = require_non_negative_int(record, "boxno", label)
            font_value = record.get("font")
            fill_value = record.get("fill", record.get("color", record.get("colour")))
        if item_page != page.index:
            raise PipelineError(f"{label} has page {item_page}, expected {page.index}.")
        if boxno not in expected_boxnos:
            raise PipelineError(f"{label} references missing boxno {boxno}.")
        if boxno in seen:
            raise PipelineError(f"{label} duplicates boxno {boxno}.")

        font_name = normalize_font_name(font_value, backup_font)
        if font_name is None:
            raise PipelineError(
                f"{label} must include the backup font or a filename from {FONT_USE_PATH}."
            )
        fill = normalize_fill(fill_value)
        if fill is None:
            raise PipelineError(f"{label} must include fill as black or white.")
        styles.append({"page": page.index, "boxno": boxno, "font": font_name, "fill": fill})
        seen.add(boxno)
    missing = sorted(expected_boxnos - seen)
    if missing:
        raise PipelineError(f"Style missing page {page.index} boxnos: {missing}")
    return sorted(styles, key=lambda item: item["boxno"])


def validate_placements_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    expected_boxnos = {record["boxno"] for record in structured_records}
    placements: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, value in enumerate(vlm_items):
        label = f"placement page {page.index} item {index}"
        record = require_object(value, label)
        item_page = require_non_negative_int(record, "page", label)
        boxno = require_non_negative_int(record, "boxno", label)
        if item_page != page.index:
            raise PipelineError(f"{label} has page {item_page}, expected {page.index}.")
        if boxno not in expected_boxnos:
            raise PipelineError(f"{label} references missing boxno {boxno}.")
        if boxno in seen:
            raise PipelineError(f"{label} duplicates boxno {boxno}.")
        box = record.get("box_2d")
        if not isinstance(box, list):
            raise PipelineError(f"{label} must include box_2d array.")
        placement = {
            "page": page.index,
            "boxno": boxno,
            "box_2d": box,
            "placementRegion": normalized_box_to_region(box, page.image_path),
        }
        for optional_key in (
            "box_widening",
            "target_width_ratio",
            "container_x_min",
            "container_x_max",
            "height_increase",
            "placementMethod",
            "fallbackReason",
            "detectedBackground",
            "tolerance",
            "componentArea",
            "searchRegion",
            "inset",
            "touchesSearchBounds",
            "raySeed",
            "rayEndpoints",
            "hitRayCount",
            "cardinalHitRayCount",
        ):
            if optional_key in record:
                placement[optional_key] = record[optional_key]
        font_name = normalize_font_name(record.get("font"))
        if font_name is not None:
            placement["font"] = font_name
        fill = normalize_fill(record.get("fill", record.get("color", record.get("colour"))))
        if fill is not None:
            placement["fill"] = fill
        placements.append(placement)
        seen.add(boxno)
    missing = sorted(expected_boxnos - seen)
    if missing:
        raise PipelineError(f"Placement missing page {page.index} boxnos: {missing}")
    return correct_placement_fills_for_legibility(
        page,
        sorted(placements, key=lambda item: item["boxno"]),
    )


def preliminary_placements(
    page: Page,
    structured_records: list[dict[str, Any]],
    open_placements: list[dict[str, Any]],
    expansions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    open_by_boxno = {placement["boxno"]: placement for placement in open_placements}
    expansion_by_label = {expansion["label"]: expansion for expansion in expansions}
    placements: list[dict[str, Any]] = []
    for record in structured_records:
        boxno = record["boxno"]
        if record["openLettering"]:
            placement = dict(open_by_boxno[boxno])
        else:
            expansion = expansion_by_label[boxno]
            placement = {
                "page": page.index,
                "boxno": boxno,
            }
            if "height_increase" in expansion:
                placement["height_increase"] = expansion["height_increase"]
            if "placementRegion" in expansion:
                placement_region = expansion["placementRegion"]
            elif "target_width_ratio" in expansion:
                placement_region = expand_region_to_container_width(
                    record["region"],
                    expansion["target_width_ratio"],
                    expansion["container_x_min"],
                    expansion["container_x_max"],
                    expansion["height_increase"],
                    page.image_path,
                )
                placement["target_width_ratio"] = expansion["target_width_ratio"]
                placement["container_x_min"] = expansion["container_x_min"]
                placement["container_x_max"] = expansion["container_x_max"]
            else:
                placement_region = expand_region(
                    record["region"],
                    expansion["box_widening"],
                    expansion["height_increase"],
                    page.image_path,
                )
                placement["box_widening"] = expansion["box_widening"]
            for optional_key in (
                "box_widening",
                "target_width_ratio",
                "container_x_min",
                "container_x_max",
                "placementMethod",
                "fallbackReason",
                "detectedBackground",
                "tolerance",
                "componentArea",
                "searchRegion",
                "inset",
                "touchesSearchBounds",
                "raySeed",
                "rayEndpoints",
                "hitRayCount",
                "cardinalHitRayCount",
            ):
                if optional_key in expansion:
                    placement[optional_key] = expansion[optional_key]
            placement["box_2d"] = region_to_normalized_box(placement_region, page.image_path)
            placement["placementRegion"] = placement_region
        placements.append(placement)
    return sorted(placements, key=lambda item: item["boxno"])


def merge_placement_styles(
    page: Page,
    placements: list[dict[str, Any]],
    styles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    style_by_boxno = {style["boxno"]: style for style in styles}
    merged: list[dict[str, Any]] = []
    for placement in placements:
        styled = dict(placement)
        style = style_by_boxno[placement["boxno"]]
        styled["font"] = style["font"]
        styled["fill"] = style["fill"]
        merged.append(styled)
    return correct_placement_fills_for_legibility(page, merged)


def apply_placements_to_records(
    records: list[dict[str, Any]],
    placements: list[dict[str, Any]],
) -> None:
    by_boxno = {item["boxno"]: item for item in placements}
    for record in records:
        placement = by_boxno[record["boxno"]]
        record["box_2d"] = placement["box_2d"]
        record["placementRegion"] = placement["placementRegion"]
        if "font" in placement:
            record["font"] = placement["font"]
        if "fill" in placement:
            record["fill"] = placement["fill"]


def placement_expand_fixture_items(
    fixture_dir: Path | None,
    page: Page,
) -> list[Any] | None:
    if fixture_dir is None:
        return None
    fixture_path = fixture_dir / "placement_expand" / f"{page_name(page.index)}.json"
    if not fixture_path.exists():
        return None
    return load_json_array(fixture_path, f"placement_expand fixture page {page.index}")


def attach_placements(
    pages: list[Page],
    output_dir: Path,
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    placements_by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        path = data_page_path(output_dir, "placements", page)
        data = load_json_array(path, f"placement page {page.index}")
        placements = validate_placements_page(page, structured_by_page[page.index], data)
        placements_by_page[page.index] = placements
        apply_placements_to_records(structured_by_page[page.index], placements)
    return placements_by_page


def ensure_placed_pages(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
) -> None:
    for page in pages:
        for record in structured_by_page[page.index]:
            if "placementRegion" not in record:
                raise PipelineError(
                    f"Missing placementRegion for page {page.index} boxno {record['boxno']}"
                )


def run_placement_phase(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    fixture_dir: Path | None,
    start_page: int = 0,
    end_page: int | None = None,
    existing_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    ensure_all_pages(structured_by_page, pages, "Structured OCR records")
    ensure_translated_pages(pages, structured_by_page)
    placements_by_page: dict[int, list[dict[str, Any]]] = dict(existing_by_page or {})
    for page in pages_in_range(pages, start_page, end_page):
        structured_records = structured_by_page[page.index]
        if not structured_records:
            placements = []
        else:
            open_records = open_lettering_records(structured_records)
            if open_records:
                print(f"VLM open placement page {page.index}", file=sys.stderr)
                open_placements = get_validated_vlm_array(
                    "placement_open",
                    page,
                    placement_open_prompt(page, structured_records, config.language),
                    output_dir,
                    config.vlm,
                    fixture_dir,
                    lambda items, current_page=page: validate_open_placements_page(
                        current_page,
                        structured_by_page[current_page.index],
                        items,
                    ),
                )
            else:
                open_placements = []
            write_json(trace_page_path(output_dir, "placement_open", page), open_placements)

            records_to_expand = non_open_records(structured_records)
            if records_to_expand:
                fixture_expansion_items = placement_expand_fixture_items(fixture_dir, page)
                if fixture_expansion_items is not None:
                    expansions = validate_expansions_page(
                        page,
                        structured_records,
                        fixture_expansion_items,
                    )
                else:
                    print(f"Detect placement containers page {page.index}", file=sys.stderr)
                    expansions = detect_expansions_page(page, structured_records)
            else:
                expansions = []
            write_json(trace_page_path(output_dir, "placement_expand", page), expansions)
            draw_region_debug_image(
                expansions,
                page,
                output_dir,
                "placement_expand_img",
                PLACEMENT_EXPAND_DEBUG_COLOR,
                "placementRegion",
            )

            preliminary = preliminary_placements(
                page,
                structured_records,
                open_placements,
                expansions,
            )

            print(f"VLM style placement page {page.index}", file=sys.stderr)
            styles = get_validated_vlm_array(
                "placement_style",
                page,
                placement_style_prompt(
                    page,
                    structured_records,
                    preliminary,
                    config.language,
                    config.render_font,
                ),
                output_dir,
                config.vlm,
                fixture_dir,
                lambda items, current_page=page: validate_styles_page(
                    current_page,
                    structured_by_page[current_page.index],
                    items,
                    config.render_font,
                ),
            )
            write_json(trace_page_path(output_dir, "placement_style", page), styles)
            placements = merge_placement_styles(page, preliminary, styles)
            apply_placements_to_records(structured_records, placements)
        placements_by_page[page.index] = placements
        write_json(data_page_path(output_dir, "placements", page), placements)
        draw_region_debug_image(
            placements,
            page,
            output_dir,
            "placements_img",
            VLM_PLACEMENT_DEBUG_COLOR,
            "placementRegion",
        )
    ensure_placed_pages(pages, structured_by_page)
    write_json(master_json_path(output_dir), flatten_pages(structured_by_page))
    return placements_by_page


def render_page(
    page: Page,
    records: list[dict[str, Any]],
    output_dir: Path,
    config: PipelineConfig,
) -> None:
    print(f"Render page {page.index}", file=sys.stderr)
    cleaned_path = cleaned_pages_dir(output_dir) / f"{page.image_path.stem}.png"
    final_path = final_pages_dir(output_dir) / f"{page.image_path.stem}.png"
    keep_mask = debug_image_path(output_dir, "masks", page)

    clean_entries = [record for record in records if not record["openLettering"]]
    clean_text_regions.clean_text_regions(
        clean_entries,
        page.image_path,
        cleaned_path,
        clean_text_regions.DEFAULT_PADDING,
        clean_text_regions.DEFAULT_DEVICE,
        None,
        clean_text_regions.DEFAULT_CROP_TRIGGER_SIZE,
        clean_text_regions.DEFAULT_CROP_MARGIN,
        keep_mask,
    )

    overlay_entries: list[dict[str, Any]] = []
    for record in records:
        english_text = record.get("englishText")
        if not isinstance(english_text, str):
            raise PipelineError(f"Missing englishText for page {page.index} boxno {record['boxno']}")
        render_region = record.get("placementRegion", record["region"])
        fill = record.get("fill", config.render_fill)
        if not isinstance(fill, str):
            raise PipelineError(f"Invalid fill for page {page.index} boxno {record['boxno']}")
        overlay_entries.append(
            {
                "page": page.index,
                "boxno": record["boxno"],
                "region": render_region,
                "englishText": english_text,
                "font": local_font_path(record.get("font"), config.render_font),
                "fill": fill,
                "stroke": outline_for_fill(fill),
                "strokeWidth": DEFAULT_RENDER_STROKE_WIDTH,
                "gravity": config.render_gravity,
            }
        )

    write_json(data_page_path(output_dir, "render_entries", page), overlay_entries)
    overlay_text.overlay_text(overlay_entries, cleaned_path, final_path)


def render_pages(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    start_page: int = 0,
    end_page: int | None = None,
) -> None:
    ensure_all_pages(structured_by_page, pages, "Structured OCR records")
    ensure_translated_pages(pages, structured_by_page)
    ensure_placed_pages(pages, structured_by_page)
    selected_pages = pages_in_range(pages, start_page, end_page)
    for page in selected_pages:
        render_page(page, structured_by_page[page.index], output_dir, config)


def copy_zipinfo_for_deflated_write(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    copied = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    copied.comment = info.comment
    copied.extra = info.extra
    copied.internal_attr = info.internal_attr
    copied.external_attr = info.external_attr
    copied.create_system = info.create_system
    copied.compress_type = zipfile.ZIP_DEFLATED
    return copied


def write_preserved_non_image_members(
    archive: zipfile.ZipFile,
    input_cbz: Path,
    written_names: set[str],
) -> None:
    try:
        with zipfile.ZipFile(input_cbz) as source_archive:
            source_members = validate_cbz_members(source_archive)
            archive.comment = source_archive.comment
            for info in source_members:
                if not is_preserved_non_image_member(info):
                    continue
                if info.filename in written_names:
                    print(
                        f"warning: skipping original non-image entry that conflicts with output: {info.filename}",
                        file=sys.stderr,
                    )
                    continue
                with source_archive.open(info) as source, archive.open(
                    copy_zipinfo_for_deflated_write(info),
                    "w",
                ) as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                written_names.add(info.filename)
    except zipfile.BadZipFile as exc:
        raise PipelineError(f"Input file is not a valid CBZ/zip: {input_cbz}") from exc


def final_page_png_path(output_dir: Path, page: Page) -> Path:
    return final_pages_dir(output_dir) / f"{page.image_path.stem}.png"


def temporary_archive_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        return Path(file.name)


def package_cbz(pages: list[Page], output_dir: Path, input_cbz: Path) -> Path:
    cbz_path = translated_cbz_path(output_dir)
    output_mode = cbz_path.stat().st_mode & 0o777 if cbz_path.exists() else 0o644
    temp_cbz_path = temporary_archive_path(cbz_path)
    try:
        with zipfile.ZipFile(temp_cbz_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            written_names: set[str] = set()
            for page in pages:
                image_path = final_page_png_path(output_dir, page)
                if not image_path.exists():
                    raise PipelineError(f"Final page image missing: {image_path}")
                archive.write(image_path, arcname=image_path.name)
                written_names.add(image_path.name)

            write_preserved_non_image_members(archive, input_cbz, written_names)
        os.chmod(temp_cbz_path, output_mode)
        os.replace(temp_cbz_path, cbz_path)
    finally:
        temp_cbz_path.unlink(missing_ok=True)
    return cbz_path


def convert_final_page_with_magick(
    source_path: Path,
    output_path: Path,
    quality: int,
) -> None:
    magick = shutil.which("magick")
    if magick is None:
        raise PipelineError("ImageMagick command not found: magick")

    command = [
        magick,
        str(source_path),
        "-strip",
        "-quality",
        str(quality),
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = (
            f"ImageMagick failed converting {source_path.name} to {output_path.suffix}."
        )
        if details:
            message += f" {details}"
        raise PipelineError(message) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise PipelineError(f"ImageMagick did not write converted image: {output_path}")


def package_converted_cbz(
    pages: list[Page],
    output_dir: Path,
    input_cbz: Path,
    cbz_path: Path,
    suffix: str,
    quality: int,
) -> Path:
    output_mode = cbz_path.stat().st_mode & 0o777 if cbz_path.exists() else 0o644
    temp_cbz_path = temporary_archive_path(cbz_path)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"tetolate_{suffix.lstrip('.')}_"
        ) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            with zipfile.ZipFile(temp_cbz_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                written_names: set[str] = set()
                for page in pages:
                    source_path = final_page_png_path(output_dir, page)
                    if not source_path.exists():
                        raise PipelineError(f"Final page image missing: {source_path}")
                    page_suffix = TRANSLATED_ALT_COVER_SUFFIX if page.index == 0 else suffix
                    converted_path = temp_dir / f"{page.image_path.stem}{page_suffix}"
                    convert_final_page_with_magick(source_path, converted_path, quality)
                    archive.write(converted_path, arcname=converted_path.name)
                    written_names.add(converted_path.name)
                    converted_path.unlink()

                write_preserved_non_image_members(archive, input_cbz, written_names)
        os.chmod(temp_cbz_path, output_mode)
        os.replace(temp_cbz_path, cbz_path)
    finally:
        temp_cbz_path.unlink(missing_ok=True)
    return cbz_path


def print_packaged_cbz(
    pages: list[Page],
    output_dir: Path,
    input_cbz: Path,
    config: PipelineConfig,
) -> None:
    cbz_path = package_cbz(pages, output_dir, input_cbz)
    print(f"Wrote {cbz_path}", file=sys.stderr)
    for variant_path, suffix, quality, label in (
        (translated_webp_cbz_path(output_dir), ".webp", config.webp_quality, "WebP"),
        (translated_jxl_cbz_path(output_dir), ".jxl", config.jxl_quality, "JXL"),
    ):
        try:
            written_path = package_converted_cbz(
                pages,
                output_dir,
                input_cbz,
                variant_path,
                suffix,
                quality,
            )
        except PipelineError as exc:
            if variant_path.exists():
                variant_path.unlink()
            print(f"warning: skipped {label} CBZ output: {exc}", file=sys.stderr)
            continue
        print(f"Wrote {written_path}", file=sys.stderr)



def count_output_pages(output_dir: Path) -> int:
    pages_path = original_pages_dir(output_dir)
    if not pages_path.is_dir():
        return 0
    return sum(
        1
        for path in pages_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_runtime_summary(started_at: float, output_dir: Path) -> None:
    elapsed = max(time.monotonic() - started_at, 0.001)
    page_count = count_output_pages(output_dir)
    if page_count:
        per_page = elapsed / page_count
        print(
            (
                f"Finished in {format_elapsed(elapsed)} "
                f"({page_count} pages, {format_elapsed(per_page)}/page)"
            ),
            file=sys.stderr,
        )
    else:
        print(f"Finished in {format_elapsed(elapsed)}", file=sys.stderr)


def resume_phase_artifact_path(output_dir: Path, phase: str, page: Page) -> Path | None:
    if phase in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements"}:
        return data_page_path(output_dir, phase, page)
    if phase == "render":
        return final_page_png_path(output_dir, page)
    return None


def rewind_resume_page_for_missing_artifacts(
    output_dir: Path,
    phase: str,
    pages: list[Page],
    requested_page: int,
) -> int:
    for page in pages_before(pages, requested_page):
        artifact = resume_phase_artifact_path(output_dir, phase, page)
        if artifact is not None and not artifact.is_file():
            print(
                (
                    f"warning: cannot resume {phase} at page {requested_page} because "
                    f"page {page.index} output is missing; rewinding to page {page.index}."
                ),
                file=sys.stderr,
            )
            return page.index
    return requested_page


def run_full_pipeline(
    args: argparse.Namespace,
    config: PipelineConfig,
    translation_notes: dict[str, Any],
) -> None:
    prepare_output_dir(args.output_dir, args.overwrite)

    pages = extract_cbz(args.input_cbz, args.output_dir)
    raw_by_page = run_ocr(pages, args.output_dir, config)
    if args.stop_after == "ocr_raw":
        print("Stopped after OCR raw", file=sys.stderr)
        return
    structured_by_page = run_structure_phase(
        pages, raw_by_page, args.output_dir, config, args.fixture_dir
    )
    run_alt_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
    run_translation_phase(
        pages,
        structured_by_page,
        args.output_dir,
        config,
        args.fixture_dir,
        translation_notes,
    )
    run_post_translation_phases(
        pages,
        structured_by_page,
        args.output_dir,
        config,
        args.fixture_dir,
        translation_notes,
    )
    run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
    render_pages(pages, structured_by_page, args.output_dir, config)
    print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)


def run_single_page_resume_pipeline(
    args: argparse.Namespace,
    config: PipelineConfig,
    pages: list[Page],
    translation_notes: dict[str, Any],
) -> None:
    phase = args.resume_from
    target_page = args.resume_page
    if phase not in SINGLE_PAGE_PHASES:
        raise PipelineError(
            "--single-page can only be used with --resume-from "
            + ", ".join(sorted(SINGLE_PAGE_PHASES))
            + "."
        )

    other_pages = pages_except(pages, target_page)

    if phase == "ocr_raw":
        raw_existing = load_phase_records(
            args.output_dir,
            "ocr_raw",
            other_pages,
            "OCR records",
        )
        raw_by_page = run_ocr(
            pages,
            args.output_dir,
            config,
            start_page=target_page,
            end_page=target_page,
            existing_by_page=raw_existing,
        )
    else:
        raw_by_page = load_phase_records(args.output_dir, "ocr_raw", pages, "OCR records")

    if phase in {"ocr_raw", "ocr_structured"}:
        structured_existing = load_structured_records(args.output_dir, other_pages)
        structured_by_page = run_structure_phase(
            pages,
            raw_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=target_page,
            end_page=target_page,
            existing_by_page=structured_existing,
            use_existing_merged=phase == "ocr_structured",
        )
    else:
        structured_by_page = load_structured_records(args.output_dir, pages)

    if phase in {"ocr_raw", "ocr_structured", "alt_placement"}:
        run_alt_placement_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=target_page,
            end_page=target_page,
        )

    if phase in {"ocr_raw", "ocr_structured", "alt_placement", "translations"}:
        translation_existing = attach_translations(
            other_pages,
            args.output_dir,
            structured_by_page,
        )
        run_translation_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
            start_page=target_page,
            end_page=target_page,
            existing_by_page=translation_existing,
        )
    else:
        attach_translations(pages, args.output_dir, structured_by_page)

    if phase in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements"}:
        placement_existing = attach_placements(
            other_pages,
            args.output_dir,
            structured_by_page,
        )
        run_placement_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=target_page,
            end_page=target_page,
            existing_by_page=placement_existing,
        )
    else:
        attach_placements(pages, args.output_dir, structured_by_page)

    render_pages(
        pages,
        structured_by_page,
        args.output_dir,
        config,
        start_page=target_page,
        end_page=target_page,
    )
    if not args.skip_package:
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)


def run_resume_pipeline(
    args: argparse.Namespace,
    config: PipelineConfig,
    translation_notes: dict[str, Any],
) -> None:
    if args.overwrite:
        raise PipelineError("--overwrite cannot be combined with --resume-from.")
    if args.resume_page < 0:
        raise PipelineError("--resume-page must be a non-negative integer.")
    if not args.output_dir.is_dir():
        raise PipelineError(f"Cannot resume; output directory does not exist: {args.output_dir}")

    phase = args.resume_from
    if phase is None:
        raise PipelineError("--resume-from is required for resume mode.")
    if args.single_page and phase not in SINGLE_PAGE_PHASES:
        raise PipelineError(
            "--single-page can only be used with --resume-from "
            + ", ".join(sorted(SINGLE_PAGE_PHASES))
            + "."
        )
    if phase in {"extract", "proofreading", "translation_notes", "package"} and args.resume_page != 0:
        raise PipelineError(f"--resume-page is not used with --resume-from {phase}.")

    if phase == "extract":
        pages = extract_cbz(args.input_cbz, args.output_dir)
        raw_by_page = run_ocr(pages, args.output_dir, config)
        structured_by_page = run_structure_phase(
            pages, raw_by_page, args.output_dir, config, args.fixture_dir
        )
        run_alt_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        run_translation_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    verify_resume_input(args.input_cbz, args.output_dir)
    pages = load_extracted_pages(args.output_dir)
    validate_resume_page(pages, args.resume_page, phase)
    start_page = args.resume_page
    if args.single_page:
        run_single_page_resume_pipeline(args, config, pages, translation_notes)
        return
    start_page = rewind_resume_page_for_missing_artifacts(
        args.output_dir,
        phase,
        pages,
        start_page,
    )

    if phase == "ocr_raw":
        raw_prior = load_phase_records(
            args.output_dir,
            "ocr_raw",
            pages_before(pages, start_page),
            "OCR records",
        )
        raw_by_page = run_ocr(
            pages,
            args.output_dir,
            config,
            start_page=start_page,
            existing_by_page=raw_prior,
        )
        structured_by_page = run_structure_phase(
            pages, raw_by_page, args.output_dir, config, args.fixture_dir
        )
        run_alt_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        run_translation_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "ocr_structured":
        raw_by_page = load_phase_records(args.output_dir, "ocr_raw", pages, "OCR records")
        structured_prior = load_structured_records(args.output_dir, pages_before(pages, start_page))
        structured_by_page = run_structure_phase(
            pages,
            raw_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=start_page,
            existing_by_page=structured_prior,
            use_existing_merged=True,
        )
        alt_placement_start = rewind_resume_page_for_missing_artifacts(
            args.output_dir,
            "alt_placement",
            pages,
            start_page,
        )
        run_alt_placement_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=alt_placement_start,
        )
        translation_start = rewind_resume_page_for_missing_artifacts(
            args.output_dir,
            "translations",
            pages,
            alt_placement_start,
        )
        translation_prior = attach_translations(
            pages_before(pages, translation_start),
            args.output_dir,
            structured_by_page,
        )
        run_translation_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
            start_page=translation_start,
            existing_by_page=translation_prior,
        )
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "alt_placement":
        structured_by_page = load_structured_records(args.output_dir, pages)
        run_alt_placement_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=start_page,
        )
        translation_start = rewind_resume_page_for_missing_artifacts(
            args.output_dir,
            "translations",
            pages,
            start_page,
        )
        translation_prior = attach_translations(
            pages_before(pages, translation_start),
            args.output_dir,
            structured_by_page,
        )
        run_translation_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
            start_page=translation_start,
            existing_by_page=translation_prior,
        )
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "translations":
        structured_by_page = load_structured_records(args.output_dir, pages)
        translation_prior = attach_translations(
            pages_before(pages, start_page),
            args.output_dir,
            structured_by_page,
        )
        run_translation_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
            start_page=start_page,
            existing_by_page=translation_prior,
        )
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "proofreading":
        structured_by_page = load_structured_records(args.output_dir, pages)
        attach_translations(pages, args.output_dir, structured_by_page)
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
            start_phase="proofreading",
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "translation_notes":
        structured_by_page = load_structured_records(args.output_dir, pages)
        attach_translations(pages, args.output_dir, structured_by_page)
        run_post_translation_phases(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            translation_notes,
            start_phase="translation_notes",
        )
        run_placement_phase(pages, structured_by_page, args.output_dir, config, args.fixture_dir)
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "placements":
        structured_by_page = load_structured_records(args.output_dir, pages)
        attach_translations(pages, args.output_dir, structured_by_page)
        placement_prior = attach_placements(
            pages_before(pages, start_page),
            args.output_dir,
            structured_by_page,
        )
        run_placement_phase(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            args.fixture_dir,
            start_page=start_page,
            existing_by_page=placement_prior,
        )
        render_pages(pages, structured_by_page, args.output_dir, config)
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "render":
        structured_by_page = load_structured_records(args.output_dir, pages)
        attach_translations(pages, args.output_dir, structured_by_page)
        attach_placements(pages, args.output_dir, structured_by_page)
        render_pages(
            pages,
            structured_by_page,
            args.output_dir,
            config,
            start_page=start_page,
        )
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    if phase == "package":
        print_packaged_cbz(pages, args.output_dir, args.input_cbz, config)
        return

    raise PipelineError(f"Unsupported resume phase: {phase}")


def needs_vlm(args: argparse.Namespace) -> bool:
    if args.fixture_dir is not None:
        return False
    if args.stop_after == "ocr_raw":
        return False
    return args.resume_from not in {"render", "package"}


def main() -> int:
    signal.signal(signal.SIGTERM, handle_cancel_signal)
    signal.signal(signal.SIGINT, handle_cancel_signal)
    args = parse_args()
    started_at = time.monotonic()
    try:
        if args.resume_from is None and args.resume_page != 0:
            raise PipelineError("--resume-page requires --resume-from.")
        if args.resume_from is None and args.single_page:
            raise PipelineError("--single-page requires --resume-from.")
        if args.skip_package and not args.single_page:
            raise PipelineError("--skip-package requires --single-page.")
        if args.stop_after is not None and args.resume_from is not None:
            raise PipelineError("--stop-after cannot be combined with --resume-from.")
        config = load_config(args.config, args.fixture_dir)
        if args.source_language is not None or args.target_language is not None:
            language = language_config_from_codes(
                args.source_language or config.language.source.code,
                args.target_language or config.language.target.code,
            )
            config = replace(
                config,
                language=language,
                ocr=(
                    replace(config.ocr, lang=language.source.ocr_lang)
                    if args.source_language is not None
                    else config.ocr
                ),
            )
        if args.webp_quality is not None or args.jxl_quality is not None:
            config = replace(
                config,
                webp_quality=(
                    args.webp_quality
                    if args.webp_quality is not None
                    else config.webp_quality
                ),
                jxl_quality=(
                    args.jxl_quality
                    if args.jxl_quality is not None
                    else config.jxl_quality
                ),
            )
        if args.vlm_base_url is not None:
            if config.vlm is None:
                raise PipelineError("--vlm-base-url requires a VLM config.")
            config = replace(
                config,
                vlm=replace(
                    config.vlm,
                    base_url=normalize_vlm_base_url(args.vlm_base_url, "--vlm-base-url"),
                ),
            )
        if args.thinking_budget_tokens is not None and config.vlm is not None:
            config = replace(
                config,
                vlm=replace(
                    config.vlm,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                ),
            )
        if args.proofread_translations is not None or args.write_translation_notes is not None:
            config = replace(
                config,
                postprocess=replace(
                    config.postprocess,
                    proofread_translations=(
                        args.proofread_translations
                        if args.proofread_translations is not None
                        else config.postprocess.proofread_translations
                    ),
                    write_translation_notes=(
                        args.write_translation_notes
                        if args.write_translation_notes is not None
                        else config.postprocess.write_translation_notes
                    ),
                ),
            )
        if args.alt_placement is not None:
            config = replace(config, alt_placement_enabled=args.alt_placement)
        if (
            args.ocr_engine is not None
            or args.ocr_service_url is not None
            or args.paddleocr_vl_server_url is not None
            or args.paddleocr_vl_model is not None
            or args.paddleocr_vl_api_key is not None
            or args.paddleocr_vl_max_concurrency is not None
        ):
            config = replace(
                config,
                ocr=replace(
                    config.ocr,
                    engine=(
                        paddle_ocr_image.normalize_ocr_engine(args.ocr_engine)
                        if args.ocr_engine is not None
                        else config.ocr.engine
                    ),
                    service_url=(
                        paddle_ocr_image.normalize_ocr_service_url(args.ocr_service_url)
                        if args.ocr_service_url is not None
                        else config.ocr.service_url
                    ),
                    paddleocr_vl_server_url=(
                        args.paddleocr_vl_server_url
                        if args.paddleocr_vl_server_url is not None
                        else config.ocr.paddleocr_vl_server_url
                    ),
                    paddleocr_vl_model=(
                        args.paddleocr_vl_model
                        if args.paddleocr_vl_model is not None
                        else config.ocr.paddleocr_vl_model
                    ),
                    paddleocr_vl_api_key=(
                        args.paddleocr_vl_api_key
                        if args.paddleocr_vl_api_key is not None
                        else config.ocr.paddleocr_vl_api_key
                    ),
                    paddleocr_vl_max_concurrency=(
                        args.paddleocr_vl_max_concurrency
                        if args.paddleocr_vl_max_concurrency is not None
                        else config.ocr.paddleocr_vl_max_concurrency
                    ),
                ),
            )
            if (
                config.ocr.paddleocr_vl_max_concurrency is not None
                and config.ocr.paddleocr_vl_max_concurrency <= 0
            ):
                raise PipelineError("--paddleocr-vl-max-concurrency must be positive when provided.")
        if needs_vlm(args):
            config = finalize_vlm_config(config)
        translation_notes = load_translation_notes(args.translation_notes_json)
        if args.resume_from is None:
            run_full_pipeline(args, config, translation_notes)
        else:
            run_resume_pipeline(args, config, translation_notes)
        print_runtime_summary(started_at, args.output_dir)
    except (
        PipelineError,
        clean_text_regions.InputError,
        overlay_text.InputError,
        paddle_ocr_image.InputError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
