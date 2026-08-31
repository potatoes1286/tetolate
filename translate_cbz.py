#!/usr/bin/env python3
"""End-to-end manga/manhwa/manhua CBZ translation pipeline."""

from __future__ import annotations

import argparse
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
import threading
import time
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from PIL import Image

import clean_text_regions
import editor_runtime
import lama_inpaint
from ocr_merge import (
    merge_ocr_records_for_page,
)
import overlay_text
import paddle_ocr_image
from placement_detection import (
    clip_region_values,
    detect_expansions_page,
    dominant_text_container_brightness,
    expand_region,
    expand_region_to_container_width,
    normalized_box_to_region,
    parse_expansion_value,
    parse_fraction_value,
    percentile,
    region_to_normalized_box,
    sampled_pixel_values,
)
from pipeline_types import (
    LanguageConfig,
    OCRConfig,
    Page,
    PipelineCancelled,
    PipelineConfig,
    PipelineError,
    PostprocessConfig,
    SourceLanguageProfile,
    TargetLanguageProfile,
    VLMConfig,
)
from prompt_templates import load_prompt
from vlm_client import call_vlm, finalize_vlm_config, format_elapsed


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

# Output archive variants accepted by the package phase.
PACKAGE_VARIANTS = ("png", "webp", "jxl")

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

# Fonts distributed with tetolate as usable stand-ins when no user fonts are installed.
BUNDLED_FONT_DIR = Path(__file__).resolve().parent / "data" / "bundled_fonts"

# Default intended uses for bundled fonts; replaced by a user font_use.txt when present.
BUNDLED_FONT_USE_PATH = BUNDLED_FONT_DIR / "font_use.txt"

# Fallback font used when a placement does not specify another available font.
DEFAULT_RENDER_FONT = "ComicNeue-Bold.ttf"

# Fallback text fill color for rendered translations when placement styling omits fill.
DEFAULT_RENDER_FILL = "black"

# Default ImageMagick caption gravity, used to center text inside placement regions.
DEFAULT_RENDER_GRAVITY = "center"

# Supported source-language profiles for the first multilingual pass.
DEFAULT_SOURCE_LANGUAGE = "jp"
DEFAULT_TARGET_LANGUAGE = "en"

# Default VLM response budget when max_tokens is omitted from the config.
DEFAULT_VLM_MAX_TOKENS = 32768

# Maximum records sent in one proofreading request. This leaves completion room
# in models where the prompt and response share one context window.
PROOFREADING_BATCH_MAX_RECORDS = 80

# Maximum compact input characters sent in one proofreading request.
PROOFREADING_BATCH_MAX_CHARACTERS = 16_000

# Number of corrected records from the previous batch included for continuity.
PROOFREADING_CONTEXT_RECORDS = 24

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

# Page-level worker defaults. Higher values increase CPU and memory use.
DEFAULT_OCR_PAGE_WORKERS = 1
DEFAULT_LAMA_WORKERS = 1
DEFAULT_IMAGEMAGICK_WORKERS = 1
MAX_PAGE_WORKERS = 32


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

# Minimum stroke width for debug bounding boxes drawn over page images.
VLM_DEBUG_BOX_WIDTH_MIN = 3

# Box stroke width as a fraction of the page's shorter dimension.
VLM_DEBUG_BOX_WIDTH_RATIO = 0.003

# Minimum font size for numeric debug labels drawn on bounding boxes.
VLM_DEBUG_FONT_SIZE_MIN = 18

# Debug label font size as a fraction of the page's shorter dimension.
VLM_DEBUG_FONT_SIZE_RATIO = 0.025


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



def handle_cancel_signal(signum: int, _frame: Any) -> None:
    raise PipelineCancelled(f"received signal {signum}; cancelling pipeline")


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


def normalize_vlm_model(value: Any, label: str = "VLM model") -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{label} must be a non-empty model ID.")
    return value.strip()


def apply_vlm_model_override(config: PipelineConfig, value: Any) -> PipelineConfig:
    if config.vlm is None:
        raise PipelineError("--vlm-model requires a VLM config.")
    return replace(
        config,
        vlm=replace(config.vlm, model=normalize_vlm_model(value, "--vlm-model")),
    )


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
            "regenerate only --resume-page through render."
        ),
    )
    parser.add_argument(
        "--translation-boxno",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help=(
            "Run the requested work without rebuilding translated CBZ archives."
        ),
    )
    parser.add_argument(
        "--package-variant",
        choices=PACKAGE_VARIANTS,
        help=(
            "Generate only this translated CBZ variant. Without this option, "
            "the package phase generates all variants."
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
        "--editor-manifest",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--editor-baseline-dir",
        type=Path,
        help=argparse.SUPPRESS,
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
        "--vlm-model",
        help="Override the VLM model ID from the config.",
    )
    parser.add_argument(
        "--vlm-api-key",
        help="Override the OpenAI-compatible VLM API key from the config.",
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
        "--ocr-workers",
        type=int,
        help="Concurrent PaddleOCR-VL page workers. Regular PaddleOCR stays serial.",
    )
    parser.add_argument(
        "--lama-workers",
        type=int,
        help="Concurrent LaMa page-cleaning workers.",
    )
    parser.add_argument(
        "--imagemagick-workers",
        type=int,
        help="Concurrent ImageMagick page-typesetting workers.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("ocr_merged",),
        help="Stop successfully after the named phase instead of running the full pipeline.",
    )
    return parser.parse_args()


def reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"non-finite number {value}")


def normalize_page_workers(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PipelineError(f"{label} must be an integer from 1 to {MAX_PAGE_WORKERS}.")
    if isinstance(value, float) and not value.is_integer():
        raise PipelineError(f"{label} must be an integer from 1 to {MAX_PAGE_WORKERS}.")
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            f"{label} must be an integer from 1 to {MAX_PAGE_WORKERS}."
        ) from exc
    if workers < 1 or workers > MAX_PAGE_WORKERS:
        raise PipelineError(f"{label} must be from 1 to {MAX_PAGE_WORKERS}.")
    return workers


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
    if isinstance(value, int):
        quality = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise PipelineError(
                f"Config field {label}.{key} must be an integer from 1 to 100."
            )
        quality = int(value)
    elif isinstance(value, str):
        try:
            quality = int(value.strip())
        except ValueError as exc:
            raise PipelineError(
                f"Config field {label}.{key} must be an integer from 1 to 100."
            ) from exc
    else:
        raise PipelineError(
            f"Config field {label}.{key} must be an integer from 1 to 100."
        )
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
            model=(model_value.strip() or None) if isinstance(model_value, str) else None,
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
        ocr_page_workers=DEFAULT_OCR_PAGE_WORKERS,
        lama_workers=DEFAULT_LAMA_WORKERS,
        imagemagick_workers=DEFAULT_IMAGEMAGICK_WORKERS,
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
    system_prompt: str | None = None,
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
    return load_prompt(
        "validation_retry.txt",
        prompt=prompt,
        error=last_error,
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
        system_prompt=load_prompt("system_plain_text_notes.txt"),
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
    system_prompt: str | None = None,
) -> Any:
    if fixture_dir is not None:
        fixture_path = fixture_dir / phase / f"{stem}.txt"
        return validator(fixture_path.read_text(encoding="utf-8"))
    if config is None:
        raise PipelineError("VLM config is missing.")
    if system_prompt is None:
        system_prompt = load_prompt("system_plain_text_lines.txt")

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
    def create_engine() -> Any:
        if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            return paddle_ocr_image.create_paddleocr_vl(
                config.ocr.device,
                config.ocr.paddleocr_vl_server_url,
                config.ocr.paddleocr_vl_model,
                api_key=config.ocr.paddleocr_vl_api_key,
                max_concurrency=config.ocr.paddleocr_vl_max_concurrency,
                service_url=config.ocr.service_url,
                service_timeout=config.ocr.service_timeout,
            )
        return paddle_ocr_image.create_paddle_ocr(
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

    def extract_page(ocr: Any, page: Page) -> list[dict[str, Any]]:
        print(f"OCR page {page.index}", file=sys.stderr)
        if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            print(
                (
                    f"OCR page {page.index}: PaddleOCR-VL 1.6 via "
                    f"{config.ocr.paddleocr_vl_server_url}"
                ),
                file=sys.stderr,
            )
            return paddle_ocr_image.extract_paddleocr_vl_image_records(
                ocr,
                page.image_path,
                page.index,
                config.ocr.min_score,
            )
        if config.ocr.tile_enabled:
            print(
                (
                    f"OCR page {page.index}: tiled "
                    f"{config.ocr.tile_width}x{config.ocr.tile_height} "
                    f"overlap {config.ocr.tile_overlap}"
                ),
                file=sys.stderr,
            )
        return paddle_ocr_image.extract_image_records(
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

    def store_page(page: Page, records: list[dict[str, Any]]) -> None:
        by_page[page.index] = editor_runtime.reconcile_records(
            "ocr_raw", page.index, records, "ocr"
        )
        write_json(data_page_path(output_dir, "ocr_raw", page), by_page[page.index])
        box_width, font_size = debug_annotation_size(page.image_path)
        paddle_ocr_image.draw_boxes(
            by_page[page.index],
            page.image_path,
            debug_image_path(output_dir, "ocr_raw_img", page),
            "#ff2d55",
            box_width,
            font_size,
        )

    selected_pages = pages_in_range(pages, start_page, end_page)
    by_page: dict[int, list[dict[str, Any]]] = dict(existing_by_page or {})
    if not selected_pages:
        ensure_all_pages(by_page, pages, "OCR records")
        return by_page
    worker_count = (
        config.ocr_page_workers
        if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL
        else 1
    )
    if config.ocr.engine != paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL and config.ocr_page_workers > 1:
        print("warning: OCR page workers are ignored for regular PaddleOCR.", file=sys.stderr)
    if config.ocr.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL and config.ocr.tile_enabled:
        print("warning: tiled OCR is ignored by PaddleOCR-VL.", file=sys.stderr)

    if worker_count == 1 or len(selected_pages) < 2:
        ocr = create_engine()
        try:
            for page in selected_pages:
                store_page(page, extract_page(ocr, page))
        finally:
            paddle_ocr_image.close_ocr_engine(ocr)
    else:
        worker_state = threading.local()
        engines: list[Any] = []
        engines_lock = threading.Lock()

        def parallel_extract(page: Page) -> list[dict[str, Any]]:
            ocr = getattr(worker_state, "ocr", None)
            if ocr is None:
                ocr = create_engine()
                worker_state.ocr = ocr
                with engines_lock:
                    engines.append(ocr)
            return extract_page(ocr, page)

        try:
            with ThreadPoolExecutor(
                max_workers=min(worker_count, len(selected_pages)),
                thread_name_prefix="tetolate-ocr",
            ) as executor:
                futures = {
                    page.index: executor.submit(parallel_extract, page)
                    for page in selected_pages
                }
                for page in selected_pages:
                    store_page(page, futures[page.index].result())
        finally:
            for ocr in engines:
                paddle_ocr_image.close_ocr_engine(ocr)
    ensure_all_pages(by_page, pages, "OCR records")
    return by_page


def flatten_pages(by_page: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in sorted(by_page):
        records.extend(by_page[page])
    return records


def debug_annotation_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        shorter_dimension = min(image.size)
    box_width = max(
        VLM_DEBUG_BOX_WIDTH_MIN,
        round(shorter_dimension * VLM_DEBUG_BOX_WIDTH_RATIO),
    )
    font_size = max(
        VLM_DEBUG_FONT_SIZE_MIN,
        round(shorter_dimension * VLM_DEBUG_FONT_SIZE_RATIO),
    )
    return box_width, font_size


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
    box_width, font_size = debug_annotation_size(page.image_path)
    paddle_ocr_image.draw_boxes(
        debug_records,
        page.image_path,
        output_path,
        color,
        box_width,
        font_size,
    )
    return output_path


def ensure_structured_debug_image(
    records: list[dict[str, Any]],
    page: Page,
    output_dir: Path,
) -> Path:
    return draw_region_debug_image(
        records,
        page,
        output_dir,
        "ocr_structured_img",
        VLM_STRUCTURED_DEBUG_COLOR,
    )


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


def available_font_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for directory in (BUNDLED_FONT_DIR, FONT_DIR):
        if not directory.is_dir():
            continue
        paths.update(
            {
                path.name: path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS
            }
        )
    return paths


def available_font_names() -> set[str]:
    return set(available_font_paths())


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
    selected = value if isinstance(value, str) and value.strip() else fallback
    name = Path(selected.strip().strip("[]")).name
    candidate = available_font_paths().get(name)
    if candidate is not None:
        return str(candidate)
    return selected


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
    font_use_path = FONT_USE_PATH if FONT_USE_PATH.is_file() else BUNDLED_FONT_USE_PATH
    if not font_use_path.is_file():
        return f"No local font_use.txt was found. Use backup font {backup_font!r}."

    lines: list[str] = []
    font_names = available_font_names()
    for raw_line in font_use_path.read_text(encoding="utf-8").splitlines():
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
    return load_prompt(
        "structure.txt",
        structure_context=language.source.structure_context,
        reading_order=language.source.reading_order,
        source_name=language.source.name,
        target_name=language.target.name,
        record_count=len(page_ocr),
        page_index=page.index,
        records_table=compact_records_table(
            page_ocr, ("boxno", "sourceBoxnos", "region", "text")
        ),
    )


def character_in_ranges(character: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    value = ord(character)
    return any(start <= value <= end for start, end in ranges)


def text_needs_translation(text: str, language: LanguageConfig) -> bool:
    value = text.strip()
    if not value:
        return False
    source_ranges = {
        "jp": (
            (0x3040, 0x30FF),
            (0x31F0, 0x31FF),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0xFF66, 0xFF9D),
        ),
        "cn": (
            (0x3100, 0x312F),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
        ),
        "kr": (
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xAC00, 0xD7AF),
            (0xF900, 0xFAFF),
        ),
    }.get(language.source.code, ())
    if any(character_in_ranges(character, source_ranges) for character in value):
        return True

    letters = [character for character in value if character.isalpha()]
    if not letters:
        return False
    if language.target.code == "en" and all(
        "LATIN" in unicodedata.name(character, "") for character in letters
    ):
        return False
    return True


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
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", stripped):
            return int(stripped)
    raise PipelineError(f"{label} must include non-negative integer `{key}`.")


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


def run_ocr_merge_phase(
    pages: list[Page],
    raw_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    config: PipelineConfig,
    start_page: int = 0,
    end_page: int | None = None,
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
    for page in selected_pages:
        merged_by_page[page.index] = editor_runtime.reconcile_records(
            "ocr_merged", page.index, merged_by_page[page.index], "ocr"
        )
        write_json(
            data_page_path(output_dir, "ocr_merged", page),
            merged_by_page[page.index],
        )
    return merged_by_page


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
    selected_pages = pages_in_range(pages, start_page, end_page)
    merged_by_page = run_ocr_merge_phase(
        pages,
        raw_by_page,
        output_dir,
        config,
        start_page=start_page,
        end_page=end_page,
        use_existing_merged=use_existing_merged,
    )

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
        skipped = [
            record
            for record in structured
            if not text_needs_translation(
                str(record.get("text", "")), config.language
            )
        ]
        if skipped:
            labels = ", ".join(str(record.get("boxno", "?")) for record in skipped)
            print(
                f"Skip translation-free text page {page.index} boxnos: {labels}",
                file=sys.stderr,
            )
            structured = [
                record
                for record in structured
                if text_needs_translation(
                    str(record.get("text", "")), config.language
                )
            ]
            for boxno, record in enumerate(structured):
                record["boxno"] = boxno
        structured_by_page[page.index] = structured
        structured = editor_runtime.reconcile_records(
            "ocr_structured", page.index, structured, "structure"
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
    return load_prompt(
        "alt_placement.txt",
        structure_context=language.source.structure_context,
        source_name=language.source.name,
        target_name=language.target.name,
        page_index=page.index,
        records_table=compact_records_table(
            structured_records, ("boxno", "region", "text", "sfx")
        ),
    )


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
        alt_placements = editor_runtime.reconcile_records(
            "alt_placement", page.index, alt_placements, "erase"
        )
        by_page[page.index] = alt_placements
        write_json(data_page_path(output_dir, "alt_placement", page), alt_placements)
        apply_alt_placements_to_records(structured_records, alt_placements)
        structured_records[:] = editor_runtime.reconcile_records(
            "ocr_structured", page.index, structured_records, "erase"
        )
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
    current_records: list[dict[str, Any]] | None = None,
) -> str:
    master = flatten_pages(structured_by_page)
    records_to_translate = (
        structured_by_page[page.index]
        if current_records is None
        else current_records
    )
    previous_translations = previous_translation_records(page, structured_by_page)
    notes_section = translation_notes_prompt_section(page, translation_notes)
    return load_prompt(
        "translation.txt",
        source_name=language.source.name,
        target_name=language.target.name,
        reading_order=language.source.reading_order,
        master_table=compact_records_table(
            master, ("page", "boxno", "text", "sfx", "openLettering")
        ),
        previous_translations_table=compact_records_table(
            previous_translations,
            ("page", "boxno", "text", "englishText", "sfx", "openLettering"),
        ),
        page_index=page.index,
        current_records_table=compact_records_table(
            records_to_translate, ("boxno", "text", "sfx", "openLettering")
        ),
        notes_section=notes_section,
    )


def validate_translation_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
    require_non_empty: bool = True,
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
        if require_non_empty and not english_text.strip():
            record_type = "SFX" if by_boxno[boxno].get("sfx") else "text"
            raise PipelineError(
                f"{label} returns an empty translation for kept {record_type} "
                f"boxno {boxno}; provide visible target-language text."
            )
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
    require_non_empty: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    translations_by_page: dict[int, list[dict[str, Any]]] = {}
    for page in pages:
        path = data_page_path(output_dir, "translations", page)
        data = load_json_array(path, f"translation page {page.index}")
        translations = validate_translation_page(
            page,
            structured_by_page[page.index],
            data,
            require_non_empty=require_non_empty,
        )
        translations = editor_runtime.apply_protected_records(
            "translations", page.index, translations, "translation"
        )
        translations_by_page[page.index] = translations
        write_json(path, translations)
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
            english_text = record.get("englishText")
            if not isinstance(english_text, str) or not english_text.strip():
                raise PipelineError(
                    f"Missing non-empty englishText for page {page.index} "
                    f"boxno {record['boxno']}"
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
    selected_boxno: int | None = None,
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

        records_to_translate = structured_records
        existing_translations: list[dict[str, Any]] | None = None
        if selected_boxno is not None:
            records_to_translate = [
                record
                for record in structured_records
                if record.get("boxno") == selected_boxno
            ]
            if not records_to_translate:
                raise PipelineError(
                    f"Translation page {page.index} has no boxno {selected_boxno}."
                )
            existing_data = load_json_array(
                data_page_path(output_dir, "translations", page),
                f"translation page {page.index}",
            )
            existing_translations = validate_translation_page(
                page,
                structured_records,
                existing_data,
            )

        translation_label = (
            f"page {page.index} boxno {selected_boxno}"
            if selected_boxno is not None
            else f"page {page.index}"
        )
        print(f"VLM translate {translation_label}", file=sys.stderr)
        structured_debug_path = ensure_structured_debug_image(
            structured_records,
            page,
            output_dir,
        )
        generated_translations = get_validated_vlm_array(
            "translations",
            page,
            translation_prompt(
                page,
                structured_by_page,
                translation_notes,
                config.language,
                records_to_translate,
            ),
            output_dir,
            config.vlm,
            fixture_dir,
            lambda items, current_page=page: validate_translation_page(
                current_page,
                records_to_translate,
                items,
            ),
            structured_debug_path,
        )
        if selected_boxno is None:
            translations = editor_runtime.reconcile_records(
                "translations", page.index, generated_translations, "translation"
            )
        else:
            assert existing_translations is not None
            translations = editor_runtime.reconcile_record_subset(
                "translations",
                page.index,
                generated_translations,
                existing_translations,
                "translation",
                "boxno",
            )
        translations_by_page[page.index] = translations
        by_boxno = {item["boxno"]: item for item in translations}
        for record in structured_records:
            record["englishText"] = by_boxno[record["boxno"]]["englishText"]
        write_json(data_page_path(output_dir, "translations", page), translations)
        data_page_path(output_dir, "translations_raw", page).unlink(missing_ok=True)
    ensure_translated_pages(
        pages_in_range(pages, start_page, end_page),
        structured_by_page,
    )
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


def compact_proofreading_batch_input(records: list[dict[str, Any]]) -> str:
    lines = ["row\tpage:box\tsfx\tsourceText\tenglishText"]
    for row, record in enumerate(records):
        lines.append(
            "\t".join(
                (
                    str(row),
                    f"{record['page']}:{record['boxno']}",
                    "S" if record.get("sfx") else "-",
                    compact_proofreading_field(record.get("text", "")),
                    compact_proofreading_field(record.get("englishText", "")),
                )
            )
        )
    return "\n".join(lines)


def compact_proofreading_context(records: list[dict[str, Any]]) -> str:
    lines = ["page:box\tsfx\tsourceText\tenglishText"]
    for record in records:
        lines.append(
            "\t".join(
                (
                    f"{record['page']}:{record['boxno']}",
                    "S" if record.get("sfx") else "-",
                    compact_proofreading_field(record.get("text", "")),
                    compact_proofreading_field(record.get("englishText", "")),
                )
            )
        )
    return "\n".join(lines)


def proofreading_record_batches(
    records: list[dict[str, Any]],
    max_records: int = PROOFREADING_BATCH_MAX_RECORDS,
    max_characters: int = PROOFREADING_BATCH_MAX_CHARACTERS,
) -> list[list[dict[str, Any]]]:
    if max_records <= 0 or max_characters <= 0:
        raise ValueError("Proofreading batch limits must be positive.")

    page_groups: list[list[dict[str, Any]]] = []
    for record in records:
        if not page_groups or page_groups[-1][0]["page"] != record["page"]:
            page_groups.append([])
        page_groups[-1].append(record)

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_characters = 0
    for page_records in page_groups:
        page_characters = len(compact_proofreading_input(page_records))
        would_exceed_limit = current and (
            len(current) + len(page_records) > max_records
            or current_characters + page_characters > max_characters
        )
        if would_exceed_limit:
            batches.append(current)
            current = []
            current_characters = 0
        current.extend(page_records)
        current_characters += page_characters
    if current:
        batches.append(current)
    return batches


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
    prior_context: list[dict[str, Any]] | None = None,
) -> str:
    records = proofread_records_in_order(structured_by_page)
    page_indexes = sorted({int(record["page"]) for record in records})
    if not page_indexes:
        batch_scope = "an empty batch"
    elif page_indexes[0] == page_indexes[-1]:
        batch_scope = f"page {page_indexes[0]}"
    else:
        batch_scope = f"pages {page_indexes[0]} through {page_indexes[-1]}"
    prior_context_table = (
        compact_proofreading_context(prior_context)
        if prior_context
        else "No earlier corrected records are available."
    )
    return load_prompt(
        "proofreading.txt",
        source_name=language.source.name,
        target_name=language.target.name,
        batch_scope=batch_scope,
        translation_notes=translation_notes_book_prompt_section(translation_notes),
        prior_context=prior_context_table,
        records_table=compact_proofreading_batch_input(records),
    )


def validate_proofread_record_batch(
    records: list[dict[str, Any]],
    raw_text: str,
) -> list[dict[str, Any]]:
    text = strip_markdown_code_fence(raw_text).strip()
    replacements: dict[int, str] = {}
    if text != "<NO_CHANGES>":
        if not text:
            raise PipelineError(
                "Proofreading returned an empty response; expected changed rows or <NO_CHANGES>."
            )
        for line_number, line in enumerate(text.splitlines(), start=1):
            fields = line.split("\t", 1)
            if len(fields) != 2:
                raise PipelineError(
                    f"Proofreading output line {line_number} must be row<TAB>englishText."
                )
            try:
                row = int(fields[0].strip())
            except ValueError as exc:
                raise PipelineError(
                    f"Proofreading output line {line_number} has invalid row {fields[0]!r}."
                ) from exc
            if row < 0 or row >= len(records):
                raise PipelineError(
                    f"Proofreading output line {line_number} references missing row {row}."
                )
            if row in replacements:
                raise PipelineError(f"Proofreading output duplicates row {row}.")
            english_text = fields[1].strip()
            if not english_text:
                raise PipelineError(
                    f"Proofreading output row {row} has empty text."
                )
            if english_text == "<EMPTY>":
                raise PipelineError(
                    f"Proofreading output row {row} must not remove translated text."
                )
            replacements[row] = english_text

    corrected: list[dict[str, Any]] = []
    for row, source in enumerate(records):
        english_text = replacements.get(row, source.get("englishText", ""))
        if not isinstance(english_text, str) or not english_text.strip():
            raise PipelineError(
                f"Proofreading row {row} has no translated text; provide a correction."
            )
        corrected.append(
            {
                "page": source["page"],
                "boxno": source["boxno"],
                "text": source.get("text", ""),
                "englishText": english_text,
                "sfx": source.get("sfx", False),
                "openLettering": source.get("openLettering", False),
            }
        )
    return corrected


def apply_translation_records(
    pages: list[Page],
    structured_by_page: dict[int, list[dict[str, Any]]],
    translations_by_page: dict[int, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    for page in pages:
        translations = editor_runtime.reconcile_records(
            "translations",
            page.index,
            translations_by_page.get(page.index, []),
            "translation",
        )
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
    records = proofread_records_in_order(structured_by_page)
    if not records:
        print("Skip VLM proofread translations: no translated records", file=sys.stderr)
        return
    batches = proofreading_record_batches(records)
    print(
        (
            f"VLM proofread {len(records)} translations in "
            f"{len(batches)} batch(es)"
        ),
        file=sys.stderr,
    )
    translations_by_page: dict[int, list[dict[str, Any]]] = {
        page.index: [] for page in pages
    }
    prior_context: list[dict[str, Any]] = []
    for batch_index, batch_records in enumerate(batches, start=1):
        page_indexes = sorted({int(record["page"]) for record in batch_records})
        batch_by_page: dict[int, list[dict[str, Any]]] = {}
        for record in batch_records:
            batch_by_page.setdefault(int(record["page"]), []).append(record)
        stem = f"batch_{batch_index:04d}_pages_{page_indexes[0]:04d}-{page_indexes[-1]:04d}"
        label = (
            f"VLM proofread translations batch {batch_index}/{len(batches)} "
            f"(pages {page_indexes[0]}-{page_indexes[-1]})"
        )
        corrected = get_validated_vlm_text_named(
            "proofreading",
            stem,
            label,
            proofreading_prompt(
                batch_by_page,
                translation_notes,
                config.language,
                prior_context,
            ),
            output_dir,
            config.vlm,
            fixture_dir,
            lambda text, expected=batch_records: validate_proofread_record_batch(
                expected, text
            ),
        )
        for record in corrected:
            translations_by_page[record["page"]].append(record)
        prior_context = (prior_context + corrected)[-PROOFREADING_CONTEXT_RECORDS:]

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
    return load_prompt(
        "translation_notes.txt",
        source_name=language.source.name,
        target_name=language.target.name,
        translation_notes=translation_notes_book_prompt_section(translation_notes),
        records_table=compact_proofreading_input(records),
    )


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


def labeled_open_placement_records(
    structured_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "region": record["region"],
            "text": record["text"],
            "englishText": record.get("englishText"),
            "sfx": record["sfx"],
        }
        for label, record in enumerate(open_lettering_records(structured_records))
    ]


def placement_open_prompt(
    page: Page,
    structured_records: list[dict[str, Any]],
    language: LanguageConfig,
) -> str:
    open_records = labeled_open_placement_records(structured_records)
    return load_prompt(
        "placement_open.txt",
        target_name=language.target.name,
        source_name=language.source.name,
        reading_order=language.source.reading_order,
        page_index=page.index,
        records_table=compact_records_table(
            open_records, ("label", "region", "text", "englishText", "sfx")
        ),
    )


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
    return load_prompt(
        "placement_style.txt",
        font_uses=font_use_prompt(backup_font),
        backup_font=backup_font,
        target_name=language.target.name,
        reading_order=language.source.reading_order,
        page_index=page.index,
        records_table=compact_records_table(
            placement_style_records(structured_records, placements),
            (
                "boxno",
                "text",
                "englishText",
                "openLettering",
                "sfx",
                "box_2d",
                "placementRegion",
            ),
        ),
    )


def validate_open_placements_page(
    page: Page,
    structured_records: list[dict[str, Any]],
    vlm_items: list[Any],
) -> list[dict[str, Any]]:
    open_records = open_lettering_records(structured_records)
    expected_labels = set(range(len(open_records)))
    placements: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, value in enumerate(vlm_items):
        label = f"open placement page {page.index} item {index}"
        if isinstance(value, list):
            row = require_row(value, label, 2)
            item_page = page.index
            item_label = parse_non_negative_int_value(row[0], label, "label")
            box = row[1]
        else:
            record = require_object(value, label)
            item_page = require_non_negative_int(record, "page", label) if "page" in record else page.index
            item_label = require_non_negative_int(record, "label", label)
            box = record.get("box_2d")
        if item_page != page.index:
            raise PipelineError(f"{label} has page {item_page}, expected {page.index}.")
        if item_label not in expected_labels:
            raise PipelineError(f"{label} references missing label {item_label}.")
        if item_label in seen:
            raise PipelineError(f"{label} duplicates label {item_label}.")
        if not isinstance(box, list):
            raise PipelineError(f"{label} must include box_2d array.")
        boxno = open_records[item_label]["boxno"]
        placement = {
            "page": page.index,
            "boxno": boxno,
            "box_2d": box,
            "placementRegion": normalized_box_to_region(box, page.image_path),
        }
        placements.append(placement)
        seen.add(item_label)
    missing = sorted(expected_labels - seen)
    if missing:
        raise PipelineError(f"Open placement missing page {page.index} labels: {missing}")
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
        placement_region = record.get("placementRegion")
        if isinstance(placement_region, list) and len(placement_region) == 4:
            box = region_to_normalized_box(placement_region, page.image_path)
        if not isinstance(box, list):
            raise PipelineError(f"{label} must include box_2d or placementRegion array.")
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
            "overlapAdjusted",
        ):
            if optional_key in record:
                placement[optional_key] = record[optional_key]
        font_name = normalize_font_name(record.get("font"))
        if font_name is not None:
            placement["font"] = font_name
        fill = normalize_fill(record.get("fill", record.get("color", record.get("colour"))))
        if fill is not None:
            placement["fill"] = fill
        for optional_key in (
            "stroke",
            "strokeWidth",
            "gravity",
            "fontSizeWidthPercent",
            "manualLineBreaks",
            "minPointSize",
            "maxPointSize",
        ):
            if optional_key in record:
                placement[optional_key] = record[optional_key]
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
                "overlapAdjusted",
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
        for field in (
            "font",
            "fill",
            "stroke",
            "strokeWidth",
            "gravity",
            "fontSizeWidthPercent",
            "manualLineBreaks",
            "minPointSize",
            "maxPointSize",
        ):
            if field in placement:
                record[field] = placement[field]


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
        placements = editor_runtime.apply_protected_records(
            "placements", page.index, placements, "placement"
        )
        write_json(path, placements)
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
    ensure_translated_pages(
        pages_in_range(pages, start_page, end_page),
        structured_by_page,
    )
    placements_by_page: dict[int, list[dict[str, Any]]] = dict(existing_by_page or {})
    for page in pages_in_range(pages, start_page, end_page):
        structured_records = structured_by_page[page.index]
        if not structured_records:
            placements = []
        else:
            open_records = open_lettering_records(structured_records)
            if open_records:
                print(f"VLM open placement page {page.index}", file=sys.stderr)
                open_placement_debug_path = draw_region_debug_image(
                    labeled_open_placement_records(structured_records),
                    page,
                    output_dir,
                    "placement_open_input_img",
                    VLM_PLACEMENT_DEBUG_COLOR,
                )
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
                    open_placement_debug_path,
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
            preliminary_debug_path = draw_region_debug_image(
                preliminary,
                page,
                output_dir,
                "placement_preliminary_img",
                VLM_PLACEMENT_DEBUG_COLOR,
                "placementRegion",
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
                preliminary_debug_path,
            )
            write_json(trace_page_path(output_dir, "placement_style", page), styles)
            placements = merge_placement_styles(page, preliminary, styles)
            apply_placements_to_records(structured_records, placements)
        placements_by_page[page.index] = placements
        placements = editor_runtime.reconcile_records(
            "placements", page.index, placements, "placement"
        )
        placements_by_page[page.index] = placements
        apply_placements_to_records(structured_records, placements)
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


def clean_render_page(
    page: Page,
    records: list[dict[str, Any]],
    output_dir: Path,
    lama_session: lama_inpaint.LaMaSession | None,
) -> None:
    print(f"Clean page {page.index}", file=sys.stderr)
    cleaned_path = cleaned_pages_dir(output_dir) / f"{page.image_path.stem}.png"
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
        lama_session,
    )


def typeset_render_page(
    page: Page,
    records: list[dict[str, Any]],
    output_dir: Path,
    config: PipelineConfig,
) -> None:
    print(f"Render page {page.index}", file=sys.stderr)
    cleaned_path = cleaned_pages_dir(output_dir) / f"{page.image_path.stem}.png"
    final_path = final_pages_dir(output_dir) / f"{page.image_path.stem}.png"
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
                **(
                    {"fontSizeWidthPercent": record["fontSizeWidthPercent"]}
                    if isinstance(record.get("fontSizeWidthPercent"), (int, float))
                    else {}
                ),
                **(
                    {"englishText": record["manualLineBreaks"]}
                    if isinstance(record.get("manualLineBreaks"), str)
                    and record["manualLineBreaks"].strip()
                    else {}
                ),
                **(
                    {"stroke": record["stroke"]}
                    if isinstance(record.get("stroke"), str)
                    else {}
                ),
                **(
                    {"strokeWidth": record["strokeWidth"]}
                    if isinstance(record.get("strokeWidth"), (int, float))
                    else {}
                ),
                **(
                    {"gravity": record["gravity"]}
                    if isinstance(record.get("gravity"), str)
                    else {}
                ),
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
    selected_pages = pages_in_range(pages, start_page, end_page)
    if not selected_pages:
        return
    ensure_translated_pages(selected_pages, structured_by_page)
    ensure_placed_pages(selected_pages, structured_by_page)

    needs_lama = any(
        not record["openLettering"]
        for page in selected_pages
        for record in structured_by_page[page.index]
    )
    lama_session = (
        lama_inpaint.LaMaSession(clean_text_regions.DEFAULT_DEVICE)
        if needs_lama
        else None
    )
    try:
        with ThreadPoolExecutor(
            max_workers=min(config.lama_workers, len(selected_pages)),
            thread_name_prefix="tetolate-lama",
        ) as executor:
            futures = {
                page.index: executor.submit(
                    clean_render_page,
                    page,
                    structured_by_page[page.index],
                    output_dir,
                    lama_session,
                )
                for page in selected_pages
            }
            for page in selected_pages:
                futures[page.index].result()
    finally:
        if lama_session is not None:
            lama_session.close()

    with ThreadPoolExecutor(
        max_workers=min(config.imagemagick_workers, len(selected_pages)),
        thread_name_prefix="tetolate-imagemagick",
    ) as executor:
        futures = {
            page.index: executor.submit(
                typeset_render_page,
                page,
                structured_by_page[page.index],
                output_dir,
                config,
            )
            for page in selected_pages
        }
        for page in selected_pages:
            futures[page.index].result()


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
    *,
    source_archive: zipfile.ZipFile | None = None,
    source_members: list[zipfile.ZipInfo] | None = None,
) -> None:
    def copy_members(
        source: zipfile.ZipFile,
        members: list[zipfile.ZipInfo],
    ) -> None:
        archive.comment = source.comment
        for info in members:
            if not is_preserved_non_image_member(info):
                continue
            if info.filename in written_names:
                print(
                    f"warning: skipping original non-image entry that conflicts with output: {info.filename}",
                    file=sys.stderr,
                )
                continue
            with source.open(info) as source_file, archive.open(
                copy_zipinfo_for_deflated_write(info),
                "w",
            ) as destination:
                shutil.copyfileobj(source_file, destination, length=1024 * 1024)
            written_names.add(info.filename)

    if source_archive is not None:
        if source_members is None:
            source_members = validate_cbz_members(source_archive)
        copy_members(source_archive, source_members)
        return

    try:
        with zipfile.ZipFile(input_cbz) as source:
            copy_members(source, validate_cbz_members(source))
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


def package_cbz(
    pages: list[Page],
    output_dir: Path,
    input_cbz: Path,
    *,
    source_archive: zipfile.ZipFile | None = None,
    source_members: list[zipfile.ZipInfo] | None = None,
) -> Path:
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

            write_preserved_non_image_members(
                archive,
                input_cbz,
                written_names,
                source_archive=source_archive,
                source_members=source_members,
            )
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
    imagemagick_workers: int = DEFAULT_IMAGEMAGICK_WORKERS,
    *,
    source_archive: zipfile.ZipFile | None = None,
    source_members: list[zipfile.ZipInfo] | None = None,
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
                conversion_jobs: list[tuple[Page, Path]] = []
                for page in pages:
                    source_path = final_page_png_path(output_dir, page)
                    if not source_path.exists():
                        raise PipelineError(f"Final page image missing: {source_path}")
                    page_suffix = TRANSLATED_ALT_COVER_SUFFIX if page.index == 0 else suffix
                    converted_path = temp_dir / f"{page.image_path.stem}{page_suffix}"
                    conversion_jobs.append((page, converted_path))

                if conversion_jobs:
                    with ThreadPoolExecutor(
                        max_workers=min(imagemagick_workers, len(conversion_jobs)),
                        thread_name_prefix="tetolate-imagemagick-package",
                    ) as executor:
                        futures = [
                            executor.submit(
                                convert_final_page_with_magick,
                                final_page_png_path(output_dir, page),
                                converted_path,
                                quality,
                            )
                            for page, converted_path in conversion_jobs
                        ]
                        for future, (_page, converted_path) in zip(
                            futures,
                            conversion_jobs,
                        ):
                            future.result()
                            archive.write(converted_path, arcname=converted_path.name)
                            written_names.add(converted_path.name)
                            converted_path.unlink()

                write_preserved_non_image_members(
                    archive,
                    input_cbz,
                    written_names,
                    source_archive=source_archive,
                    source_members=source_members,
                )
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
    package_variant: str | None = None,
) -> None:
    if package_variant is not None and package_variant not in PACKAGE_VARIANTS:
        raise PipelineError(f"Unsupported package variant: {package_variant}")

    try:
        with zipfile.ZipFile(input_cbz) as source_archive:
            source_members = validate_cbz_members(source_archive)
            variants = {
                "png": (
                    translated_cbz_path(output_dir),
                    None,
                    None,
                    "PNG",
                ),
                "webp": (
                    translated_webp_cbz_path(output_dir),
                    ".webp",
                    config.webp_quality,
                    "WebP",
                ),
                "jxl": (
                    translated_jxl_cbz_path(output_dir),
                    ".jxl",
                    config.jxl_quality,
                    "JXL",
                ),
            }
            variant_names = (
                (package_variant,)
                if package_variant is not None
                else PACKAGE_VARIANTS
            )
            for variant_name in variant_names:
                variant_path, suffix, quality, label = variants[variant_name]
                try:
                    if variant_name == "png":
                        written_path = package_cbz(
                            pages,
                            output_dir,
                            input_cbz,
                            source_archive=source_archive,
                            source_members=source_members,
                        )
                    else:
                        assert suffix is not None
                        assert quality is not None
                        written_path = package_converted_cbz(
                            pages,
                            output_dir,
                            input_cbz,
                            variant_path,
                            suffix,
                            quality,
                            config.imagemagick_workers,
                            source_archive=source_archive,
                            source_members=source_members,
                        )
                except PipelineError as exc:
                    if package_variant is not None or variant_name == "png":
                        raise
                    print(f"warning: skipped {label} CBZ output: {exc}", file=sys.stderr)
                    continue
                print(f"Wrote {written_path}", file=sys.stderr)
    except zipfile.BadZipFile as exc:
        raise PipelineError(f"Input file is not a valid CBZ/zip: {input_cbz}") from exc


def maybe_print_packaged_cbz(
    args: argparse.Namespace,
    pages: list[Page],
    config: PipelineConfig,
) -> None:
    """Package output unless the caller explicitly requested rendered pages only."""
    if getattr(args, "skip_package", False):
        return
    print_packaged_cbz(
        pages,
        args.output_dir,
        args.input_cbz,
        config,
        package_variant=getattr(args, "package_variant", None),
    )



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
    if args.stop_after == "ocr_merged":
        run_ocr_merge_phase(pages, raw_by_page, args.output_dir, config)
        print("Stopped after OCR merge", file=sys.stderr)
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
    maybe_print_packaged_cbz(args, pages, config)


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
    selected_pages = pages_in_range(pages, target_page, target_page)

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
            require_non_empty=False,
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
            selected_boxno=args.translation_boxno,
        )
    else:
        attach_translations(selected_pages, args.output_dir, structured_by_page)

    if args.translation_boxno is not None:
        return

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
        attach_placements(selected_pages, args.output_dir, structured_by_page)

    render_pages(
        pages,
        structured_by_page,
        args.output_dir,
        config,
        start_page=target_page,
        end_page=target_page,
    )
    maybe_print_packaged_cbz(args, pages, config)


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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
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
        maybe_print_packaged_cbz(args, pages, config)
        return

    if phase == "package":
        print_packaged_cbz(
            pages,
            args.output_dir,
            args.input_cbz,
            config,
            package_variant=getattr(args, "package_variant", None),
        )
        return

    raise PipelineError(f"Unsupported resume phase: {phase}")


def needs_vlm(args: argparse.Namespace) -> bool:
    if args.fixture_dir is not None:
        return False
    if args.stop_after == "ocr_merged":
        return False
    return args.resume_from not in {"render", "package"}


def main() -> int:
    signal.signal(signal.SIGTERM, handle_cancel_signal)
    signal.signal(signal.SIGINT, handle_cancel_signal)
    args = parse_args()
    if (args.editor_manifest is None) != (args.editor_baseline_dir is None):
        print(
            "error: --editor-manifest and --editor-baseline-dir must be provided together.",
            file=sys.stderr,
        )
        return 1
    editor_runtime.configure(args.editor_manifest, args.editor_baseline_dir)
    started_at = time.monotonic()
    try:
        if args.resume_from is None and args.resume_page != 0:
            raise PipelineError("--resume-page requires --resume-from.")
        if args.resume_from is None and args.single_page:
            raise PipelineError("--single-page requires --resume-from.")
        if args.translation_boxno is not None:
            if args.translation_boxno < 0:
                raise PipelineError("--translation-boxno must be zero or greater.")
            if args.resume_from != "translations" or not args.single_page:
                raise PipelineError(
                    "--translation-boxno requires --resume-from translations and --single-page."
                )
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
        if any(
            value is not None
            for value in (
                args.ocr_workers,
                args.lama_workers,
                args.imagemagick_workers,
            )
        ):
            config = replace(
                config,
                ocr_page_workers=(
                    normalize_page_workers(args.ocr_workers, "--ocr-workers")
                    if args.ocr_workers is not None
                    else config.ocr_page_workers
                ),
                lama_workers=(
                    normalize_page_workers(args.lama_workers, "--lama-workers")
                    if args.lama_workers is not None
                    else config.lama_workers
                ),
                imagemagick_workers=(
                    normalize_page_workers(
                        args.imagemagick_workers,
                        "--imagemagick-workers",
                    )
                    if args.imagemagick_workers is not None
                    else config.imagemagick_workers
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
        if args.vlm_model is not None:
            config = apply_vlm_model_override(config, args.vlm_model)
        vlm_api_key = args.vlm_api_key
        if vlm_api_key is None:
            vlm_api_key = os.environ.get("TETOLATE_VLM_API_KEY")
        if vlm_api_key is not None:
            if config.vlm is None:
                raise PipelineError("A VLM API key requires a VLM config.")
            config = replace(config, vlm=replace(config.vlm, api_key=vlm_api_key))
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
                        else os.environ.get("TETOLATE_PADDLEOCR_VL_API_KEY")
                        or config.ocr.paddleocr_vl_api_key
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
