#!/usr/bin/env python3
"""Minimal admin-only web UI for managing CBZ translation jobs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import queue
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

import paddle_ocr_image
import translate_cbz
import web_security


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = translate_cbz.DATA_DIR
DEFAULT_WEB_CONFIG = DEFAULT_DATA_DIR / "config" / "web_config.json"
DEFAULT_JOBS_DIR = DEFAULT_DATA_DIR / "jobs"
DEFAULT_PIPELINE_CONFIG = DEFAULT_DATA_DIR / "config" / "vlm_config.json"
DEFAULT_TRANSLATE_SCRIPT = REPO_DIR / "translate_cbz.py"
TRANSLATED_CBZ_FILENAMES = {
    "png": "translated.cbz",
    "webp": "translated_webp.cbz",
    "jxl": "translated_jxl.cbz",
}
DEFAULT_WEBP_QUALITY = translate_cbz.TRANSLATED_WEBP_QUALITY
DEFAULT_JXL_QUALITY = translate_cbz.TRANSLATED_JXL_QUALITY
DEFAULT_THINKING_BUDGET_TOKENS = translate_cbz.DEFAULT_VLM_THINKING_BUDGET_TOKENS
DEFAULT_PROOFREAD_TRANSLATIONS = True
DEFAULT_WRITE_TRANSLATION_NOTES = True
DEFAULT_ALT_PLACEMENT_ENABLED = translate_cbz.DEFAULT_ALT_PLACEMENT_ENABLED
DEFAULT_OCR_ENGINE = paddle_ocr_image.DEFAULT_OCR_ENGINE
DEFAULT_PADDLEOCR_VL_SERVER_URL = paddle_ocr_image.DEFAULT_PADDLEOCR_VL_SERVER_URL
DEFAULT_PADDLEOCR_VL_MODEL = paddle_ocr_image.DEFAULT_PADDLEOCR_VL_MODEL
DEFAULT_SOURCE_LANGUAGE = translate_cbz.DEFAULT_SOURCE_LANGUAGE
GENERATED_TRANSLATION_NOTES_NAME = translate_cbz.GENERATED_TRANSLATION_NOTES_NAME
DEFAULT_LISTEN = "127.0.0.1:8088"
DEFAULT_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_PAGE_SELECTION_ITEMS = 10_000
ADMIN_COOKIE_NAME = "tetolate_admin"
ADMIN_SESSION_SECONDS = 12 * 60 * 60
ADMIN_LOGIN_FAILURE_LIMIT = 5
ADMIN_LOGIN_FAILURE_WINDOW_SECONDS = 60
WEB_STATE_FILENAME = ".tetolate-web-state.json"
WEB_INSTANCE_LOCK_FILENAME = ".tetolate-web.lock"
WEB_CONFIG_FIELDS = {
    "listen",
    "jobs_dir",
    "max_upload_bytes",
}
UPLOAD_CHUNK_SIZE = 1024 * 1024
LOG_TAIL_BYTES = 256 * 1024
UPLOAD_PAGE_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".jxl",
    ".bmp",
    ".tif",
    ".tiff",
    ".avif",
    ".gif",
}
UPLOAD_PAGE_IMAGE_ACCEPT = (
    ".png,.jpg,.jpeg,.webp,.jxl,.bmp,.tif,.tiff,.avif,.gif,"
    "image/png,image/jpeg,image/webp,image/jxl,image/bmp,image/tiff,image/avif,image/gif"
)
CATEGORY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
JOB_ID_BYTES = 4
VLM_LIVE_PROGRESS_PATTERN = re.compile(r"^VLM .+: (?:waiting|reasoning|answering) \| ")
RESUME_PHASE_BY_PROGRESS = {
    "OCR": "ocr_raw",
    "Structure": "ocr_structured",
    "Alt placement": "alt_placement",
    "Translation": "translations",
    "Proofreading": "proofreading",
    "Translation notes": "translation_notes",
    "Open placement": "placements",
    "Placement expansion": "placements",
    "Placement style": "placements",
    "Render": "render",
    "Package": "package",
}
RESUME_PHASES = frozenset(RESUME_PHASE_BY_PROGRESS.values())
TERMINAL_JOB_STATUSES = frozenset(("complete", "failed", "cancelled"))
OCR_MERGE_EDITOR_STAGE = "ocr_merge"
EDITABLE_STAGES = frozenset(
    ("ocr_raw", "ocr_merged", "ocr_structured", "translations", "placements")
)
EDITOR_UI_STAGES = frozenset((OCR_MERGE_EDITOR_STAGE, "ocr_structured", "translations", "placements"))
EDITOR_DATA_STAGES = EDITABLE_STAGES | frozenset((OCR_MERGE_EDITOR_STAGE,))
EDITOR_RERUN_STAGES = frozenset(
    (OCR_MERGE_EDITOR_STAGE, "ocr_raw", "ocr_merged", "ocr_structured", "translations", "placements", "render")
)
RERUN_STAGE_MAP = {
    OCR_MERGE_EDITOR_STAGE: "ocr_structured",
    "ocr_raw": "ocr_raw",
    "ocr_merged": "ocr_structured",
    "ocr_structured": "ocr_structured",
    "alt_placement": "alt_placement",
    "translations": "translations",
    "placements": "placements",
    "render": "render",
}
RERUN_JOB_STAGE_ORDER = (
    OCR_MERGE_EDITOR_STAGE,
    "ocr_structured",
    "alt_placement",
    "translations",
    "placements",
    "render",
)
RERUN_JOB_STAGE_RESUME = {
    OCR_MERGE_EDITOR_STAGE: "ocr_structured",
    "ocr_structured": "ocr_structured",
    "alt_placement": "alt_placement",
    "translations": "translations",
    "placements": "placements",
    "render": "render",
}
RERUN_JOB_PACKAGE_STAGE = "package"
EDITOR_META_DIRNAME = "web_meta"
EDITOR_META_FILENAME = "editor.json"
TRANSLATION_NOTES_FILENAME = "translation_notes.json"
CATEGORY_ADVANCED_OPTIONS_FILENAME = "advanced_options.json"
EDITOR_STAGE_ORDER = (OCR_MERGE_EDITOR_STAGE, "ocr_structured", "translations", "placements")
EDITOR_STAGE_LABELS = {
    OCR_MERGE_EDITOR_STAGE: "OCR merge",
    "ocr_structured": "Structured",
    "translations": "Translations",
    "placements": "Placements",
}
EDITOR_STAGE_RERUN_FROM = {
    OCR_MERGE_EDITOR_STAGE: "ocr_structured",
    "ocr_structured": "translations",
    "translations": "placements",
    "placements": "render",
}
EDITOR_STAGE_UPSTREAM = {
    OCR_MERGE_EDITOR_STAGE: (),
    "ocr_structured": (OCR_MERGE_EDITOR_STAGE,),
    "translations": (OCR_MERGE_EDITOR_STAGE, "ocr_structured"),
    "placements": (OCR_MERGE_EDITOR_STAGE, "ocr_structured", "translations"),
}
PAGE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DELETE_JOB_CONFIRM = "Delete this job and its generated files?"
CATEGORY_DELETE_CONFIRM = (
    "Delete this category and every job, input, output, log, and download inside it?"
)
TERMINATE_JOB_CONFIRM = (
    "Terminate this running job? The local VLM request stream will be closed, "
    "but external providers may handle cancellation differently."
)


@dataclass(frozen=True)
class WebConfig:
    listen_host: str
    listen_port: int
    pipeline_config: Path
    translate_script: Path
    jobs_dir: Path
    max_upload_bytes: int
    default_webp_quality: int
    default_jxl_quality: int
    default_thinking_budget_tokens: int
    default_vlm_base_url: str
    default_alt_placement_enabled: bool
    default_source_language: str
    default_ocr_engine: str
    default_paddleocr_vl_server_url: str
    default_paddleocr_vl_model: str


@dataclass(frozen=True)
class JobTiming:
    age_seconds: float | None
    elapsed_seconds: float | None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"non-finite number {value}")


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def resolve_config_path(value: Any, default: Path, base_dir: Path) -> Path:
    if value is None or value == "":
        return default.expanduser().resolve()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def integer_value(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{label} must be an integer.")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
    raise ValueError(f"{label} must be an integer.")


def validate_output_quality(value: Any, label: str) -> int:
    try:
        quality = integer_value(value, label)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer from 1 to 100.") from exc
    if quality < 1 or quality > 100:
        raise ValueError(f"{label} must be an integer from 1 to 100.")
    return quality


def parse_quality_form(value: str, label: str, default: int) -> int:
    value = value.strip()
    if not value:
        return default
    try:
        return validate_output_quality(value, label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def validate_thinking_budget_tokens(value: Any, label: str = "thinking_budget_tokens") -> int:
    return integer_value(value, label)


def parse_thinking_budget_form(value: str, default: int) -> int:
    value = value.strip()
    if not value:
        return default
    try:
        return validate_thinking_budget_tokens(value, "Thinking tokens")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_ocr_engine_form(value: str, default: str) -> str:
    value = value.strip()
    try:
        return paddle_ocr_image.normalize_ocr_engine(value or default)
    except paddle_ocr_image.InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_source_language_form(value: str, default: str) -> str:
    try:
        return translate_cbz.normalize_source_language(value.strip() or default)
    except translate_cbz.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_optional_text_form(value: str, default: str) -> str:
    value = value.strip()
    return value or default


def validate_vlm_base_url(value: str, default: str) -> str:
    endpoint = value.strip() or default
    try:
        return translate_cbz.normalize_vlm_base_url(endpoint)
    except translate_cbz.PipelineError as exc:
        raise ValueError(str(exc)) from exc


def parse_vlm_base_url_form(value: str, config: WebConfig) -> str:
    try:
        return validate_vlm_base_url(value, config.default_vlm_base_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def http_origin(value: str, label: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid HTTP(S) URL.") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be a valid HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain URL credentials.")
    host = parsed.hostname
    host_text = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{host_text}" + (f":{port}" if port is not None else "")


def validate_paddleocr_vl_server_url(
    value: str,
    default: str,
) -> str:
    endpoint = value.strip() or default
    http_origin(endpoint, "PaddleOCR-VL server URL")
    return endpoint.rstrip("/")


def parse_paddleocr_vl_server_url_form(
    value: str,
    config: WebConfig,
) -> str:
    try:
        return validate_paddleocr_vl_server_url(
            value,
            config.default_paddleocr_vl_server_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_checkbox(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def thinking_budget_text(value: Any) -> str:
    try:
        budget = validate_thinking_budget_tokens(value)
    except ValueError:
        budget = DEFAULT_THINKING_BUDGET_TOKENS
    if budget < 0:
        return "unlimited"
    if budget == 0:
        return "off"
    return f"{budget} tokens"


def load_pipeline_defaults(path: Path) -> tuple[int, int, int, str, bool, str, str, str, str]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise RuntimeError(f"Invalid JSON in pipeline config {path}: {detail}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Pipeline config root must be a JSON object.")
    output = data.get("output", {})
    if output is None:
        output = {}
    if not isinstance(output, dict):
        raise RuntimeError("Pipeline config field output must be an object when provided.")
    ocr = data.get("ocr", {})
    if ocr is None:
        ocr = {}
    if not isinstance(ocr, dict):
        raise RuntimeError("Pipeline config field ocr must be an object when provided.")
    language = data.get("default_language", {})
    if language is None:
        language = {}
    if not isinstance(language, dict):
        raise RuntimeError(
            "Pipeline config field default_language must be an object when provided."
        )
    alt_placement = data.get("alt_placement", {})
    if alt_placement is None:
        alt_placement = {}
    if not isinstance(alt_placement, dict):
        raise RuntimeError("Pipeline config field alt_placement must be an object when provided.")
    default_alt_placement_enabled = alt_placement.get(
        "enabled",
        DEFAULT_ALT_PLACEMENT_ENABLED,
    )
    if not isinstance(default_alt_placement_enabled, bool):
        raise RuntimeError("Pipeline config field alt_placement.enabled must be true or false.")
    try:
        default_source_language = translate_cbz.normalize_source_language(
            language.get("source", DEFAULT_SOURCE_LANGUAGE)
        )
    except translate_cbz.PipelineError as exc:
        raise RuntimeError(str(exc)) from exc
    try:
        return (
            validate_output_quality(
                output.get("webp_quality", DEFAULT_WEBP_QUALITY),
                "output.webp_quality",
            ),
            validate_output_quality(
                output.get("jxl_quality", DEFAULT_JXL_QUALITY),
                "output.jxl_quality",
            ),
            validate_thinking_budget_tokens(
                data.get("thinking_budget_tokens", DEFAULT_THINKING_BUDGET_TOKENS),
                "thinking_budget_tokens",
            ),
            validate_vlm_base_url(str(data.get("base_url", "")), ""),
            default_alt_placement_enabled,
            default_source_language,
            DEFAULT_OCR_ENGINE,
            DEFAULT_PADDLEOCR_VL_SERVER_URL,
            DEFAULT_PADDLEOCR_VL_MODEL,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def parse_listen(value: Any) -> tuple[str, int]:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Web config listen must be a non-empty host:port string.")
    listen = value.strip()
    try:
        parsed = urlsplit(f"//{listen}")
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("Web config listen must be a valid host:port string.") from exc
    if (
        not parsed.hostname
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Web config listen must be a valid host:port string.")
    if port <= 0 or port > 65535:
        raise RuntimeError("Web config listen port must be between 1 and 65535.")
    return parsed.hostname, port


def load_web_config(path: Path | None = None) -> WebConfig:
    config_path = path or Path(
        os.environ.get("TETOLATE_WEB_CONFIG", DEFAULT_WEB_CONFIG)
    )
    if not config_path.exists():
        raise RuntimeError(
            (
                f"Web config not found: {config_path}. Copy "
                "data/config/web_config.example.json to "
                "data/config/web_config.json."
            )
        )

    try:
        data = json.loads(
            config_path.read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
        raise RuntimeError(f"Invalid JSON in {config_path}: {detail}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Web config root must be a JSON object.")
    unknown_fields = sorted(set(data) - WEB_CONFIG_FIELDS)
    if unknown_fields:
        raise RuntimeError(
            "Unknown web config field(s): " + ", ".join(unknown_fields)
        )

    config_path = config_path.resolve()
    base_dir = config_path.parent

    listen_host, listen_port = parse_listen(data.get("listen", DEFAULT_LISTEN))

    try:
        max_upload_bytes = integer_value(
            data.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES),
            "max_upload_bytes",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if max_upload_bytes <= 0:
        raise RuntimeError("Web config max_upload_bytes must be positive.")

    pipeline_config = base_dir / DEFAULT_PIPELINE_CONFIG.name
    translate_script = DEFAULT_TRANSLATE_SCRIPT
    if not pipeline_config.is_file():
        raise RuntimeError(f"Pipeline config not found: {pipeline_config}")
    if not translate_script.is_file():
        raise RuntimeError(f"Translation script not found: {translate_script}")
    (
        default_webp_quality,
        default_jxl_quality,
        default_thinking_budget_tokens,
        default_vlm_base_url,
        default_alt_placement_enabled,
        default_source_language,
        default_ocr_engine,
        default_paddleocr_vl_server_url,
        default_paddleocr_vl_model,
    ) = load_pipeline_defaults(pipeline_config)

    return WebConfig(
        listen_host=listen_host,
        listen_port=listen_port,
        pipeline_config=pipeline_config,
        translate_script=translate_script,
        jobs_dir=resolve_config_path(data.get("jobs_dir"), DEFAULT_JOBS_DIR, base_dir),
        max_upload_bytes=max_upload_bytes,
        default_webp_quality=default_webp_quality,
        default_jxl_quality=default_jxl_quality,
        default_thinking_budget_tokens=default_thinking_budget_tokens,
        default_vlm_base_url=default_vlm_base_url,
        default_alt_placement_enabled=default_alt_placement_enabled,
        default_source_language=default_source_language,
        default_ocr_engine=default_ocr_engine,
        default_paddleocr_vl_server_url=default_paddleocr_vl_server_url,
        default_paddleocr_vl_model=default_paddleocr_vl_model,
    )


def duration_text(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}m {remaining:.0f}s"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m"


def age_text(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return "less than 1m"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h {minutes % 60}m"
    days = int(hours // 24)
    remaining_hours = hours % 24
    return f"{days}d {remaining_hours}h"


def job_timing(status: dict[str, Any], now: datetime | None = None) -> JobTiming:
    now = now or datetime.now(timezone.utc)
    created_at = (
        parse_timestamp(status.get("createdAt"))
        or parse_timestamp(status.get("updatedAt"))
        or parse_timestamp(status.get("startedAt"))
        or parse_timestamp(status.get("finishedAt"))
    )
    started_at = parse_timestamp(status.get("startedAt"))
    finished_at = parse_timestamp(status.get("finishedAt"))
    if finished_at is not None and started_at is not None:
        elapsed_seconds = (finished_at - started_at).total_seconds()
    elif started_at is not None:
        elapsed_seconds = (now - started_at).total_seconds()
    else:
        elapsed_seconds = None
    age_seconds = (now - created_at).total_seconds() if created_at is not None else None
    return JobTiming(age_seconds=age_seconds, elapsed_seconds=elapsed_seconds)


def file_size_text(size_bytes: int | None) -> str:
    if size_bytes is None:
        return ""
    size = max(float(size_bytes), 0.0)
    units = ("B", "KB", "MB", "GB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def file_info(path: Path) -> dict[str, Any]:
    size_bytes: int | None = None
    mtime_ns: int | None = None
    available = path.is_file()
    if available:
        try:
            stat = path.stat()
            size_bytes = stat.st_size
            mtime_ns = stat.st_mtime_ns
        except OSError:
            available = False
            size_bytes = None
            mtime_ns = None
    return {
        "available": available,
        "sizeBytes": size_bytes,
        "mtimeNs": mtime_ns,
        "size": file_size_text(size_bytes),
        "downloadToken": f"{size_bytes}-{mtime_ns}" if available else "",
    }


def upload_filename(upload: UploadFile | None) -> str:
    if upload is None:
        return ""
    return (upload.filename or "").strip()


def non_empty_uploads(uploads: list[UploadFile] | None) -> list[UploadFile]:
    return [upload for upload in uploads or [] if upload_filename(upload)]


def upload_names_summary(names: list[str]) -> str:
    if not names:
        return "Image pages"
    if len(names) == 1:
        return f"Image page: {names[0]}"
    if len(names) == 2:
        return f"2 image pages: {names[0]}, {names[1]}"
    return f"{len(names)} image pages: {names[0]} ... {names[-1]}"


async def write_uploaded_file_to_path(
    upload: UploadFile,
    output_path: Path,
    max_bytes: int,
    size_label: str,
) -> int:
    written = 0
    with output_path.open("wb") as output:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(status_code=413, detail=f"{size_label} is too large.")
            output.write(chunk)
    if written == 0:
        raise HTTPException(status_code=400, detail=f"{size_label} was empty.")
    return written


def verify_output_png(path: Path, filename: str) -> None:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise HTTPException(
                    status_code=500,
                    detail=f"Converted page was not written as PNG: {filename}",
                )
            image.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Converted page was not a readable PNG image: {filename}",
        ) from exc


def convert_uploaded_image_to_png(source_path: Path, output_path: Path, filename: str) -> None:
    def convert_with_pillow() -> None:
        with Image.open(source_path) as image:
            image.convert("RGB").save(output_path, format="PNG")
        verify_output_png(output_path, filename)

    magick = shutil.which("magick")
    magick_details = ""
    if magick is not None:
        command = [
            magick,
            f"{source_path}[0]",
            "-auto-orient",
            "-background",
            "white",
            "-alpha",
            "remove",
            "-alpha",
            "off",
            str(output_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode == 0 and output_path.is_file():
            verify_output_png(output_path, filename)
            return
        magick_details = (completed.stderr or completed.stdout or "").strip()

    try:
        convert_with_pillow()
        return
    except Exception as exc:
        detail = (
            f"Page upload could not be converted to PNG: {filename}. "
            "Install ImageMagick with support for this format."
        )
        if magick_details:
            detail += f" ImageMagick said: {magick_details}"
        raise HTTPException(
            status_code=400,
            detail=detail,
        ) from exc


async def write_uploaded_images_as_cbz(
    uploads: list[UploadFile],
    output_path: Path,
    max_bytes: int,
) -> tuple[int, str]:
    # Preserve multipart order. Mobile pickers may expose a deliberate selection
    # order that differs from filename order.
    ordered_uploads = list(uploads)
    names = [upload_filename(upload) for upload in ordered_uploads]
    with tempfile.TemporaryDirectory(prefix="uploaded_pages_", dir=output_path.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        page_paths: list[Path] = []
        total_written = 0
        for index, upload in enumerate(ordered_uploads):
            filename = upload_filename(upload)
            suffix = Path(filename).suffix.lower()
            if suffix not in UPLOAD_PAGE_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported page image file type: {filename}",
                )
            source_path = temp_dir / f"upload-{index:04d}{suffix}"
            page_path = temp_dir / f"{index:04d}.png"
            written = 0
            with source_path.open("wb") as output:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    written += len(chunk)
                    total_written += len(chunk)
                    if total_written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="Uploaded page images are too large.",
                        )
                    output.write(chunk)
            if written == 0:
                raise HTTPException(status_code=400, detail=f"Page image was empty: {filename}")
            await run_in_threadpool(
                convert_uploaded_image_to_png,
                source_path,
                page_path,
                filename,
            )
            page_paths.append(page_path)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for page_path in page_paths:
                archive.write(page_path, arcname=page_path.name)
    return total_written, upload_names_summary(names)


def quality_for_display(value: Any, default: int) -> int:
    try:
        return validate_output_quality(value, "quality")
    except ValueError:
        return default


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def write_json_atomic(path: Path, data: Any) -> None:
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
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def safe_log_lines(path: Path, limit: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            start = max(0, size - LOG_TAIL_BYTES)
            file.seek(start)
            data = file.read()
    except OSError:
        return []
    if start > 0:
        first_break = min(
            (index for index in (data.find(b"\n"), data.find(b"\r")) if index >= 0),
            default=-1,
        )
        if first_break >= 0:
            data = data[first_break + 1 :]
    text = data.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.replace("\r", "\n").splitlines()]
    lines = [line for line in lines if line and not VLM_LIVE_PROGRESS_PATTERN.match(line)]
    return lines[-limit:]


def parse_progress_line(line: str) -> tuple[str, int | None] | None:
    patterns = (
        (re.compile(r"^OCR page (\d+)"), "OCR"),
        (re.compile(r"^VLM structure page (\d+)"), "Structure"),
        (re.compile(r"^(?:VLM|Skip VLM) alt-placement page (\d+)"), "Alt placement"),
        (re.compile(r"^VLM translate page (\d+)"), "Translation"),
        (re.compile(r"^VLM proofread translations"), "Proofreading"),
        (re.compile(r"^VLM write translation notes"), "Translation notes"),
        (re.compile(r"^VLM open placement page (\d+)"), "Open placement"),
        (re.compile(r"^Detect placement containers page (\d+)"), "Placement expansion"),
        (re.compile(r"^VLM style placement page (\d+)"), "Placement style"),
        (re.compile(r"^Render page (\d+)"), "Render"),
        (re.compile(r"^Wrote .+translated\.cbz"), "Package"),
    )
    for pattern, phase in patterns:
        match = pattern.search(line)
        if match is None:
            continue
        page = int(match.group(1)) if match.groups() else None
        return phase, page
    return None


def progress_resume_target(phase: Any, page: Any) -> tuple[str, int] | None:
    resume_from = RESUME_PHASE_BY_PROGRESS.get(str(phase))
    if resume_from in {"proofreading", "translation_notes", "package"}:
        return stored_resume_target(resume_from, 0)
    return stored_resume_target(resume_from, page)


def stored_resume_target(resume_from: Any, page: Any) -> tuple[str, int] | None:
    if resume_from not in RESUME_PHASES:
        return None
    if resume_from in {"proofreading", "translation_notes", "package"}:
        return resume_from, 0
    if page is None:
        return None
    try:
        resume_page = int(page)
    except (TypeError, ValueError):
        return None
    if resume_page < 0:
        return None
    return resume_from, resume_page


def parse_page_selection(value: str, allow_empty: bool = False) -> list[int]:
    pages: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = [item.strip() for item in part.split("-", 1)]
            if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
                raise HTTPException(status_code=400, detail="Page ranges must look like 0-3.")
            start = int(bounds[0])
            end = int(bounds[1])
            if end < start:
                raise HTTPException(status_code=400, detail="Page ranges must be ascending.")
            if end - start + 1 > MAX_PAGE_SELECTION_ITEMS:
                raise HTTPException(
                    status_code=400,
                    detail=f"A page selection cannot contain more than {MAX_PAGE_SELECTION_ITEMS} pages.",
                )
            pages.update(range(start, end + 1))
        else:
            if not part.isdigit():
                raise HTTPException(status_code=400, detail="Pages must be numbers or ranges.")
            pages.add(int(part))
        if len(pages) > MAX_PAGE_SELECTION_ITEMS:
            raise HTTPException(
                status_code=400,
                detail=f"A page selection cannot contain more than {MAX_PAGE_SELECTION_ITEMS} pages.",
            )
    if not pages and allow_empty:
        return []
    if not pages:
        raise HTTPException(status_code=400, detail="Enter at least one page.")
    return sorted(pages)


def parse_rerun_job_stages(values: list[str]) -> tuple[str | None, bool]:
    selected = {str(value).strip() for value in values if str(value).strip()}
    allowed = set(RERUN_JOB_STAGE_ORDER) | {RERUN_JOB_PACKAGE_STAGE}
    unsupported = sorted(selected - allowed)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="Unsupported rerun pass: " + ", ".join(unsupported),
        )
    if not selected:
        raise HTTPException(status_code=400, detail="Select at least one pass to rerun.")

    resume_from = None
    for stage in RERUN_JOB_STAGE_ORDER:
        if stage in selected:
            resume_from = RERUN_JOB_STAGE_RESUME[stage]
            break
    return resume_from, RERUN_JOB_PACKAGE_STAGE in selected


def validate_stage(stage: str, allowed: frozenset[str] = EDITABLE_STAGES) -> str:
    stage = stage.strip()
    if stage not in allowed:
        raise HTTPException(status_code=404, detail="Unknown editor stage.")
    return stage


def lock_stages_for_editor_stage(stage: str) -> tuple[str, ...]:
    stage = validate_stage(stage, EDITOR_DATA_STAGES)
    if stage == OCR_MERGE_EDITOR_STAGE:
        return ("ocr_raw", "ocr_merged")
    return (validate_stage(stage),)


def page_key(page: int) -> str:
    return str(page)


def default_editor_meta() -> dict[str, Any]:
    return {
        "locks": {},
        "changedStages": {},
        "translationNotes": {
            "job": "",
            "pages": {},
        },
    }


def normalize_editor_meta(value: Any) -> dict[str, Any]:
    meta = default_editor_meta()
    if not isinstance(value, dict):
        return meta

    locks = value.get("locks")
    if isinstance(locks, dict):
        normalized_locks: dict[str, dict[str, bool]] = {}
        for raw_page, raw_stages in locks.items():
            if not isinstance(raw_stages, dict):
                continue
            page_locks = {
                stage: bool(raw_stages.get(stage))
                for stage in EDITABLE_STAGES
                if raw_stages.get(stage) is not None
            }
            if page_locks:
                normalized_locks[str(raw_page)] = page_locks
        meta["locks"] = normalized_locks

    changed_stages = value.get("changedStages")
    if isinstance(changed_stages, dict):
        normalized_changed: dict[str, dict[str, str]] = {}
        for raw_page, raw_stages in changed_stages.items():
            if not isinstance(raw_stages, dict):
                continue
            page_changes: dict[str, str] = {}
            for stage in EDITOR_STAGE_ORDER:
                changed_value = raw_stages.get(stage)
                if changed_value is None or changed_value is False:
                    continue
                page_changes[stage] = (
                    changed_value if isinstance(changed_value, str) else now_utc()
                )
            if page_changes:
                normalized_changed[str(raw_page)] = page_changes
        meta["changedStages"] = normalized_changed

    notes = value.get("translationNotes")
    if isinstance(notes, dict):
        job_note = notes.get("job", "")
        pages = notes.get("pages", {})
        meta["translationNotes"] = {
            "job": job_note if isinstance(job_note, str) else "",
            "pages": {
                str(raw_page): note
                for raw_page, note in pages.items()
                if isinstance(note, str)
            }
            if isinstance(pages, dict)
            else {},
        }
    return meta


def parse_region(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise HTTPException(status_code=400, detail=f"{label} must be a four-number array.")
    try:
        left, top, right, bottom = [round(float(item)) for item in value]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} values must be numbers.") from exc
    if right <= left or bottom <= top:
        raise HTTPException(status_code=400, detail=f"{label} must have positive width and height.")
    return [left, top, right, bottom]


def optional_non_negative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{label} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be a non-negative integer.") from exc
    if parsed < 0:
        raise HTTPException(status_code=400, detail=f"{label} must be a non-negative integer.")
    return parsed


def offset_record_region(record: dict[str, Any], left: int, top: int, width: int, height: int) -> dict[str, Any] | None:
    shifted = dict(record)
    region = shifted.get("region")
    if not isinstance(region, list) or len(region) != 4:
        return None
    try:
        x0, y0, x1, y1 = [round(float(value)) for value in region]
    except (TypeError, ValueError):
        return None
    x0 += left
    x1 += left
    y0 += top
    y1 += top
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    shifted["region"] = [x0, y0, x1, y1]

    polygon = shifted.get("polygon")
    if isinstance(polygon, list):
        shifted_polygon: list[list[int]] = []
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                shifted_polygon = []
                break
            try:
                px = max(0, min(width, round(float(point[0])) + left))
                py = max(0, min(height, round(float(point[1])) + top))
            except (TypeError, ValueError):
                shifted_polygon = []
                break
            shifted_polygon.append([px, py])
        if shifted_polygon:
            shifted["polygon"] = shifted_polygon
    shifted["ocrSource"] = "manual_crop"
    return shifted


class JobManager:
    def __init__(self, config: WebConfig) -> None:
        self.config = config
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._queued_jobs: set[tuple[str, str]] = set()
        self._lock = threading.RLock()
        self._pause_condition = threading.Condition(self._lock)
        self._paused = False
        self._stopping = False
        self._active_processes: dict[tuple[str, str], subprocess.Popen[str]] = {}
        self._worker: threading.Thread | None = None
        self._password_hash = ""
        self._categories: set[str] = set()
        self._admin_sessions: dict[str, float] = {}
        self._admin_login_failures: dict[str, list[float]] = {}
        self._password_workers = threading.BoundedSemaphore(2)
        self._instance_lock_file: Any | None = None
        self._instance_lock_kind = ""

    def start(self) -> None:
        self.config.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.acquire_instance_lock()
        try:
            self.initialize_web_state()
            self.requeue_interrupted_jobs()
            self._worker = threading.Thread(
                target=self.worker_loop,
                name="tetolate-web-worker",
                daemon=True,
            )
            self._worker.start()
        except Exception:
            self.release_instance_lock()
            raise

    def stop(self) -> None:
        with self._pause_condition:
            self._stopping = True
            self._paused = False
            active_processes = list(self._active_processes.values())
            self._pause_condition.notify_all()
        continue_signal = getattr(signal, "SIGCONT", None)
        for process in active_processes:
            if continue_signal is not None:
                self.signal_process_group_or_process(process, continue_signal)
            self.signal_process_group_or_process(process, signal.SIGTERM)
        for process in active_processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    self.signal_process_group_or_process(process, signal.SIGKILL)
                else:
                    process.kill()
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
        self.release_instance_lock()

    def instance_lock_path(self) -> Path:
        return self.config.jobs_dir / WEB_INSTANCE_LOCK_FILENAME

    def acquire_instance_lock(self) -> None:
        if self._instance_lock_file is not None:
            return
        path = self.instance_lock_path()
        lock_file = path.open("a+", encoding="utf-8")
        os.chmod(path, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                if not lock_file.read(1):
                    lock_file.write("\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                lock_kind = "msvcrt"
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_kind = "fcntl"
        except (ImportError, OSError) as exc:
            lock_file.close()
            raise RuntimeError(
                f"Another tetolate web server is using {self.config.jobs_dir}, "
                "or locking is unavailable."
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        self._instance_lock_file = lock_file
        self._instance_lock_kind = lock_kind

    def release_instance_lock(self) -> None:
        lock_file = self._instance_lock_file
        if lock_file is None:
            return
        try:
            if self._instance_lock_kind == "msvcrt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            elif self._instance_lock_kind == "fcntl":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
            self._instance_lock_file = None
            self._instance_lock_kind = ""

    def web_state_path(self) -> Path:
        return self.config.jobs_dir / WEB_STATE_FILENAME

    def web_state_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "adminPasswordHash": self._password_hash,
            "categories": sorted(self._categories, key=str.casefold),
            "updatedAt": now_utc(),
        }

    def save_web_state(self) -> None:
        web_security.write_private_json_atomic(self.web_state_path(), self.web_state_payload())

    def initialize_web_state(self) -> None:
        path = self.web_state_path()
        state: dict[str, Any] = {}
        if path.exists():
            try:
                state = web_security.read_json_object(path)
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot load web state {path}: {exc}") from exc
        password_hash = state.get("adminPasswordHash", "")
        if password_hash is None:
            password_hash = ""
        if not isinstance(password_hash, str):
            raise RuntimeError("Web state adminPasswordHash must be a string.")
        if password_hash:
            try:
                web_security.parse_password_hash(password_hash)
            except ValueError as exc:
                raise RuntimeError(f"Web state has an invalid admin password hash: {exc}") from exc

        configured_categories = state.get("categories", [])
        if not isinstance(configured_categories, list):
            raise RuntimeError("Web state categories must be an array.")
        categories: set[str] = set()
        category_names: dict[str, str] = {}
        for category in configured_categories:
            if not isinstance(category, str) or not CATEGORY_PATTERN.fullmatch(category):
                raise RuntimeError(f"Web state contains an invalid category: {category!r}")
            folded = category.casefold()
            previous = category_names.get(folded)
            if previous is not None and previous != category:
                raise RuntimeError(
                    f"Job categories differ only by letter case: {previous!r} and {category!r}."
                )
            category_names[folded] = category
            categories.add(category)

        generated_password: str | None = None
        if not password_hash:
            generated_password = web_security.generate_password()
            password_hash = web_security.hash_password(generated_password)

        if generated_password is not None:
            print(
                "tetolate generated admin password (shown once): "
                + generated_password,
                file=sys.stderr,
                flush=True,
            )

        with self._lock:
            self._password_hash = password_hash
            self._categories = categories
            self.save_web_state()

    def password_matches_hash(self, password: str, password_hash: str) -> bool:
        with self._password_workers:
            try:
                return web_security.verify_password(password, password_hash)
            except ValueError:
                return False

    def admin_password_matches(self, password: str) -> bool:
        with self._lock:
            password_hash = self._password_hash
        return self.password_matches_hash(password, password_hash)

    def authenticate_admin(self, password: str) -> str | None:
        with self._lock:
            password_hash = self._password_hash
        if not self.password_matches_hash(password, password_hash):
            return None
        with self._lock:
            if self._password_hash != password_hash:
                return None
            return self.create_admin_session()

    def session_digest(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def purge_expired_admin_sessions(self) -> None:
        now = time.monotonic()
        expired = [digest for digest, expiry in self._admin_sessions.items() if expiry <= now]
        for digest in expired:
            self._admin_sessions.pop(digest, None)

    def create_admin_session(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self.purge_expired_admin_sessions()
            self._admin_sessions[self.session_digest(token)] = (
                time.monotonic() + ADMIN_SESSION_SECONDS
            )
        return token

    def admin_session_is_valid(self, token: str) -> bool:
        if not token:
            return False
        with self._lock:
            self.purge_expired_admin_sessions()
            return self.session_digest(token) in self._admin_sessions

    def revoke_admin_session(self, token: str) -> None:
        if not token:
            return
        with self._lock:
            self._admin_sessions.pop(self.session_digest(token), None)

    def admin_login_retry_after(self, client_key: str) -> int:
        now = time.monotonic()
        cutoff = now - ADMIN_LOGIN_FAILURE_WINDOW_SECONDS
        with self._lock:
            attempts = [
                timestamp
                for timestamp in self._admin_login_failures.get(client_key, [])
                if timestamp > cutoff
            ]
            if attempts:
                self._admin_login_failures[client_key] = attempts
            else:
                self._admin_login_failures.pop(client_key, None)
            if len(attempts) < ADMIN_LOGIN_FAILURE_LIMIT:
                return 0
            return max(
                1,
                round(ADMIN_LOGIN_FAILURE_WINDOW_SECONDS - (now - attempts[0])),
            )

    def record_admin_login_failure(self, client_key: str) -> None:
        with self._lock:
            self._admin_login_failures.setdefault(client_key, []).append(time.monotonic())

    def clear_admin_login_failures(self, client_key: str) -> None:
        with self._lock:
            self._admin_login_failures.pop(client_key, None)

    def change_admin_password(
        self,
        current_password: str,
        new_password: str,
        confirm_password: str,
    ) -> str:
        if new_password != confirm_password:
            raise ValueError("New password and confirmation do not match.")
        with self._lock:
            previous_hash = self._password_hash
        if not self.password_matches_hash(current_password, previous_hash):
            raise ValueError("Current password is incorrect.")
        with self._password_workers:
            new_hash = web_security.hash_password(new_password)
        with self._lock:
            if self._password_hash != previous_hash:
                raise ValueError("Admin password changed during this request; try again.")
            self._password_hash = new_hash
            try:
                self.save_web_state()
            except Exception:
                self._password_hash = previous_hash
                raise
            self._admin_sessions.clear()
        return self.create_admin_session()

    def categories(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._categories, key=str.casefold))

    def create_category(self, category: str) -> str:
        category = category.strip()
        if not CATEGORY_PATTERN.fullmatch(category):
            raise HTTPException(
                status_code=400,
                detail="Category must be 1-64 characters using letters, numbers, '_' or '-'.",
            )
        with self._lock:
            if any(existing.casefold() == category.casefold() for existing in self._categories):
                raise HTTPException(status_code=409, detail="That category already exists.")
            path = self.category_dir(category)
            if path.exists():
                raise HTTPException(status_code=409, detail="That category directory already exists.")
            path.mkdir(parents=True)
            self._categories.add(category)
            try:
                self.save_web_state()
            except Exception:
                self._categories.remove(category)
                path.rmdir()
                raise
        return category

    def category_job_counts(self, category: str) -> dict[str, int]:
        category = self.validate_category(category)
        counts: dict[str, int] = {}
        for job_id in self.iter_job_ids(category):
            status = self.load_status(category, job_id) or {}
            value = str(status.get("status") or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    def delete_category(self, category: str, confirmation: str) -> None:
        category = self.validate_category(category)
        if confirmation != category:
            raise HTTPException(
                status_code=400,
                detail="Type the exact category name to confirm deletion.",
            )
        trash_path: Path | None = None
        with self._lock:
            if any(code == category for code, _job_id in self._active_processes):
                raise HTTPException(status_code=409, detail="Stop active jobs before deleting this category.")
            if any(code == category for code, _job_id in self._queued_jobs):
                raise HTTPException(status_code=409, detail="Stop queued jobs before deleting this category.")
            for job_id in self.iter_job_ids(category):
                status = self.load_status(category, job_id) or {}
                if status.get("status") not in TERMINAL_JOB_STATUSES:
                    raise HTTPException(
                        status_code=409,
                        detail="Only categories whose jobs are complete, failed, or cancelled can be deleted.",
                    )
            source_path = self.category_dir(category)
            if source_path.exists():
                trash_path = self.config.jobs_dir / (
                    f".deleted-{category}-{secrets.token_hex(4)}"
                )
                os.replace(source_path, trash_path)
            self._categories.remove(category)
            try:
                self.save_web_state()
            except Exception:
                self._categories.add(category)
                if trash_path is not None:
                    os.replace(trash_path, source_path)
                raise
        if trash_path is not None:
            try:
                shutil.rmtree(trash_path)
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Category was removed but its files could not be fully deleted: {exc}",
                ) from exc

    def wait_until_unpaused(self) -> bool:
        with self._pause_condition:
            while self._paused and not self._stopping:
                self._pause_condition.wait()
            return not self._stopping

    def signal_process_group(self, process: subprocess.Popen[str], sig: signal.Signals) -> bool:
        if process.poll() is not None or os.name != "posix":
            return False
        try:
            os.killpg(process.pid, sig)
        except OSError:
            return False
        return True

    def signal_process(self, process: subprocess.Popen[str], sig: signal.Signals) -> bool:
        if process.poll() is not None:
            return False
        try:
            process.send_signal(sig)
        except OSError:
            return False
        return True

    def signal_process_group_or_process(
        self,
        process: subprocess.Popen[str],
        sig: signal.Signals,
    ) -> bool:
        return self.signal_process_group(process, sig) or self.signal_process(process, sig)

    def recorded_pid(self, status: dict[str, Any]) -> int | None:
        try:
            pid = int(status.get("pid"))
        except (TypeError, ValueError):
            return None
        return pid if pid > 0 else None

    def recorded_process_matches_pipeline(self, pid: int) -> bool:
        if os.name != "posix":
            return False
        cmdline_path = Path("/proc") / str(pid) / "cmdline"
        try:
            raw_cmdline = cmdline_path.read_bytes()
        except OSError:
            return False
        parts = [part.decode("utf-8", errors="replace") for part in raw_cmdline.split(b"\0") if part]
        translate_script = self.config.translate_script.resolve()
        for part in parts:
            try:
                if Path(part).resolve() == translate_script:
                    return True
            except OSError:
                pass
            if part == self.config.translate_script.name:
                return True
        return False

    def terminate_recorded_pipeline_process(self, status: dict[str, Any]) -> bool:
        pid = self.recorded_pid(status)
        if pid is None or not self.recorded_process_matches_pipeline(pid):
            return False
        sent = False
        try:
            os.killpg(pid, signal.SIGTERM)
            sent = True
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
                sent = True
            except OSError:
                return False
        continue_signal = getattr(signal, "SIGCONT", None)
        if continue_signal is not None:
            try:
                os.killpg(pid, continue_signal)
            except OSError:
                try:
                    os.kill(pid, continue_signal)
                except OSError:
                    pass
        return sent

    def stop_recorded_pipeline_process(
        self,
        status: dict[str, Any],
        timeout_seconds: float = 5.0,
    ) -> None:
        pid = self.recorded_pid(status)
        if pid is None or not self.recorded_process_matches_pipeline(pid):
            return
        self.terminate_recorded_pipeline_process(status)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not self.recorded_process_matches_pipeline(pid):
                return
            time.sleep(0.05)
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def mark_active_job_paused(self, code: str, job_id: str, signal_sent: bool) -> None:
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        if status.get("pendingTermination"):
            return
        if signal_sent:
            status["status"] = "paused"
            status["isPaused"] = True
            status["pausedAt"] = now_utc()
            status["message"] = "Paused by admin."
        else:
            status["message"] = "Pause requested, but the active process could not be suspended."
        self.save_status(code, job_id, status)

    def mark_active_job_resumed(self, code: str, job_id: str, signal_sent: bool) -> None:
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        if status.get("pendingTermination"):
            return
        if signal_sent or status.get("status") == "paused":
            status["status"] = "running"
            status["isPaused"] = False
            status.pop("pausedAt", None)
            phase = status.get("phase") or "Running"
            page = status.get("page")
            status["message"] = f"Resumed {phase}" + (f" page {page}" if page is not None else "")
        else:
            status["message"] = "Resume requested, but the active process could not be continued."
        self.save_status(code, job_id, status)

    def force_kill_process_later(
        self,
        code: str,
        job_id: str,
        process: subprocess.Popen[str],
        timeout_seconds: float = 10.0,
    ) -> None:
        time.sleep(timeout_seconds)
        if process.poll() is not None:
            return
        if os.name == "posix":
            killed = self.signal_process_group_or_process(process, signal.SIGKILL)
        else:
            try:
                process.kill()
                killed = True
            except OSError:
                killed = False
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        if status.get("pendingTermination"):
            status["message"] = (
                "Termination did not finish after SIGTERM; sent SIGKILL."
                if killed
                else "Termination did not finish and SIGKILL failed."
            )
            self.save_status(code, job_id, status)

    def pause_jobs(self) -> None:
        with self._pause_condition:
            self._paused = True
            active_processes = list(self._active_processes.items())
        stop_signal = getattr(signal, "SIGSTOP", None)
        for (code, job_id), process in active_processes:
            signal_sent = (
                self.signal_process_group_or_process(process, stop_signal)
                if stop_signal is not None
                else False
            )
            self.mark_active_job_paused(code, job_id, signal_sent)

    def resume_jobs(self) -> None:
        with self._pause_condition:
            self._paused = False
            active_processes = list(self._active_processes.items())
            self._pause_condition.notify_all()
        continue_signal = getattr(signal, "SIGCONT", None)
        for (code, job_id), process in active_processes:
            signal_sent = (
                self.signal_process_group_or_process(process, continue_signal)
                if continue_signal is not None
                else False
            )
            self.mark_active_job_resumed(code, job_id, signal_sent)

    def admin_status(self) -> dict[str, Any]:
        with self._lock:
            active: list[dict[str, Any]] = []
            for code, job_id in self._active_processes:
                status = self.load_status(code, job_id) or {}
                active.append(
                    {
                        "category": code,
                        "jobId": job_id,
                        "status": status.get("status", "running"),
                        "phase": status.get("phase"),
                        "page": status.get("page"),
                        "inputFilename": status.get("inputFilename", ""),
                        "url": f"/job/{code}/{job_id}",
                    }
                )
            categories = [
                {
                    "category": category,
                    "url": f"/category/{category}",
                    "jobCount": len(self.iter_job_ids(category)),
                }
                for category in self.categories()
            ]
            return {
                "paused": self._paused,
                "workerRunning": self._worker is not None and self._worker.is_alive(),
                "queuedCount": len(self._queued_jobs),
                "active": active,
                "categories": categories,
                "posixSignals": os.name == "posix",
            }

    def validate_category(self, code: str) -> str:
        code = code.strip()
        with self._lock:
            exists = code in self._categories
        if not exists:
            raise HTTPException(status_code=404, detail="Unknown job category.")
        return code

    def validate_job_id(self, job_id: str) -> str:
        job_id = job_id.strip()
        if not JOB_ID_PATTERN.match(job_id):
            raise HTTPException(status_code=400, detail="Invalid job id.")
        return job_id

    def category_dir(self, code: str) -> Path:
        return self.config.jobs_dir / code

    def category_advanced_options_path(self, code: str) -> Path:
        return self.category_dir(code) / CATEGORY_ADVANCED_OPTIONS_FILENAME

    def default_category_advanced_options(self) -> dict[str, Any]:
        return {
            "translationNotes": "",
            "thinkingBudgetTokens": self.config.default_thinking_budget_tokens,
            "vlmBaseUrl": self.config.default_vlm_base_url,
            "pauseAfterOcr": False,
            "proofreadTranslations": DEFAULT_PROOFREAD_TRANSLATIONS,
            "writeTranslationNotes": DEFAULT_WRITE_TRANSLATION_NOTES,
            "altPlacementEnabled": self.config.default_alt_placement_enabled,
            "sourceLanguage": self.config.default_source_language,
            "ocrEngine": self.config.default_ocr_engine,
            "paddleocrVlServerUrl": self.config.default_paddleocr_vl_server_url,
            "paddleocrVlModel": self.config.default_paddleocr_vl_model,
        }

    def normalize_category_advanced_options(self, data: Any) -> dict[str, Any]:
        defaults = self.default_category_advanced_options()
        if not isinstance(data, dict):
            return defaults

        normalized = dict(defaults)
        notes = data.get("translationNotes")
        if isinstance(notes, str):
            normalized["translationNotes"] = notes
        try:
            normalized["thinkingBudgetTokens"] = validate_thinking_budget_tokens(
                data.get("thinkingBudgetTokens", defaults["thinkingBudgetTokens"])
            )
        except ValueError:
            pass
        try:
            normalized["vlmBaseUrl"] = validate_vlm_base_url(
                str(data.get("vlmBaseUrl", "")),
                defaults["vlmBaseUrl"],
            )
        except ValueError:
            pass
        for key in (
            "pauseAfterOcr",
            "proofreadTranslations",
            "writeTranslationNotes",
            "altPlacementEnabled",
        ):
            if isinstance(data.get(key), bool):
                normalized[key] = data[key]
        try:
            normalized["sourceLanguage"] = translate_cbz.normalize_source_language(
                data.get("sourceLanguage", defaults["sourceLanguage"])
            )
        except translate_cbz.PipelineError:
            pass
        try:
            normalized["ocrEngine"] = paddle_ocr_image.normalize_ocr_engine(
                data.get("ocrEngine", defaults["ocrEngine"])
            )
        except paddle_ocr_image.InputError:
            pass
        for key in ("paddleocrVlServerUrl", "paddleocrVlModel"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                normalized[key] = value.strip()
        return normalized

    def load_category_advanced_options(self, code: str) -> dict[str, Any]:
        path = self.category_advanced_options_path(code)
        if not path.is_file():
            return self.default_category_advanced_options()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.default_category_advanced_options()
        return self.normalize_category_advanced_options(data)

    def remember_category_advanced_options(
        self,
        code: str,
        *,
        thinking_budget_tokens: int,
        vlm_base_url: str,
        pause_after_ocr: bool,
        proofread_translations: bool,
        write_translation_notes: bool,
        alt_placement_enabled: bool,
        source_language: str,
        ocr_engine: str,
        paddleocr_vl_server_url: str,
        paddleocr_vl_model: str,
        translation_notes: str | None = None,
    ) -> None:
        with self._lock:
            options = self.load_category_advanced_options(code)
            options.update(
                {
                    "thinkingBudgetTokens": thinking_budget_tokens,
                    "vlmBaseUrl": vlm_base_url,
                    "pauseAfterOcr": bool(pause_after_ocr),
                    "proofreadTranslations": bool(proofread_translations),
                    "writeTranslationNotes": bool(write_translation_notes),
                    "altPlacementEnabled": bool(alt_placement_enabled),
                    "sourceLanguage": source_language,
                    "ocrEngine": ocr_engine,
                    "paddleocrVlServerUrl": paddleocr_vl_server_url,
                    "paddleocrVlModel": paddleocr_vl_model,
                }
            )
            if translation_notes is not None:
                options["translationNotes"] = translation_notes.strip()
            write_json_atomic(
                self.category_advanced_options_path(code),
                self.normalize_category_advanced_options(options),
            )

    def jobs_dir(self, code: str) -> Path:
        return self.category_dir(code) / "jobs"

    def job_dir(self, code: str, job_id: str) -> Path:
        return self.jobs_dir(code) / job_id

    def status_path(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / "status.json"

    def log_path(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / "run.log"

    def input_path(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / "input.cbz"

    def output_dir(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / "output"

    def translated_cbz_path(self, code: str, job_id: str) -> Path:
        return self.translated_cbz_variant_path(code, job_id, "png")

    def translated_cbz_variant_path(self, code: str, job_id: str, variant: str) -> Path:
        filename = TRANSLATED_CBZ_FILENAMES.get(variant)
        if filename is None:
            raise HTTPException(status_code=404, detail="Unknown translated CBZ variant.")
        return self.output_dir(code, job_id) / filename

    def editor_meta_dir(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / EDITOR_META_DIRNAME

    def editor_meta_path(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / EDITOR_META_FILENAME

    def translation_notes_path(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / TRANSLATION_NOTES_FILENAME

    def generated_translation_notes_path(self, code: str, job_id: str) -> Path:
        return self.output_dir(code, job_id) / GENERATED_TRANSLATION_NOTES_NAME

    def read_generated_translation_notes(self, code: str, job_id: str) -> str:
        path = self.generated_translation_notes_path(code, job_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace").strip()

    def data_page_path(self, code: str, job_id: str, stage: str, page: int) -> Path:
        stage = validate_stage(stage)
        return self.output_dir(code, job_id) / "data" / stage / f"page_{page:04d}.json"

    def original_pages_dir(self, code: str, job_id: str) -> Path:
        return self.output_dir(code, job_id) / "pages" / "original"

    def final_pages_dir(self, code: str, job_id: str) -> Path:
        return self.output_dir(code, job_id) / "pages" / "final"

    def original_page_files(self, code: str, job_id: str) -> list[Path]:
        path = self.original_pages_dir(code, job_id)
        if not path.is_dir():
            return []
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in PAGE_IMAGE_EXTENSIONS
        )

    def final_page_files(self, code: str, job_id: str) -> list[Path]:
        path = self.final_pages_dir(code, job_id)
        if not path.is_dir():
            return []
        return sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in PAGE_IMAGE_EXTENSIONS
        )

    def original_page_path(self, code: str, job_id: str, page: int) -> Path:
        pages = self.original_page_files(code, job_id)
        if page < 0 or page >= len(pages):
            raise HTTPException(status_code=404, detail="Unknown page.")
        return pages[page]

    def final_page_path(self, code: str, job_id: str, page: int) -> Path:
        pages = self.final_page_files(code, job_id)
        if page < 0 or page >= len(pages):
            raise HTTPException(status_code=404, detail="Unknown translated page.")
        return pages[page]

    def require_viewable_job(self, code: str, job_id: str) -> dict[str, Any]:
        self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        status = self.load_status(code, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        if status.get("status") != "complete":
            raise HTTPException(status_code=400, detail="Only complete jobs can be viewed.")
        if not self.final_page_files(code, job_id):
            raise HTTPException(status_code=404, detail="No translated page images are available.")
        return status

    def require_original_viewable_job(self, code: str, job_id: str) -> dict[str, Any]:
        self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        status = self.load_status(code, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        if not self.original_page_files(code, job_id):
            raise HTTPException(status_code=404, detail="No extracted original page images are available.")
        return status

    def page_infos(self, paths: list[Path]) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            info = file_info(path)
            infos.append(
                {
                    "index": index,
                    "filename": path.name,
                    "token": info["downloadToken"],
                    "size": info["size"],
                }
            )
        return infos

    def original_page_infos(self, code: str, job_id: str) -> list[dict[str, Any]]:
        return self.page_infos(self.original_page_files(code, job_id))

    def final_page_infos(self, code: str, job_id: str) -> list[dict[str, Any]]:
        return self.page_infos(self.final_page_files(code, job_id))

    def load_editor_meta(self, code: str, job_id: str) -> dict[str, Any]:
        path = self.editor_meta_path(code, job_id)
        if not path.exists():
            return default_editor_meta()
        try:
            return normalize_editor_meta(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return default_editor_meta()

    def save_editor_meta(self, code: str, job_id: str, meta: dict[str, Any]) -> None:
        with self._lock:
            normalized = normalize_editor_meta(meta)
            path = self.editor_meta_path(code, job_id)
            write_json_atomic(path, normalized)
            self.write_translation_notes(code, job_id, normalized)

    def write_translation_notes(self, code: str, job_id: str, meta: dict[str, Any] | None = None) -> None:
        meta = normalize_editor_meta(meta if meta is not None else self.load_editor_meta(code, job_id))
        notes = meta.get("translationNotes", {})
        path = self.translation_notes_path(code, job_id)
        write_json_atomic(path, notes)

    def set_initial_translation_notes(self, code: str, job_id: str, notes: str) -> None:
        notes = notes.strip()
        if not notes:
            return
        with self._lock:
            meta = self.load_editor_meta(code, job_id)
            translation_notes = meta.setdefault("translationNotes", {"job": "", "pages": {}})
            if isinstance(translation_notes, dict):
                translation_notes["job"] = notes
            self.save_editor_meta(code, job_id, meta)

    def require_editable_job(self, code: str, job_id: str) -> dict[str, Any]:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        status = self.load_status(code, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        if status.get("status") != "complete":
            raise HTTPException(status_code=400, detail="Only complete jobs can be edited.")
        if not self.output_dir(code, job_id).is_dir():
            raise HTTPException(status_code=400, detail="Job output directory is missing.")
        return status

    def is_stage_locked(self, code: str, job_id: str, page: int, stage: str) -> bool:
        stages = lock_stages_for_editor_stage(stage)
        meta = self.load_editor_meta(code, job_id)
        locks = meta.get("locks", {})
        if not isinstance(locks, dict):
            return False
        page_locks = locks.get(page_key(page), {})
        return isinstance(page_locks, dict) and all(bool(page_locks.get(stage_name)) for stage_name in stages)

    def set_stage_lock(self, code: str, job_id: str, page: int, stage: str, locked: bool) -> dict[str, Any]:
        stages = lock_stages_for_editor_stage(stage)
        self.require_editable_job(code, job_id)
        self.original_page_path(code, job_id, page)
        with self._lock:
            meta = self.load_editor_meta(code, job_id)
            locks = meta.setdefault("locks", {})
            if not isinstance(locks, dict):
                locks = {}
                meta["locks"] = locks
            key = page_key(page)
            page_locks = locks.setdefault(key, {})
            if not isinstance(page_locks, dict):
                page_locks = {}
                locks[key] = page_locks
            for stage_name in stages:
                page_locks[stage_name] = bool(locked)
            self.save_editor_meta(code, job_id, meta)
            return meta

    def editor_stage_key(self, stage: str) -> str:
        stage = validate_stage(stage, EDITOR_UI_STAGES)
        if stage in {"ocr_raw", "ocr_merged", OCR_MERGE_EDITOR_STAGE}:
            return OCR_MERGE_EDITOR_STAGE
        if stage in EDITOR_UI_STAGES:
            return stage
        raise HTTPException(status_code=400, detail="Unsupported editor change stage.")

    def mark_editor_stage_changed(self, code: str, job_id: str, page: int, stage: str) -> dict[str, Any]:
        self.require_editable_job(code, job_id)
        self.original_page_path(code, job_id, page)
        stage_key = self.editor_stage_key(stage)
        with self._lock:
            meta = self.load_editor_meta(code, job_id)
            changed = meta.setdefault("changedStages", {})
            if not isinstance(changed, dict):
                changed = {}
                meta["changedStages"] = changed
            page_changes = changed.setdefault(page_key(page), {})
            if not isinstance(page_changes, dict):
                page_changes = {}
                changed[page_key(page)] = page_changes
            page_changes[stage_key] = now_utc()
            self.save_editor_meta(code, job_id, meta)
            return meta

    def editor_changed_pages(self, meta: dict[str, Any]) -> dict[int, set[str]]:
        changed = meta.get("changedStages", {})
        result: dict[int, set[str]] = {}
        if not isinstance(changed, dict):
            return result
        for raw_page, raw_stages in changed.items():
            try:
                page = int(raw_page)
            except (TypeError, ValueError):
                continue
            if not isinstance(raw_stages, dict):
                continue
            stages = {
                stage
                for stage in EDITOR_STAGE_ORDER
                if raw_stages.get(stage) is not None
            }
            if stages:
                result[page] = stages
        return result

    def earliest_editor_rerun_stage(self, stages: set[str]) -> str | None:
        order = ["ocr_structured", "translations", "placements", "render"]
        candidates = [
            EDITOR_STAGE_RERUN_FROM[stage]
            for stage in EDITOR_STAGE_ORDER
            if stage in stages and stage in EDITOR_STAGE_RERUN_FROM
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda stage: order.index(stage))

    def editor_change_info(
        self,
        code: str,
        job_id: str,
        page: int,
        stage: str,
    ) -> dict[str, Any]:
        stage_key = self.editor_stage_key(stage)
        meta = self.load_editor_meta(code, job_id)
        changed_by_page = self.editor_changed_pages(meta)
        current_stages = changed_by_page.get(page, set())
        upstream = [
            stage_name
            for stage_name in EDITOR_STAGE_UPSTREAM.get(stage_key, ())
            if stage_name in current_stages
        ]
        all_pages = sorted(changed_by_page)
        all_stages = set().union(*changed_by_page.values()) if changed_by_page else set()
        return {
            "page": page,
            "stage": stage_key,
            "changedStages": sorted(current_stages, key=EDITOR_STAGE_ORDER.index),
            "changedStageLabels": [
                EDITOR_STAGE_LABELS[stage_name]
                for stage_name in sorted(current_stages, key=EDITOR_STAGE_ORDER.index)
            ],
            "outdatedBecause": upstream,
            "outdatedBecauseLabels": [EDITOR_STAGE_LABELS[stage_name] for stage_name in upstream],
            "currentStageChanged": stage_key in current_stages,
            "allChangedPages": all_pages,
            "allChangedCount": len(all_pages),
            "allChangedResumeFrom": self.earliest_editor_rerun_stage(all_stages),
        }

    def clear_editor_changes_for_pages(self, code: str, job_id: str, pages: list[int]) -> None:
        with self._lock:
            meta = self.load_editor_meta(code, job_id)
            changed = meta.get("changedStages", {})
            if not isinstance(changed, dict):
                return
            for page in pages:
                changed.pop(page_key(page), None)
            self.save_editor_meta(code, job_id, meta)

    def locked_stage_paths_for_page(self, code: str, job_id: str, page: int) -> dict[str, Path]:
        meta = self.load_editor_meta(code, job_id)
        locks = meta.get("locks", {})
        page_locks = locks.get(page_key(page), {}) if isinstance(locks, dict) else {}
        result: dict[str, Path] = {}
        if not isinstance(page_locks, dict):
            return result
        for stage in EDITABLE_STAGES:
            if page_locks.get(stage):
                path = self.data_page_path(code, job_id, stage, page)
                if path.exists():
                    result[stage] = path
        return result

    def load_stage_records(self, code: str, job_id: str, stage: str, page: int) -> list[Any]:
        path = self.data_page_path(code, job_id, stage, page)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Invalid JSON in {stage} page {page}: {exc.msg}") from exc
        if not isinstance(data, list):
            raise HTTPException(status_code=500, detail=f"{stage} page {page} is not a JSON array.")
        return data

    def save_stage_records(self, code: str, job_id: str, stage: str, page: int, records: list[Any]) -> None:
        stage = validate_stage(stage)
        self.require_editable_job(code, job_id)
        self.original_page_path(code, job_id, page)
        if stage == "placements":
            records = self.normalize_saved_placements(code, job_id, page, records)
        path = self.data_page_path(code, job_id, stage, page)
        write_json_atomic(path, records)
        if stage in {"translations", "placements"}:
            self.apply_classification_from_stage_records(code, job_id, page, records)

    def clamp_editor_region(
        self,
        region: Any,
        width: int,
        height: int,
        label: str,
    ) -> list[int]:
        left, top, right, bottom = parse_region(region, label)
        left = max(0, min(width, left))
        right = max(0, min(width, right))
        top = max(0, min(height, top))
        bottom = max(0, min(height, bottom))
        if right <= left or bottom <= top:
            raise HTTPException(status_code=400, detail=f"{label} does not overlap the page.")
        return [left, top, right, bottom]

    def normalize_ocr_merge_records(
        self,
        code: str,
        job_id: str,
        page: int,
        raw_records: list[Any],
        merged_records: list[Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        image_path = self.original_page_path(code, job_id, page)
        with Image.open(image_path) as image:
            width, height = image.size

        normalized_raw: list[dict[str, Any]] = []
        used_raw_boxnos: set[int] = set()
        next_raw_boxno = 0
        for index, item in enumerate(raw_records):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"rawRecords[{index}] must be an object.")
            record = dict(item)
            raw_boxno = record.get("boxno")
            if not isinstance(raw_boxno, int) or raw_boxno in used_raw_boxnos:
                while next_raw_boxno in used_raw_boxnos:
                    next_raw_boxno += 1
                raw_boxno = next_raw_boxno
            used_raw_boxnos.add(raw_boxno)
            next_raw_boxno = max(next_raw_boxno, raw_boxno + 1)
            record["page"] = page
            record["boxno"] = raw_boxno
            record["region"] = self.clamp_editor_region(record.get("region"), width, height, f"rawRecords[{index}].region")
            record["text"] = str(record.get("text", ""))
            normalized_raw.append(record)

        raw_by_boxno = {
            int(record["boxno"]): record
            for record in normalized_raw
            if isinstance(record.get("boxno"), int)
        }

        normalized_merged: list[dict[str, Any]] = []
        for index, item in enumerate(merged_records):
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail=f"mergedRecords[{index}] must be an object.")
            record = dict(item)
            seen_sources: set[int] = set()
            source_boxnos: list[int] = []
            for raw_source in record.get("sourceBoxnos", []):
                if not isinstance(raw_source, int) or raw_source in seen_sources or raw_source not in raw_by_boxno:
                    continue
                seen_sources.add(raw_source)
                source_boxnos.append(raw_source)
            if not source_boxnos:
                continue

            source_regions = [raw_by_boxno[boxno]["region"] for boxno in source_boxnos]
            source_texts = [str(raw_by_boxno[boxno].get("text", "")) for boxno in source_boxnos]
            region = record.get("region")
            if isinstance(region, list) and len(region) == 4:
                region = self.clamp_editor_region(region, width, height, f"mergedRecords[{index}].region")
            else:
                region = [
                    min(item[0] for item in source_regions),
                    min(item[1] for item in source_regions),
                    max(item[2] for item in source_regions),
                    max(item[3] for item in source_regions),
                ]
            record["page"] = page
            record["boxno"] = len(normalized_merged)
            record["sourceBoxnos"] = source_boxnos
            record["sourceTexts"] = source_texts
            record["text"] = "".join(source_texts)
            record["region"] = region
            normalized_merged.append(record)

        return normalized_raw, normalized_merged

    def save_ocr_merge_records(
        self,
        code: str,
        job_id: str,
        page: int,
        raw_records: list[Any],
        merged_records: list[Any],
    ) -> None:
        raw_records, merged_records = self.normalize_ocr_merge_records(
            code,
            job_id,
            page,
            raw_records,
            merged_records,
        )
        self.save_stage_records(code, job_id, "ocr_raw", page, raw_records)
        self.save_stage_records(code, job_id, "ocr_merged", page, merged_records)
        self.set_stage_lock(code, job_id, page, OCR_MERGE_EDITOR_STAGE, True)

    def normalize_saved_placements(
        self,
        code: str,
        job_id: str,
        page: int,
        records: list[Any],
    ) -> list[Any]:
        image_path = self.original_page_path(code, job_id, page)
        with Image.open(image_path) as image:
            width, height = image.size
        normalized: list[Any] = []
        for item in records:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            record = dict(item)
            region = record.get("placementRegion", record.get("region"))
            if isinstance(region, list) and len(region) == 4:
                try:
                    left, top, right, bottom = [round(float(value)) for value in region]
                except (TypeError, ValueError):
                    normalized.append(record)
                    continue
                left = max(0, min(width, left))
                right = max(0, min(width, right))
                top = max(0, min(height, top))
                bottom = max(0, min(height, bottom))
                if right > left and bottom > top:
                    record["placementRegion"] = [left, top, right, bottom]
                    record["box_2d"] = [
                        round(top / height * 1000),
                        round(left / width * 1000),
                        round(bottom / height * 1000),
                        round(right / width * 1000),
                    ]
            normalized.append(record)
        return normalized

    def apply_classification_from_stage_records(
        self,
        code: str,
        job_id: str,
        page: int,
        stage_records: list[Any],
    ) -> None:
        structured_path = self.data_page_path(code, job_id, "ocr_structured", page)
        if not structured_path.exists():
            return
        try:
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(structured, list):
            return
        stage_by_boxno = {
            item.get("boxno"): item
            for item in stage_records
            if isinstance(item, dict) and isinstance(item.get("boxno"), int)
        }
        changed = False
        for record in structured:
            if not isinstance(record, dict) or not isinstance(record.get("boxno"), int):
                continue
            stage_record = stage_by_boxno.get(record["boxno"])
            if not isinstance(stage_record, dict):
                continue
            for field in ("sfx", "openLettering"):
                if isinstance(stage_record.get(field), bool) and record.get(field) != stage_record[field]:
                    record[field] = stage_record[field]
                    changed = True
        if changed:
            write_json_atomic(structured_path, structured)

    def snapshot_locked_stage_files(self, code: str, job_id: str, page: int) -> dict[str, bytes]:
        snapshot: dict[str, bytes] = {}
        for stage, path in self.locked_stage_paths_for_page(code, job_id, page).items():
            try:
                snapshot[stage] = path.read_bytes()
            except OSError:
                continue
        return snapshot

    def restore_locked_stage_files(
        self,
        code: str,
        job_id: str,
        page: int,
        snapshot: dict[str, bytes],
    ) -> bool:
        restored = False
        for stage, content in snapshot.items():
            path = self.data_page_path(code, job_id, stage, page)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            restored = True
        return restored

    def post_restore_rerun_sequence(self, resume_from: str, restored_stages: set[str]) -> list[str]:
        overwritten_by_resume = {
            "ocr_raw": {"ocr_raw", "ocr_merged", "ocr_structured", "translations", "placements"},
            "ocr_structured": {"ocr_structured", "translations", "placements"},
            "translations": {"translations", "placements"},
            "placements": {"placements"},
            "render": set(),
        }
        overwritten = overwritten_by_resume.get(resume_from, set())
        restored_overwritten = restored_stages & overwritten
        followup_for_stage = {
            "ocr_raw": "ocr_structured",
            "ocr_merged": "ocr_structured",
            "ocr_structured": "translations",
            "translations": "placements",
            "placements": "render",
        }
        phase_order = ["ocr_structured", "translations", "placements", "render"]
        needed = {
            followup_for_stage[stage]
            for stage in restored_overwritten
            if stage in followup_for_stage
        }
        return [phase for phase in phase_order if phase in needed]

    def editor_payload(self, code: str, job_id: str, stage: str, page: int) -> dict[str, Any]:
        stage = validate_stage(stage, EDITOR_UI_STAGES)
        self.require_editable_job(code, job_id)
        page_path = self.original_page_path(code, job_id, page)
        records: list[Any] = []
        raw_records: list[Any] = []
        merged_records: list[Any] = []
        reference_records: list[Any] = []
        if stage == OCR_MERGE_EDITOR_STAGE:
            raw_records = self.load_stage_records(code, job_id, "ocr_raw", page)
            merged_records = self.load_stage_records(code, job_id, "ocr_merged", page)
            records = merged_records
            reference_records = raw_records
        else:
            records = self.load_stage_records(code, job_id, stage, page)
        if stage in {"translations", "placements"}:
            reference_records = self.load_stage_records(code, job_id, "ocr_structured", page)
        elif stage == "ocr_merged":
            reference_records = self.load_stage_records(code, job_id, "ocr_raw", page)
        meta = self.load_editor_meta(code, job_id)
        locks = meta.get("locks", {})
        page_locks = locks.get(page_key(page), {}) if isinstance(locks, dict) else {}
        notes = meta.get("translationNotes", {})
        page_notes = notes.get("pages", {}) if isinstance(notes, dict) else {}
        return {
            "category": code,
            "jobId": job_id,
            "stage": stage,
            "page": page,
            "pageCount": len(self.original_page_files(code, job_id)),
            "imageUrl": f"/job/{code}/{job_id}/image/{page}",
            "imageName": page_path.name,
            "records": records,
            "rawRecords": raw_records,
            "mergedRecords": merged_records,
            "referenceRecords": reference_records,
            "locked": self.is_stage_locked(code, job_id, page, stage),
            "locks": page_locks if isinstance(page_locks, dict) else {},
            "translationNotes": {
                "job": notes.get("job", "") if isinstance(notes, dict) else "",
                "page": page_notes.get(page_key(page), "") if isinstance(page_notes, dict) else "",
            },
            "stages": list(EDITOR_UI_STAGES),
            "changeInfo": self.editor_change_info(code, job_id, page, stage),
        }

    def save_translation_notes(
        self,
        code: str,
        job_id: str,
        page: int,
        job_note: str | None,
        page_note: str | None,
    ) -> dict[str, Any]:
        self.require_editable_job(code, job_id)
        self.original_page_path(code, job_id, page)
        with self._lock:
            meta = self.load_editor_meta(code, job_id)
            notes = meta.setdefault("translationNotes", {"job": "", "pages": {}})
            if not isinstance(notes, dict):
                notes = {"job": "", "pages": {}}
                meta["translationNotes"] = notes
            pages = notes.setdefault("pages", {})
            if not isinstance(pages, dict):
                pages = {}
                notes["pages"] = pages
            if job_note is not None:
                notes["job"] = job_note
            if page_note is not None:
                if page_note:
                    pages[page_key(page)] = page_note
                else:
                    pages.pop(page_key(page), None)
            self.save_editor_meta(code, job_id, meta)
            return meta

    def run_crop_ocr(
        self,
        code: str,
        job_id: str,
        page: int,
        region: list[int],
        raw_boxno_start: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        self.require_editable_job(code, job_id)
        image_path = self.original_page_path(code, job_id, page)
        with Image.open(image_path) as image:
            width, height = image.size
            left, top, right, bottom = region
            left = max(0, min(width, left))
            right = max(0, min(width, right))
            top = max(0, min(height, top))
            bottom = max(0, min(height, bottom))
            if right <= left or bottom <= top:
                raise HTTPException(status_code=400, detail="OCR crop does not overlap the page.")
            crop = image.convert("RGB").crop((left, top, right, bottom))

        config = translate_cbz.load_config(self.config.pipeline_config, None)
        status = self.load_status(code, job_id) or {}
        ocr_config = self.ocr_config_for_status(config.ocr, status)
        if ocr_config.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            ocr = paddle_ocr_image.create_paddleocr_vl(
                ocr_config.device,
                ocr_config.paddleocr_vl_server_url,
                ocr_config.paddleocr_vl_model,
                api_key=ocr_config.paddleocr_vl_api_key,
                max_concurrency=ocr_config.paddleocr_vl_max_concurrency,
                service_url=ocr_config.service_url,
                service_timeout=ocr_config.service_timeout,
            )
        else:
            ocr = paddle_ocr_image.create_paddle_ocr(
                ocr_config.lang,
                ocr_config.device,
                ocr_config.text_det_limit_side_len,
                ocr_config.text_det_limit_type,
                ocr_config.use_doc_preprocessor,
                ocr_config.use_textline_orientation,
                ocr_version=ocr_config.ocr_version,
                text_detection_model_name=ocr_config.text_detection_model_name,
                text_recognition_model_name=ocr_config.text_recognition_model_name,
                text_detection_model_dir=ocr_config.text_detection_model_dir,
                text_recognition_model_dir=ocr_config.text_recognition_model_dir,
                text_det_thresh=ocr_config.text_det_thresh,
                text_det_box_thresh=ocr_config.text_det_box_thresh,
                text_det_unclip_ratio=ocr_config.text_det_unclip_ratio,
                text_rec_score_thresh=ocr_config.text_rec_score_thresh,
                service_url=ocr_config.service_url,
                service_timeout=ocr_config.service_timeout,
            )

        try:
            with tempfile.TemporaryDirectory(
                prefix="tetolate_crop_ocr_"
            ) as temp_dir_name:
                crop_path = Path(temp_dir_name) / "crop.png"
                crop.save(crop_path, format="PNG")
                if ocr_config.engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
                    crop_records = paddle_ocr_image.extract_paddleocr_vl_image_records(
                        ocr,
                        crop_path,
                        page,
                        ocr_config.min_score,
                    )
                else:
                    crop_records = paddle_ocr_image.extract_image_records(
                        ocr,
                        crop_path,
                        page,
                        ocr_config.min_score,
                        tile_enabled=False,
                    )
        finally:
            paddle_ocr_image.close_ocr_engine(ocr)

        next_boxno = raw_boxno_start
        if next_boxno is None:
            existing = self.load_stage_records(code, job_id, "ocr_raw", page)
            next_boxno = 0
            for record in existing:
                if isinstance(record, dict) and isinstance(record.get("boxno"), int):
                    next_boxno = max(next_boxno, record["boxno"] + 1)

        shifted_records: list[dict[str, Any]] = []
        for record in crop_records:
            shifted = offset_record_region(record, left, top, width, height)
            if shifted is None:
                continue
            shifted["boxno"] = next_boxno + len(shifted_records)
            shifted_records.append(shifted)
        page_ref = translate_cbz.Page(index=page, image_path=image_path)
        status = self.load_status(code, job_id) or {}
        merged_records = translate_cbz.merge_ocr_records_for_page(
            page_ref,
            shifted_records,
            right_to_left=self.status_source_language(status) != "kr",
        )
        return {
            "rawRecords": shifted_records,
            "mergedRecords": merged_records,
            "records": shifted_records,
        }

    def download_info(self, code: str, job_id: str) -> dict[str, dict[str, Any]]:
        downloads: dict[str, dict[str, Any]] = {}
        for variant in TRANSLATED_CBZ_FILENAMES:
            downloads[variant] = file_info(
                self.translated_cbz_variant_path(code, job_id, variant)
            )
        return downloads

    def iter_job_ids(self, code: str) -> list[str]:
        jobs_path = self.jobs_dir(code)
        if not jobs_path.is_dir():
            return []
        return sorted(
            (
                path.name
                for path in jobs_path.iterdir()
                if path.is_dir() and JOB_ID_PATTERN.match(path.name)
            ),
            reverse=True,
        )

    def create_job_id(self, code: str) -> str:
        with self._lock:
            self.jobs_dir(code).mkdir(parents=True, exist_ok=True)
            while True:
                job_id = secrets.token_hex(JOB_ID_BYTES)
                if not self.job_dir(code, job_id).exists():
                    return job_id

    def infer_resume_target_from_log(self, code: str, job_id: str) -> tuple[str, int] | None:
        path = self.log_path(code, job_id)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.replace("\r", "\n").splitlines()
        for line in reversed(lines):
            parsed = parse_progress_line(line.strip())
            if parsed is None:
                continue
            target = progress_resume_target(*parsed)
            if target is not None:
                return target
        return None

    def restart_target_for_status(
        self,
        code: str,
        job_id: str,
        status: dict[str, Any],
    ) -> tuple[str, int] | None:
        target = stored_resume_target(status.get("lastResumeFrom"), status.get("lastResumePage"))
        if target is not None:
            return target
        target = progress_resume_target(status.get("lastPhase"), status.get("lastPage"))
        if target is not None:
            return target
        target = progress_resume_target(status.get("phase"), status.get("page"))
        if target is not None:
            return target
        return self.infer_resume_target_from_log(code, job_id)

    def pending_resume_target_for_status(self, status: dict[str, Any]) -> tuple[str, int] | None:
        return stored_resume_target(status.get("pendingResumeFrom"), status.get("pendingResumePage"))

    def apply_pending_resume_target(
        self,
        status: dict[str, Any],
        target: tuple[str, int] | None,
    ) -> None:
        if target is None:
            status.pop("pendingResumeFrom", None)
            status.pop("pendingResumePage", None)
            status.pop("pendingPackageOnly", None)
            status.pop("pendingWebpQuality", None)
            status.pop("pendingJxlQuality", None)
            return

        resume_from, resume_page = target
        status["pendingResumeFrom"] = resume_from
        status["pendingResumePage"] = resume_page
        status["lastResumeFrom"] = resume_from
        status["lastResumePage"] = resume_page
        if resume_from == "package":
            status["pendingPackageOnly"] = True
            status["pendingWebpQuality"] = quality_for_display(
                status.get("webpQuality"),
                self.config.default_webp_quality,
            )
            status["pendingJxlQuality"] = quality_for_display(
                status.get("jxlQuality"),
                self.config.default_jxl_quality,
            )
        else:
            status.pop("pendingPackageOnly", None)
            status.pop("pendingWebpQuality", None)
            status.pop("pendingJxlQuality", None)

    def load_status(self, code: str, job_id: str) -> dict[str, Any] | None:
        path = self.status_path(code, job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def save_status(self, code: str, job_id: str, status: dict[str, Any]) -> None:
        status.pop("code", None)
        status["category"] = code
        status["jobId"] = job_id
        status["updatedAt"] = now_utc()
        path = self.status_path(code, job_id)
        write_json_atomic(path, status)

    def status_thinking_budget_tokens(self, status: dict[str, Any]) -> int:
        try:
            return validate_thinking_budget_tokens(
                status.get(
                    "thinkingBudgetTokens",
                    self.config.default_thinking_budget_tokens,
                )
            )
        except ValueError:
            return self.config.default_thinking_budget_tokens

    def status_vlm_base_url(self, status: dict[str, Any]) -> str:
        try:
            return validate_vlm_base_url(
                str(status.get("vlmBaseUrl", "")),
                self.config.default_vlm_base_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def status_pause_after_ocr(self, status: dict[str, Any]) -> bool:
        return bool(status.get("pauseAfterOcr"))

    def status_proofread_translations(self, status: dict[str, Any]) -> bool:
        value = status.get("proofreadTranslations", DEFAULT_PROOFREAD_TRANSLATIONS)
        return bool(value)

    def status_write_translation_notes(self, status: dict[str, Any]) -> bool:
        value = status.get("writeTranslationNotes", DEFAULT_WRITE_TRANSLATION_NOTES)
        return bool(value)

    def status_alt_placement_enabled(self, status: dict[str, Any]) -> bool:
        value = status.get("altPlacementEnabled", self.config.default_alt_placement_enabled)
        return bool(value)

    def status_source_language(self, status: dict[str, Any]) -> str:
        try:
            return translate_cbz.normalize_source_language(
                status.get("sourceLanguage", self.config.default_source_language)
            )
        except translate_cbz.PipelineError:
            return self.config.default_source_language

    def status_ocr_engine(self, status: dict[str, Any]) -> str:
        try:
            return paddle_ocr_image.normalize_ocr_engine(
                status.get("ocrEngine", self.config.default_ocr_engine)
            )
        except paddle_ocr_image.InputError:
            return self.config.default_ocr_engine

    def status_paddleocr_vl_server_url(self, status: dict[str, Any]) -> str:
        value = str(
            status.get(
                "paddleocrVlServerUrl",
                self.config.default_paddleocr_vl_server_url,
            )
            or self.config.default_paddleocr_vl_server_url
        )
        try:
            return validate_paddleocr_vl_server_url(
                value,
                self.config.default_paddleocr_vl_server_url,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def status_paddleocr_vl_model(self, status: dict[str, Any]) -> str:
        return str(
            status.get(
                "paddleocrVlModel",
                self.config.default_paddleocr_vl_model,
            )
            or self.config.default_paddleocr_vl_model
        )

    def ocr_config_for_status(
        self,
        ocr_config: translate_cbz.OCRConfig,
        status: dict[str, Any],
    ) -> translate_cbz.OCRConfig:
        return replace(
            ocr_config,
            lang=translate_cbz.SOURCE_LANGUAGE_PROFILES[
                self.status_source_language(status)
            ].ocr_lang,
            engine=self.status_ocr_engine(status),
            paddleocr_vl_server_url=self.status_paddleocr_vl_server_url(status),
            paddleocr_vl_model=self.status_paddleocr_vl_model(status),
        )

    def public_status(
        self,
        code: str,
        job_id: str,
        include_log: bool = True,
    ) -> dict[str, Any]:
        self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        status = self.load_status(code, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown job.")

        timing = job_timing(status)
        status_value = status.get("status", "unknown")
        with self._lock:
            is_active = (code, job_id) in self._active_processes
        logical_pause = status_value == "paused" and not is_active
        can_restart = (
            status_value in {"failed", "cancelled"} or logical_pause
        ) and self.input_path(code, job_id).is_file()
        can_terminate = status_value in {"running", "paused", "terminating"} and is_active
        can_delete = status_value in TERMINAL_JOB_STATUSES and not is_active
        restart_target = (
            self.restart_target_for_status(code, job_id, status) if can_restart else None
        )
        input_info = file_info(self.input_path(code, job_id))
        downloads = self.download_info(code, job_id)
        original_page_count = len(self.original_page_files(code, job_id))
        final_page_count = (
            len(self.final_page_files(code, job_id))
            if status.get("status") == "complete"
            else 0
        )
        webp_quality = quality_for_display(
            status.get("webpQuality"),
            self.config.default_webp_quality,
        )
        jxl_quality = quality_for_display(
            status.get("jxlQuality"),
            self.config.default_jxl_quality,
        )
        thinking_budget_tokens = self.status_thinking_budget_tokens(status)
        vlm_base_url = self.status_vlm_base_url(status)
        pause_after_ocr = self.status_pause_after_ocr(status)
        proofread_translations = self.status_proofread_translations(status)
        write_translation_notes = self.status_write_translation_notes(status)
        alt_placement_enabled = self.status_alt_placement_enabled(status)
        source_language = self.status_source_language(status)
        ocr_engine = self.status_ocr_engine(status)
        paddleocr_vl_server_url = self.status_paddleocr_vl_server_url(status)
        paddleocr_vl_model = self.status_paddleocr_vl_model(status)
        generated_translation_notes = self.read_generated_translation_notes(code, job_id)

        payload = {
            "category": code,
            "jobId": job_id,
            "status": status_value,
            "phase": status.get("phase"),
            "page": status.get("page"),
            "message": status.get("message", ""),
            "isPaused": bool(status.get("isPaused")),
            "inputFilename": status.get("inputFilename", ""),
            "inputSizeBytes": input_info["sizeBytes"],
            "inputSize": input_info["size"],
            "inputDownloadToken": input_info["downloadToken"],
            "createdAt": status.get("createdAt"),
            "startedAt": status.get("startedAt"),
            "finishedAt": status.get("finishedAt"),
            "updatedAt": status.get("updatedAt"),
            "ageSeconds": timing.age_seconds,
            "age": age_text(timing.age_seconds),
            "elapsedSeconds": timing.elapsed_seconds,
            "elapsed": duration_text(timing.elapsed_seconds),
            "hasDownload": self.translated_cbz_path(code, job_id).is_file()
            and status.get("status") == "complete",
            "downloads": downloads if status.get("status") == "complete" else {},
            "hasOriginalDownload": bool(input_info["available"]),
            "originalDownloadUrl": f"/job/{code}/{job_id}/download-original",
            "canViewOriginal": original_page_count > 0,
            "originalViewUrl": f"/job/{code}/{job_id}/view-original",
            "originalPageCount": original_page_count,
            "canView": status.get("status") == "complete" and final_page_count > 0,
            "viewUrl": f"/job/{code}/{job_id}/view",
            "finalPageCount": final_page_count,
            "canDelete": can_delete,
            "canRestart": can_restart,
            "canTerminate": can_terminate,
            "canRerunPages": status.get("status") == "complete",
            "canRegenerateDownloads": status.get("status") == "complete",
            "canEdit": status.get("status") == "complete",
            "webpQuality": webp_quality,
            "jxlQuality": jxl_quality,
            "defaultWebpQuality": self.config.default_webp_quality,
            "defaultJxlQuality": self.config.default_jxl_quality,
            "thinkingBudgetTokens": thinking_budget_tokens,
            "thinkingBudget": thinking_budget_text(thinking_budget_tokens),
            "defaultThinkingBudgetTokens": self.config.default_thinking_budget_tokens,
            "vlmBaseUrl": vlm_base_url,
            "defaultVlmBaseUrl": self.config.default_vlm_base_url,
            "pauseAfterOcr": pause_after_ocr,
            "proofreadTranslations": proofread_translations,
            "writeTranslationNotes": write_translation_notes,
            "altPlacementEnabled": alt_placement_enabled,
            "sourceLanguage": source_language,
            "ocrEngine": ocr_engine,
            "paddleocrVlServerUrl": paddleocr_vl_server_url,
            "paddleocrVlModel": paddleocr_vl_model,
            "defaultAltPlacementEnabled": self.config.default_alt_placement_enabled,
            "defaultSourceLanguage": self.config.default_source_language,
            "defaultOcrEngine": self.config.default_ocr_engine,
            "defaultPaddleocrVlServerUrl": self.config.default_paddleocr_vl_server_url,
            "defaultPaddleocrVlModel": self.config.default_paddleocr_vl_model,
            "generatedTranslationNotes": generated_translation_notes,
            "hasGeneratedTranslationNotes": bool(generated_translation_notes),
            "canUpdateAdvancedOptions": status_value in {"failed", "cancelled"} or logical_pause,
            "restartResumeFrom": restart_target[0] if restart_target is not None else None,
            "restartResumePage": restart_target[1] if restart_target is not None else None,
            "url": f"/job/{code}/{job_id}",
        }
        if include_log:
            payload["recentLog"] = safe_log_lines(self.log_path(code, job_id))
        return payload

    def public_category_jobs(self, code: str) -> dict[str, Any]:
        self.validate_category(code)
        options = self.load_category_advanced_options(code)
        jobs = []
        for job_id in self.iter_job_ids(code):
            try:
                jobs.append(self.public_status(code, job_id, include_log=False))
            except HTTPException:
                continue
        jobs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
        return {
            "category": code,
            "jobs": jobs,
            "translationNotes": options["translationNotes"],
            "defaultThinkingBudgetTokens": options["thinkingBudgetTokens"],
            "defaultVlmBaseUrl": options["vlmBaseUrl"],
            "pauseAfterOcr": options["pauseAfterOcr"],
            "proofreadTranslations": options["proofreadTranslations"],
            "writeTranslationNotes": options["writeTranslationNotes"],
            "defaultAltPlacementEnabled": options["altPlacementEnabled"],
            "defaultSourceLanguage": options["sourceLanguage"],
            "defaultOcrEngine": options["ocrEngine"],
            "defaultPaddleocrVlServerUrl": options["paddleocrVlServerUrl"],
            "defaultPaddleocrVlModel": options["paddleocrVlModel"],
        }

    def requeue_interrupted_jobs(self) -> None:
        saw_paused_job = False
        for code in self.categories():
            for job_id in self.iter_job_ids(code):
                status = self.load_status(code, job_id)
                if status is None:
                    continue
                previous_status = status.get("status")
                if previous_status not in {"queued", "running", "paused", "terminating"}:
                    continue
                if previous_status == "terminating":
                    self.stop_recorded_pipeline_process(status)
                    status.update(
                        {
                            "status": "cancelled",
                            "phase": "Cancelled",
                            "page": None,
                            "message": "Termination completed during web server startup.",
                            "finishedAt": now_utc(),
                            "isPaused": False,
                        }
                    )
                    for key in (
                        "pid",
                        "pausedAt",
                        "pendingTermination",
                        "terminatingAt",
                    ):
                        status.pop(key, None)
                    self.save_status(code, job_id, status)
                    continue
                if previous_status == "paused":
                    if status.get("pauseAfterOcr") and status.get("pendingResumeFrom"):
                        continue
                    saw_paused_job = True
                resume_target = self.pending_resume_target_for_status(status)
                if previous_status != "queued" and resume_target is None:
                    resume_target = self.restart_target_for_status(code, job_id, status)
                if previous_status != "queued":
                    self.stop_recorded_pipeline_process(status)
                status["status"] = "queued"
                status["phase"] = "Queued"
                status["page"] = None
                status["message"] = (
                    f"Queued after web server startup to resume from {resume_target[0]} "
                    f"page {resume_target[1]}."
                    if resume_target is not None
                    else "Queued after web server startup."
                )
                status["isPaused"] = False
                status.pop("pid", None)
                status.pop("pausedAt", None)
                status.pop("pendingTermination", None)
                status.pop("terminatingAt", None)
                if previous_status != "queued":
                    self.apply_pending_resume_target(status, resume_target)
                self.save_status(code, job_id, status)
                self.enqueue(code, job_id)
        if saw_paused_job:
            with self._pause_condition:
                self._paused = True

    def enqueue(self, code: str, job_id: str) -> None:
        key = (code, job_id)
        with self._lock:
            if key in self._queued_jobs:
                return
            self._queued_jobs.add(key)
            self._queue.put(key)

    def submit_job(
        self,
        code: str,
        job_id: str,
        original_filename: str,
        translation_notes: str = "",
        thinking_budget_tokens: int | None = None,
        vlm_base_url: str | None = None,
        pause_after_ocr: bool = False,
        proofread_translations: bool = DEFAULT_PROOFREAD_TRANSLATIONS,
        write_translation_notes: bool = DEFAULT_WRITE_TRANSLATION_NOTES,
        alt_placement_enabled: bool = DEFAULT_ALT_PLACEMENT_ENABLED,
        source_language: str | None = None,
        ocr_engine: str | None = None,
        paddleocr_vl_server_url: str | None = None,
        paddleocr_vl_model: str | None = None,
    ) -> None:
        created_at = now_utc()
        if thinking_budget_tokens is None:
            thinking_budget_tokens = self.config.default_thinking_budget_tokens
        thinking_budget_tokens = validate_thinking_budget_tokens(thinking_budget_tokens)
        vlm_base_url = validate_vlm_base_url(
            vlm_base_url or "",
            self.config.default_vlm_base_url,
        )
        if source_language is None:
            source_language = self.config.default_source_language
        source_language = translate_cbz.normalize_source_language(source_language)
        if ocr_engine is None:
            ocr_engine = self.config.default_ocr_engine
        ocr_engine = paddle_ocr_image.normalize_ocr_engine(ocr_engine)
        paddleocr_vl_server_url = (
            paddleocr_vl_server_url or self.config.default_paddleocr_vl_server_url
        )
        paddleocr_vl_model = paddleocr_vl_model or self.config.default_paddleocr_vl_model
        status = {
            "category": code,
            "jobId": job_id,
            "status": "queued",
            "phase": "Queued",
            "page": None,
            "message": "Waiting for the worker.",
            "createdAt": created_at,
            "inputFilename": original_filename,
            "thinkingBudgetTokens": thinking_budget_tokens,
            "vlmBaseUrl": vlm_base_url,
            "pauseAfterOcr": bool(pause_after_ocr),
            "proofreadTranslations": bool(proofread_translations),
            "writeTranslationNotes": bool(write_translation_notes),
            "altPlacementEnabled": bool(alt_placement_enabled),
            "sourceLanguage": source_language,
            "ocrEngine": ocr_engine,
            "paddleocrVlServerUrl": paddleocr_vl_server_url,
            "paddleocrVlModel": paddleocr_vl_model,
        }
        self.save_status(code, job_id, status)
        self.set_initial_translation_notes(code, job_id, translation_notes)
        self.remember_category_advanced_options(
            code,
            translation_notes=translation_notes,
            thinking_budget_tokens=thinking_budget_tokens,
            vlm_base_url=vlm_base_url,
            pause_after_ocr=pause_after_ocr,
            proofread_translations=proofread_translations,
            write_translation_notes=write_translation_notes,
            alt_placement_enabled=alt_placement_enabled,
            source_language=source_language,
            ocr_engine=ocr_engine,
            paddleocr_vl_server_url=paddleocr_vl_server_url,
            paddleocr_vl_model=paddleocr_vl_model,
        )
        self.enqueue(code, job_id)

    def update_job_advanced_options(
        self,
        code: str,
        job_id: str,
        thinking_budget_tokens: int,
        vlm_base_url: str,
        pause_after_ocr: bool,
        proofread_translations: bool,
        write_translation_notes: bool,
        alt_placement_enabled: bool,
        source_language: str,
        ocr_engine: str,
        paddleocr_vl_server_url: str,
        paddleocr_vl_model: str,
    ) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        thinking_budget_tokens = validate_thinking_budget_tokens(thinking_budget_tokens)
        vlm_base_url = validate_vlm_base_url(
            vlm_base_url,
            self.config.default_vlm_base_url,
        )
        source_language = translate_cbz.normalize_source_language(source_language)
        ocr_engine = paddle_ocr_image.normalize_ocr_engine(ocr_engine)
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") not in {"failed", "cancelled", "paused"}:
                raise HTTPException(
                    status_code=400,
                    detail="Only failed, cancelled, or paused jobs can update advanced options.",
                )
            status["thinkingBudgetTokens"] = thinking_budget_tokens
            status["vlmBaseUrl"] = vlm_base_url
            status["pauseAfterOcr"] = bool(pause_after_ocr)
            status["proofreadTranslations"] = bool(proofread_translations)
            status["writeTranslationNotes"] = bool(write_translation_notes)
            status["altPlacementEnabled"] = bool(alt_placement_enabled)
            status["sourceLanguage"] = source_language
            status["ocrEngine"] = ocr_engine
            status["paddleocrVlServerUrl"] = paddleocr_vl_server_url
            status["paddleocrVlModel"] = paddleocr_vl_model
            status["message"] = "Advanced options updated."
            self.save_status(code, job_id, status)
            self.remember_category_advanced_options(
                code,
                thinking_budget_tokens=thinking_budget_tokens,
                vlm_base_url=vlm_base_url,
                pause_after_ocr=pause_after_ocr,
                proofread_translations=proofread_translations,
                write_translation_notes=write_translation_notes,
                alt_placement_enabled=alt_placement_enabled,
                source_language=source_language,
                ocr_engine=ocr_engine,
                paddleocr_vl_server_url=paddleocr_vl_server_url,
                paddleocr_vl_model=paddleocr_vl_model,
            )

    def restart_failed_job(self, code: str, job_id: str) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if (code, job_id) in self._active_processes:
                raise HTTPException(status_code=409, detail="Job is already running or paused.")
            if status.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Job is already queued or running.")
            if status.get("status") not in {"failed", "cancelled", "paused"}:
                raise HTTPException(
                    status_code=400,
                    detail="Only failed, cancelled, or paused jobs can be restarted.",
                )
            if not self.input_path(code, job_id).is_file():
                raise HTTPException(status_code=400, detail="Original uploaded CBZ is missing.")

            restart_target = self.restart_target_for_status(code, job_id, status)
            try:
                restart_count = int(status.get("restartCount", 0)) + 1
            except (TypeError, ValueError):
                restart_count = 1
            status.update(
                {
                    "status": "queued",
                    "phase": "Queued",
                    "page": None,
                    "message": (
                        "Queued to resume from "
                        f"{restart_target[0]} page {restart_target[1]}."
                        if restart_target is not None
                        else "Queued to restart from the beginning."
                    ),
                    "finishedAt": None,
                    "returnCode": None,
                    "restartCount": restart_count,
                }
            )
            status.pop("pid", None)
            if restart_target is not None:
                status["pendingResumeFrom"] = restart_target[0]
                status["pendingResumePage"] = restart_target[1]
                if restart_target[0] == "package":
                    status["pendingPackageOnly"] = True
                    status["pendingWebpQuality"] = quality_for_display(
                        status.get("webpQuality"),
                        self.config.default_webp_quality,
                    )
                    status["pendingJxlQuality"] = quality_for_display(
                        status.get("jxlQuality"),
                        self.config.default_jxl_quality,
                    )
                else:
                    status.pop("pendingPackageOnly", None)
                    status.pop("pendingWebpQuality", None)
                    status.pop("pendingJxlQuality", None)
            else:
                status.pop("pendingResumeFrom", None)
                status.pop("pendingResumePage", None)
                status.pop("pendingPackageOnly", None)
                status.pop("pendingWebpQuality", None)
                status.pop("pendingJxlQuality", None)
            self.save_status(code, job_id, status)
            self.enqueue(code, job_id)

    def rerun_completed_job_pages(
        self,
        code: str,
        job_id: str,
        pages: list[int],
        resume_from: str = "ocr_raw",
        webp_quality: int | None = None,
        jxl_quality: int | None = None,
    ) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        resume_from = RERUN_STAGE_MAP.get(resume_from, resume_from)
        if resume_from not in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements", "render"}:
            raise HTTPException(status_code=400, detail="Unsupported rerun stage.")
        if webp_quality is not None:
            webp_quality = validate_output_quality(webp_quality, "WebP quality")
        if jxl_quality is not None:
            jxl_quality = validate_output_quality(jxl_quality, "JXL quality")
        if (webp_quality is None) != (jxl_quality is None):
            raise HTTPException(status_code=400, detail="Both WebP and JXL quality must be provided together.")
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Job is already queued or running.")
            if status.get("status") != "complete":
                raise HTTPException(status_code=400, detail="Only complete jobs can rerun pages.")
            if not self.input_path(code, job_id).is_file():
                raise HTTPException(status_code=400, detail="Original uploaded CBZ is missing.")
            if not self.output_dir(code, job_id).is_dir():
                raise HTTPException(status_code=400, detail="Existing output directory is missing.")

            page_count = len(self.original_page_files(code, job_id))
            if page_count <= 0:
                raise HTTPException(status_code=400, detail="No extracted pages were found for this job.")
            if pages:
                pages = sorted(set(pages))
                out_of_range = [page for page in pages if page < 0 or page >= page_count]
                if out_of_range:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Page is outside this job's range 0-{page_count - 1}: {out_of_range[0]}",
                    )
                page_message = ", ".join(str(page) for page in pages)
            else:
                pages = list(range(page_count))
                page_message = f"all pages (0-{page_count - 1})"

            try:
                rerun_count = int(status.get("rerunCount", 0)) + 1
            except (TypeError, ValueError):
                rerun_count = 1
            status.update(
                {
                    "status": "queued",
                    "phase": "Queued",
                    "page": None,
                    "message": (
                        f"Queued to rerun {page_message} from {resume_from}."
                    ),
                    "finishedAt": None,
                    "returnCode": None,
                    "rerunCount": rerun_count,
                    "pendingSinglePages": pages,
                    "pendingSinglePageResumeFrom": resume_from,
                    "pendingResumeFrom": resume_from,
                    "pendingResumePage": pages[0],
                    "lastResumeFrom": resume_from,
                    "lastResumePage": pages[0],
                }
            )
            if webp_quality is not None and jxl_quality is not None:
                status["pendingWebpQuality"] = webp_quality
                status["pendingJxlQuality"] = jxl_quality
            else:
                status.pop("pendingWebpQuality", None)
                status.pop("pendingJxlQuality", None)
            status.pop("pid", None)
            self.save_status(code, job_id, status)
            self.enqueue(code, job_id)

    def rerun_changed_editor_pages(self, code: str, job_id: str) -> tuple[list[int], str]:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        meta = self.load_editor_meta(code, job_id)
        changed_by_page = self.editor_changed_pages(meta)
        if not changed_by_page:
            raise HTTPException(status_code=400, detail="No saved editor changes need regeneration.")
        pages = sorted(changed_by_page)
        all_stages = set().union(*changed_by_page.values())
        resume_from = self.earliest_editor_rerun_stage(all_stages)
        if resume_from is None:
            raise HTTPException(status_code=400, detail="No supported editor changes need regeneration.")
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Job is already queued or running.")
            if status.get("status") != "complete":
                raise HTTPException(status_code=400, detail="Only complete jobs can rerun pages.")
            if not self.input_path(code, job_id).is_file():
                raise HTTPException(status_code=400, detail="Original uploaded CBZ is missing.")
            if not self.output_dir(code, job_id).is_dir():
                raise HTTPException(status_code=400, detail="Existing output directory is missing.")

            try:
                rerun_count = int(status.get("rerunCount", 0)) + 1
            except (TypeError, ValueError):
                rerun_count = 1
            status.update(
                {
                    "status": "queued",
                    "phase": "Queued",
                    "page": None,
                    "message": (
                        "Queued to regenerate changed editor pages "
                        + ", ".join(str(page) for page in pages)
                        + f" from {resume_from}."
                    ),
                    "finishedAt": None,
                    "returnCode": None,
                    "rerunCount": rerun_count,
                    "pendingSinglePages": pages,
                    "pendingSinglePageResumeFrom": resume_from,
                    "pendingEditorChangedPages": pages,
                    "pendingResumeFrom": resume_from,
                    "pendingResumePage": pages[0],
                    "lastResumeFrom": resume_from,
                    "lastResumePage": pages[0],
                }
            )
            status.pop("pid", None)
            self.save_status(code, job_id, status)
            self.enqueue(code, job_id)
        return pages, resume_from

    def terminate_active_job(self, code: str, job_id: str) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        key = (code, job_id)
        with self._lock:
            process = self._active_processes.get(key)
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if process is None or process.poll() is not None:
                raise HTTPException(status_code=400, detail="Job is not currently running.")
            if status.get("status") not in {"running", "paused", "terminating"}:
                raise HTTPException(status_code=400, detail="Only running or paused jobs can be terminated.")

            status.update(
                {
                    "status": "terminating",
                    "phase": "Terminating",
                    "message": (
                        "Terminating job. Any active VLM HTTP stream will be closed "
                        "when the local pipeline process exits."
                    ),
                    "pendingTermination": True,
                    "terminatingAt": now_utc(),
                    "isPaused": False,
                }
            )
            status.pop("pausedAt", None)
            self.save_status(code, job_id, status)

        if os.name == "posix":
            terminated = self.signal_process_group_or_process(process, signal.SIGTERM)
            continue_signal = getattr(signal, "SIGCONT", None)
            if continue_signal is not None:
                self.signal_process_group_or_process(process, continue_signal)
        else:
            try:
                process.terminate()
                terminated = True
            except OSError:
                terminated = False
        if not terminated:
            status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
            status["message"] = "Terminate requested, but SIGTERM could not be sent."
            self.save_status(code, job_id, status)
            raise HTTPException(status_code=500, detail="Could not signal active job.")

        threading.Thread(
            target=self.force_kill_process_later,
            args=(code, job_id, process),
            name=f"tetolate-kill-{code}-{job_id}",
            daemon=True,
        ).start()

    def regenerate_completed_job_downloads(
        self,
        code: str,
        job_id: str,
        webp_quality: int,
        jxl_quality: int,
    ) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        webp_quality = validate_output_quality(webp_quality, "WebP quality")
        jxl_quality = validate_output_quality(jxl_quality, "JXL quality")
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Job is already queued or running.")
            if status.get("status") != "complete":
                raise HTTPException(
                    status_code=400,
                    detail="Only complete jobs can regenerate downloads.",
                )
            if not self.input_path(code, job_id).is_file():
                raise HTTPException(status_code=400, detail="Original uploaded CBZ is missing.")
            if not self.output_dir(code, job_id).is_dir():
                raise HTTPException(status_code=400, detail="Existing output directory is missing.")

            try:
                package_count = int(status.get("packageRegenerationCount", 0)) + 1
            except (TypeError, ValueError):
                package_count = 1
            status.update(
                {
                    "status": "queued",
                    "phase": "Queued",
                    "page": None,
                    "message": (
                        "Queued to regenerate downloads "
                        f"(WebP quality {webp_quality}, JXL quality {jxl_quality})."
                    ),
                    "finishedAt": None,
                    "returnCode": None,
                    "packageRegenerationCount": package_count,
                    "pendingPackageOnly": True,
                    "pendingWebpQuality": webp_quality,
                    "pendingJxlQuality": jxl_quality,
                    "pendingResumeFrom": "package",
                    "pendingResumePage": 0,
                    "lastResumeFrom": "package",
                    "lastResumePage": 0,
                }
            )
            status.pop("pid", None)
            status.pop("pendingSinglePages", None)
            self.save_status(code, job_id, status)
            self.enqueue(code, job_id)

    def delete_job(self, code: str, job_id: str) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") in {"queued", "running", "paused", "terminating"}:
                raise HTTPException(status_code=409, detail="Job is queued or running.")
            if status.get("status") not in TERMINAL_JOB_STATUSES:
                raise HTTPException(status_code=400, detail="Only finished jobs can be deleted from the web UI.")
            shutil.rmtree(self.job_dir(code, job_id), ignore_errors=True)

    def worker_loop(self) -> None:
        while True:
            key = self._queue.get()
            if key is None:
                self._queue.task_done()
                return
            code, job_id = key
            if not self.wait_until_unpaused():
                self._queue.task_done()
                return
            with self._lock:
                self._queued_jobs.discard(key)
            try:
                self.run_job(code, job_id)
            except Exception as exc:
                print(
                    f"error: unexpected web worker failure for {code}/{job_id}: {exc}",
                    file=sys.stderr,
                )
                try:
                    status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
                    status.update(
                        {
                            "status": "failed",
                            "phase": "Failed",
                            "message": f"Unexpected worker failure: {exc}",
                            "finishedAt": now_utc(),
                        }
                    )
                    status.pop("pid", None)
                    self.save_status(code, job_id, status)
                except Exception as status_exc:
                    print(
                        f"error: could not persist worker failure for {code}/{job_id}: {status_exc}",
                        file=sys.stderr,
                    )
            finally:
                self._queue.task_done()

    def update_from_log_line(self, code: str, job_id: str, line: str) -> None:
        parsed = parse_progress_line(line)
        if parsed is None:
            return
        with self._lock:
            phase, page = parsed
            status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
            if status.get("pendingTermination"):
                return
            status["phase"] = phase
            status["page"] = page
            status["lastPhase"] = phase
            status["lastPage"] = page
            pending_single_pages = status.get("pendingSinglePages")
            if (
                isinstance(pending_single_pages, list)
                and pending_single_pages
                and isinstance(page, int)
            ):
                status["pendingResumePage"] = page
            target = progress_resume_target(phase, page)
            if target is not None:
                status["lastResumeFrom"] = target[0]
                status["lastResumePage"] = target[1]
            status["message"] = f"{phase}" + (f" page {page}" if page is not None else "")
            self.save_status(code, job_id, status)

    def build_command(
        self,
        code: str,
        job_id: str,
        resume_from: str | None = None,
        resume_page: int | None = None,
        single_page: bool = False,
        skip_package: bool = False,
        webp_quality: int | None = None,
        jxl_quality: int | None = None,
    ) -> list[str]:
        command = [
            sys.executable,
            str(self.config.translate_script),
            str(self.input_path(code, job_id)),
            str(self.output_dir(code, job_id)),
            "--config",
            str(self.config.pipeline_config),
        ]
        translation_notes_path = self.translation_notes_path(code, job_id)
        if translation_notes_path.is_file():
            command.extend(["--translation-notes-json", str(translation_notes_path)])
        status = self.load_status(code, job_id) or {}
        command.extend(
            [
                "--thinking-budget-tokens",
                str(self.status_thinking_budget_tokens(status)),
                "--vlm-base-url",
                self.status_vlm_base_url(status),
            ]
        )
        ocr_engine = self.status_ocr_engine(status)
        command.extend(["--source-language", self.status_source_language(status)])
        command.extend(["--ocr-engine", ocr_engine])
        if ocr_engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL:
            command.extend(
                [
                    "--paddleocr-vl-server-url",
                    self.status_paddleocr_vl_server_url(status),
                    "--paddleocr-vl-model",
                    self.status_paddleocr_vl_model(status),
                ]
            )
        command.append(
            "--proofread-translations"
            if self.status_proofread_translations(status)
            else "--no-proofread-translations"
        )
        command.append(
            "--write-translation-notes"
            if self.status_write_translation_notes(status)
            else "--no-write-translation-notes"
        )
        command.append(
            "--alt-placement"
            if self.status_alt_placement_enabled(status)
            else "--no-alt-placement"
        )
        if webp_quality is not None:
            command.extend(["--webp-quality", str(webp_quality)])
        if jxl_quality is not None:
            command.extend(["--jxl-quality", str(jxl_quality)])
        if resume_from is None:
            if self.status_pause_after_ocr(status):
                command.extend(["--stop-after", "ocr_raw"])
            command.append("--overwrite")
            return command

        command.extend(["--resume-from", resume_from])
        if resume_from not in {"extract", "package"}:
            command.extend(["--resume-page", str(resume_page or 0)])
        if single_page and resume_from in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements", "render"}:
            command.append("--single-page")
        if skip_package:
            command.append("--skip-package")
        return command

    def run_pipeline_process(
        self,
        code: str,
        job_id: str,
        command: list[str],
        label: str,
    ) -> tuple[int, float]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_path = self.log_path(code, job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        started_monotonic = time.monotonic()
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(f"$ {' '.join(command)}\n")
            popen_args: dict[str, Any] = {
                "cwd": REPO_DIR,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "env": env,
            }
            if os.name == "posix":
                popen_args["start_new_session"] = True
            try:
                process = subprocess.Popen(command, **popen_args)
            except OSError as exc:
                status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
                status.update(
                    {
                        "status": "failed",
                        "phase": "Failed",
                        "message": f"Could not start translation pipeline for {label}: {exc}",
                        "finishedAt": now_utc(),
                        "elapsedSeconds": round(time.monotonic() - started_monotonic, 3),
                    }
                )
                self.save_status(code, job_id, status)
                return 1, round(time.monotonic() - started_monotonic, 3)

            with self._lock:
                self._active_processes[(code, job_id)] = process
                should_pause = self._paused

            status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
            status["pid"] = process.pid
            self.save_status(code, job_id, status)
            if should_pause:
                stop_signal = getattr(signal, "SIGSTOP", None)
                signal_sent = (
                    self.signal_process_group_or_process(process, stop_signal)
                    if stop_signal is not None
                    else False
                )
                self.mark_active_job_paused(code, job_id, signal_sent)

            try:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    log_file.write(raw_line)
                    for line in raw_line.replace("\r", "\n").splitlines():
                        line = line.strip()
                        if line:
                            self.update_from_log_line(code, job_id, line)

                return_code = process.wait()
            finally:
                with self._lock:
                    self._active_processes.pop((code, job_id), None)

        return return_code, round(time.monotonic() - started_monotonic, 3)

    def run_package_regeneration_job(
        self,
        code: str,
        job_id: str,
        status: dict[str, Any],
    ) -> None:
        webp_quality = quality_for_display(
            status.get("pendingWebpQuality"),
            self.config.default_webp_quality,
        )
        jxl_quality = quality_for_display(
            status.get("pendingJxlQuality"),
            self.config.default_jxl_quality,
        )
        status.update(
            {
                "status": "running",
                "phase": "Starting",
                "page": None,
                "message": (
                    "Regenerating downloads "
                    f"(WebP quality {webp_quality}, JXL quality {jxl_quality})."
                ),
                "startedAt": now_utc(),
                "finishedAt": None,
                "pendingResumeFrom": "package",
                "pendingResumePage": 0,
                "lastResumeFrom": "package",
                "lastResumePage": 0,
            }
        )
        self.save_status(code, job_id, status)

        return_code, elapsed = self.run_pipeline_process(
            code,
            job_id,
            self.build_command(
                code,
                job_id,
                "package",
                0,
                webp_quality=webp_quality,
                jxl_quality=jxl_quality,
            ),
            "regenerate downloads",
        )

        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        status["returnCode"] = return_code
        status["finishedAt"] = now_utc()
        status["elapsedSeconds"] = elapsed
        status["webpQuality"] = webp_quality
        status["jxlQuality"] = jxl_quality
        status.pop("pid", None)
        status.pop("pendingPackageOnly", None)
        status.pop("pendingWebpQuality", None)
        status.pop("pendingJxlQuality", None)
        status.pop("pendingResumeFrom", None)
        status.pop("pendingResumePage", None)
        was_terminated = bool(status.pop("pendingTermination", None))
        status.pop("terminatingAt", None)
        status.pop("isPaused", None)
        status.pop("pausedAt", None)

        if was_terminated:
            status.update(
                {
                    "status": "cancelled",
                    "phase": "Cancelled",
                    "page": None,
                    "message": "Download regeneration terminated.",
                }
            )
        elif return_code == 0 and self.translated_cbz_path(code, job_id).is_file():
            status.update(
                {
                    "status": "complete",
                    "phase": "Complete",
                    "page": None,
                    "message": "Download regeneration complete.",
                }
            )
        else:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": f"Download regeneration failed with exit code {return_code}.",
                }
            )
        self.save_status(code, job_id, status)

    def run_single_page_batch_job(
        self,
        code: str,
        job_id: str,
        status: dict[str, Any],
        pending_single_pages: list[Any],
    ) -> None:
        pages: list[int] = []
        for item in pending_single_pages:
            try:
                page = int(item)
            except (TypeError, ValueError):
                continue
            if page >= 0:
                pages.append(page)
        pages = sorted(set(pages))
        if not pages:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": "No valid pages were queued for rerun.",
                    "finishedAt": now_utc(),
                }
            )
            status.pop("pendingSinglePages", None)
            self.save_status(code, job_id, status)
            return

        total_started_at = time.monotonic()
        final_return_code = 0
        resume_from = str(status.get("pendingSinglePageResumeFrom") or "ocr_raw")
        resume_from = RERUN_STAGE_MAP.get(resume_from, resume_from)
        if resume_from not in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements", "render"}:
            resume_from = "ocr_raw"
        package_webp_quality = quality_for_display(
            status.get("pendingWebpQuality"),
            quality_for_display(status.get("webpQuality"), self.config.default_webp_quality),
        )
        package_jxl_quality = quality_for_display(
            status.get("pendingJxlQuality"),
            quality_for_display(status.get("jxlQuality"), self.config.default_jxl_quality),
        )
        for index, page in enumerate(pages, start=1):
            locked_snapshot = self.snapshot_locked_stage_files(code, job_id, page)
            status = self.load_status(code, job_id) or status
            status.update(
                {
                    "status": "running",
                    "phase": "Starting",
                    "page": page,
                    "message": f"Rerunning page {page} from {resume_from} ({index}/{len(pages)}).",
                    "startedAt": now_utc(),
                    "finishedAt": None,
                    "pendingResumeFrom": resume_from,
                    "pendingResumePage": page,
                    "lastResumeFrom": resume_from,
                    "lastResumePage": page,
                }
            )
            self.save_status(code, job_id, status)
            final_return_code, _elapsed = self.run_pipeline_process(
                code,
                job_id,
                self.build_command(
                    code,
                    job_id,
                    resume_from,
                    page,
                    single_page=True,
                    skip_package=True,
                ),
                f"rerun page {page}",
            )
            if final_return_code == 0 and locked_snapshot:
                self.restore_locked_stage_files(code, job_id, page, locked_snapshot)
                for followup_stage in self.post_restore_rerun_sequence(
                    resume_from,
                    set(locked_snapshot),
                ):
                    self.restore_locked_stage_files(code, job_id, page, locked_snapshot)
                    final_return_code, _elapsed = self.run_pipeline_process(
                        code,
                        job_id,
                        self.build_command(
                            code,
                            job_id,
                            followup_stage,
                            page,
                            single_page=True,
                            skip_package=True,
                        ),
                        f"rerun page {page} from {followup_stage} after restoring locked edits",
                    )
                    self.restore_locked_stage_files(code, job_id, page, locked_snapshot)
                    if final_return_code != 0:
                        break
            if final_return_code != 0:
                break

        if final_return_code == 0:
            status = self.load_status(code, job_id) or status
            status.update(
                {
                    "status": "running",
                    "phase": "Package",
                    "page": None,
                    "message": "Packaging rerun output.",
                    "pendingResumeFrom": "package",
                    "pendingResumePage": 0,
                    "lastResumeFrom": "package",
                    "lastResumePage": 0,
                }
            )
            self.save_status(code, job_id, status)
            final_return_code, _elapsed = self.run_pipeline_process(
                code,
                job_id,
                self.build_command(
                    code,
                    job_id,
                    "package",
                    0,
                    webp_quality=package_webp_quality,
                    jxl_quality=package_jxl_quality,
                ),
                "package rerun output",
            )

        elapsed = round(time.monotonic() - total_started_at, 3)
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        status["returnCode"] = final_return_code
        status["finishedAt"] = now_utc()
        status["elapsedSeconds"] = elapsed
        status.pop("pid", None)
        status.pop("pendingResumeFrom", None)
        status.pop("pendingResumePage", None)
        status.pop("pendingSinglePages", None)
        status.pop("pendingSinglePageResumeFrom", None)
        status.pop("pendingWebpQuality", None)
        status.pop("pendingJxlQuality", None)
        pending_editor_changed_pages = status.pop("pendingEditorChangedPages", None)
        was_terminated = bool(status.pop("pendingTermination", None))
        status.pop("terminatingAt", None)
        status.pop("isPaused", None)
        status.pop("pausedAt", None)

        if was_terminated:
            status.update(
                {
                    "status": "cancelled",
                    "phase": "Cancelled",
                    "page": None,
                    "message": "Page rerun terminated.",
                }
            )
        elif final_return_code == 0 and self.translated_cbz_path(code, job_id).is_file():
            if isinstance(pending_editor_changed_pages, list):
                pages_to_clear: list[int] = []
                for item in pending_editor_changed_pages:
                    try:
                        pages_to_clear.append(int(item))
                    except (TypeError, ValueError):
                        continue
                self.clear_editor_changes_for_pages(code, job_id, pages_to_clear)
            status["webpQuality"] = package_webp_quality
            status["jxlQuality"] = package_jxl_quality
            status.update(
                {
                    "status": "complete",
                    "phase": "Complete",
                    "page": None,
                    "message": "Page rerun complete.",
                }
            )
        else:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": f"Page rerun failed with exit code {final_return_code}.",
                }
            )
        self.save_status(code, job_id, status)

    def run_job(self, code: str, job_id: str) -> None:
        status = self.load_status(code, job_id) or {
            "category": code,
            "jobId": job_id,
            "createdAt": now_utc(),
        }
        if status.get("pendingPackageOnly"):
            self.run_package_regeneration_job(code, job_id, status)
            return

        pending_single_pages = status.get("pendingSinglePages")
        if isinstance(pending_single_pages, list) and pending_single_pages:
            self.run_single_page_batch_job(code, job_id, status, pending_single_pages)
            return

        resume_from = status.get("pendingResumeFrom")
        resume_page = status.get("pendingResumePage")
        if not isinstance(resume_from, str):
            resume_from = None
            resume_page = None
        if resume_from is not None:
            try:
                resume_page = int(resume_page or 0)
            except (TypeError, ValueError):
                resume_page = 0
        status.update(
            {
                "status": "running",
                "phase": "Starting",
                "page": None,
                "message": (
                    f"Resuming translation pipeline from {resume_from} page {resume_page}."
                    if resume_from is not None
                    else "Starting translation pipeline."
                ),
                "startedAt": now_utc(),
                "finishedAt": None,
            }
        )
        if resume_from is not None:
            status["lastResumeFrom"] = resume_from
            status["lastResumePage"] = resume_page
        self.save_status(code, job_id, status)

        command = self.build_command(code, job_id, resume_from, resume_page)
        return_code, elapsed = self.run_pipeline_process(
            code,
            job_id,
            command,
            "translation pipeline",
        )
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        status["returnCode"] = return_code
        status["finishedAt"] = now_utc()
        status["elapsedSeconds"] = elapsed
        status.pop("pid", None)
        status.pop("pendingResumeFrom", None)
        status.pop("pendingResumePage", None)
        status.pop("pendingSinglePages", None)
        was_terminated = bool(status.pop("pendingTermination", None))
        status.pop("terminatingAt", None)
        status.pop("isPaused", None)
        status.pop("pausedAt", None)

        if was_terminated:
            status.update(
                {
                    "status": "cancelled",
                    "phase": "Cancelled",
                    "page": None,
                    "message": "Translation terminated.",
                }
            )
        elif return_code == 0 and self.status_pause_after_ocr(status) and not self.translated_cbz_path(code, job_id).is_file():
            status.update(
                {
                    "status": "paused",
                    "phase": "Paused",
                    "page": None,
                    "message": "Paused after OCR pass. Restart run to continue from structure.",
                    "isPaused": True,
                    "pausedAt": now_utc(),
                    "pendingResumeFrom": "ocr_structured",
                    "pendingResumePage": 0,
                    "lastResumeFrom": "ocr_structured",
                    "lastResumePage": 0,
                }
            )
        elif return_code == 0 and self.translated_cbz_path(code, job_id).is_file():
            status["webpQuality"] = self.config.default_webp_quality
            status["jxlQuality"] = self.config.default_jxl_quality
            status.update(
                {
                    "status": "complete",
                    "phase": "Complete",
                    "page": None,
                    "message": "Translation complete.",
                }
            )
        else:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": f"Translation failed with exit code {return_code}.",
                }
            )
        self.save_status(code, job_id, status)


def manager_from_request(request: Request) -> JobManager:
    manager = getattr(request.app.state, "manager", None)
    if not isinstance(manager, JobManager):
        raise HTTPException(status_code=503, detail="Web worker is not ready.")
    return manager


def admin_is_authenticated(request: Request, manager: JobManager) -> bool:
    token = request.cookies.get(ADMIN_COOKIE_NAME, "")
    return manager.admin_session_is_valid(token)


def require_admin(request: Request) -> JobManager:
    manager = manager_from_request(request)
    if not admin_is_authenticated(request, manager):
        raise HTTPException(status_code=401, detail="Admin login required.")
    return manager


def set_admin_session_cookie(
    response: RedirectResponse,
    request: Request,
    token: str,
) -> None:
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        max_age=ADMIN_SESSION_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )


def mutation_request_is_same_origin(request: Request) -> bool:
    fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
    if fetch_site == "cross-site":
        return False
    origin = request.headers.get("origin", "").strip()
    if not origin:
        return True
    try:
        supplied = http_origin(origin, "Origin header")
        expected = http_origin(str(request.base_url), "Request URL")
    except ValueError:
        return False
    return secrets.compare_digest(supplied, expected)


def base_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 56rem; line-height: 1.45; }}
    label {{ display: block; margin: 0 0 0.75rem; }}
    input, button, select, textarea {{ font: inherit; padding: 0.45rem; }}
    input[type="text"] {{ width: min(24rem, 100%); }}
    textarea {{ width: min(40rem, 100%); }}
    button {{ cursor: pointer; }}
    .row {{ margin: 1rem 0; }}
    .muted {{ color: #555; }}
    .status {{ font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 0.45rem; text-align: left; vertical-align: top; }}
    pre {{ background: #f5f5f5; padding: 1rem; overflow: auto; max-height: 24rem; }}
    a.button {{ display: inline-block; padding: 0.5rem 0.7rem; border: 1px solid #222; text-decoration: none; color: #111; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    )


def admin_login_page(message: str = "") -> HTMLResponse:
    message_html = f"<p>{escape(message)}</p>" if message else ""
    return base_page(
        "Admin Login",
        f"""
<h1>Admin</h1>
{message_html}
<form action="/admin/login" method="post">
  <label>Password<br><input name="password" type="password" required autocomplete="current-password"></label>
  <button type="submit">Log in</button>
</form>
""",
    )


def admin_state_label(status: dict[str, Any]) -> str:
    if status.get("paused"):
        return "Paused"
    if status.get("active"):
        return "Processing"
    if not status.get("workerRunning"):
        return "Stopped"
    if status.get("queuedCount"):
        return "Queued"
    return "Idle"


def admin_dashboard_page(manager: JobManager, message: str = "") -> HTMLResponse:
    status = manager.admin_status()
    state = admin_state_label(status)
    message_html = f'<p class="status">{escape(message)}</p>' if message else ""
    signal_note = (
        ""
        if status.get("posixSignals")
        else '<p class="muted">This platform can pause the queue, but may not suspend a running process.</p>'
    )
    active_rows: list[str] = []
    for item in status.get("active", []):
        page = "" if item.get("page") is None else str(item.get("page"))
        active_rows.append(
            "<tr>"
            f'<td><a href="{escape(item.get("url"))}">{escape(item.get("inputFilename") or item.get("jobId"))}</a></td>'
            f'<td>{escape(item.get("category"))}</td>'
            f'<td>{escape(item.get("status"))}</td>'
            f'<td>{escape(item.get("phase"))}</td>'
            f"<td>{escape(page)}</td>"
            "</tr>"
        )
    active_html = (
        """
<table>
  <thead><tr><th>Job</th><th>Category</th><th>Status</th><th>Phase</th><th>Page</th></tr></thead>
  <tbody>
"""
        + "\n".join(active_rows)
        + """
  </tbody>
</table>
"""
        if active_rows
        else '<p class="muted">No active process.</p>'
    )
    category_rows: list[str] = []
    for item in status.get("categories", []):
        category = str(item.get("category") or "")
        url = str(item.get("url") or f"/category/{category}")
        category_rows.append(
            "<tr>"
            f'<td><a href="{escape(url)}">{escape(category)}</a></td>'
            f'<td>{escape(item.get("jobCount", 0))}</td>'
            f'<td><a class="button" href="/admin/categories/{escape(category)}/delete">Delete...</a></td>'
            "</tr>"
        )
    categories_html = (
        """
<table>
  <thead><tr><th>Category</th><th>Jobs</th><th>Actions</th></tr></thead>
  <tbody>
"""
        + "\n".join(category_rows)
        + """
  </tbody>
</table>
"""
        if category_rows
        else '<p class="muted">No job categories. Create one to submit a job.</p>'
    )
    return base_page(
        "Admin",
        f"""
<h1>Admin</h1>
{message_html}
<dl>
  <dt>Worker State</dt><dd class="status">{escape(state)}</dd>
  <dt>Queued Jobs</dt><dd>{escape(status.get("queuedCount"))}</dd>
</dl>
<form action="/admin/pause" method="post">
  <button type="submit">Pause all jobs</button>
</form>
<form action="/admin/resume" method="post">
  <button type="submit">Resume all jobs</button>
</form>
<form action="/admin/logout" method="post">
  <button type="submit">Log out</button>
</form>
{signal_note}
<h2>Job Categories</h2>
<form action="/admin/categories" method="post">
  <label>New category<br><input name="category" type="text" required pattern="[A-Za-z0-9_-]{{1,64}}" maxlength="64" autocomplete="off"></label>
  <button type="submit">Create category</button>
</form>
{categories_html}
<h2>Active Job</h2>
{active_html}
<h2>Change Admin Password</h2>
<form action="/admin/password" method="post">
  <label>Current password<br><input name="current_password" type="password" required autocomplete="current-password"></label>
  <label>New password<br><input name="new_password" type="password" required autocomplete="new-password"></label>
  <label>Confirm new password<br><input name="confirm_password" type="password" required autocomplete="new-password"></label>
  <button type="submit">Change password</button>
</form>
""",
    )


def category_delete_page(category: str, counts: dict[str, int], message: str = "") -> HTMLResponse:
    total = sum(counts.values())
    count_rows = "".join(
        f"<li>{escape(status)}: {escape(count)}</li>"
        for status, count in sorted(counts.items())
    )
    message_html = f'<p class="status">{escape(message)}</p>' if message else ""
    return base_page(
        f"Delete {category}",
        f"""
<h1>Delete Category: {escape(category)}</h1>
{message_html}
<p><strong>{escape(CATEGORY_DELETE_CONFIRM)}</strong></p>
<p>This permanently removes {total} job(s), including every original upload, translated image, log, editor change, and download.</p>
<ul>{count_rows}</ul>
<form action="/admin/categories/{escape(category)}/delete" method="post">
  <label>Type <code>{escape(category)}</code> to confirm<br><input name="confirmation" type="text" required autocomplete="off"></label>
  <button type="submit">Delete category and all jobs</button>
</form>
<p><a href="/admin">Cancel</a></p>
""",
    )


def download_links_html(code: str, job_id: str, status: dict[str, Any]) -> str:
    downloads = status.get("downloads")
    if not isinstance(downloads, dict):
        downloads = {"png": status.get("hasDownload")}
    labels = {
        "png": "Download PNG CBZ",
        "webp": "Download WebP CBZ",
        "jxl": "Download JXL CBZ",
    }
    summary_parts: list[str] = []
    input_size = str(status.get("inputSize") or "")
    if input_size:
        summary_parts.append(f"Original: {input_size}")
    links: list[str] = []
    original_links: list[str] = []
    if status.get("canViewOriginal"):
        view_url = str(
            status.get("originalViewUrl") or f"/job/{code}/{job_id}/view-original"
        )
        page_count = status.get("originalPageCount")
        page_text = f" ({page_count} pages)" if page_count else ""
        original_links.append(
            f'<a class="button" href="{escape(view_url)}">View original{escape(page_text)}</a>'
        )
    if status.get("hasOriginalDownload"):
        href = str(
            status.get("originalDownloadUrl") or f"/job/{code}/{job_id}/download-original"
        )
        token = str(status.get("inputDownloadToken") or "")
        if token:
            href += f"?v={escape(token)}"
        suffix = f" ({input_size})" if input_size else ""
        original_links.append(
            f'<a class="button" href="{escape(href)}">Download original CBZ{escape(suffix)}</a>'
        )
    for variant, label in labels.items():
        item = downloads.get(variant)
        if isinstance(item, dict):
            available = bool(item.get("available"))
            size = str(item.get("size") or "")
            token = str(item.get("downloadToken") or "")
        else:
            available = bool(item)
            size = ""
            token = ""
        if not available:
            continue
        summary_label = label.removeprefix("Download ")
        summary_parts.append(f"{summary_label}: {size}" if size else summary_label)
        href = f"/job/{escape(code)}/{escape(job_id)}/download/{escape(variant)}"
        if token:
            href += f"?v={escape(token)}"
        suffix = f" ({size})" if size else ""
        links.append(f'<a class="button" href="{href}">{escape(label + suffix)}</a>')
    if status.get("canView"):
        view_url = str(status.get("viewUrl") or f"/job/{code}/{job_id}/view")
        page_count = status.get("finalPageCount")
        page_text = f" ({page_count} pages)" if page_count else ""
        links.insert(
            0,
            f'<a class="button" href="{escape(view_url)}">View in browser{escape(page_text)}</a>',
        )
    links = original_links + links
    summary_html = (
        f'<p class="muted">{escape("; ".join(summary_parts))}</p>'
        if summary_parts
        else ""
    )
    if not links:
        return summary_html
    return summary_html + "<p>" + " ".join(links) + "</p>"


def input_display_html(status: dict[str, Any]) -> str:
    filename = escape(status.get("inputFilename"))
    size = str(status.get("inputSize") or "")
    if not size:
        return filename
    return f"{filename} <span class=\"muted\">({escape(size)})</span>"


def generated_translation_notes_html(status: dict[str, Any]) -> str:
    notes = str(status.get("generatedTranslationNotes") or "")
    if not notes:
        return ""
    return f"""
<details>
  <summary>Translation notes</summary>
  <pre>{escape(notes)}</pre>
</details>
"""


def job_viewer_page(
    code: str,
    job_id: str,
    status: dict[str, Any],
    pages: list[dict[str, Any]],
    *,
    original: bool = False,
) -> HTMLResponse:
    view_route = "view-original" if original else "view"
    page_kind = "Original" if original else "Translated"
    page_items: list[str] = []
    for page in pages:
        index = int(page["index"])
        token = str(page.get("token") or "")
        src = f"/job/{escape(code)}/{escape(job_id)}/{view_route}/image/{index}"
        if token:
            src += f"?v={escape(token)}"
        page_items.append(
            f"""
<figure id="page-{index}">
  <figcaption>Page {index}</figcaption>
  <img src="{src}" alt="{page_kind} page {index}" loading="lazy" decoding="async">
</figure>
"""
        )
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(status.get("inputFilename") or job_id)} - {page_kind} Browser View</title>
  <style>
    body {{ margin: 0; background: #222; color: #eee; font-family: system-ui, sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 1; background: #111; border-bottom: 1px solid #444; padding: 0.7rem 1rem; }}
    header a {{ color: #fff; }}
    .meta {{ color: #bbb; margin-left: 0.75rem; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 1rem; }}
    figure {{ margin: 0 0 1rem; }}
    figcaption {{ color: #bbb; font-size: 0.9rem; margin: 0 0 0.35rem; }}
    img {{ display: block; max-width: 100%; height: auto; margin: 0 auto; background: #fff; }}
  </style>
</head>
<body>
  <header>
    <a href="/job/{escape(code)}/{escape(job_id)}">Back to job</a>
    <span class="meta">{page_kind} - {escape(status.get("inputFilename") or job_id)} - {len(pages)} pages</span>
  </header>
  <main>
    {"".join(page_items)}
  </main>
</body>
</html>""",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def source_language_options(selected: str) -> str:
    try:
        selected = translate_cbz.normalize_source_language(selected)
    except translate_cbz.PipelineError:
        selected = DEFAULT_SOURCE_LANGUAGE
    options: list[str] = []
    for code in ("jp", "kr", "cn"):
        profile = translate_cbz.SOURCE_LANGUAGE_PROFILES[code]
        selected_attr = " selected" if code == selected else ""
        options.append(
            f'<option value="{escape(code)}"{selected_attr}>{escape(profile.name)}</option>'
        )
    return "\n".join(options)


def advanced_options_fields(
    thinking_budget_tokens: Any,
    vlm_base_url: str,
    summary: str = "Advanced options",
    open_details: bool = False,
    include_translation_notes: bool = False,
    translation_notes: str = "",
    pause_after_ocr: bool = False,
    proofread_translations: bool = DEFAULT_PROOFREAD_TRANSLATIONS,
    write_translation_notes: bool = DEFAULT_WRITE_TRANSLATION_NOTES,
    alt_placement_enabled: bool = DEFAULT_ALT_PLACEMENT_ENABLED,
    source_language: str = DEFAULT_SOURCE_LANGUAGE,
    ocr_engine: str = DEFAULT_OCR_ENGINE,
    paddleocr_vl_server_url: str = DEFAULT_PADDLEOCR_VL_SERVER_URL,
    paddleocr_vl_model: str = DEFAULT_PADDLEOCR_VL_MODEL,
) -> str:
    try:
        budget = validate_thinking_budget_tokens(thinking_budget_tokens)
    except ValueError:
        budget = DEFAULT_THINKING_BUDGET_TOKENS
    try:
        selected_ocr_engine = paddle_ocr_image.normalize_ocr_engine(ocr_engine)
    except paddle_ocr_image.InputError:
        selected_ocr_engine = DEFAULT_OCR_ENGINE
    paddle_selected = " selected" if selected_ocr_engine == paddle_ocr_image.OCR_ENGINE_PADDLE else ""
    vl_selected = " selected" if selected_ocr_engine == paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL else ""
    open_attr = " open" if open_details else ""
    pause_checked = " checked" if pause_after_ocr else ""
    proofread_checked = " checked" if proofread_translations else ""
    write_notes_checked = " checked" if write_translation_notes else ""
    alt_placement_checked = " checked" if alt_placement_enabled else ""
    translation_notes_html = (
        '<label>Translation notes<br><textarea name="translation_notes" rows="4" '
        f'placeholder="Names, terms, tone, style guidance">{escape(translation_notes)}</textarea></label>'
        if include_translation_notes
        else ""
    )
    return f"""
<details{open_attr}>
  <summary>{escape(summary)}</summary>
  {translation_notes_html}
  <label><input name="pause_after_ocr" type="checkbox" value="1"{pause_checked}> Pause after OCR pass</label>
  <label><input name="enable_alt_placement" type="checkbox" value="1"{alt_placement_checked}> Enable alt-placement</label>
  <label><input name="enable_proofreading" type="checkbox" value="1"{proofread_checked}> Enable proofreading</label>
  <label><input name="enable_translation_notes" type="checkbox" value="1"{write_notes_checked}> Enable translation notes</label>
  <label>Source language<br><select name="source_language">
    {source_language_options(source_language)}
  </select></label>
  <label>OCR engine<br><select name="ocr_engine">
    <option value="{paddle_ocr_image.OCR_ENGINE_PADDLE}"{paddle_selected}>PaddleOCR</option>
    <option value="{paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL}"{vl_selected}>PaddleOCR-VL 1.6</option>
  </select></label>
  <label>PaddleOCR-VL server URL<br><input name="paddleocr_vl_server_url" type="text" value="{escape(paddleocr_vl_server_url)}"></label>
  <label>PaddleOCR-VL model<br><input name="paddleocr_vl_model" type="text" value="{escape(paddleocr_vl_model)}"></label>
  <label>Translation VLM endpoint<br><input name="vlm_base_url" type="url" value="{escape(vlm_base_url)}" required></label>
  <label>No. of thinking tokens<br><input name="thinking_budget_tokens" type="number" step="1" value="{escape(budget)}"></label>
  <p class="muted">Use 0 to disable thinking where supported. Use a negative value for unlimited/server-defined thinking.</p>
</details>
"""


def category_jobs_page(code: str, data: dict[str, Any]) -> HTMLResponse:
    jobs = data.get("jobs", [])
    translation_notes = str(data.get("translationNotes") or "")
    default_thinking_budget_tokens = data.get(
        "defaultThinkingBudgetTokens",
        DEFAULT_THINKING_BUDGET_TOKENS,
    )
    default_vlm_base_url = str(data.get("defaultVlmBaseUrl") or "")
    pause_after_ocr = bool(data.get("pauseAfterOcr", False))
    proofread_translations = bool(
        data.get("proofreadTranslations", DEFAULT_PROOFREAD_TRANSLATIONS)
    )
    write_translation_notes = bool(
        data.get("writeTranslationNotes", DEFAULT_WRITE_TRANSLATION_NOTES)
    )
    default_alt_placement_enabled = bool(
        data.get("defaultAltPlacementEnabled", DEFAULT_ALT_PLACEMENT_ENABLED)
    )
    default_source_language = str(data.get("defaultSourceLanguage") or DEFAULT_SOURCE_LANGUAGE)
    default_ocr_engine = str(data.get("defaultOcrEngine") or DEFAULT_OCR_ENGINE)
    default_paddleocr_vl_server_url = str(
        data.get("defaultPaddleocrVlServerUrl") or DEFAULT_PADDLEOCR_VL_SERVER_URL
    )
    default_paddleocr_vl_model = str(
        data.get("defaultPaddleocrVlModel") or DEFAULT_PADDLEOCR_VL_MODEL
    )
    rows: list[str] = []
    for job in jobs:
        job_id = str(job.get("jobId", ""))
        filename = str(job.get("inputFilename") or job_id)
        page = "" if job.get("page") is None else str(job.get("page"))
        view_html = (
            f'<a class="button" href="/job/{escape(code)}/{escape(job_id)}/view">View</a>'
            if job.get("canView")
            else ""
        )
        delete_html = (
            f'<form action="/job/{escape(code)}/{escape(job_id)}/delete" method="post" '
            f"onsubmit=\"return confirm({escape(json.dumps(DELETE_JOB_CONFIRM))});\">"
            '<button type="submit">Delete</button></form>'
            if job.get("canDelete")
            else ""
        )
        actions_html = " ".join(item for item in (view_html, delete_html) if item)
        rows.append(
            "<tr>"
            f'<td><a href="/job/{escape(code)}/{escape(job_id)}">{escape(filename)}</a><br>'
            f'<span class="muted">{escape(job_id)}</span></td>'
            f'<td>{escape(job.get("status"))}</td>'
            f'<td>{escape(job.get("phase"))}</td>'
            f"<td>{escape(page)}</td>"
            f'<td>{escape(job.get("age"))}</td>'
            f'<td>{escape(job.get("elapsed"))}</td>'
            f"<td>{actions_html}</td>"
            "</tr>"
        )
    jobs_html = (
        """
<table>
  <thead><tr><th>Job</th><th>Status</th><th>Phase</th><th>Page</th><th>Age</th><th>Elapsed</th><th>Actions</th></tr></thead>
  <tbody>
"""
        + "\n".join(rows)
        + """
  </tbody>
</table>
"""
        if rows
        else '<p class="muted">No jobs have been submitted in this category.</p>'
    )
    return base_page(
        f"Category {code}",
        f"""
<h1>Category: {escape(code)}</h1>
<p><a href="/admin">Admin</a></p>
<form action="/upload" method="post" enctype="multipart/form-data">
  <input name="category" type="hidden" value="{escape(code)}">
  <label>CBZ file<br><input name="cbz" type="file" accept=".cbz,application/zip"></label>
  <label>Page image files<br><input name="page_images" type="file" accept="{UPLOAD_PAGE_IMAGE_ACCEPT}" multiple></label>
  <p class="muted">Upload either one CBZ or multiple image pages. Images use the picker/upload order and are converted to PNG internally.</p>
  {advanced_options_fields(
      default_thinking_budget_tokens,
      default_vlm_base_url,
      include_translation_notes=True,
      translation_notes=translation_notes,
      pause_after_ocr=pause_after_ocr,
      proofread_translations=proofread_translations,
      write_translation_notes=write_translation_notes,
      alt_placement_enabled=default_alt_placement_enabled,
      source_language=default_source_language,
      ocr_engine=default_ocr_engine,
      paddleocr_vl_server_url=default_paddleocr_vl_server_url,
      paddleocr_vl_model=default_paddleocr_vl_model,
  )}
  <button type="submit">Queue new job</button>
</form>
<h2>Jobs</h2>
{jobs_html}
""",
    )


def job_page(code: str, job_id: str, status: dict[str, Any]) -> HTMLResponse:
    log_text = "\n".join(status.get("recentLog", []))
    page_value = status.get("page")
    page_text = "" if page_value is None else str(page_value)
    restart_from = status.get("restartResumeFrom")
    restart_page = status.get("restartResumePage")
    if restart_from is None:
        restart_target_text = "from the beginning"
    elif restart_from == "package":
        restart_target_text = "from package"
    else:
        restart_target_text = f"from {restart_from} page {restart_page}"
    download_html = download_links_html(code, job_id, status)
    webp_quality = status.get("webpQuality") or status.get("defaultWebpQuality") or DEFAULT_WEBP_QUALITY
    jxl_quality = status.get("jxlQuality") or status.get("defaultJxlQuality") or DEFAULT_JXL_QUALITY
    thinking_budget = status.get("thinkingBudgetTokens")
    if thinking_budget is None:
        thinking_budget = status.get(
            "defaultThinkingBudgetTokens",
            DEFAULT_THINKING_BUDGET_TOKENS,
        )
    vlm_base_url = str(
        status.get("vlmBaseUrl")
        or status.get("defaultVlmBaseUrl")
        or ""
    )
    pause_after_ocr = bool(status.get("pauseAfterOcr"))
    proofread_translations = bool(
        status.get("proofreadTranslations", DEFAULT_PROOFREAD_TRANSLATIONS)
    )
    write_translation_notes = bool(
        status.get("writeTranslationNotes", DEFAULT_WRITE_TRANSLATION_NOTES)
    )
    alt_placement_enabled = bool(
        status.get("altPlacementEnabled", status.get("defaultAltPlacementEnabled", DEFAULT_ALT_PLACEMENT_ENABLED))
    )
    source_language = str(
        status.get("sourceLanguage")
        or status.get("defaultSourceLanguage")
        or DEFAULT_SOURCE_LANGUAGE
    )
    ocr_engine = str(status.get("ocrEngine") or status.get("defaultOcrEngine") or DEFAULT_OCR_ENGINE)
    paddleocr_vl_server_url = str(
        status.get("paddleocrVlServerUrl")
        or status.get("defaultPaddleocrVlServerUrl")
        or DEFAULT_PADDLEOCR_VL_SERVER_URL
    )
    paddleocr_vl_model = str(
        status.get("paddleocrVlModel")
        or status.get("defaultPaddleocrVlModel")
        or DEFAULT_PADDLEOCR_VL_MODEL
    )
    advanced_options_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/advanced-options" method="post">
  {advanced_options_fields(
      thinking_budget,
      vlm_base_url,
      open_details=True,
      pause_after_ocr=pause_after_ocr,
      proofread_translations=proofread_translations,
      write_translation_notes=write_translation_notes,
      alt_placement_enabled=alt_placement_enabled,
      source_language=source_language,
      ocr_engine=ocr_engine,
      paddleocr_vl_server_url=paddleocr_vl_server_url,
      paddleocr_vl_model=paddleocr_vl_model,
  )}
  <button type="submit">Save advanced options</button>
</form>
"""
        if status.get("canUpdateAdvancedOptions")
        else ""
    )
    restart_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/restart" method="post">
  <button type="submit">Restart run</button>
  <span class="muted">Resume {escape(restart_target_text)}</span>
</form>
"""
        if status.get("canRestart")
        else ""
    )
    terminate_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/terminate" method="post" onsubmit="return confirm({escape(json.dumps(TERMINATE_JOB_CONFIRM))});">
  <button type="submit">Terminate job</button>
</form>
"""
        if status.get("canTerminate")
        else ""
    )
    delete_html = (
        f"""
<form action="/job/{escape(code)}/{escape(job_id)}/delete" method="post" onsubmit="return confirm({escape(json.dumps(DELETE_JOB_CONFIRM))});">
  <button type="submit">Delete job</button>
</form>
"""
        if status.get("canDelete")
        else ""
    )
    rerun_job_html = (
        f"""
<details>
  <summary>Rerun job</summary>
  <form action="/job/{escape(code)}/{escape(job_id)}/rerun-job" method="post">
    <fieldset>
      <legend>Passes</legend>
      <label><input name="rerun_stage" type="checkbox" value="{OCR_MERGE_EDITOR_STAGE}"> OCR merge</label>
      <label><input name="rerun_stage" type="checkbox" value="ocr_structured"> Structured</label>
      <label><input name="rerun_stage" type="checkbox" value="alt_placement"> Alt-placement</label>
      <label><input name="rerun_stage" type="checkbox" value="translations"> Translations</label>
      <label><input name="rerun_stage" type="checkbox" value="placements"> Placements</label>
      <label><input name="rerun_stage" type="checkbox" value="render"> Rendering</label>
      <label><input name="rerun_stage" type="checkbox" value="{RERUN_JOB_PACKAGE_STAGE}"> Regeneration</label>
    </fieldset>
    <label>Pages<br><input name="page_spec" type="text" placeholder="0-3,6,8"></label>
    <p class="muted">Leave pages empty to rerun every page. Regeneration alone rebuilds download archives.</p>
    <label>WebP quality<br><input name="webp_quality" type="number" min="1" max="100" value="{escape(webp_quality)}"></label>
    <label>JXL quality<br><input name="jxl_quality" type="number" min="1" max="100" value="{escape(jxl_quality)}"></label>
    <button type="submit">Queue rerun</button>
  </form>
</details>
"""
        if status.get("canRerunPages") or status.get("canRegenerateDownloads")
        else ""
    )
    edit_html = (
        f'<p><a class="button" href="/job/{escape(code)}/{escape(job_id)}/edit">Edit job</a></p>'
        if status.get("canEdit")
        else ""
    )
    generated_notes_html = generated_translation_notes_html(status)
    return base_page(
        f"Job {code} {job_id}",
        f"""
<h1>Job {escape(job_id)}</h1>
<p><a href="/category/{escape(code)}">Back to category {escape(code)}</a></p>
<dl>
  <dt>Category</dt><dd>{escape(code)}</dd>
  <dt>Input</dt><dd>{input_display_html(status)}</dd>
  <dt>Status</dt><dd id="status" class="status">{escape(status.get("status"))}</dd>
  <dt>Age</dt><dd id="age">{escape(status.get("age"))}</dd>
  <dt>Phase</dt><dd id="phase">{escape(status.get("phase"))}</dd>
  <dt>Page</dt><dd id="page">{escape(page_text)}</dd>
  <dt>Elapsed</dt><dd id="elapsed">{escape(status.get("elapsed"))}</dd>
  <dt>Thinking Tokens</dt><dd id="thinking-budget">{escape(status.get("thinkingBudget"))}</dd>
  <dt>Message</dt><dd id="message">{escape(status.get("message"))}</dd>
</dl>
<div id="advanced-options">{advanced_options_html}</div>
<div id="restart">{restart_html}</div>
<div id="terminate">{terminate_html}</div>
<div id="edit">{edit_html}</div>
<div id="download">{download_html}</div>
<div id="generated-translation-notes">{generated_notes_html}</div>
<div id="rerun-job">{rerun_job_html}</div>
<div id="delete">{delete_html}</div>
<h2>Recent Log</h2>
<pre id="log">{escape(log_text)}</pre>
<script>
const code = {json.dumps(code)};
const jobId = {json.dumps(job_id)};
const deleteJobConfirm = {json.dumps(DELETE_JOB_CONFIRM)};
const terminateJobConfirm = {json.dumps(TERMINATE_JOB_CONFIRM)};
function htmlEscape(value) {{
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }}[char]));
}}
function restartMarkup(data) {{
  if (!data.canRestart) return "";
  let target = "from the beginning";
  if (data.restartResumeFrom === "package") {{
    target = "from package";
  }} else if (data.restartResumeFrom) {{
    target = `from ${{data.restartResumeFrom}} page ${{data.restartResumePage ?? 0}}`;
  }}
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/restart" method="post"><button type="submit">Restart run</button> <span class="muted">Resume ${{target}}</span></form>`;
}}
function downloadMarkup(data) {{
  const labels = {{png: "Download PNG CBZ", webp: "Download WebP CBZ", jxl: "Download JXL CBZ"}};
  const downloads = data.downloads || {{}};
  const summary = [];
  if (data.inputSize) {{
    summary.push(`Original: ${{data.inputSize}}`);
  }}
  const links = Object.keys(labels)
    .filter((variant) => {{
      const item = downloads[variant];
      return item && (typeof item !== "object" || item.available);
    }})
    .map((variant) => {{
      const item = downloads[variant];
      const size = item && typeof item === "object" && item.size ? ` (${{item.size}})` : "";
      const summaryLabel = labels[variant].replace("Download ", "");
      summary.push(size ? `${{summaryLabel}}: ${{item.size}}` : summaryLabel);
      const token = item && typeof item === "object" && item.downloadToken ? `?v=${{encodeURIComponent(item.downloadToken)}}` : "";
      const href = `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/download/${{variant}}${{token}}`;
      return `<a class="button" href="${{href}}">${{labels[variant]}}${{size}}</a>`;
    }});
  if (data.canView) {{
    const viewUrl = data.viewUrl || `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/view`;
    const pageCount = data.finalPageCount ? ` (${{data.finalPageCount}} pages)` : "";
    links.unshift(`<a class="button" href="${{viewUrl}}">View in browser${{pageCount}}</a>`);
  }}
  if (data.hasOriginalDownload) {{
    const inputSize = data.inputSize ? ` (${{data.inputSize}})` : "";
    const token = data.inputDownloadToken ? `?v=${{encodeURIComponent(data.inputDownloadToken)}}` : "";
    const url = data.originalDownloadUrl || `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/download-original`;
    links.unshift(`<a class="button" href="${{url}}${{token}}">Download original CBZ${{inputSize}}</a>`);
  }}
  if (data.canViewOriginal) {{
    const viewUrl = data.originalViewUrl || `/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/view-original`;
    const pageCount = data.originalPageCount ? ` (${{data.originalPageCount}} pages)` : "";
    links.unshift(`<a class="button" href="${{viewUrl}}">View original${{pageCount}}</a>`);
  }}
  const summaryHtml = summary.length ? `<p class="muted">${{summary.join("; ")}}</p>` : "";
  const linksHtml = links.length ? `<p>${{links.join(" ")}}</p>` : "";
  return summaryHtml + linksHtml;
}}
function deleteMarkup(data) {{
  if (!data.canDelete) return "";
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/delete" method="post" onsubmit="return confirm(${{JSON.stringify(deleteJobConfirm)}});"><button type="submit">Delete job</button></form>`;
}}
function terminateMarkup(data) {{
  if (!data.canTerminate) return "";
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/terminate" method="post" onsubmit="return confirm(${{JSON.stringify(terminateJobConfirm)}});"><button type="submit">Terminate job</button></form>`;
}}
function advancedOptionsMarkup(data) {{
  if (!data.canUpdateAdvancedOptions) return "";
  const thinkingBudget = data.thinkingBudgetTokens ?? data.defaultThinkingBudgetTokens ?? 2048;
  const vlmBaseUrl = htmlEscape(data.vlmBaseUrl || data.defaultVlmBaseUrl || "");
  const pauseChecked = data.pauseAfterOcr ? " checked" : "";
  const proofreadChecked = data.proofreadTranslations ? " checked" : "";
  const notesChecked = data.writeTranslationNotes ? " checked" : "";
  const altPlacementChecked = (data.altPlacementEnabled ?? data.defaultAltPlacementEnabled ?? true) ? " checked" : "";
  const sourceLanguage = data.sourceLanguage || data.defaultSourceLanguage || "jp";
  const jpSelected = sourceLanguage === "jp" ? " selected" : "";
  const krSelected = sourceLanguage === "kr" ? " selected" : "";
  const cnSelected = sourceLanguage === "cn" ? " selected" : "";
  const ocrEngine = data.ocrEngine || data.defaultOcrEngine || "paddle";
  const paddleSelected = ocrEngine === "paddle" ? " selected" : "";
  const vlSelected = ocrEngine === "paddleocr_vl" ? " selected" : "";
  const vlServerUrl = htmlEscape(data.paddleocrVlServerUrl || data.defaultPaddleocrVlServerUrl || "http://127.0.0.1:8081/v1");
  const vlModel = htmlEscape(data.paddleocrVlModel || data.defaultPaddleocrVlModel || "PaddlePaddle/PaddleOCR-VL-1.6");
  return `<form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/advanced-options" method="post"><details open><summary>Advanced options</summary><label><input name="pause_after_ocr" type="checkbox" value="1"${{pauseChecked}}> Pause after OCR pass</label><label><input name="enable_alt_placement" type="checkbox" value="1"${{altPlacementChecked}}> Enable alt-placement</label><label><input name="enable_proofreading" type="checkbox" value="1"${{proofreadChecked}}> Enable proofreading</label><label><input name="enable_translation_notes" type="checkbox" value="1"${{notesChecked}}> Enable translation notes</label><label>Source language<br><select name="source_language"><option value="jp"${{jpSelected}}>Japanese</option><option value="kr"${{krSelected}}>Korean</option><option value="cn"${{cnSelected}}>Chinese</option></select></label><label>OCR engine<br><select name="ocr_engine"><option value="paddle"${{paddleSelected}}>PaddleOCR</option><option value="paddleocr_vl"${{vlSelected}}>PaddleOCR-VL 1.6</option></select></label><label>PaddleOCR-VL server URL<br><input name="paddleocr_vl_server_url" type="text" value="${{vlServerUrl}}"></label><label>PaddleOCR-VL model<br><input name="paddleocr_vl_model" type="text" value="${{vlModel}}"></label><label>Translation VLM endpoint<br><input name="vlm_base_url" type="url" value="${{vlmBaseUrl}}" required></label><label>No. of thinking tokens<br><input name="thinking_budget_tokens" type="number" step="1" value="${{thinkingBudget}}"></label><p class="muted">Use 0 to disable thinking where supported. Use a negative value for unlimited/server-defined thinking.</p></details><button type="submit">Save advanced options</button></form>`;
}}
function rerunJobMarkup(data) {{
  if (!data.canRerunPages && !data.canRegenerateDownloads) return "";
  const webpQuality = data.webpQuality ?? data.defaultWebpQuality ?? 90;
  const jxlQuality = data.jxlQuality ?? data.defaultJxlQuality ?? 90;
  return `<details><summary>Rerun job</summary><form action="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/rerun-job" method="post"><fieldset><legend>Passes</legend><label><input name="rerun_stage" type="checkbox" value="ocr_merge"> OCR merge</label><label><input name="rerun_stage" type="checkbox" value="ocr_structured"> Structured</label><label><input name="rerun_stage" type="checkbox" value="alt_placement"> Alt-placement</label><label><input name="rerun_stage" type="checkbox" value="translations"> Translations</label><label><input name="rerun_stage" type="checkbox" value="placements"> Placements</label><label><input name="rerun_stage" type="checkbox" value="render"> Rendering</label><label><input name="rerun_stage" type="checkbox" value="package"> Regeneration</label></fieldset><label>Pages<br><input name="page_spec" type="text" placeholder="0-3,6,8"></label><p class="muted">Leave pages empty to rerun every page. Regeneration alone rebuilds download archives.</p><label>WebP quality<br><input name="webp_quality" type="number" min="1" max="100" value="${{webpQuality}}"></label><label>JXL quality<br><input name="jxl_quality" type="number" min="1" max="100" value="${{jxlQuality}}"></label><button type="submit">Queue rerun</button></form></details>`;
}}
function editMarkup(data) {{
  if (!data.canEdit) return "";
  return `<p><a class="button" href="/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}/edit">Edit job</a></p>`;
}}
function generatedTranslationNotesMarkup(data) {{
  if (!data.generatedTranslationNotes) return "";
  return `<details><summary>Translation notes</summary><pre>${{htmlEscape(data.generatedTranslationNotes)}}</pre></details>`;
}}
async function refreshStatus() {{
  const response = await fetch(`/api/job/${{encodeURIComponent(code)}}/${{encodeURIComponent(jobId)}}`, {{cache: "no-store"}});
  if (!response.ok) return;
  const data = await response.json();
  const previousStatus = document.getElementById("status").textContent;
  document.getElementById("status").textContent = data.status || "";
  document.getElementById("age").textContent = data.age || "";
  document.getElementById("phase").textContent = data.phase || "";
  document.getElementById("page").textContent = data.page ?? "";
  document.getElementById("elapsed").textContent = data.elapsed || "";
  document.getElementById("thinking-budget").textContent = data.thinkingBudget || "";
  document.getElementById("message").textContent = data.message || "";
  document.getElementById("log").textContent = (data.recentLog || []).join("\\n");
  document.getElementById("download").innerHTML = downloadMarkup(data);
  document.getElementById("generated-translation-notes").innerHTML = generatedTranslationNotesMarkup(data);
  if (!(previousStatus === "complete" && data.status === "complete")) {{
    document.getElementById("advanced-options").innerHTML = advancedOptionsMarkup(data);
    document.getElementById("restart").innerHTML = restartMarkup(data);
    document.getElementById("terminate").innerHTML = terminateMarkup(data);
    document.getElementById("edit").innerHTML = editMarkup(data);
    document.getElementById("rerun-job").innerHTML = rerunJobMarkup(data);
    document.getElementById("delete").innerHTML = deleteMarkup(data);
  }}
  if (data.status === "complete" || data.status === "failed" || data.status === "cancelled" || (data.status === "paused" && data.canRestart)) {{
    window.clearInterval(statusTimer);
  }}
}}
const statusTimer = window.setInterval(refreshStatus, 5000);
</script>
""",
    )


def editor_page(manager: JobManager, code: str, job_id: str) -> HTMLResponse:
    status = manager.require_editable_job(code, job_id)
    page_count = len(manager.original_page_files(code, job_id))
    page_options = "\n".join(
        f'<option value="{index}">Page {index}</option>' for index in range(page_count)
    )
    stage_options = "\n".join(
        f'<option value="{stage}">{label}</option>'
        for stage, label in (
            (OCR_MERGE_EDITOR_STAGE, "OCR merge"),
            ("ocr_structured", "Structured"),
            ("translations", "Translations"),
            ("placements", "Placements"),
        )
    )
    return base_page(
        f"Edit {code} {job_id}",
        f"""
<style>
  body {{ max-width: none; }}
  .editor {{ display: grid; grid-template-columns: minmax(0, 1fr) 22rem; gap: 1rem; align-items: start; }}
  .toolbar {{ display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: end; margin: 1rem 0; }}
  .canvas-wrap {{ border: 1px solid #ccc; overflow: auto; max-height: 82vh; background: #eee; display: flex; justify-content: center; align-items: flex-start; }}
  canvas {{ display: block; max-width: 100%; max-height: 78vh; width: auto; height: auto; background: white; }}
  .panel {{ border-left: 1px solid #ddd; padding-left: 1rem; }}
  .panel input[type="number"], .panel input[type="text"] {{ width: 100%; box-sizing: border-box; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }}
  .selected-list {{ max-height: 8rem; overflow: auto; font-size: 0.9rem; }}
  #record-json {{ width: 100%; box-sizing: border-box; font-family: ui-monospace, monospace; }}
  .ocr-merge-only {{ display: none; }}
  body.ocr-merge-stage .ocr-merge-only {{ display: inline-block; }}
  .warn {{ color: #8a4b00; }}
</style>
<h1>Edit Job {escape(job_id)}</h1>
<p><a href="/job/{escape(code)}/{escape(job_id)}">Back to job</a></p>
<p class="muted">Input: {input_display_html(status)}</p>
<div class="toolbar">
  <label>Page ([ / ])<br><select id="page-select">{page_options}</select></label>
  <button id="prev-page-btn" type="button">Previous page ([)</button>
  <button id="next-page-btn" type="button">Next page (])</button>
  <label>Stage (, / .)<br><select id="stage-select">{stage_options}</select></label>
  <label><input id="lock-stage" type="checkbox"> Lock Stage &amp; Page (K)</label>
  <button id="load-btn" type="button">Load (L)</button>
  <button id="save-btn" type="button">Save (Ctrl+S)</button>
  <button id="draw-btn" type="button">Draw box (D)</button>
  <button id="delete-btn" type="button">Delete selected (Del)</button>
  <button id="merge-btn" type="button">Merge selected (M)</button>
  <button id="unmerge-btn" class="ocr-merge-only" type="button">Unmerge selected group (U)</button>
  <label class="ocr-merge-only"><input id="show-raw" type="checkbox" checked> Show raw (1)</label>
  <label class="ocr-merge-only"><input id="show-merged" type="checkbox" checked> Show merged (2)</label>
  <button id="ocr-crop-btn" type="button">OCR new area (O)</button>
  <button id="regen-btn" type="button">Regenerate this page (G)</button>
  <button id="regen-changed-btn" type="button">Regenerate changed pages (R)</button>
</div>
<div class="editor">
  <div>
    <p id="outdated-notice" class="muted"></p>
    <div class="canvas-wrap"><canvas id="page-canvas"></canvas></div>
    <p id="editor-status" class="muted"></p>
  </div>
  <aside class="panel">
    <h2>Selected Record</h2>
    <p class="muted">Click a box to select. Repeated clicks in the same spot cycle through stacked merged and raw OCR boxes. Shift-click selects multiple boxes. Drag the selected box to move it; drag its lower-right handle to resize.</p>
    <div class="selected-list" id="selected-list"></div>
    <div class="grid2">
      <label>Left<input id="field-left" type="number"></label>
      <label>Top<input id="field-top" type="number"></label>
      <label>Right<input id="field-right" type="number"></label>
      <label>Bottom<input id="field-bottom" type="number"></label>
    </div>
    <label>Source/OCR text<input id="field-text" type="text"></label>
    <label>Translated text<input id="field-english" type="text"></label>
    <label><input id="field-sfx" type="checkbox"> SFX</label>
    <label><input id="field-open" type="checkbox"> Open lettering</label>
    <div class="grid2">
      <label>Fill<input id="field-fill" type="text" placeholder="black or white"></label>
      <label>Font<input id="field-font" type="text" placeholder="font filename"></label>
    </div>
    <label>Raw record JSON<textarea id="record-json" rows="12"></textarea></label>
    <button id="apply-record-btn" type="button">Apply selected form (A)</button>
    <h2>Translation Notes</h2>
    <label>Job notes<textarea id="job-notes" rows="5"></textarea></label>
    <label>Page notes<textarea id="page-notes" rows="5"></textarea></label>
  </aside>
</div>
<script>
window.TETOLATE_EDITOR = {{
  category: {json.dumps(code)},
  jobId: {json.dumps(job_id)},
  pageCount: {page_count},
}};
</script>
<script src="/assets/editor.js?v=ocr-merge-6"></script>
""",
    )


def create_app(config_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = load_web_config(config_path)
        manager = JobManager(config)
        app.state.manager = manager
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def require_admin_session(request: Request, call_next: Any) -> Any:
        public_login = (
            request.url.path == "/admin" and request.method == "GET"
        ) or (
            request.url.path == "/admin/login" and request.method == "POST"
        )
        if not public_login:
            manager = getattr(request.app.state, "manager", None)
            authenticated = isinstance(manager, JobManager) and admin_is_authenticated(
                request,
                manager,
            )
            if not authenticated:
                if request.url.path.startswith("/api/"):
                    return JSONResponse(
                        {"detail": "Admin login required."},
                        status_code=401,
                        headers={"Cache-Control": "no-store"},
                    )
                return RedirectResponse("/admin", status_code=303)
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not mutation_request_is_same_origin(
                request
            ):
                if request.url.path.startswith("/api/"):
                    return JSONResponse(
                        {"detail": "Cross-origin mutation rejected."},
                        status_code=403,
                        headers={"Cache-Control": "no-store"},
                    )
                return HTMLResponse(
                    "Cross-origin mutation rejected.",
                    status_code=403,
                    headers={"Cache-Control": "no-store"},
                )
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/")
    async def index() -> RedirectResponse:
        return RedirectResponse("/admin", status_code=303)

    @app.get("/assets/editor.js")
    async def editor_asset() -> FileResponse:
        return FileResponse(REPO_DIR / "web_editor.js", media_type="application/javascript")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin(request: Request, message: str = "") -> HTMLResponse:
        manager = manager_from_request(request)
        if not admin_is_authenticated(request, manager):
            return admin_login_page()
        return admin_dashboard_page(manager, message)

    @app.post("/admin/login")
    async def admin_login(request: Request, password: str = Form(...)) -> Any:
        manager = manager_from_request(request)
        client_key = request.client.host if request.client is not None else "unknown"
        retry_after = manager.admin_login_retry_after(client_key)
        if retry_after:
            response = admin_login_page(
                f"Too many failed attempts. Try again in {retry_after} seconds."
            )
            response.status_code = 429
            response.headers["Retry-After"] = str(retry_after)
            return response
        token = await run_in_threadpool(manager.authenticate_admin, password)
        if token is None:
            manager.record_admin_login_failure(client_key)
            return admin_login_page("Invalid password.")
        manager.clear_admin_login_failures(client_key)
        response = RedirectResponse("/admin", status_code=303)
        set_admin_session_cookie(response, request, token)
        return response

    @app.post("/admin/logout")
    async def admin_logout(request: Request) -> RedirectResponse:
        manager = require_admin(request)
        manager.revoke_admin_session(request.cookies.get(ADMIN_COOKIE_NAME, ""))
        response = RedirectResponse("/admin", status_code=303)
        response.delete_cookie(ADMIN_COOKIE_NAME, path="/")
        return response

    @app.post("/admin/pause")
    async def admin_pause(request: Request) -> RedirectResponse:
        manager = require_admin(request)
        manager.pause_jobs()
        return RedirectResponse("/admin", status_code=303)

    @app.post("/admin/resume")
    async def admin_resume(request: Request) -> RedirectResponse:
        manager = require_admin(request)
        manager.resume_jobs()
        return RedirectResponse("/admin", status_code=303)

    @app.post("/admin/categories")
    async def create_category(request: Request, category: str = Form(...)) -> RedirectResponse:
        manager = require_admin(request)
        category = manager.create_category(category)
        return RedirectResponse(f"/category/{category}", status_code=303)

    @app.get("/admin/categories/{category}/delete", response_class=HTMLResponse)
    async def confirm_delete_category(request: Request, category: str) -> HTMLResponse:
        manager = require_admin(request)
        return category_delete_page(category, manager.category_job_counts(category))

    @app.post("/admin/categories/{category}/delete")
    async def delete_category(
        request: Request,
        category: str,
        confirmation: str = Form(...),
    ) -> RedirectResponse:
        manager = require_admin(request)
        manager.delete_category(category, confirmation)
        return RedirectResponse("/admin?message=Category+deleted.", status_code=303)

    @app.post("/admin/password")
    async def change_admin_password(
        request: Request,
        current_password: str = Form(...),
        new_password: str = Form(...),
        confirm_password: str = Form(...),
    ) -> Any:
        manager = require_admin(request)
        try:
            token = await run_in_threadpool(
                manager.change_admin_password,
                current_password,
                new_password,
                confirm_password,
            )
        except ValueError as exc:
            return admin_dashboard_page(manager, str(exc))
        response = RedirectResponse("/admin?message=Password+changed.", status_code=303)
        set_admin_session_cookie(response, request, token)
        return response

    @app.post("/upload")
    async def upload(
        request: Request,
        category: str = Form(...),
        cbz: UploadFile | None = File(None),
        page_images: list[UploadFile] | None = File(None),
        translation_notes: str = Form(""),
        thinking_budget_tokens: str = Form(""),
        vlm_base_url: str = Form(""),
        pause_after_ocr: str | None = Form(None),
        enable_alt_placement: str | None = Form(None),
        enable_proofreading: str | None = Form(None),
        enable_translation_notes: str | None = Form(None),
        source_language: str = Form(""),
        ocr_engine: str = Form(""),
        paddleocr_vl_server_url: str = Form(""),
        paddleocr_vl_model: str = Form(""),
    ) -> Any:
        manager = manager_from_request(request)
        code = manager.validate_category(category)
        thinking_budget = parse_thinking_budget_form(
            thinking_budget_tokens,
            manager.config.default_thinking_budget_tokens,
        )
        parsed_vlm_base_url = parse_vlm_base_url_form(vlm_base_url, manager.config)
        parsed_ocr_engine = parse_ocr_engine_form(
            ocr_engine,
            manager.config.default_ocr_engine,
        )
        parsed_source_language = parse_source_language_form(
            source_language,
            manager.config.default_source_language,
        )
        parsed_paddleocr_vl_server_url = parse_paddleocr_vl_server_url_form(
            paddleocr_vl_server_url,
            manager.config,
        )
        parsed_paddleocr_vl_model = parse_optional_text_form(
            paddleocr_vl_model,
            manager.config.default_paddleocr_vl_model,
        )
        cbz_filename = upload_filename(cbz)
        image_uploads = non_empty_uploads(page_images)
        has_cbz = bool(cbz_filename)
        has_images = bool(image_uploads)
        if has_cbz == has_images:
            raise HTTPException(
                status_code=400,
                detail="Upload either one CBZ file or one or more page image files.",
            )

        job_id = manager.create_job_id(code)
        manager.job_dir(code, job_id).mkdir(parents=True, exist_ok=True)
        input_path = manager.input_path(code, job_id)
        original_filename = cbz_filename
        try:
            if has_cbz:
                if not cbz_filename.lower().endswith(".cbz"):
                    raise HTTPException(status_code=400, detail="Upload must be a .cbz file.")
                if cbz is None:
                    raise HTTPException(status_code=400, detail="Upload must include a CBZ file.")
                await write_uploaded_file_to_path(
                    cbz,
                    input_path,
                    manager.config.max_upload_bytes,
                    "Uploaded CBZ",
                )
            else:
                _, original_filename = await write_uploaded_images_as_cbz(
                    image_uploads,
                    input_path,
                    manager.config.max_upload_bytes,
                )
        except HTTPException:
            input_path.unlink(missing_ok=True)
            shutil.rmtree(manager.job_dir(code, job_id), ignore_errors=True)
            raise
        finally:
            if cbz is not None:
                await cbz.close()
            for upload in page_images or []:
                await upload.close()

        manager.submit_job(
            code,
            job_id,
            original_filename,
            translation_notes,
            thinking_budget,
            parsed_vlm_base_url,
            parse_checkbox(pause_after_ocr),
            parse_checkbox(enable_proofreading),
            parse_checkbox(enable_translation_notes),
            parse_checkbox(enable_alt_placement),
            parsed_source_language,
            parsed_ocr_engine,
            parsed_paddleocr_vl_server_url,
            parsed_paddleocr_vl_model,
        )
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.get("/category/{code}", response_class=HTMLResponse)
    async def category_jobs(request: Request, code: str) -> HTMLResponse:
        manager = manager_from_request(request)
        return category_jobs_page(code, manager.public_category_jobs(code))

    @app.get("/job/{code}/{job_id}", response_class=HTMLResponse)
    async def job(request: Request, code: str, job_id: str) -> HTMLResponse:
        manager = manager_from_request(request)
        status = manager.public_status(code, job_id)
        return job_page(code, job_id, status)

    @app.get("/job/{code}/{job_id}/view", response_class=HTMLResponse)
    async def view_job(request: Request, code: str, job_id: str) -> HTMLResponse:
        manager = manager_from_request(request)
        status = manager.require_viewable_job(code, job_id)
        return job_viewer_page(code, job_id, status, manager.final_page_infos(code, job_id))

    @app.get("/job/{code}/{job_id}/view/image/{page}")
    async def view_job_page_image(
        request: Request,
        code: str,
        job_id: str,
        page: int,
    ) -> FileResponse:
        manager = manager_from_request(request)
        manager.require_viewable_job(code, job_id)
        image_path = manager.final_page_path(code, job_id, page)
        return FileResponse(
            image_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/job/{code}/{job_id}/view-original", response_class=HTMLResponse)
    async def view_original_job(request: Request, code: str, job_id: str) -> HTMLResponse:
        manager = manager_from_request(request)
        status = manager.require_original_viewable_job(code, job_id)
        return job_viewer_page(
            code,
            job_id,
            status,
            manager.original_page_infos(code, job_id),
            original=True,
        )

    @app.get("/job/{code}/{job_id}/view-original/image/{page}")
    async def view_original_job_page_image(
        request: Request,
        code: str,
        job_id: str,
        page: int,
    ) -> FileResponse:
        manager = manager_from_request(request)
        manager.require_original_viewable_job(code, job_id)
        image_path = manager.original_page_path(code, job_id, page)
        return FileResponse(
            image_path,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/job/{code}/{job_id}/edit", response_class=HTMLResponse)
    async def edit_job(request: Request, code: str, job_id: str) -> HTMLResponse:
        manager = manager_from_request(request)
        return editor_page(manager, code, job_id)

    @app.get("/job/{code}/{job_id}/image/{page}")
    async def job_page_image(request: Request, code: str, job_id: str, page: int) -> FileResponse:
        manager = manager_from_request(request)
        manager.require_editable_job(code, job_id)
        image_path = manager.original_page_path(code, job_id, page)
        return FileResponse(image_path)

    @app.get("/api/job/{code}/{job_id}")
    async def job_status(request: Request, code: str, job_id: str) -> JSONResponse:
        manager = manager_from_request(request)
        return JSONResponse(manager.public_status(code, job_id))

    @app.get("/api/job/{code}/{job_id}/edit/{stage}/{page}")
    async def edit_stage_data(
        request: Request,
        code: str,
        job_id: str,
        stage: str,
        page: int,
    ) -> JSONResponse:
        manager = manager_from_request(request)
        return JSONResponse(manager.editor_payload(code, job_id, stage, page))

    @app.post("/api/job/{code}/{job_id}/edit/{stage}/{page}")
    async def save_edit_stage_data(
        request: Request,
        code: str,
        job_id: str,
        stage: str,
        page: int,
    ) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        stage = validate_stage(stage, EDITOR_UI_STAGES)
        if stage == OCR_MERGE_EDITOR_STAGE:
            raw_records = data.get("rawRecords")
            merged_records = data.get("mergedRecords")
            if not isinstance(raw_records, list):
                raise HTTPException(status_code=400, detail="rawRecords must be a JSON array.")
            if not isinstance(merged_records, list):
                raise HTTPException(status_code=400, detail="mergedRecords must be a JSON array.")
            manager.save_ocr_merge_records(code, job_id, page, raw_records, merged_records)
        else:
            records = data.get("records")
            if not isinstance(records, list):
                raise HTTPException(status_code=400, detail="records must be a JSON array.")
            manager.save_stage_records(code, job_id, stage, page, records)
        notes = data.get("translationNotes")
        if isinstance(notes, dict):
            job_note = notes.get("job")
            page_note = notes.get("page")
            manager.save_translation_notes(
                code,
                job_id,
                page,
                job_note if isinstance(job_note, str) else None,
                page_note if isinstance(page_note, str) else None,
            )
        manager.mark_editor_stage_changed(code, job_id, page, stage)
        return JSONResponse(
            {
                "ok": True,
                "changeInfo": manager.editor_change_info(code, job_id, page, stage),
            }
        )

    @app.post("/api/job/{code}/{job_id}/edit/{stage}/{page}/lock")
    async def lock_edit_stage(
        request: Request,
        code: str,
        job_id: str,
        stage: str,
        page: int,
    ) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        meta = manager.set_stage_lock(code, job_id, page, stage, bool(data.get("locked")))
        locks = meta.get("locks", {})
        page_locks = locks.get(page_key(page), {}) if isinstance(locks, dict) else {}
        return JSONResponse({"ok": True, "locks": page_locks})

    @app.post("/api/job/{code}/{job_id}/edit/ocr-crop")
    async def edit_ocr_crop(request: Request, code: str, job_id: str) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        try:
            page = int(data.get("page", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="page must be an integer.") from exc
        region = parse_region(data.get("region"), "region")
        raw_boxno_start = optional_non_negative_int(data.get("rawBoxnoStart"), "rawBoxnoStart")
        try:
            result = await run_in_threadpool(
                manager.run_crop_ocr,
                code,
                job_id,
                page,
                region,
                raw_boxno_start,
            )
        except (paddle_ocr_image.InputError, translate_cbz.PipelineError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/job/{code}/{job_id}/edit/regenerate")
    async def regenerate_edit_page(request: Request, code: str, job_id: str) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
        try:
            page = int(data.get("page", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="page must be an integer.") from exc
        stage = validate_stage(str(data.get("stage", OCR_MERGE_EDITOR_STAGE)), EDITOR_RERUN_STAGES)
        manager.rerun_completed_job_pages(code, job_id, [page], stage)
        return JSONResponse({"ok": True, "url": f"/job/{code}/{job_id}"})

    @app.post("/api/job/{code}/{job_id}/edit/regenerate-changes")
    async def regenerate_changed_edit_pages(request: Request, code: str, job_id: str) -> JSONResponse:
        manager = manager_from_request(request)
        pages, resume_from = manager.rerun_changed_editor_pages(code, job_id)
        return JSONResponse(
            {
                "ok": True,
                "url": f"/job/{code}/{job_id}",
                "pages": pages,
                "resumeFrom": resume_from,
            }
        )

    @app.post("/job/{code}/{job_id}/restart")
    async def restart_job(request: Request, code: str, job_id: str) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.restart_failed_job(code, job_id)
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/advanced-options")
    async def update_advanced_options(
        request: Request,
        code: str,
        job_id: str,
        thinking_budget_tokens: str = Form(""),
        vlm_base_url: str = Form(""),
        pause_after_ocr: str | None = Form(None),
        enable_alt_placement: str | None = Form(None),
        enable_proofreading: str | None = Form(None),
        enable_translation_notes: str | None = Form(None),
        source_language: str = Form(""),
        ocr_engine: str = Form(""),
        paddleocr_vl_server_url: str = Form(""),
        paddleocr_vl_model: str = Form(""),
    ) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.update_job_advanced_options(
            code,
            job_id,
            parse_thinking_budget_form(
                thinking_budget_tokens,
                manager.config.default_thinking_budget_tokens,
            ),
            parse_vlm_base_url_form(vlm_base_url, manager.config),
            parse_checkbox(pause_after_ocr),
            parse_checkbox(enable_proofreading),
            parse_checkbox(enable_translation_notes),
            parse_checkbox(enable_alt_placement),
            parse_source_language_form(
                source_language,
                manager.config.default_source_language,
            ),
            parse_ocr_engine_form(ocr_engine, manager.config.default_ocr_engine),
            parse_paddleocr_vl_server_url_form(
                paddleocr_vl_server_url,
                manager.config,
            ),
            parse_optional_text_form(
                paddleocr_vl_model,
                manager.config.default_paddleocr_vl_model,
            ),
        )
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/terminate")
    async def terminate_job(request: Request, code: str, job_id: str) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.terminate_active_job(code, job_id)
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/rerun-pages")
    async def rerun_pages(
        request: Request,
        code: str,
        job_id: str,
        page_spec: str = Form(...),
    ) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.rerun_completed_job_pages(code, job_id, parse_page_selection(page_spec))
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/rerun-job")
    async def rerun_job(request: Request, code: str, job_id: str) -> RedirectResponse:
        manager = manager_from_request(request)
        form = await request.form()
        resume_from, package_selected = parse_rerun_job_stages(
            [str(value) for value in form.getlist("rerun_stage")]
        )
        webp_quality = parse_quality_form(
            str(form.get("webp_quality") or ""),
            "WebP quality",
            manager.config.default_webp_quality,
        )
        jxl_quality = parse_quality_form(
            str(form.get("jxl_quality") or ""),
            "JXL quality",
            manager.config.default_jxl_quality,
        )
        if resume_from is None:
            if not package_selected:
                raise HTTPException(status_code=400, detail="Select at least one pass to rerun.")
            manager.regenerate_completed_job_downloads(code, job_id, webp_quality, jxl_quality)
            return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

        page_spec = str(form.get("page_spec") or "")
        manager.rerun_completed_job_pages(
            code,
            job_id,
            parse_page_selection(page_spec, allow_empty=True),
            resume_from,
            webp_quality if package_selected else None,
            jxl_quality if package_selected else None,
        )
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/regenerate-downloads")
    async def regenerate_downloads(
        request: Request,
        code: str,
        job_id: str,
        webp_quality: str = Form(""),
        jxl_quality: str = Form(""),
    ) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.regenerate_completed_job_downloads(
            code,
            job_id,
            parse_quality_form(
                webp_quality,
                "WebP quality",
                manager.config.default_webp_quality,
            ),
            parse_quality_form(
                jxl_quality,
                "JXL quality",
                manager.config.default_jxl_quality,
            ),
        )
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/delete")
    async def delete_job(request: Request, code: str, job_id: str) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.delete_job(code, job_id)
        return RedirectResponse(f"/category/{code}", status_code=303)

    @app.get("/job/{code}/{job_id}/download")
    async def download(request: Request, code: str, job_id: str) -> FileResponse:
        return await download_variant(request, code, job_id, "png")

    @app.get("/job/{code}/{job_id}/download-original")
    async def download_original(request: Request, code: str, job_id: str) -> FileResponse:
        manager = manager_from_request(request)
        manager.validate_category(code)
        job_id = manager.validate_job_id(job_id)
        status = manager.load_status(code, job_id)
        input_path = manager.input_path(code, job_id)
        if status is None or not input_path.is_file():
            raise HTTPException(status_code=404, detail="Original input CBZ is not available.")
        return FileResponse(
            input_path,
            media_type="application/vnd.comicbook+zip",
            filename=f"{code}_{job_id}_original.cbz",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    @app.get("/job/{code}/{job_id}/download/{variant}")
    async def download_variant(
        request: Request,
        code: str,
        job_id: str,
        variant: str,
    ) -> FileResponse:
        manager = manager_from_request(request)
        manager.validate_category(code)
        job_id = manager.validate_job_id(job_id)
        if variant not in TRANSLATED_CBZ_FILENAMES:
            raise HTTPException(status_code=404, detail="Unknown translated CBZ variant.")
        status = manager.load_status(code, job_id)
        cbz_path = manager.translated_cbz_variant_path(code, job_id, variant)
        if status is None or status.get("status") != "complete" or not cbz_path.is_file():
            raise HTTPException(status_code=404, detail="Translated CBZ is not available.")
        return FileResponse(
            cbz_path,
            media_type="application/vnd.comicbook+zip",
            filename=f"{code}_{job_id}_translated_{variant}.cbz",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tetolate web UI.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("TETOLATE_WEB_CONFIG", DEFAULT_WEB_CONFIG)),
        help="Path to web_config.json.",
    )
    parser.add_argument("--host", help="Override the listen host from the web config.")
    parser.add_argument("--port", type=int, help="Override the listen port from the web config.")
    args = parser.parse_args()

    config = load_web_config(args.config)
    host = args.host or config.listen_host
    port = args.port or config.listen_port

    import uvicorn

    uvicorn.run(create_app(args.config), host=host, port=port)


if __name__ == "__main__":
    main()
