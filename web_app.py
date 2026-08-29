#!/usr/bin/env python3
"""Minimal admin-only web UI for managing CBZ translation jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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
import editor_v2
import lama_inpaint
import translate_cbz
from web_editor_backend import (
    EditorManagerMixin,
    parse_region,
)
from web_pages import (
    admin_dashboard_page,
    admin_login_page,
    category_delete_page,
    category_jobs_page,
    editor_page,
    job_page,
    job_viewer_page,
)
import web_security
from web_storage import write_json_atomic


LOGGER = logging.getLogger(__name__)
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
OCR_REVIEW_CHECKPOINT = "ocr"
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
LOG_TAIL_BYTES = 2 * 1024 * 1024
UPLOAD_COMIC_ARCHIVE_EXTENSIONS = {".cbz", ".zip"}
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


class UploadRequestTooLarge(Exception):
    """Raised when an upload request exceeds the configured byte limit."""


class UploadSizeLimitMiddleware:
    """Reject oversized upload bodies before multipart parsing buffers them."""

    def __init__(self, app: Any) -> None:
        self.app = app

    @staticmethod
    async def _send_too_large(send: Any) -> None:
        body = b'{"detail":"Upload request is too large."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/upload" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        manager = getattr(getattr(self.app, "state", None), "manager", None)
        config = getattr(manager, "config", None)
        max_bytes = getattr(config, "max_upload_bytes", None)
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        content_length = dict(scope.get("headers", [])).get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError):
                declared_length = None
            if declared_length is not None and declared_length > max_bytes:
                await self._send_too_large(send)
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > max_bytes:
                    raise UploadRequestTooLarge
            return message

        response_started = False

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except UploadRequestTooLarge:
            if not response_started:
                await self._send_too_large(send)
            else:
                raise
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
    "Clean": "render",
    "Render": "render",
    "Package": "package",
}
RESUME_PHASES = frozenset(RESUME_PHASE_BY_PROGRESS.values())
TERMINAL_JOB_STATUSES = frozenset(("complete", "failed", "cancelled"))
OCR_MERGE_EDITOR_STAGE = "ocr_merge"
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
EDITOR_META_DIRNAME = "web_meta"
TRANSLATION_NOTES_FILENAME = "translation_notes.json"
JOB_SECRETS_FILENAME = ".job-secrets.json"
CATEGORY_ADVANCED_OPTIONS_FILENAME = "advanced_options.json"
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


def parse_auth_token_form(value: str) -> str | None:
    token = value.strip()
    if not token:
        return None
    if len(token) > 16_384:
        raise HTTPException(status_code=400, detail="Auth token is too long.")
    if any(character in token for character in ("\0", "\r", "\n")):
        raise HTTPException(status_code=400, detail="Auth token contains an invalid character.")
    return token


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


def parse_page_workers_form(value: str, label: str) -> int:
    try:
        return translate_cbz.normalize_page_workers(value, label)
    except translate_cbz.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def stored_elapsed_seconds(status: dict[str, Any]) -> float:
    try:
        value = float(status.get("elapsedSeconds", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, value)


def start_active_runtime(
    status: dict[str, Any],
    now: datetime | None = None,
) -> None:
    now = now or datetime.now(timezone.utc)
    timestamp = now.isoformat()
    if parse_timestamp(status.get("startedAt")) is None:
        status["startedAt"] = timestamp
    if parse_timestamp(status.get("activeStartedAt")) is None:
        status["activeStartedAt"] = timestamp


def stop_active_runtime(
    status: dict[str, Any],
    now: datetime | None = None,
) -> float:
    now = now or datetime.now(timezone.utc)
    elapsed = stored_elapsed_seconds(status)
    active_started_at = parse_timestamp(status.get("activeStartedAt"))
    if active_started_at is not None:
        elapsed += max(0.0, (now - active_started_at).total_seconds())
    status["elapsedSeconds"] = round(elapsed, 3)
    status.pop("activeStartedAt", None)
    return elapsed


def job_timing(status: dict[str, Any], now: datetime | None = None) -> JobTiming:
    now = now or datetime.now(timezone.utc)
    created_at = (
        parse_timestamp(status.get("createdAt"))
        or parse_timestamp(status.get("updatedAt"))
        or parse_timestamp(status.get("startedAt"))
        or parse_timestamp(status.get("finishedAt"))
    )
    elapsed_seconds = stored_elapsed_seconds(status)
    active_started_at = parse_timestamp(status.get("activeStartedAt"))
    if active_started_at is not None:
        elapsed_seconds += max(0.0, (now - active_started_at).total_seconds())
    if (
        elapsed_seconds == 0
        and active_started_at is None
        and parse_timestamp(status.get("startedAt")) is None
    ):
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


def validate_uploaded_comic_archive(path: Path, filename: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = translate_cbz.validate_cbz_members(archive)
            image_members = [
                info
                for info in members
                if not info.is_dir() and translate_cbz.is_image_member(info.filename)
            ]
            if not image_members:
                raise HTTPException(
                    status_code=400,
                    detail="Comic archive must contain at least one supported image page.",
                )
            total_pixels = 0
            for info in image_members:
                try:
                    with archive.open(info) as source, Image.open(source) as image:
                        page_pixels = image.width * image.height
                        if page_pixels > translate_cbz.MAX_PAGE_PIXELS:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Archive page is too large: {info.filename}",
                            )
                        total_pixels += page_pixels
                        if total_pixels > translate_cbz.MAX_TOTAL_PAGE_PIXELS:
                            raise HTTPException(
                                status_code=400,
                                detail="Archive image pages contain too many pixels.",
                            )
                        image.verify()
                except HTTPException:
                    raise
                except (OSError, RuntimeError, Image.DecompressionBombError) as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Archive page is not a readable image: {info.filename}",
                    ) from exc
    except HTTPException:
        raise
    except translate_cbz.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail=f"File is not a valid CBZ or ZIP archive: {filename}",
        ) from exc


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


def safe_log_lines(path: Path, limit: int = 1000) -> list[str]:
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
        (re.compile(r"^Clean page (\d+)"), "Clean"),
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


def parse_rerun_job_stages(values: list[str]) -> str:
    selected = {str(value).strip() for value in values if str(value).strip()}
    allowed = set(RERUN_JOB_STAGE_ORDER)
    unsupported = sorted(selected - allowed)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="Unsupported rerun pass: " + ", ".join(unsupported),
        )
    if not selected:
        raise HTTPException(status_code=400, detail="Select at least one pass to rerun.")

    for stage in RERUN_JOB_STAGE_ORDER:
        if stage in selected:
            return RERUN_JOB_STAGE_RESUME[stage]
    raise HTTPException(status_code=400, detail="Select at least one processing pass to rerun.")


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


class JobManager(EditorManagerMixin):
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
        self._editor_lama_lock = threading.Lock()
        self._editor_lama_session: lama_inpaint.LaMaSession | None = None
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
        with self._editor_lama_lock:
            if self._editor_lama_session is not None:
                self._editor_lama_session.close()
                self._editor_lama_session = None
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
            paused_at = datetime.now(timezone.utc)
            stop_active_runtime(status, paused_at)
            status["status"] = "paused"
            status["isPaused"] = True
            status["pausedAt"] = paused_at.isoformat()
            status["message"] = "Paused by admin."
        else:
            status["message"] = "Pause requested, but the active process could not be suspended."
        self.save_status(code, job_id, status)

    def mark_active_job_resumed(self, code: str, job_id: str, signal_sent: bool) -> None:
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        if status.get("pendingTermination"):
            return
        if signal_sent or status.get("status") == "paused":
            start_active_runtime(status)
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
            "ocrPageWorkers": translate_cbz.DEFAULT_OCR_PAGE_WORKERS,
            "lamaWorkers": translate_cbz.DEFAULT_LAMA_WORKERS,
            "imagemagickWorkers": translate_cbz.DEFAULT_IMAGEMAGICK_WORKERS,
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
        for key, label in (
            ("ocrPageWorkers", "OCR workers"),
            ("lamaWorkers", "LaMa workers"),
            ("imagemagickWorkers", "ImageMagick workers"),
        ):
            try:
                normalized[key] = translate_cbz.normalize_page_workers(
                    data.get(key, defaults[key]), label
                )
            except translate_cbz.PipelineError:
                pass
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
        ocr_page_workers: int = translate_cbz.DEFAULT_OCR_PAGE_WORKERS,
        lama_workers: int = translate_cbz.DEFAULT_LAMA_WORKERS,
        imagemagick_workers: int = translate_cbz.DEFAULT_IMAGEMAGICK_WORKERS,
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
                    "ocrPageWorkers": ocr_page_workers,
                    "lamaWorkers": lama_workers,
                    "imagemagickWorkers": imagemagick_workers,
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

    def job_secrets_path(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / JOB_SECRETS_FILENAME

    def load_job_secrets(self, code: str, job_id: str) -> dict[str, str]:
        path = self.job_secrets_path(code, job_id)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            key: value
            for key, value in data.items()
            if key in {"vlmAuthToken", "paddleocrVlAuthToken"}
            and isinstance(value, str)
            and value
        }

    def update_job_auth_tokens(
        self,
        code: str,
        job_id: str,
        *,
        vlm_auth_token: str | None,
        paddleocr_vl_auth_token: str | None,
        clear_vlm_auth_token: bool = False,
        clear_paddleocr_vl_auth_token: bool = False,
        preserve_existing: bool = True,
    ) -> None:
        secrets = self.load_job_secrets(code, job_id) if preserve_existing else {}
        for key, token, clear in (
            ("vlmAuthToken", vlm_auth_token, clear_vlm_auth_token),
            (
                "paddleocrVlAuthToken",
                paddleocr_vl_auth_token,
                clear_paddleocr_vl_auth_token,
            ),
        ):
            if clear:
                secrets.pop(key, None)
            elif token is not None:
                secrets[key] = token
        path = self.job_secrets_path(code, job_id)
        if secrets:
            write_json_atomic(path, secrets, mode=0o600)
        else:
            path.unlink(missing_ok=True)

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

    def translation_notes_path(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / TRANSLATION_NOTES_FILENAME

    def generated_translation_notes_path(self, code: str, job_id: str) -> Path:
        return self.output_dir(code, job_id) / GENERATED_TRANSLATION_NOTES_NAME

    def read_generated_translation_notes(self, code: str, job_id: str) -> str:
        path = self.generated_translation_notes_path(code, job_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace").strip()

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

    def download_info(self, code: str, job_id: str) -> dict[str, dict[str, Any]]:
        downloads: dict[str, dict[str, Any]] = {}
        for variant in TRANSLATED_CBZ_FILENAMES:
            info = file_info(
                self.translated_cbz_variant_path(code, job_id, variant)
            )
            info["variant"] = variant
            info["generateUrl"] = f"/job/{code}/{job_id}/generate-download/{variant}"
            downloads[variant] = info
        return downloads

    def invalidate_translated_cbz_archives(self, code: str, job_id: str) -> None:
        """Remove archives whose rendered page inputs are no longer current."""
        for variant in TRANSLATED_CBZ_FILENAMES:
            self.translated_cbz_variant_path(code, job_id, variant).unlink(missing_ok=True)

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
            status.pop("pendingPackageVariant", None)
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
            status.pop("pendingPackageVariant", None)
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

    def status_is_ocr_review_checkpoint(self, status: dict[str, Any]) -> bool:
        return (
            status.get("status") == "paused"
            and status.get("reviewCheckpoint") == OCR_REVIEW_CHECKPOINT
        )

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

    def status_page_workers(self, status: dict[str, Any], key: str, label: str) -> int:
        try:
            return translate_cbz.normalize_page_workers(status.get(key, 1), label)
        except translate_cbz.PipelineError:
            return 1

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
        ocr_review_checkpoint = (
            self.status_is_ocr_review_checkpoint(status) and not is_active
        )
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
        ocr_page_workers = self.status_page_workers(status, "ocrPageWorkers", "OCR workers")
        lama_workers = self.status_page_workers(status, "lamaWorkers", "LaMa workers")
        imagemagick_workers = self.status_page_workers(
            status, "imagemagickWorkers", "ImageMagick workers"
        )
        job_secrets = self.load_job_secrets(code, job_id)
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
            "hasDownload": any(
                item.get("available") for item in downloads.values()
            ) and status.get("status") == "complete",
            "downloads": downloads if status.get("status") == "complete" else {},
            "downloadGenerationUrls": {
                variant: f"/job/{code}/{job_id}/generate-download/{variant}"
                for variant in TRANSLATED_CBZ_FILENAMES
            }
            if status.get("status") == "complete"
            else {},
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
            "canGenerateDownloads": status.get("status") == "complete",
            "canEdit": status.get("status") == "complete" or ocr_review_checkpoint,
            "ocrReviewCheckpoint": ocr_review_checkpoint,
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
            "ocrPageWorkers": ocr_page_workers,
            "lamaWorkers": lama_workers,
            "imagemagickWorkers": imagemagick_workers,
            "hasVlmAuthToken": bool(job_secrets.get("vlmAuthToken")),
            "hasPaddleocrVlAuthToken": bool(
                job_secrets.get("paddleocrVlAuthToken")
            ),
            "defaultAltPlacementEnabled": self.config.default_alt_placement_enabled,
            "defaultSourceLanguage": self.config.default_source_language,
            "defaultOcrEngine": self.config.default_ocr_engine,
            "defaultPaddleocrVlServerUrl": self.config.default_paddleocr_vl_server_url,
            "defaultPaddleocrVlModel": self.config.default_paddleocr_vl_model,
            "defaultOcrPageWorkers": translate_cbz.DEFAULT_OCR_PAGE_WORKERS,
            "defaultLamaWorkers": translate_cbz.DEFAULT_LAMA_WORKERS,
            "defaultImagemagickWorkers": translate_cbz.DEFAULT_IMAGEMAGICK_WORKERS,
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
            "defaultOcrPageWorkers": options["ocrPageWorkers"],
            "defaultLamaWorkers": options["lamaWorkers"],
            "defaultImagemagickWorkers": options["imagemagickWorkers"],
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
                    stopped_at = (
                        parse_timestamp(status.get("updatedAt"))
                        or datetime.now(timezone.utc)
                    )
                    stop_active_runtime(status, stopped_at)
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
                    if self.status_is_ocr_review_checkpoint(status):
                        continue
                    saw_paused_job = True
                resume_target = self.pending_resume_target_for_status(status)
                if previous_status != "queued" and resume_target is None:
                    resume_target = self.restart_target_for_status(code, job_id, status)
                if previous_status != "queued":
                    self.stop_recorded_pipeline_process(status)
                    stopped_at = (
                        parse_timestamp(status.get("pausedAt"))
                        or parse_timestamp(status.get("updatedAt"))
                        or datetime.now(timezone.utc)
                    )
                    stop_active_runtime(status, stopped_at)
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
        ocr_page_workers: int = translate_cbz.DEFAULT_OCR_PAGE_WORKERS,
        lama_workers: int = translate_cbz.DEFAULT_LAMA_WORKERS,
        imagemagick_workers: int = translate_cbz.DEFAULT_IMAGEMAGICK_WORKERS,
        vlm_auth_token: str | None = None,
        paddleocr_vl_auth_token: str | None = None,
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
        ocr_page_workers = translate_cbz.normalize_page_workers(ocr_page_workers, "OCR workers")
        lama_workers = translate_cbz.normalize_page_workers(lama_workers, "LaMa workers")
        imagemagick_workers = translate_cbz.normalize_page_workers(
            imagemagick_workers, "ImageMagick workers"
        )
        status = {
            "category": code,
            "jobId": job_id,
            "status": "queued",
            "phase": "Queued",
            "page": None,
            "message": "Waiting for the worker.",
            "createdAt": created_at,
            "elapsedSeconds": 0.0,
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
            "ocrPageWorkers": ocr_page_workers,
            "lamaWorkers": lama_workers,
            "imagemagickWorkers": imagemagick_workers,
        }
        self.save_status(code, job_id, status)
        self.update_job_auth_tokens(
            code,
            job_id,
            vlm_auth_token=vlm_auth_token,
            paddleocr_vl_auth_token=paddleocr_vl_auth_token,
            preserve_existing=False,
        )
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
            ocr_page_workers=ocr_page_workers,
            lama_workers=lama_workers,
            imagemagick_workers=imagemagick_workers,
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
        ocr_page_workers: int,
        lama_workers: int,
        imagemagick_workers: int,
        vlm_auth_token: str | None,
        paddleocr_vl_auth_token: str | None,
        clear_vlm_auth_token: bool,
        clear_paddleocr_vl_auth_token: bool,
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
        ocr_page_workers = translate_cbz.normalize_page_workers(ocr_page_workers, "OCR workers")
        lama_workers = translate_cbz.normalize_page_workers(lama_workers, "LaMa workers")
        imagemagick_workers = translate_cbz.normalize_page_workers(
            imagemagick_workers, "ImageMagick workers"
        )
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
            status["ocrPageWorkers"] = ocr_page_workers
            status["lamaWorkers"] = lama_workers
            status["imagemagickWorkers"] = imagemagick_workers
            status["message"] = "Advanced options updated."
            self.save_status(code, job_id, status)
            self.update_job_auth_tokens(
                code,
                job_id,
                vlm_auth_token=vlm_auth_token,
                paddleocr_vl_auth_token=paddleocr_vl_auth_token,
                clear_vlm_auth_token=clear_vlm_auth_token,
                clear_paddleocr_vl_auth_token=clear_paddleocr_vl_auth_token,
            )
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
                ocr_page_workers=ocr_page_workers,
                lama_workers=lama_workers,
                imagemagick_workers=imagemagick_workers,
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
                raise HTTPException(status_code=400, detail="Original uploaded archive is missing.")

            pending_reruns = status.get("pendingPageReruns")
            first_pending = (
                pending_reruns[0]
                if isinstance(pending_reruns, list)
                and pending_reruns
                and isinstance(pending_reruns[0], dict)
                else None
            )
            pending_phase = (
                RERUN_STAGE_MAP.get(
                    str(first_pending.get("resumeFrom") or ""),
                    str(first_pending.get("resumeFrom") or ""),
                )
                if first_pending is not None
                else ""
            )
            pending_target = (
                stored_resume_target(pending_phase, first_pending.get("page"))
                if first_pending is not None
                else None
            )
            if self.status_is_ocr_review_checkpoint(status):
                restart_target = ("ocr_structured", 0)
            elif (
                pending_target is not None
                and pending_phase in RERUN_STAGE_MAP.values()
            ):
                restart_target = pending_target
            else:
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
            status.pop("reviewCheckpoint", None)
            if restart_target is not None:
                status["pendingResumeFrom"] = restart_target[0]
                status["pendingResumePage"] = restart_target[1]
                if restart_target[0] == "package":
                    status["pendingPackageOnly"] = True
                    if status.get("pendingPackageVariant") not in TRANSLATED_CBZ_FILENAMES:
                        status["pendingPackageVariant"] = "png"
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
                    status.pop("pendingPackageVariant", None)
                    status.pop("pendingWebpQuality", None)
                    status.pop("pendingJxlQuality", None)
            else:
                status.pop("pendingResumeFrom", None)
                status.pop("pendingResumePage", None)
                status.pop("pendingPackageOnly", None)
                status.pop("pendingPackageVariant", None)
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
        translation_boxno: int | None = None,
        page_resume_from: dict[int, str] | None = None,
    ) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        resume_from = RERUN_STAGE_MAP.get(resume_from, resume_from)
        if resume_from not in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements", "render"}:
            raise HTTPException(status_code=400, detail="Unsupported rerun stage.")
        if translation_boxno is not None:
            if resume_from != "translations" or len(pages) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="A selected-record translation requires one page from translations.",
                )
            if (
                not isinstance(translation_boxno, int)
                or isinstance(translation_boxno, bool)
                or translation_boxno < 0
            ):
                raise HTTPException(status_code=400, detail="Invalid translation box number.")
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Job is already queued or running.")
            if status.get("status") != "complete":
                raise HTTPException(status_code=400, detail="Only complete jobs can rerun pages.")
            if not self.input_path(code, job_id).is_file():
                raise HTTPException(status_code=400, detail="Original uploaded archive is missing.")
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

            page_reruns: list[dict[str, Any]] = []
            for page in pages:
                page_stage = (
                    page_resume_from.get(page, resume_from)
                    if page_resume_from is not None
                    else resume_from
                )
                page_stage = RERUN_STAGE_MAP.get(page_stage, page_stage)
                if page_stage not in {
                    "ocr_raw",
                    "ocr_structured",
                    "alt_placement",
                    "translations",
                    "placements",
                    "render",
                }:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported rerun stage for page {page}: {page_stage}.",
                    )
                page_reruns.append({"page": page, "resumeFrom": page_stage})
            first_rerun = page_reruns[0]

            if self.has_editor_v2(code, job_id):
                manifest = self.load_editor_v2_manifest(code, job_id)
                for page in pages:
                    self.materialize_editor_v2_page(code, job_id, manifest, page)

            # Any page rerun changes the rendered source for every archive
            # variant. Keep no stale translated archive available.
            self.invalidate_translated_cbz_archives(code, job_id)

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
                        f"Queued to rerun {page_message} from each page's required stage."
                        if page_resume_from is not None
                        else f"Queued to rerun {page_message} from {resume_from}."
                    ),
                    "finishedAt": None,
                    "returnCode": None,
                    "rerunCount": rerun_count,
                    "pendingPageReruns": page_reruns,
                    "pendingResumeFrom": first_rerun["resumeFrom"],
                    "pendingResumePage": first_rerun["page"],
                    "lastResumeFrom": first_rerun["resumeFrom"],
                    "lastResumePage": first_rerun["page"],
                }
            )
            status.pop("pendingWebpQuality", None)
            status.pop("pendingJxlQuality", None)
            status.pop("pid", None)
            status.pop("pendingPackageOnly", None)
            status.pop("pendingPackageVariant", None)
            if translation_boxno is None:
                status.pop("pendingTranslationBoxno", None)
            else:
                status["pendingTranslationBoxno"] = translation_boxno
            self.save_status(code, job_id, status)
            self.enqueue(code, job_id)

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

    def generate_download(self, code: str, job_id: str, variant: str) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        if variant not in TRANSLATED_CBZ_FILENAMES:
            raise HTTPException(status_code=404, detail="Unknown translated CBZ variant.")
        with self._lock:
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(status_code=404, detail="Unknown job.")
            if status.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="Job is already queued or running.")
            if status.get("status") != "complete":
                raise HTTPException(
                    status_code=400,
                    detail="Only complete jobs can generate downloads.",
                )
            if not self.input_path(code, job_id).is_file():
                raise HTTPException(status_code=400, detail="Original uploaded archive is missing.")
            if not self.output_dir(code, job_id).is_dir():
                raise HTTPException(status_code=400, detail="Existing output directory is missing.")

            try:
                package_count = int(status.get("packageGenerationCount", 0)) + 1
            except (TypeError, ValueError):
                package_count = 1
            status.update(
                {
                    "status": "queued",
                    "phase": "Queued",
                    "page": None,
                    "message": (
                        f"Queued to generate {variant.upper()} CBZ download."
                    ),
                    "finishedAt": None,
                    "returnCode": None,
                    "packageGenerationCount": package_count,
                    "pendingPackageOnly": True,
                    "pendingPackageVariant": variant,
                    "pendingWebpQuality": quality_for_display(
                        status.get("webpQuality"), self.config.default_webp_quality
                    ),
                    "pendingJxlQuality": quality_for_display(
                        status.get("jxlQuality"), self.config.default_jxl_quality
                    ),
                    "pendingResumeFrom": "package",
                    "pendingResumePage": 0,
                    "lastResumeFrom": "package",
                    "lastResumePage": 0,
                }
            )
            status.pop("pid", None)
            status.pop("pendingPageReruns", None)
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
            try:
                shutil.rmtree(self.job_dir(code, job_id))
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404,
                    detail="Job directory is missing.",
                ) from None
            except OSError as exc:
                raise HTTPException(
                    status_code=500,
                    detail="Could not delete job.",
                ) from exc

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
                    finished_at = datetime.now(timezone.utc)
                    stop_active_runtime(status, finished_at)
                    status.update(
                        {
                            "status": "failed",
                            "phase": "Failed",
                            "message": f"Unexpected worker failure: {exc}",
                            "finishedAt": finished_at.isoformat(),
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
            pending_page_reruns = status.get("pendingPageReruns")
            if (
                isinstance(pending_page_reruns, list)
                and pending_page_reruns
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
        package_variant: str | None = None,
        webp_quality: int | None = None,
        jxl_quality: int | None = None,
        translation_boxno: int | None = None,
    ) -> list[str]:
        command = [
            sys.executable,
            str(self.config.translate_script),
            str(self.input_path(code, job_id)),
            str(self.output_dir(code, job_id)),
            "--config",
            str(self.config.pipeline_config),
        ]
        if resume_from is not None and self.has_editor_v2(code, job_id):
            command.extend(
                [
                    "--editor-manifest",
                    str(self.editor_v2_manifest_path(code, job_id)),
                    "--editor-baseline-dir",
                    str(self.editor_meta_dir(code, job_id) / "generated_v2"),
                ]
            )
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
                "--ocr-workers",
                str(self.status_page_workers(status, "ocrPageWorkers", "OCR workers")),
                "--lama-workers",
                str(self.status_page_workers(status, "lamaWorkers", "LaMa workers")),
                "--imagemagick-workers",
                str(
                    self.status_page_workers(
                        status, "imagemagickWorkers", "ImageMagick workers"
                    )
                ),
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
        if package_variant is not None:
            if package_variant not in TRANSLATED_CBZ_FILENAMES:
                raise HTTPException(status_code=400, detail="Unknown translated CBZ variant.")
            command.extend(["--package-variant", package_variant])
        if resume_from is None:
            if self.status_pause_after_ocr(status):
                command.extend(["--stop-after", "ocr_merged"])
            command.append("--overwrite")
            # Web jobs keep rendered pages as the canonical result. Archives are
            # generated only when a user requests a specific download.
            command.append("--skip-package")
            return command

        command.extend(["--resume-from", resume_from])
        if resume_from not in {"extract", "package"}:
            command.extend(["--resume-page", str(resume_page or 0)])
        if single_page and resume_from in {"ocr_raw", "ocr_structured", "alt_placement", "translations", "placements", "render"}:
            command.append("--single-page")
        if translation_boxno is not None:
            command.extend(["--translation-boxno", str(translation_boxno)])
        if skip_package or resume_from != "package":
            command.append("--skip-package")
        return command

    def run_pipeline_process(
        self,
        code: str,
        job_id: str,
        command: list[str],
        label: str,
    ) -> int:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        secrets = self.load_job_secrets(code, job_id)
        if secrets.get("vlmAuthToken"):
            env["TETOLATE_VLM_API_KEY"] = secrets["vlmAuthToken"]
        if secrets.get("paddleocrVlAuthToken"):
            env["TETOLATE_PADDLEOCR_VL_API_KEY"] = secrets[
                "paddleocrVlAuthToken"
            ]
        log_path = self.log_path(code, job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

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
                finished_at = datetime.now(timezone.utc)
                stop_active_runtime(status, finished_at)
                status.update(
                    {
                        "status": "failed",
                        "phase": "Failed",
                        "message": f"Could not start translation pipeline for {label}: {exc}",
                        "finishedAt": finished_at.isoformat(),
                    }
                )
                self.save_status(code, job_id, status)
                return 1

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

        return return_code

    def run_package_generation_job(
        self,
        code: str,
        job_id: str,
        status: dict[str, Any],
    ) -> None:
        variant = str(status.get("pendingPackageVariant") or "")
        if variant not in TRANSLATED_CBZ_FILENAMES:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": "No valid CBZ variant was selected for generation.",
                    "finishedAt": now_utc(),
                }
            )
            self.save_status(code, job_id, status)
            return
        webp_quality = quality_for_display(
            status.get("pendingWebpQuality"),
            self.config.default_webp_quality,
        )
        jxl_quality = quality_for_display(
            status.get("pendingJxlQuality"),
            self.config.default_jxl_quality,
        )
        start_active_runtime(status)
        status.update(
            {
                "status": "running",
                "phase": "Starting",
                "page": None,
                "message": f"Generating {variant.upper()} CBZ download.",
                "finishedAt": None,
                "pendingResumeFrom": "package",
                "pendingResumePage": 0,
                "lastResumeFrom": "package",
                "lastResumePage": 0,
            }
        )
        self.save_status(code, job_id, status)

        return_code = self.run_pipeline_process(
            code,
            job_id,
            self.build_command(
                code,
                job_id,
                "package",
                0,
                webp_quality=webp_quality,
                jxl_quality=jxl_quality,
                package_variant=variant,
            ),
            f"generate {variant} download",
        )

        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        finished_at = datetime.now(timezone.utc)
        stop_active_runtime(status, finished_at)
        status["returnCode"] = return_code
        status["finishedAt"] = finished_at.isoformat()
        status["webpQuality"] = webp_quality
        status["jxlQuality"] = jxl_quality
        status.pop("pid", None)
        status.pop("pendingTranslationBoxno", None)
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
                    "message": "Download generation terminated.",
                }
            )
        elif return_code == 0 and self.translated_cbz_variant_path(code, job_id, variant).is_file():
            status.update(
                {
                    "status": "complete",
                    "phase": "Complete",
                    "page": None,
                    "message": f"{variant.upper()} CBZ download generation complete.",
                }
            )
            status.pop("pendingPackageOnly", None)
            status.pop("pendingPackageVariant", None)
            status.pop("pendingWebpQuality", None)
            status.pop("pendingJxlQuality", None)
            status.pop("pendingResumeFrom", None)
            status.pop("pendingResumePage", None)
        else:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": f"Download generation failed with exit code {return_code}.",
                }
            )
            status["pendingPackageOnly"] = True
            status["pendingPackageVariant"] = variant
            status["pendingResumeFrom"] = "package"
            status["pendingResumePage"] = 0
        self.save_status(code, job_id, status)

    def run_page_rerun_batch_job(
        self,
        code: str,
        job_id: str,
        status: dict[str, Any],
        pending_page_reruns: list[Any],
    ) -> None:
        page_reruns: list[dict[str, Any]] = []
        for item in pending_page_reruns:
            if not isinstance(item, dict):
                continue
            try:
                page = int(item.get("page"))
            except (TypeError, ValueError):
                continue
            resume_from = RERUN_STAGE_MAP.get(
                str(item.get("resumeFrom") or ""),
                str(item.get("resumeFrom") or ""),
            )
            if page < 0 or resume_from not in {
                "ocr_raw",
                "ocr_structured",
                "alt_placement",
                "translations",
                "placements",
                "render",
            }:
                continue
            page_reruns.append({"page": page, "resumeFrom": resume_from})
        if not page_reruns:
            status.update(
                {
                    "status": "failed",
                    "phase": "Failed",
                    "message": "No valid pages were queued for rerun.",
                    "finishedAt": now_utc(),
                }
            )
            status.pop("pendingPageReruns", None)
            self.save_status(code, job_id, status)
            return

        # Also enforce this invariant for callers that invoke the worker
        # directly (for example, a recovered queued job).
        self.invalidate_translated_cbz_archives(code, job_id)

        final_return_code = 0
        pages = [item["page"] for item in page_reruns]
        translation_boxno_value = status.get("pendingTranslationBoxno")
        try:
            translation_boxno = (
                int(translation_boxno_value)
                if translation_boxno_value is not None
                else None
            )
        except (TypeError, ValueError):
            translation_boxno = None
        start_active_runtime(status)
        for index, page_rerun in enumerate(page_reruns, start=1):
            page = page_rerun["page"]
            resume_from = page_rerun["resumeFrom"]
            status = self.load_status(code, job_id) or status
            status.update(
                {
                    "status": "running",
                    "phase": "Starting",
                    "page": page,
                    "message": f"Rerunning page {page} from {resume_from} ({index}/{len(pages)}).",
                    "finishedAt": None,
                    "pendingResumeFrom": resume_from,
                    "pendingResumePage": page,
                    "lastResumeFrom": resume_from,
                    "lastResumePage": page,
                    "pendingPageReruns": page_reruns[index - 1 :],
                }
            )
            status.pop("pendingPackageOnly", None)
            self.save_status(code, job_id, status)
            final_return_code = self.run_pipeline_process(
                code,
                job_id,
                self.build_command(
                    code,
                    job_id,
                    resume_from,
                    page,
                    single_page=True,
                    skip_package=True,
                    translation_boxno=translation_boxno,
                ),
                f"rerun page {page}",
            )
            if final_return_code != 0:
                break

        selected_translation_only = translation_boxno is not None

        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        finished_at = datetime.now(timezone.utc)
        stop_active_runtime(status, finished_at)
        status["returnCode"] = final_return_code
        status["finishedAt"] = finished_at.isoformat()
        status.pop("pid", None)
        was_terminated = bool(status.pop("pendingTermination", None))
        status.pop("terminatingAt", None)
        status.pop("isPaused", None)
        status.pop("pausedAt", None)

        # Selected retranslation updates translation data only. The existing
        # rendered pages remain the valid job result until typesetting reruns.
        completed = final_return_code == 0 and (
            selected_translation_only or bool(self.final_page_files(code, job_id))
        )
        if was_terminated:
            status.update(
                {
                    "status": "cancelled",
                    "phase": "Cancelled",
                    "page": None,
                    "message": "Page rerun terminated.",
                }
            )
        elif completed:
            if selected_translation_only:
                self.mark_editor_v2_translation_ready_for_typesetting(
                    code, job_id, pages[0]
                )
            else:
                for page_rerun in page_reruns:
                    self.clear_editor_changes_for_pages(
                        code,
                        job_id,
                        [page_rerun["page"]],
                        resume_from=page_rerun["resumeFrom"],
                    )
            status.update(
                {
                    "status": "complete",
                    "phase": "Complete",
                    "page": None,
                    "message": (
                        f"Translation page {pages[0]} boxno {translation_boxno} complete. "
                        "Typesetting is out of date."
                        if selected_translation_only
                        else "Page rerun complete."
                    ),
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

        if completed:
            status.pop("pendingResumeFrom", None)
            status.pop("pendingResumePage", None)
            status.pop("pendingPageReruns", None)
            status.pop("pendingPackageOnly", None)
            status.pop("pendingPackageVariant", None)
            status.pop("pendingWebpQuality", None)
            status.pop("pendingJxlQuality", None)
            status.pop("pendingEditorChangedPages", None)
            status.pop("pendingTranslationBoxno", None)
        elif not status.get("pendingPackageOnly"):
            remaining_reruns = status.get("pendingPageReruns")
            if (
                isinstance(remaining_reruns, list)
                and remaining_reruns
                and isinstance(remaining_reruns[0], dict)
            ):
                retry_page = int(remaining_reruns[0]["page"])
                retry_phase = str(remaining_reruns[0]["resumeFrom"])
                status["pendingResumeFrom"] = retry_phase
                status["pendingResumePage"] = retry_page
                status["lastResumeFrom"] = retry_phase
                status["lastResumePage"] = retry_page
        self.save_status(code, job_id, status)

    def run_job(self, code: str, job_id: str) -> None:
        status = self.load_status(code, job_id) or {
            "category": code,
            "jobId": job_id,
            "createdAt": now_utc(),
        }
        if status.get("pendingPackageOnly"):
            self.run_package_generation_job(code, job_id, status)
            return

        pending_page_reruns = status.get("pendingPageReruns")
        if isinstance(pending_page_reruns, list) and pending_page_reruns:
            self.run_page_rerun_batch_job(code, job_id, status, pending_page_reruns)
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
        start_active_runtime(status)
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
                "finishedAt": None,
            }
        )
        if resume_from is not None:
            status["lastResumeFrom"] = resume_from
            status["lastResumePage"] = resume_page
        self.save_status(code, job_id, status)

        command = self.build_command(code, job_id, resume_from, resume_page)
        return_code = self.run_pipeline_process(
            code,
            job_id,
            command,
            "translation pipeline",
        )
        status = self.load_status(code, job_id) or {"category": code, "jobId": job_id}
        finished_at = datetime.now(timezone.utc)
        stop_active_runtime(status, finished_at)
        status["returnCode"] = return_code
        status["finishedAt"] = finished_at.isoformat()
        status.pop("pid", None)
        status.pop("pendingResumeFrom", None)
        status.pop("pendingResumePage", None)
        status.pop("pendingPageReruns", None)
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
        elif return_code == 0 and self.status_pause_after_ocr(status) and not self.final_page_files(code, job_id):
            self.initialize_editor_v2(code, job_id)
            status.update(
                {
                    "status": "paused",
                    "phase": "OCR review",
                    "page": None,
                    "message": "OCR is ready for review. Edit OCR & Merge, then continue processing.",
                    "isPaused": True,
                    "pausedAt": now_utc(),
                    "reviewCheckpoint": OCR_REVIEW_CHECKPOINT,
                    "pendingResumeFrom": "ocr_structured",
                    "pendingResumePage": 0,
                    "lastResumeFrom": "ocr_structured",
                    "lastResumePage": 0,
                }
            )
        elif return_code == 0 and self.final_page_files(code, job_id):
            had_editor = self.has_editor_v2(code, job_id)
            self.initialize_editor_v2(code, job_id)
            if had_editor and status.get("lastResumeFrom") == "ocr_structured":
                self.clear_editor_changes_for_pages(
                    code,
                    job_id,
                    list(range(len(self.original_page_files(code, job_id)))),
                    "ocr_structured",
                )
            status.pop("reviewCheckpoint", None)
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
    app.add_middleware(UploadSizeLimitMiddleware)

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

    @app.get("/assets/editor-v2/{asset_name}")
    async def editor_v2_asset(asset_name: str) -> FileResponse:
        allowed = {
            "app.js",
            "api.js",
            "store.js",
            "canvas.js",
            "screens.js",
            "editor.css",
        }
        if asset_name not in allowed:
            raise HTTPException(status_code=404, detail="Unknown editor asset.")
        media_type = "text/css" if asset_name.endswith(".css") else "application/javascript"
        return FileResponse(REPO_DIR / "web_editor" / asset_name, media_type=media_type)

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
        ocr_page_workers: str = Form("1"),
        lama_workers: str = Form("1"),
        imagemagick_workers: str = Form("1"),
        vlm_auth_token: str = Form(""),
        paddleocr_vl_auth_token: str = Form(""),
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
                detail="Upload one CBZ or ZIP archive, or upload one or more page images.",
            )

        job_id = manager.create_job_id(code)
        job_path = manager.job_dir(code, job_id)
        input_path = manager.input_path(code, job_id)
        original_filename = cbz_filename
        try:
            job_path.mkdir(parents=True, exist_ok=True)
            try:
                if has_cbz:
                    if Path(cbz_filename).suffix.lower() not in UPLOAD_COMIC_ARCHIVE_EXTENSIONS:
                        raise HTTPException(
                            status_code=400,
                            detail="Comic archive must use the .cbz or .zip extension.",
                        )
                    if cbz is None:
                        raise HTTPException(
                            status_code=400,
                            detail="Upload must include a CBZ or ZIP archive.",
                        )
                    await write_uploaded_file_to_path(
                        cbz,
                        input_path,
                        manager.config.max_upload_bytes,
                        "Uploaded archive",
                    )
                    await run_in_threadpool(
                        validate_uploaded_comic_archive,
                        input_path,
                        cbz_filename,
                    )
                else:
                    _, original_filename = await write_uploaded_images_as_cbz(
                        image_uploads,
                        input_path,
                        manager.config.max_upload_bytes,
                    )
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
                parse_page_workers_form(ocr_page_workers, "OCR workers"),
                parse_page_workers_form(lama_workers, "LaMa workers"),
                parse_page_workers_form(imagemagick_workers, "ImageMagick workers"),
                parse_auth_token_form(vlm_auth_token),
                parse_auth_token_form(paddleocr_vl_auth_token),
            )
        except Exception:
            try:
                shutil.rmtree(job_path)
            except FileNotFoundError:
                pass
            except Exception:
                LOGGER.exception("Could not clean up failed upload job %s", job_path)
            raise

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

    @app.get("/api/job/{code}/{job_id}/editor/v2/pages/{page}/stages/{stage}")
    async def editor_v2_data(
        request: Request, code: str, job_id: str, stage: str, page: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        return JSONResponse(manager.editor_v2_payload(code, job_id, stage, page))

    @app.post("/api/job/{code}/{job_id}/editor/v2/pages/{page}/stages/{stage}")
    async def save_editor_v2_data(
        request: Request, code: str, job_id: str, stage: str, page: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object.")
        return JSONResponse(
            manager.save_editor_v2_update(code, job_id, page, stage, data)
        )

    @app.post("/api/job/{code}/{job_id}/editor/v2/pages/{page}/stages/{stage}/protection")
    async def editor_v2_protection(
        request: Request, code: str, job_id: str, stage: str, page: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object.")
        return JSONResponse(
            manager.set_editor_v2_protection(code, job_id, page, stage, data)
        )

    @app.post("/api/job/{code}/{job_id}/editor/v2/pages/{page}/stages/{stage}/regenerate")
    async def regenerate_editor_v2_page(
        request: Request, code: str, job_id: str, stage: str, page: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        try:
            resume_from = editor_v2.RERUN_FROM[editor_v2.validate_stage(stage)]
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        manager.rerun_completed_job_pages(code, job_id, [page], resume_from)
        return JSONResponse(
            {"ok": True, "url": f"/job/{code}/{job_id}", "resumeFrom": resume_from}
        )

    @app.post("/api/job/{code}/{job_id}/editor/v2/regenerate-changes")
    async def regenerate_editor_v2_changes(
        request: Request, code: str, job_id: str
    ) -> JSONResponse:
        manager = manager_from_request(request)
        pages, resume_by_page = manager.rerun_editor_v2_changes(code, job_id)
        return JSONResponse(
            {
                "ok": True,
                "url": f"/job/{code}/{job_id}",
                "pages": pages,
                "resumeByPage": resume_by_page,
            }
        )

    @app.post("/api/job/{code}/{job_id}/editor/v2/continue")
    async def continue_from_ocr_review(
        request: Request, code: str, job_id: str
    ) -> JSONResponse:
        manager = manager_from_request(request)
        status = manager.require_editable_job(code, job_id, "ocr")
        if not manager.status_is_ocr_review_checkpoint(status):
            raise HTTPException(
                status_code=409,
                detail="This job is not waiting for OCR review.",
            )
        manager.restart_failed_job(code, job_id)
        return JSONResponse({"ok": True, "url": f"/job/{code}/{job_id}"})

    @app.post("/api/job/{code}/{job_id}/editor/v2/pages/{page}/retranslate")
    async def retranslate_editor_v2_page(
        request: Request, code: str, job_id: str, page: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        data = await request.json()
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Request body must be an object.")
        record_id_value = data.get("recordId")
        boxno = data.get("boxno")
        if record_id_value is not None and not isinstance(record_id_value, str):
            raise HTTPException(status_code=400, detail="recordId must be a string.")
        manager.retranslate_editor_v2_page(
            code,
            job_id,
            page,
            record_id_value,
            boxno,
        )
        return JSONResponse(
            {
                "ok": True,
                "url": f"/job/{code}/{job_id}",
                "boxno": boxno if record_id_value is not None else None,
            }
        )

    @app.post("/api/job/{code}/{job_id}/editor/v2/pages/{page}/stages/{stage}/preview")
    async def render_editor_v2_preview(
        request: Request, code: str, job_id: str, stage: str, page: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        if stage != "placement":
            raise HTTPException(status_code=400, detail="Preview is only available for placement.")
        manager.require_editable_job(code, job_id, stage)
        data = await request.json()
        records = data.get("records") if isinstance(data, dict) else None
        clean_records = data.get("cleanRecords") if isinstance(data, dict) else None
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="records must be an array.")
        if not isinstance(clean_records, list):
            raise HTTPException(status_code=400, detail="cleanRecords must be an array.")
        _path, cached, cleaned_with_lama = await run_in_threadpool(
            manager.render_editor_v2_preview,
            code,
            job_id,
            page,
            records,
            clean_records,
        )
        return JSONResponse(
            {
                "ok": True,
                "cached": cached,
                "cleanedWithLama": cleaned_with_lama,
                "url": f"/api/job/{code}/{job_id}/editor/v2/preview-image/{page}?v={int(time.time())}",
            }
        )

    @app.get("/api/job/{code}/{job_id}/editor/v2/preview-image/{page}")
    async def editor_v2_preview_image(
        request: Request, code: str, job_id: str, page: int
    ) -> FileResponse:
        manager = manager_from_request(request)
        path = manager.editor_v2_preview_path(code, job_id, page)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No exact preview has been rendered.")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/job/{code}/{job_id}/editor/v2/image/{kind}/{page}")
    async def editor_v2_stage_image(
        request: Request, code: str, job_id: str, kind: str, page: int
    ) -> FileResponse:
        manager = manager_from_request(request)
        manager.require_editable_job(code, job_id)
        original = manager.original_page_path(code, job_id, page)
        if kind == "original":
            path = original
        elif kind == "cleaned":
            path = manager.output_dir(code, job_id) / "pages" / "cleaned" / f"{original.stem}.png"
        elif kind == "final":
            path = manager.output_dir(code, job_id) / "pages" / "final" / f"{original.stem}.png"
        else:
            raise HTTPException(status_code=404, detail="Unknown editor image.")
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"The {kind} page image is missing.")
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    @app.get("/api/job/{code}/{job_id}/editor/v2/revisions")
    async def editor_v2_revisions(
        request: Request, code: str, job_id: str
    ) -> JSONResponse:
        manager = manager_from_request(request)
        return JSONResponse({"revisions": manager.editor_v2_revisions(code, job_id)})

    @app.post("/api/job/{code}/{job_id}/editor/v2/revisions/{revision}/restore")
    async def restore_editor_v2_revision(
        request: Request, code: str, job_id: str, revision: int
    ) -> JSONResponse:
        manager = manager_from_request(request)
        return JSONResponse(manager.restore_editor_v2_revision(code, job_id, revision))

    @app.post("/api/job/{code}/{job_id}/editor/v2/ocr-crop")
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
        ocr_page_workers: str = Form("1"),
        lama_workers: str = Form("1"),
        imagemagick_workers: str = Form("1"),
        vlm_auth_token: str = Form(""),
        paddleocr_vl_auth_token: str = Form(""),
        clear_vlm_auth_token: str | None = Form(None),
        clear_paddleocr_vl_auth_token: str | None = Form(None),
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
            parse_page_workers_form(ocr_page_workers, "OCR workers"),
            parse_page_workers_form(lama_workers, "LaMa workers"),
            parse_page_workers_form(imagemagick_workers, "ImageMagick workers"),
            parse_auth_token_form(vlm_auth_token),
            parse_auth_token_form(paddleocr_vl_auth_token),
            parse_checkbox(clear_vlm_auth_token),
            parse_checkbox(clear_paddleocr_vl_auth_token),
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
        resume_from = parse_rerun_job_stages(
            [str(value) for value in form.getlist("rerun_stage")]
        )

        page_spec = str(form.get("page_spec") or "")
        manager.rerun_completed_job_pages(
            code,
            job_id,
            parse_page_selection(page_spec, allow_empty=True),
            resume_from,
        )
        return RedirectResponse(f"/job/{code}/{job_id}", status_code=303)

    @app.post("/job/{code}/{job_id}/generate-download/{variant}")
    async def generate_download(
        request: Request,
        code: str,
        job_id: str,
        variant: str,
    ) -> RedirectResponse:
        manager = manager_from_request(request)
        manager.generate_download(code, job_id, variant)
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
            raise HTTPException(status_code=404, detail="Original input archive is not available.")
        original_suffix = (
            ".zip"
            if str(status.get("inputFilename") or "").lower().endswith(".zip")
            else ".cbz"
        )
        return FileResponse(
            input_path,
            media_type=(
                "application/zip"
                if original_suffix == ".zip"
                else "application/vnd.comicbook+zip"
            ),
            filename=f"{code}_{job_id}_original{original_suffix}",
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
