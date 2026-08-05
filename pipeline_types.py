"""Shared immutable types and errors for the translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot continue safely."""


class PipelineCancelled(PipelineError):
    """Raised when the pipeline receives a termination signal."""


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
    ocr_page_workers: int
    lama_workers: int
    imagemagick_workers: int
    ocr: OCRConfig
    postprocess: PostprocessConfig
