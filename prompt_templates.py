"""Load and render editable VLM prompt templates from disk."""

from __future__ import annotations

import os
from pathlib import Path
from string import Template
from typing import Any

from pipeline_types import PipelineError


DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "TETOLATE_DATA_DIR",
        Path(__file__).resolve().parent / "data",
    )
).expanduser()
DEFAULT_PROMPTS_DIR = DEFAULT_DATA_DIR / "prompts"
PROMPTS_DIR_ENV = "TETOLATE_PROMPTS_DIR"


def prompts_dir() -> Path:
    configured = os.environ.get(PROMPTS_DIR_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_PROMPTS_DIR


def prompt_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".txt":
        raise PipelineError(f"Invalid prompt template name: {name!r}")
    return prompts_dir() / relative


def load_prompt(name: str, /, **values: Any) -> str:
    path = prompt_path(name)
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PipelineError(f"Prompt template is missing: {path}") from exc
    except OSError as exc:
        raise PipelineError(f"Could not read prompt template {path}: {exc}") from exc

    try:
        rendered = Template(source).substitute(
            {key: str(value) for key, value in values.items()}
        )
    except KeyError as exc:
        raise PipelineError(
            f"Prompt template {path} requires missing value {exc.args[0]!r}."
        ) from exc
    except ValueError as exc:
        raise PipelineError(f"Prompt template {path} is invalid: {exc}") from exc
    return rendered.strip()
