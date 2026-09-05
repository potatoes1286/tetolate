"""OpenAI-compatible VLM transport, streaming, retry, and status reporting."""

from __future__ import annotations

import base64
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from pipeline_types import (
    PipelineCancelled,
    PipelineConfig,
    PipelineError,
    VLMConfig,
    VLMStreamResult,
)
from prompt_templates import load_prompt


DEFAULT_VLM_MODEL_LOADING_RETRIES = 6
VLM_MODEL_LOADING_BACKOFF_SECONDS = (10, 20, 40, 60, 60, 60)
REASONING_FIELD_NAMES = (
    "reasoning_content",
    "reasoning",
    "reasoning_delta",
    "thoughts",
    "thinking",
    "analysis",
)


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

    model = resolve_vlm_model(config.vlm)
    return replace(config, vlm=replace(config.vlm, model=model))


def call_vlm(
    config: VLMConfig,
    prompt: str,
    image_path: Path | None,
    label: str,
    system_prompt: str | None = None,
) -> VLMStreamResult:
    if system_prompt is None:
        system_prompt = load_prompt("system_json.txt")
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


def list_vlm_model_ids(config: VLMConfig) -> list[str]:
    """List usable VLM model IDs in the endpoint's reported order."""
    client = openai_client(config)
    try:
        try:
            models = client.models.list()
        except PipelineCancelled:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Could not list VLM models from {config.base_url}."
            ) from exc
    finally:
        close_openai_client(client)

    data = getattr(models, "data", None)
    if not data:
        raise PipelineError(f"No VLM models were returned by {config.base_url}.")

    model_ids: list[str] = []
    seen: set[str] = set()
    for model in data:
        model_id = object_field(model, "id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        model_ids.append(model_id)

    if not model_ids:
        raise PipelineError(f"No usable VLM model IDs were returned by {config.base_url}.")
    return model_ids


def resolve_vlm_model(config: VLMConfig) -> str:
    if config.model:
        return config.model

    model_id = list_vlm_model_ids(config)[0]
    print(f"Using discovered VLM model: {model_id}", file=sys.stderr)
    return model_id
