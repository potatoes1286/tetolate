"""Lightweight pipeline hook for applying protected web-editor overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import editor_v2


_manifest_path: Path | None = None
_baseline_dir: Path | None = None


def configure(manifest_path: Path | None, baseline_dir: Path | None) -> None:
    global _manifest_path, _baseline_dir
    _manifest_path = manifest_path
    _baseline_dir = baseline_dir


def enabled() -> bool:
    return _manifest_path is not None and _baseline_dir is not None


def _load_manifest() -> dict[str, Any] | None:
    if not enabled():
        return None
    assert _manifest_path is not None
    try:
        return editor_v2.normalize_manifest(
            json.loads(_manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def reconcile_records(
    artifact: str,
    page: int,
    records: list[dict[str, Any]],
    producing_stage: str,
) -> list[dict[str, Any]]:
    manifest = _load_manifest()
    if manifest is None:
        return records
    assert _manifest_path is not None
    assert _baseline_dir is not None

    generated = editor_v2.hydrate_ids(artifact, page, records)
    _write_json(_baseline_dir / artifact / f"page_{page:04d}.json", generated)
    if artifact in editor_v2.STAGE_ARTIFACTS.get(producing_stage, ()):
        editor_v2.discard_unprotected(manifest, page, producing_stage, artifact)
    effective = editor_v2.effective_records(
        manifest,
        page,
        artifact,
        generated,
        protected_only=True,
        through_stage=producing_stage,
    )
    _write_json(_manifest_path, manifest)
    return effective


def reconcile_record_subset(
    artifact: str,
    page: int,
    replacement_records: list[dict[str, Any]],
    existing_records: list[dict[str, Any]],
    producing_stage: str,
    key_field: str,
) -> list[dict[str, Any]]:
    manifest = _load_manifest()
    if manifest is None:
        replacements = {item.get(key_field): item for item in replacement_records}
        result = [
            replacements.get(item.get(key_field), item)
            for item in existing_records
        ]
        existing_keys = {item.get(key_field) for item in existing_records}
        result.extend(
            item for key, item in replacements.items() if key not in existing_keys
        )
        return result
    assert _manifest_path is not None
    assert _baseline_dir is not None

    baseline_path = _baseline_dir / artifact / f"page_{page:04d}.json"
    try:
        baseline_value = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        baseline_value = existing_records
    if not isinstance(baseline_value, list):
        baseline_value = existing_records

    baseline = editor_v2.hydrate_ids(artifact, page, baseline_value)
    replacements = editor_v2.hydrate_ids(artifact, page, replacement_records)
    replacement_by_key = {item.get(key_field): item for item in replacements}
    generated: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for item in baseline:
        key = item.get(key_field)
        generated.append(replacement_by_key.get(key, item))
        seen.add(key)
    generated.extend(
        item for key, item in replacement_by_key.items() if key not in seen
    )
    _write_json(baseline_path, generated)

    for record in replacements:
        record_id_value = record.get("recordId")
        if isinstance(record_id_value, str):
            editor_v2.discard_unprotected_record(
                manifest,
                page,
                producing_stage,
                artifact,
                record_id_value,
            )
    effective = editor_v2.effective_records(
        manifest,
        page,
        artifact,
        generated,
        protected_only=True,
        through_stage=producing_stage,
    )
    _write_json(_manifest_path, manifest)
    return effective


def apply_protected_records(
    artifact: str,
    page: int,
    records: list[dict[str, Any]],
    through_stage: str,
) -> list[dict[str, Any]]:
    manifest = _load_manifest()
    if manifest is None:
        return records
    return editor_v2.effective_records(
        manifest,
        page,
        artifact,
        records,
        protected_only=True,
        through_stage=through_stage,
    )
