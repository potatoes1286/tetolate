"""Versioned, field-level editor overrides and pipeline reconciliation."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Iterable


SCHEMA_VERSION = 2
STAGE_ORDER = ("ocr", "structure", "erase", "translation", "placement")
STAGE_LABELS = {
    "ocr": "OCR & Merge",
    "structure": "Structure",
    "erase": "Erase & Alternate Placement",
    "translation": "Translation",
    "placement": "Typesetting",
}
STAGE_ARTIFACTS = {
    "ocr": ("ocr_raw", "ocr_merged"),
    "structure": ("ocr_structured",),
    "erase": ("ocr_structured", "alt_placement"),
    "translation": ("translations",),
    "placement": ("placements",),
}
ARTIFACT_STAGES = {
    artifact: tuple(stage for stage in STAGE_ORDER if artifact in STAGE_ARTIFACTS[stage])
    for artifact in {item for values in STAGE_ARTIFACTS.values() for item in values}
}
STAGE_FIELDS: dict[str, dict[str, frozenset[str] | None]] = {
    "ocr": {"ocr_raw": None, "ocr_merged": None},
    "structure": {
        "ocr_structured": frozenset(
            {"text", "sfx", "reject", "region", "sourceBoxnos", "sourceTexts"}
        )
    },
    "erase": {
        "ocr_structured": frozenset(
            {"openLettering", "safeToEraseOriginal", "altPlacementReason"}
        ),
        "alt_placement": None,
    },
    "translation": {"translations": frozenset({"englishText", "text"})},
    "placement": {
        "placements": frozenset(
            {
                "placementRegion",
                "box_2d",
                "font",
                "fill",
                "stroke",
                "strokeWidth",
                "gravity",
                "fontSizeWidthPercent",
                "manualLineBreaks",
                "minPointSize",
                "maxPointSize",
            }
        )
    },
}
RERUN_FROM = {
    "ocr": "ocr_structured",
    "structure": "alt_placement",
    "erase": "translations",
    "translation": "placements",
    "placement": "render",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def validate_stage(stage: str) -> str:
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown editor stage: {stage}")
    return stage


def downstream_stages(stage: str) -> tuple[str, ...]:
    stage = validate_stage(stage)
    return STAGE_ORDER[STAGE_ORDER.index(stage) + 1 :]


def default_manifest() -> dict[str, Any]:
    now = utc_now()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "revision": 0,
        "createdAt": now,
        "updatedAt": now,
        "pages": {},
        "translationNotes": {"job": "", "pages": {}},
    }


def normalize_manifest(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("schemaVersion") != SCHEMA_VERSION:
        return None
    result = default_manifest()
    result["revision"] = max(0, int(value.get("revision", 0)))
    for key in ("createdAt", "updatedAt"):
        if isinstance(value.get(key), str):
            result[key] = value[key]
    if isinstance(value.get("pages"), dict):
        result["pages"] = copy.deepcopy(value["pages"])
    notes = value.get("translationNotes")
    if isinstance(notes, dict):
        result["translationNotes"] = copy.deepcopy(notes)
    return result


def stage_definitions() -> list[dict[str, Any]]:
    return [
        {
            "key": stage,
            "label": STAGE_LABELS[stage],
            "artifacts": list(STAGE_ARTIFACTS[stage]),
            "invalidates": list(downstream_stages(stage)),
            "rerunFrom": RERUN_FROM[stage],
        }
        for stage in STAGE_ORDER
    ]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_id(stage: str, page: int, record: dict[str, Any], index: int) -> str:
    existing = record.get("recordId")
    if isinstance(existing, str) and existing:
        return existing
    boxno = record.get("boxno", record.get("mergedBoxno", index))
    if stage == "ocr_raw":
        identity = [boxno, record.get("region"), record.get("text")]
        prefix = "raw"
    elif stage == "ocr_merged":
        identity = [record.get("sourceBoxnos", []), boxno]
        prefix = "group"
    else:
        identity = [boxno, record.get("sourceBoxnos", [])]
        prefix = "text"
    digest = hashlib.sha256(_canonical([page, identity]).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def hydrate_ids(stage: str, page: int, records: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        record = copy.deepcopy(item)
        record["recordId"] = record_id(stage, page, record, index)
        result.append(record)
    return result


def empty_override() -> dict[str, Any]:
    return {
        "fields": {},
        "added": {},
        "deleted": {},
        "order": None,
        "freeze": None,
    }


def _page(manifest: dict[str, Any], page: int) -> dict[str, Any]:
    return manifest.setdefault("pages", {}).setdefault(
        str(page), {"stages": {}, "stale": [], "pendingStages": [], "savedRevision": 0}
    )


def stage_override(manifest: dict[str, Any], page: int, stage: str) -> dict[str, Any]:
    validate_stage(stage)
    stages = _page(manifest, page).setdefault("stages", {})
    value = stages.setdefault(stage, {"artifacts": {}, "frozen": False})
    value.setdefault("artifacts", {})
    value.setdefault("frozen", False)
    return value


def artifact_override(
    manifest: dict[str, Any], page: int, stage: str, artifact: str
) -> dict[str, Any]:
    value = stage_override(manifest, page, stage)["artifacts"].setdefault(
        artifact, empty_override()
    )
    for key, default in empty_override().items():
        value.setdefault(key, copy.deepcopy(default))
    return value


def _record_map(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        item["recordId"]: item
        for item in records
        if isinstance(item.get("recordId"), str)
    }


def _field_value(value: dict[str, Any], field: str) -> Any:
    return value[field] if field in value else None


def update_artifact_override(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    baseline_records: list[dict[str, Any]],
    submitted_records: list[dict[str, Any]],
) -> None:
    override = artifact_override(manifest, page, stage, artifact)
    baseline = hydrate_ids(artifact, page, baseline_records)
    submitted = hydrate_ids(artifact, page, submitted_records)
    baseline_by_id = _record_map(baseline)
    submitted_by_id = _record_map(submitted)
    editable = STAGE_FIELDS[stage].get(artifact)

    old_fields = override.get("fields", {})
    fields: dict[str, dict[str, Any]] = {}
    for rid, current in submitted_by_id.items():
        generated = baseline_by_id.get(rid)
        if generated is None:
            continue
        names = set(current) | set(generated)
        names.discard("recordId")
        if editable is not None:
            names &= set(editable)
        changed: dict[str, Any] = {}
        for name in sorted(names):
            current_value = _field_value(current, name)
            baseline_value = _field_value(generated, name)
            if current_value == baseline_value:
                continue
            prior = old_fields.get(rid, {}).get(name)
            protected = bool(
                isinstance(prior, dict)
                and prior.get("value") == current_value
                and prior.get("protected") is False
            ) is False
            changed[name] = {"value": copy.deepcopy(current_value), "protected": protected}
        if changed:
            fields[rid] = changed
    override["fields"] = fields

    old_added = override.get("added", {})
    override["added"] = {
        rid: {
            "record": copy.deepcopy(record),
            "protected": not (
                isinstance(old_added.get(rid), dict)
                and old_added[rid].get("record") == record
                and old_added[rid].get("protected") is False
            ),
        }
        for rid, record in submitted_by_id.items()
        if rid not in baseline_by_id
    }
    old_deleted = override.get("deleted", {})
    deleted: dict[str, Any] = {}
    for rid, record in baseline_by_id.items():
        if rid in submitted_by_id:
            continue
        prior = old_deleted.get(rid)
        deleted[rid] = {
            "record": copy.deepcopy(record),
            "region": copy.deepcopy(record.get("region")),
            "protected": not (
                isinstance(prior, dict) and prior.get("protected") is False
            ),
        }
    override["deleted"] = deleted
    baseline_order = list(baseline_by_id)
    current_order = list(submitted_by_id)
    if current_order != baseline_order:
        prior_order = override.get("order")
        override["order"] = {
            "recordIds": current_order,
            "protected": not (
                isinstance(prior_order, dict)
                and prior_order.get("recordIds") == current_order
                and prior_order.get("protected") is False
            ),
        }
    else:
        override["order"] = None


def _iou(first: Any, second: Any) -> float:
    if not (
        isinstance(first, list)
        and isinstance(second, list)
        and len(first) == 4
        and len(second) == 4
    ):
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def apply_artifact_override(
    records: list[dict[str, Any]],
    override: dict[str, Any],
    artifact: str,
    page: int,
    *,
    protected_only: bool = False,
) -> list[dict[str, Any]]:
    hydrated = hydrate_ids(artifact, page, records)
    freeze = override.get("freeze")
    if isinstance(freeze, dict) and (not protected_only or freeze.get("protected")):
        return hydrate_ids(artifact, page, freeze.get("records", []))

    deleted = override.get("deleted", {})
    kept: list[dict[str, Any]] = []
    for record in hydrated:
        rid = record["recordId"]
        tombstone = deleted.get(rid)
        should_delete = isinstance(tombstone, dict) and (
            not protected_only or tombstone.get("protected")
        )
        if not should_delete and artifact == "ocr_raw":
            should_delete = any(
                (not protected_only or item.get("protected"))
                and _iou(record.get("region"), item.get("region")) >= 0.5
                for item in deleted.values()
                if isinstance(item, dict)
            )
        if not should_delete:
            kept.append(record)

    by_id = _record_map(kept)
    for rid, field_values in override.get("fields", {}).items():
        target = by_id.get(rid)
        if target is None or not isinstance(field_values, dict):
            continue
        for name, item in field_values.items():
            if not isinstance(item, dict) or (protected_only and not item.get("protected")):
                continue
            target[name] = copy.deepcopy(item.get("value"))

    for rid, item in override.get("added", {}).items():
        if not isinstance(item, dict) or (protected_only and not item.get("protected")):
            continue
        record = item.get("record")
        if isinstance(record, dict):
            by_id[rid] = copy.deepcopy(record)

    result = list(by_id.values())
    order = override.get("order")
    if isinstance(order, dict) and (not protected_only or order.get("protected")):
        positions = {rid: index for index, rid in enumerate(order.get("recordIds", []))}
        result.sort(key=lambda item: positions.get(item.get("recordId"), len(positions)))
    return result


def effective_records(
    manifest: dict[str, Any],
    page: int,
    artifact: str,
    baseline: list[dict[str, Any]],
    *,
    protected_only: bool = False,
    through_stage: str | None = None,
) -> list[dict[str, Any]]:
    records = hydrate_ids(artifact, page, baseline)
    page_value = manifest.get("pages", {}).get(str(page), {})
    stages = page_value.get("stages", {}) if isinstance(page_value, dict) else {}
    for stage in ARTIFACT_STAGES.get(artifact, ()):
        if through_stage is not None and STAGE_ORDER.index(stage) > STAGE_ORDER.index(through_stage):
            continue
        stage_value = stages.get(stage, {}) if isinstance(stages, dict) else {}
        artifacts = stage_value.get("artifacts", {}) if isinstance(stage_value, dict) else {}
        override = artifacts.get(artifact)
        if isinstance(override, dict):
            records = apply_artifact_override(
                records, override, artifact, page, protected_only=protected_only
            )
    return records


def discard_unprotected(
    manifest: dict[str, Any], page: int, stage: str, artifact: str
) -> None:
    value = artifact_override(manifest, page, stage, artifact)
    fields: dict[str, Any] = {}
    for rid, record_fields in value.get("fields", {}).items():
        protected_fields = {
            name: item
            for name, item in record_fields.items()
            if isinstance(item, dict) and item.get("protected")
        }
        if protected_fields:
            fields[rid] = protected_fields
    value["fields"] = fields
    for key in ("added", "deleted"):
        value[key] = {
            rid: item
            for rid, item in value.get(key, {}).items()
            if isinstance(item, dict) and item.get("protected")
        }
    order = value.get("order")
    if isinstance(order, dict) and not order.get("protected"):
        value["order"] = None


def set_artifact_protection(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    protected: bool,
) -> None:
    value = artifact_override(manifest, page, stage, artifact)
    for record_fields in value.get("fields", {}).values():
        if not isinstance(record_fields, dict):
            continue
        for item in record_fields.values():
            if isinstance(item, dict):
                item["protected"] = bool(protected)
    for key in ("added", "deleted"):
        for item in value.get(key, {}).values():
            if isinstance(item, dict):
                item["protected"] = bool(protected)
    order = value.get("order")
    if isinstance(order, dict):
        order["protected"] = bool(protected)


def discard_unprotected_record(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    record_id_value: str,
) -> None:
    value = artifact_override(manifest, page, stage, artifact)
    record_fields = value.get("fields", {}).get(record_id_value)
    if isinstance(record_fields, dict):
        protected_fields = {
            name: item
            for name, item in record_fields.items()
            if isinstance(item, dict) and item.get("protected")
        }
        if protected_fields:
            value["fields"][record_id_value] = protected_fields
        else:
            value["fields"].pop(record_id_value, None)
    for key in ("added", "deleted"):
        item = value.get(key, {}).get(record_id_value)
        if isinstance(item, dict) and not item.get("protected"):
            value[key].pop(record_id_value, None)


def set_field_protection(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    record_id_value: str,
    field: str,
    protected: bool,
) -> bool:
    value = artifact_override(manifest, page, stage, artifact)
    item = value.get("fields", {}).get(record_id_value, {}).get(field)
    if not isinstance(item, dict):
        return False
    item["protected"] = bool(protected)
    return True


def remove_field_override(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    record_id_value: str,
    field: str,
) -> bool:
    value = artifact_override(manifest, page, stage, artifact)
    record_fields = value.get("fields", {}).get(record_id_value)
    if not isinstance(record_fields, dict) or field not in record_fields:
        return False
    del record_fields[field]
    if not record_fields:
        value["fields"].pop(record_id_value, None)
    return True


def set_record_protection(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    record_id_value: str,
    protected: bool,
) -> bool:
    value = artifact_override(manifest, page, stage, artifact)
    changed = False
    for field in value.get("fields", {}).get(record_id_value, {}).values():
        if isinstance(field, dict):
            field["protected"] = bool(protected)
            changed = True
    for key in ("added", "deleted"):
        item = value.get(key, {}).get(record_id_value)
        if isinstance(item, dict):
            item["protected"] = bool(protected)
            changed = True
    return changed


def remove_record_override(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    artifact: str,
    record_id_value: str,
) -> bool:
    value = artifact_override(manifest, page, stage, artifact)
    changed = False
    for key in ("fields", "added", "deleted"):
        mapping = value.get(key, {})
        if record_id_value in mapping:
            mapping.pop(record_id_value, None)
            changed = True
    order = value.get("order")
    if isinstance(order, dict) and record_id_value in order.get("recordIds", []):
        value["order"] = None
        changed = True
    return changed


def freeze_stage(
    manifest: dict[str, Any],
    page: int,
    stage: str,
    records_by_artifact: dict[str, list[dict[str, Any]]],
    frozen: bool,
) -> None:
    value = stage_override(manifest, page, stage)
    value["frozen"] = bool(frozen)
    for artifact in STAGE_ARTIFACTS[stage]:
        override = artifact_override(manifest, page, stage, artifact)
        override["freeze"] = (
            {
                "records": hydrate_ids(artifact, page, records_by_artifact.get(artifact, [])),
                "protected": True,
            }
            if frozen
            else None
        )


def mark_saved(manifest: dict[str, Any], page: int, stage: str) -> int:
    manifest["revision"] = int(manifest.get("revision", 0)) + 1
    manifest["updatedAt"] = utc_now()
    page_value = _page(manifest, page)
    page_value["savedRevision"] = manifest["revision"]
    stale = set(page_value.get("stale", []))
    pending = set(page_value.get("pendingStages", []))
    stage_index = STAGE_ORDER.index(stage)
    accepted_stages = STAGE_ORDER[: stage_index + 1]
    stale.difference_update(accepted_stages)
    pending.difference_update(accepted_stages)
    pending.add(stage)
    page_value["pendingStages"] = [item for item in STAGE_ORDER if item in pending]
    stale.update(downstream_stages(stage))
    page_value["stale"] = [item for item in STAGE_ORDER if item in stale]
    stage_override(manifest, page, stage)["savedRevision"] = manifest["revision"]
    return manifest["revision"]


def mark_regenerated(manifest: dict[str, Any], page: int, resume_from: str) -> None:
    start = next(
        (index for index, stage in enumerate(STAGE_ORDER) if RERUN_FROM[stage] == resume_from),
        len(STAGE_ORDER) - 1,
    )
    page_value = _page(manifest, page)
    stale = set(page_value.get("stale", []))
    stale.difference_update(STAGE_ORDER[start:])
    page_value["stale"] = [item for item in STAGE_ORDER if item in stale]
    pending = set(page_value.get("pendingStages", []))
    pending.difference_update(STAGE_ORDER[start:])
    page_value["pendingStages"] = [item for item in STAGE_ORDER if item in pending]
    for pending_stage in pending:
        stale.update(downstream_stages(pending_stage))
    page_value["stale"] = [item for item in STAGE_ORDER if item in stale]


def stage_status(manifest: dict[str, Any], page: int, stage: str) -> dict[str, Any]:
    page_value = manifest.get("pages", {}).get(str(page), {})
    stages = page_value.get("stages", {}) if isinstance(page_value, dict) else {}
    value = stages.get(stage, {}) if isinstance(stages, dict) else {}
    overrides = value.get("artifacts", {}) if isinstance(value, dict) else {}
    changed = any(
        bool(item.get("fields") or item.get("added") or item.get("deleted") or item.get("order") or item.get("freeze"))
        for item in overrides.values()
        if isinstance(item, dict)
    )
    return {
        "stage": stage,
        "label": STAGE_LABELS[stage],
        "changed": changed,
        "frozen": bool(value.get("frozen")) if isinstance(value, dict) else False,
        "stale": stage in page_value.get("stale", []) if isinstance(page_value, dict) else False,
        "pending": stage in page_value.get("pendingStages", []) if isinstance(page_value, dict) else False,
        "savedRevision": int(value.get("savedRevision", 0)) if isinstance(value, dict) else 0,
    }


def protection_payload(manifest: dict[str, Any], page: int, stage: str) -> dict[str, Any]:
    value = stage_override(manifest, page, stage)
    records: dict[str, dict[str, bool]] = {}
    record_states: dict[str, dict[str, Any]] = {}
    for override in value.get("artifacts", {}).values():
        if not isinstance(override, dict):
            continue
        for rid, fields in override.get("fields", {}).items():
            target = records.setdefault(rid, {})
            for name, item in fields.items():
                if isinstance(item, dict):
                    target[name] = bool(item.get("protected"))
        for kind in ("added", "deleted"):
            for rid, item in override.get(kind, {}).items():
                if isinstance(item, dict):
                    record_states[rid] = {
                        "kind": kind,
                        "protected": bool(item.get("protected")),
                    }
    return {
        "frozen": bool(value.get("frozen")),
        "records": records,
        "recordStates": record_states,
    }


def earliest_rerun(stages: Iterable[str]) -> str:
    selected = set(stages)
    return next((RERUN_FROM[stage] for stage in STAGE_ORDER if stage in selected), "render")


def pending_changes(manifest: dict[str, Any]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for raw_page, value in manifest.get("pages", {}).items():
        if not isinstance(value, dict):
            continue
        stages = {item for item in value.get("pendingStages", []) if item in STAGE_ORDER}
        if stages:
            try:
                result[int(raw_page)] = stages
            except (TypeError, ValueError):
                continue
    return result
