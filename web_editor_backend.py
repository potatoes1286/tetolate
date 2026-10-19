"""Editor persistence and OCR operations mixed into the web job manager."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from PIL import Image

import clean_text_regions
import editor_v2
import lama_inpaint
import overlay_text
import paddle_ocr_image
import translate_cbz
from web_storage import write_json_atomic


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


def offset_record_region(
    record: dict[str, Any],
    left: int,
    top: int,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    shifted = dict(record)
    region = shifted.get("region")
    if not isinstance(region, list) or len(region) != 4:
        return None
    try:
        x0, y0, x1, y1 = [round(float(value)) for value in region]
    except (TypeError, ValueError):
        return None
    x0 = max(0, min(width, x0 + left))
    x1 = max(0, min(width, x1 + left))
    y0 = max(0, min(height, y0 + top))
    y1 = max(0, min(height, y1 + top))
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


class EditorManagerMixin:
    """Provides editor persistence, validation, and OCR operations for JobManager."""

    def editor_v2_manifest_path(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / "editor_v2.json"

    def editor_v2_generated_path(
        self, code: str, job_id: str, artifact: str, page: int
    ) -> Path:
        if artifact not in editor_v2.ARTIFACT_STAGES:
            raise HTTPException(status_code=404, detail="Unknown editor artifact.")
        return (
            self.editor_meta_dir(code, job_id)
            / "generated_v2"
            / artifact
            / f"page_{page:04d}.json"
        )

    def editor_v2_history_dir(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / "history_v2"

    def editor_v2_preview_path(self, code: str, job_id: str, page: int) -> Path:
        return self.editor_meta_dir(code, job_id) / "previews" / f"page_{page:04d}.png"

    def editor_v2_clean_preview_path(self, code: str, job_id: str, page: int, key: str) -> Path:
        return (
            self.editor_meta_dir(code, job_id)
            / "previews"
            / f"page_{page:04d}.{key}.cleaned.png"
        )

    def has_editor_v2(self, code: str, job_id: str) -> bool:
        return self.editor_v2_manifest_path(code, job_id).is_file()

    def load_editor_v2_manifest(self, code: str, job_id: str) -> dict[str, Any]:
        path = self.editor_v2_manifest_path(code, job_id)
        if not path.is_file():
            raise HTTPException(
                status_code=409,
                detail="Editor data is unavailable for this job.",
            )
        try:
            manifest = editor_v2.normalize_manifest(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="The editor manifest is invalid.") from exc
        if manifest is None:
            raise HTTPException(status_code=500, detail="Unsupported editor manifest version.")
        return manifest

    def save_editor_v2_manifest(
        self,
        code: str,
        job_id: str,
        manifest: dict[str, Any],
        *,
        keep_history: bool = True,
    ) -> None:
        normalized = editor_v2.normalize_manifest(manifest)
        if normalized is None:
            raise HTTPException(status_code=500, detail="Cannot save invalid editor state.")
        write_json_atomic(self.editor_v2_manifest_path(code, job_id), normalized)
        if keep_history:
            history_path = (
                self.editor_v2_history_dir(code, job_id)
                / f"revision_{normalized['revision']:08d}.json"
            )
            write_json_atomic(history_path, normalized)

    def initialize_editor_v2(self, code: str, job_id: str) -> None:
        """Create editor state when a job completes."""
        with self._lock:
            if self.has_editor_v2(code, job_id):
                return
            page_files = self.original_page_files(code, job_id)
            if not page_files:
                return
            manifest = editor_v2.default_manifest()
            notes_path = self.translation_notes_path(code, job_id)
            if notes_path.is_file():
                try:
                    notes = json.loads(notes_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    notes = None
                if isinstance(notes, dict):
                    manifest["translationNotes"] = {
                        "job": notes.get("job", "") if isinstance(notes.get("job"), str) else "",
                        "pages": notes.get("pages", {}) if isinstance(notes.get("pages"), dict) else {},
                    }
            for page in range(len(page_files)):
                for artifact in editor_v2.ARTIFACT_STAGES:
                    source = (
                        self.output_dir(code, job_id)
                        / "data"
                        / artifact
                        / f"page_{page:04d}.json"
                    )
                    if source.is_file():
                        target = self.editor_v2_generated_path(code, job_id, artifact, page)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            self.save_editor_v2_manifest(code, job_id, manifest)

    def load_editor_v2_baseline(
        self, code: str, job_id: str, artifact: str, page: int
    ) -> list[dict[str, Any]]:
        if artifact == "clean_boxes":
            return []  # Manual-only records must never become a generated baseline.
        path = self.editor_v2_generated_path(code, job_id, artifact, page)
        if not path.is_file():
            path = (
                self.output_dir(code, job_id)
                / "data"
                / artifact
                / f"page_{page:04d}.json"
            )
        if not path.is_file():
            return []
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Invalid generated {artifact} data for page {page}."
            ) from exc
        if not isinstance(value, list):
            raise HTTPException(status_code=500, detail=f"Generated {artifact} data is not a list.")
        return editor_v2.hydrate_ids(artifact, page, value)

    def editor_v2_effective_records(
        self,
        code: str,
        job_id: str,
        manifest: dict[str, Any],
        artifact: str,
        page: int,
    ) -> list[dict[str, Any]]:
        records = editor_v2.effective_records(
            manifest,
            page,
            artifact,
            self.load_editor_v2_baseline(code, job_id, artifact, page),
        )
        if artifact == "ocr_merged":
            records = editor_v2.complete_manual_ocr_groups(manifest, page,
                self.editor_v2_effective_records(code, job_id, manifest, "ocr_raw", page), records)
        return records

    def add_pending_ocr_descendants(
        self,
        artifacts: dict[str, list[dict[str, Any]]],
        page: int,
        rejected_sources: set[int],
        reserved_boxnos: set[int],
    ) -> None:
        structured = artifacts.setdefault("ocr_structured", [])
        represented_sources = {source for record in structured for source in record.get("sourceBoxnos", [])}
        raw_by_boxno = {record["boxno"]: record for record in artifacts.get("ocr_raw", [])}
        for merged in artifacts.get("ocr_merged", []):
            sources = [source for source in merged.get("sourceBoxnos", [])
                       if source not in represented_sources and source not in rejected_sources]
            if not sources:
                continue
            pending = merged
            if len(sources) != len(merged.get("sourceBoxnos", [])):
                # A regrouped box can contain both old and newly added text.
                # Show only missing sources until Structure is regenerated.
                raw = [raw_by_boxno[source] for source in sources if source in raw_by_boxno]
                regions = [record["region"] for record in raw]
                texts = [str(record.get("text", "")) for record in raw]
                pending = {**merged, "sourceTexts": texts, "text": "".join(texts)}
                if regions:
                    pending["region"] = [min(region[0] for region in regions), min(region[1] for region in regions),
                                         max(region[2] for region in regions), max(region[3] for region in regions)]
            boxno = merged.get("boxno")
            if boxno in reserved_boxnos:
                boxno = max(reserved_boxnos, default=-1) + 1
            reserved_boxnos.add(boxno)
            structured.append(
                {
                    "page": page,
                    "boxno": boxno,
                    "sourceBoxnos": sources,
                    "sourceTexts": copy.deepcopy(pending.get("sourceTexts", [])),
                    "region": copy.deepcopy(pending.get("region")),
                    "text": str(pending.get("text", "")),
                    "sfx": False,
                    "openLettering": False,
                    "safeToEraseOriginal": True,
                    "altPlacementReason": "unclear",
                }
            )
            represented_sources.update(sources)
        artifacts["ocr_structured"] = editor_v2.hydrate_ids(
            "ocr_structured", page, structured
        )

        structured_by_boxno = {
            record.get("boxno"): record
            for record in artifacts["ocr_structured"]
            if isinstance(record, dict)
        }
        translations = artifacts.setdefault("translations", [])
        translated_boxnos = {
            record.get("boxno") for record in translations if isinstance(record, dict)
        }
        placements = artifacts.setdefault("placements", [])
        placed_boxnos = {
            record.get("boxno") for record in placements if isinstance(record, dict)
        }
        alt_placements = artifacts.setdefault("alt_placement", [])
        alt_boxnos = {
            record.get("boxno") for record in alt_placements if isinstance(record, dict)
        }
        for boxno, record in structured_by_boxno.items():
            if boxno not in translated_boxnos:
                translations.append(
                    {
                        "page": page,
                        "boxno": boxno,
                        "text": record.get("text", ""),
                        "englishText": "",
                    }
                )
            if boxno not in placed_boxnos:
                placements.append(
                    {
                        "page": page,
                        "boxno": boxno,
                        "placementRegion": copy.deepcopy(record.get("region")),
                    }
                )
            if boxno not in alt_boxnos:
                alt_placements.append(
                    {
                        "page": page,
                        "boxno": boxno,
                        "safeToEraseOriginal": True,
                        "openLettering": False,
                        "altPlacementReason": "unclear",
                    }
                )
        for artifact in ("translations", "placements", "alt_placement"):
            artifacts[artifact] = editor_v2.hydrate_ids(
                artifact, page, artifacts[artifact]
            )

    def editor_v2_payload(
        self, code: str, job_id: str, stage: str, page: int
    ) -> dict[str, Any]:
        try:
            editor_v2.editor_stage_members(stage)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        status = self.require_editable_job(code, job_id, stage)
        page_path = self.original_page_path(code, job_id, page)
        manifest = self.load_editor_v2_manifest(code, job_id)
        artifacts = {
            artifact: self.editor_v2_effective_records(
                code, job_id, manifest, artifact, page
            )
            for artifact in editor_v2.ARTIFACT_STAGES
        }
        reserved_boxnos = {record["boxno"] for artifact in ("ocr_structured", "translations", "placements", "alt_placement")
                           for record in artifacts[artifact] if isinstance(record.get("boxno"), int)}
        artifacts = editor_v2.visible_artifacts(manifest, page, artifacts)
        page_state = manifest.get("pages", {}).get(str(page), {})
        pending_stages = set(page_state.get("pendingStages", []))
        represented = {source for record in artifacts["ocr_structured"] for source in record.get("sourceBoxnos", [])}
        pending_manual = any(group.get("ocrSource") == "manual" and
                             not represented.intersection(group.get("sourceBoxnos", []))
                             for group in artifacts["ocr_merged"])
        if pending_manual or pending_stages.intersection(editor_v2.STAGE_ORDER[:-1]):
            rejected_sources = {
                source
                for member in editor_v2.TEXT_STAGES[:2]
                for record in editor_v2.deleted_records(manifest, page, member, "ocr_structured")
                for source in record.get("sourceBoxnos", [])
            }
            self.add_pending_ocr_descendants(artifacts, page, rejected_sources, reserved_boxnos)
        with Image.open(page_path) as image:
            width, height = image.size
        default_font = translate_cbz.load_config(self.config.pipeline_config, None).render_font
        self.add_editor_layout_hints(artifacts, width, height, default_font)
        notes = manifest.get("translationNotes", {})
        page_notes = notes.get("pages", {}) if isinstance(notes, dict) else {}
        cleaned_path = translate_cbz.cleaned_pages_dir(self.output_dir(code, job_id)) / f"{page_path.stem}.png"
        return {
            "schemaVersion": editor_v2.SCHEMA_VERSION,
            "revision": manifest["revision"],
            "category": code,
            "jobId": job_id,
            "page": page,
            "pageCount": len(self.original_page_files(code, job_id)),
            "stage": stage,
            "availableStages": self.editor_available_stages(status),
            "reviewCheckpoint": status.get("reviewCheckpoint"),
            "stageDefinitions": editor_v2.stage_definitions(),
            "stageStates": {
                item: editor_v2.stage_status(manifest, page, item)
                for item in (*editor_v2.STAGE_ORDER, "text")
            },
            "image": {
                "name": page_path.name,
                "url": f"/job/{code}/{job_id}/image/{page}",
                "cleanedUrl": (f"/api/job/{code}/{job_id}/editor/v2/image/cleaned/{page}"
                               if cleaned_path.is_file() else None),
                "width": width,
                "height": height,
            },
            "recordsByArtifact": artifacts,
            "protection": self.editor_v2_protection_payload(manifest, page, stage),
            "rerunFrom": (editor_v2.earliest_rerun(editor_v2.pending_changes(manifest)[page])
                          if page in editor_v2.pending_changes(manifest) else None),
            "translationNotes": {
                "job": notes.get("job", "") if isinstance(notes, dict) else "",
                "page": page_notes.get(str(page), "") if isinstance(page_notes, dict) else "",
            },
            "fonts": self.editor_v2_fonts(),
            "defaultFont": default_font,
        }

    def editor_v2_fonts(self) -> list[str]:
        config = translate_cbz.load_config(self.config.pipeline_config, None)
        local_fonts = sorted(translate_cbz.available_font_names())
        return list(dict.fromkeys([config.render_font, *local_fonts]))

    def add_editor_layout_hints(
        self,
        artifacts: dict[str, list[dict[str, Any]]],
        image_width: int,
        image_height: int,
        default_font: str,
    ) -> None:
        translations = {
            record.get("boxno"): record.get("englishText", "")
            for record in artifacts.get("translations", [])
            if isinstance(record, dict)
        }
        for record in artifacts.get("placements", []):
            if not isinstance(record, dict):
                continue
            region = record.get("placementRegion", record.get("region"))
            if not isinstance(region, list) or len(region) != 4:
                continue
            try:
                region_width = max(1, round(float(region[2]) - float(region[0])))
                region_height = max(1, round(float(region[3]) - float(region[1])))
            except (TypeError, ValueError):
                continue
            text = str(
                record.get("manualLineBreaks")
                or translations.get(record.get("boxno"), "")
            )
            entry = dict(record)
            entry["region"] = region
            entry["font"] = translate_cbz.local_font_path(record.get("font"), default_font)
            outline_width = overlay_text.stroke_width(entry, image_height)
            padding = max(0, round(outline_width * 2))
            inner_width = max(1, region_width - padding * 2)
            inner_height = max(1, region_height - padding * 2)

            automatic_entry = dict(entry)
            automatic_entry.pop("fontSizeWidthPercent", None)
            _automatic_text, automatic_size, _technical = (
                overlay_text.explicit_caption_layout(
                    automatic_entry,
                    overlay_text.normalize_caption_text(text),
                    inner_width,
                    inner_height,
                    image_width,
                    image_height,
                )
            )
            record["_autoFontSizeWidthPercent"] = round(
                automatic_size / max(1, image_width) * 100,
                3,
            )

    def editor_v2_protection_payload(self, manifest: dict[str, Any], page: int, stage: str) -> dict[str, Any]:
        if stage != "text":
            return editor_v2.protection_payload(manifest, page, stage)
        records: dict[str, dict[str, bool]] = {}
        record_states: dict[str, Any] = {}
        states = [editor_v2.protection_payload(manifest, page, member) for member in editor_v2.TEXT_STAGES]
        for member, state in zip(editor_v2.TEXT_STAGES, states):
            for rid, fields in state["records"].items():
                records.setdefault(rid, {}).update(fields)
            if member != "erase":
                record_states.update(state["recordStates"])
        return {"frozen": all(state["frozen"] for state in states), "records": records,
                "recordStates": record_states}

    def editor_v2_snapshot(
        self, code: str, job_id: str, manifest: dict[str, Any], page: int,
        stages: tuple[str, ...] = editor_v2.TEXT_STAGES,
    ) -> dict[str, list[str]]:
        return {
            stage: [editor_v2.record_signature(stage, artifact, self.editor_v2_effective_records(
                code, job_id, manifest, artifact, page)) for artifact in editor_v2.STAGE_ARTIFACTS[stage]]
            for stage in stages
        }

    def save_editor_v2_text(self, code: str, job_id: str, page: int, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.require_editable_job(code, job_id, "text")
            self.original_page_path(code, job_id, page)
            manifest = self.load_editor_v2_manifest(code, job_id)
            revision = data.get("baseRevision")
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise HTTPException(400, "baseRevision must be an integer.")
            if revision != manifest["revision"]:
                raise HTTPException(409, "The editor changed after this page was loaded. Reload before saving.")
            supplied = data.get("recordsByArtifact")
            if not isinstance(supplied, dict) or any(not isinstance(supplied.get(key), list)
                                                   for key in ("ocr_structured", "translations")):
                raise HTTPException(400, "Text saves require ocr_structured and translations arrays.")
            if any(not isinstance(record, dict) for key in ("ocr_structured", "translations") for record in supplied[key]):
                raise HTTPException(400, "Records must be objects.")
            before = self.editor_v2_snapshot(code, job_id, manifest, page)
            structured = copy.deepcopy(supplied["ocr_structured"])
            for records in (structured, supplied["translations"]):
                ids = [record.get("boxno") for record in records]
                if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids) or len(set(ids)) != len(ids):
                    raise HTTPException(400, "Record box numbers must be unique non-negative integers.")
            boxnos = {record.get("boxno") for record in structured}
            translations = copy.deepcopy(supplied["translations"])
            # Display-only OCR descendants must not become protected manual records.
            display = self.editor_v2_payload(code, job_id, "text", page)["recordsByArtifact"]
            submitted_source_ids = {record.get("recordId") for record in structured}
            rejected_sources = [record for record in display["ocr_structured"] if record["recordId"] not in submitted_source_ids]
            for artifact, records, members in (("ocr_structured", structured, ("structure", "erase")),
                                               ("translations", translations, ("translation",))):
                actual_ids = {record["recordId"] for record in self.editor_v2_effective_records(code, job_id, manifest, artifact, page)}
                displayed = {record["recordId"]: record for record in display[artifact]}
                records[:] = [record for record in records if record.get("recordId") in actual_ids
                    or record.get("recordId") not in displayed
                    or any(editor_v2.record_signature(member, artifact, [record]) !=
                           editor_v2.record_signature(member, artifact, [displayed[record["recordId"]]]) for member in members)]
            structured = self.preserve_hidden_editor_records(code, job_id, manifest, page, "ocr_structured", structured, display)
            related_display = {**display, **{artifact: [record for record in display[artifact] if record.get("boxno") in boxnos]
                                           for artifact in ("translations", "alt_placement")}}
            translations = self.preserve_hidden_editor_records(code, job_id, manifest, page, "translations", translations, related_display)
            for stage, artifact, records in (
                ("structure", "ocr_structured", structured),
                ("erase", "ocr_structured", structured),
                ("translation", "translations", translations),
            ):
                editor_v2.update_artifact_override(manifest, page, stage, artifact,
                    self.load_editor_v2_baseline(code, job_id, artifact, page), records)
                if editor_v2.stage_override(manifest, page, stage).get("frozen"):
                    editor_v2.artifact_override(manifest, page, stage, artifact)["freeze"] = {
                        "records": editor_v2.hydrate_ids(artifact, page, records), "protected": True,
                    }
            rejected_override = editor_v2.artifact_override(manifest, page, "structure", "ocr_structured")
            for record in rejected_sources:
                rejected_override["deleted"].setdefault(record["recordId"], {
                    "record": copy.deepcopy(record), "region": copy.deepcopy(record.get("region")), "protected": True,
                })
            after = self.editor_v2_snapshot(code, job_id, manifest, page)
            if before["erase"] != after["erase"]:
                previous_alt = {record["boxno"]: record for record in self.editor_v2_effective_records(
                    code, job_id, manifest, "alt_placement", page)}
                alt_records = [{**previous_alt.get(record["boxno"], {}), "page": page, "boxno": record["boxno"],
                                "openLettering": bool(record.get("openLettering")),
                                "safeToEraseOriginal": not bool(record.get("openLettering")),
                                "altPlacementReason": record.get("altPlacementReason", "unclear")}
                               for record in structured if isinstance(record.get("boxno"), int)]
                alt_records = self.preserve_hidden_editor_records(code, job_id, manifest, page, "alt_placement", alt_records, related_display)
                editor_v2.update_artifact_override(manifest, page, "erase", "alt_placement",
                    self.load_editor_v2_baseline(code, job_id, "alt_placement", page), alt_records)
                if editor_v2.stage_override(manifest, page, "erase").get("frozen"):
                    editor_v2.artifact_override(manifest, page, "erase", "alt_placement")["freeze"] = {
                        "records": editor_v2.hydrate_ids("alt_placement", page, alt_records), "protected": True,
                    }
            after = self.editor_v2_snapshot(code, job_id, manifest, page)
            changed = {stage for stage in before if before[stage] != after[stage]}
            if rejected_sources:
                changed.add("structure")
            notes = manifest.setdefault("translationNotes", {"job": "", "pages": {}})
            old_notes = copy.deepcopy(notes)
            supplied_notes = data.get("translationNotes")
            if isinstance(supplied_notes, dict):
                if isinstance(supplied_notes.get("job"), str):
                    notes["job"] = supplied_notes["job"]
                if isinstance(supplied_notes.get("page"), str):
                    pages = notes.setdefault("pages", {})
                    if supplied_notes["page"]:
                        pages[str(page)] = supplied_notes["page"]
                    else:
                        pages.pop(str(page), None)
            if changed or notes != old_notes:
                editor_v2.mark_stages_saved(manifest, page, changed, before=before)
                self.save_editor_v2_manifest(code, job_id, manifest)
                write_json_atomic(self.translation_notes_path(code, job_id), notes)
                self.materialize_editor_v2_page(code, job_id, manifest, page)
            return self.editor_v2_payload(code, job_id, "text", page)

    def preserve_hidden_editor_records(
        self, code: str, job_id: str, manifest: dict[str, Any], page: int, artifact: str,
        submitted: list[dict[str, Any]], display: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        visible_ids = {record["recordId"] for record in display[artifact]}
        current = self.editor_v2_effective_records(code, job_id, manifest, artifact, page)
        hidden_ids = {record["recordId"] for record in current} - visible_ids
        submitted = [record for record in submitted if record.get("recordId") not in hidden_ids]
        submitted_ids = {record.get("recordId") for record in submitted}
        pending = iter(submitted)
        result = []
        for record in current:
            if record["recordId"] in hidden_ids:
                result.append(record)
            elif record["recordId"] in submitted_ids:
                result.append(next(pending))
        result.extend(pending)
        return result

    def save_editor_v2_update(
        self,
        code: str,
        job_id: str,
        page: int,
        stage: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if stage == "text":
            return self.save_editor_v2_text(code, job_id, page, data)
        try:
            stage = editor_v2.validate_stage(stage)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        self.require_editable_job(code, job_id, stage)
        self.original_page_path(code, job_id, page)
        with self._lock:
            manifest = self.load_editor_v2_manifest(code, job_id)
            base_revision = data.get("baseRevision")
            if isinstance(base_revision, bool) or not isinstance(base_revision, int):
                raise HTTPException(status_code=400, detail="baseRevision must be an integer.")
            if base_revision != manifest["revision"]:
                raise HTTPException(
                    status_code=409,
                    detail="The editor changed after this page was loaded. Reload before saving.",
                )
            before = self.editor_v2_snapshot(code, job_id, manifest, page, (stage,))[stage]
            supplied = data.get("recordsByArtifact")
            if not isinstance(supplied, dict):
                raise HTTPException(status_code=400, detail="recordsByArtifact must be an object.")
            display = self.editor_v2_payload(code, job_id, stage, page)["recordsByArtifact"]
            if stage == "ocr":
                deleted_groups = data.get("deletedOcrGroupIds", [])
                if not isinstance(deleted_groups, list) or any(not isinstance(value, str) for value in deleted_groups):
                    raise HTTPException(400, "deletedOcrGroupIds must be an array of record IDs.")
                raw_records = supplied.get("ocr_raw")
                merged_records = supplied.get("ocr_merged")
                if not isinstance(raw_records, list) or not isinstance(merged_records, list):
                    raise HTTPException(
                        status_code=400,
                        detail="OCR saves require ocr_raw and ocr_merged arrays.",
                    )
                normalized_raw, normalized_merged = self.normalize_ocr_merge_records(
                    code, job_id, page, raw_records, merged_records
                )
                supplied = {
                    **supplied,
                    "ocr_raw": normalized_raw,
                    "ocr_merged": normalized_merged,
                }
            for artifact in editor_v2.STAGE_ARTIFACTS[stage]:
                records = supplied.get(artifact)
                if records is None:
                    continue
                if not isinstance(records, list):
                    raise HTTPException(
                        status_code=400, detail=f"{artifact} records must be an array."
                    )
                if artifact == "clean_boxes":
                    records = self.normalize_clean_boxes(code, job_id, page, records)
                elif artifact == "placements":
                    records = self.normalize_saved_placements(
                        code, job_id, page, records
                    )
                    pipeline_config = translate_cbz.load_config(
                        self.config.pipeline_config, None
                    )
                    allowed_gravity = {
                        "north", "center", "south", "west", "east",
                        "northwest", "northeast", "southwest", "southeast",
                    }
                    for record in records:
                        if not isinstance(record, dict):
                            continue
                        try:
                            overlay_text.validate_text_transform(0, record)
                        except overlay_text.InputError as exc:
                            raise HTTPException(status_code=400, detail=str(exc)) from exc
                        try:
                            record["font"] = translate_cbz.normalize_font_name(
                                record.get("font"), pipeline_config.render_font
                            )
                            record["fill"] = translate_cbz.normalize_fill(
                                record.get("fill", pipeline_config.render_fill)
                            )
                        except translate_cbz.PipelineError as exc:
                            raise HTTPException(status_code=400, detail=str(exc)) from exc
                        stroke = record.get("stroke", translate_cbz.outline_for_fill(record["fill"]))
                        if stroke not in {"black", "white"}:
                            raise HTTPException(status_code=400, detail="Outline colour must be black or white.")
                        record["stroke"] = stroke
                        gravity = str(record.get("gravity", pipeline_config.render_gravity)).lower()
                        if gravity not in allowed_gravity:
                            raise HTTPException(status_code=400, detail="Unknown text alignment.")
                        record["gravity"] = gravity
                        percent = record.get("fontSizeWidthPercent")
                        if percent is not None and (
                            isinstance(percent, bool)
                            or not isinstance(percent, (int, float))
                            or percent <= 0
                            or percent > 100
                        ):
                            raise HTTPException(
                                status_code=400,
                                detail="Fixed font size must be between 0 and 100 percent.",
                            )
                elif stage == "erase" and artifact == "ocr_structured":
                    records = [dict(item) if isinstance(item, dict) else item for item in records]
                    for record in records:
                        if isinstance(record, dict) and isinstance(record.get("openLettering"), bool):
                            record["safeToEraseOriginal"] = not record["openLettering"]
                    supplied[artifact] = records
                baseline = self.load_editor_v2_baseline(code, job_id, artifact, page)
                if stage != "ocr":
                    records = self.preserve_hidden_editor_records(code, job_id, manifest, page, artifact, records, display)
                editor_v2.update_artifact_override(
                    manifest,
                    page,
                    stage,
                    artifact,
                    baseline,
                    records,
                )
                if stage == "ocr" and artifact == "ocr_merged":
                    # A removed derived singleton is a deliberate group deletion,
                    # not another orphan to recreate on the next page load.
                    retained = {record.get("recordId") for record in records}
                    deleted = editor_v2.artifact_override(manifest, page, stage, artifact)["deleted"]
                    for record in display[artifact]:
                        if record["recordId"] not in retained and record["recordId"] in deleted_groups:
                            deleted.setdefault(record["recordId"], {
                                "record": copy.deepcopy(record), "region": copy.deepcopy(record.get("region")),
                                "protected": True,
                            })
                if editor_v2.stage_override(manifest, page, stage).get("frozen"):
                    editor_v2.artifact_override(manifest, page, stage, artifact)["freeze"] = {
                        "records": editor_v2.hydrate_ids(artifact, page, records), "protected": True,
                    }
            if stage == "erase" and isinstance(supplied.get("ocr_structured"), list):
                alt_records = [
                    {
                        "page": page,
                        "boxno": record.get("boxno"),
                        "safeToEraseOriginal": bool(record.get("safeToEraseOriginal")),
                        "openLettering": bool(record.get("openLettering")),
                        "altPlacementReason": str(record.get("altPlacementReason", "unclear")),
                    }
                    for record in supplied["ocr_structured"]
                    if isinstance(record, dict) and isinstance(record.get("boxno"), int)
                ]
                editor_v2.update_artifact_override(
                    manifest,
                    page,
                    stage,
                    "alt_placement",
                    self.load_editor_v2_baseline(code, job_id, "alt_placement", page),
                    alt_records,
                )
            notes = data.get("translationNotes")
            if isinstance(notes, dict):
                target = manifest.setdefault("translationNotes", {"job": "", "pages": {}})
                if isinstance(notes.get("job"), str):
                    target["job"] = notes["job"]
                if isinstance(notes.get("page"), str):
                    pages = target.setdefault("pages", {})
                    if notes["page"]:
                        pages[str(page)] = notes["page"]
                    else:
                        pages.pop(str(page), None)
            if stage == "placement":
                editor_v2.mark_stages_saved(manifest, page, (stage,), before={stage: before})
            else:
                editor_v2.mark_saved(manifest, page, stage, before=before)
            self.save_editor_v2_manifest(code, job_id, manifest)
            write_json_atomic(
                self.translation_notes_path(code, job_id),
                manifest.get("translationNotes", {"job": "", "pages": {}}),
            )
            self.materialize_editor_v2_page(code, job_id, manifest, page)
        return self.editor_v2_payload(code, job_id, stage, page)

    def reset_editor_v2_stage(
        self, code: str, job_id: str, page: int, stage: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            editor_v2.editor_stage_members(stage)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        with self._lock:
            self.require_editable_job(code, job_id, stage)
            self.original_page_path(code, job_id, page)
            manifest = self.load_editor_v2_manifest(code, job_id)
            revision = data.get("baseRevision")
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise HTTPException(status_code=400, detail="baseRevision must be an integer.")
            if revision != manifest["revision"]:
                raise HTTPException(
                    status_code=409,
                    detail="The editor changed after this page was loaded. Reload before resetting.",
                )
            members = editor_v2.editor_stage_members(stage)
            before = self.editor_v2_snapshot(code, job_id, manifest, page, members)
            for member in members:
                manifest.get("pages", {}).get(str(page), {}).get("stages", {}).pop(member, None)
            after = self.editor_v2_snapshot(code, job_id, manifest, page, members)
            editor_v2.mark_reset(manifest, page, members, before, after)
            self.save_editor_v2_manifest(code, job_id, manifest)
            self.materialize_editor_v2_page(code, job_id, manifest, page)
            return self.editor_v2_payload(code, job_id, stage, page)

    def materialize_editor_v2_page(
        self, code: str, job_id: str, manifest: dict[str, Any], page: int
    ) -> None:
        for artifact in editor_v2.ARTIFACT_STAGES:
            records = self.editor_v2_effective_records(
                code, job_id, manifest, artifact, page
            )
            path = (
                self.output_dir(code, job_id)
                / "data"
                / artifact
                / f"page_{page:04d}.json"
            )
            write_json_atomic(path, records)

    def set_editor_v2_text_protection(self, code: str, job_id: str, page: int, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.require_editable_job(code, job_id, "text")
            self.original_page_path(code, job_id, page)
            manifest = self.load_editor_v2_manifest(code, job_id)
            revision = data.get("baseRevision")
            if isinstance(revision, bool) or not isinstance(revision, int):
                raise HTTPException(400, "baseRevision must be an integer.")
            if revision != manifest["revision"]:
                raise HTTPException(409, "Editor revision is stale.")
            before = self.editor_v2_snapshot(code, job_id, manifest, page)
            action = data.get("action")
            protected = bool(data.get("protected"))
            if action in {"freeze-stage", "freeze-page"}:
                for member in (editor_v2.STAGE_ORDER if action == "freeze-page" else editor_v2.TEXT_STAGES):
                    records = {artifact: self.editor_v2_effective_records(code, job_id, manifest, artifact, page)
                               for artifact in editor_v2.STAGE_ARTIFACTS[member]}
                    editor_v2.freeze_stage(manifest, page, member, records, protected)
            elif action in {"field", "revert-field"}:
                member, artifact = data.get("pipelineStage"), data.get("artifact")
                field, rid = str(data.get("field", "")), str(data.get("recordId", ""))
                if member not in editor_v2.TEXT_STAGES or artifact not in editor_v2.STAGE_ARTIFACTS[member]:
                    raise HTTPException(400, "Field is not part of this editor tab.")
                fields = editor_v2.STAGE_FIELDS[member][artifact]
                if fields is not None and field not in fields:
                    raise HTTPException(400, "Field is not part of this pipeline step.")
                changed = (editor_v2.set_field_protection(manifest, page, member, artifact, rid, field, protected)
                           if action == "field" else editor_v2.remove_field_override(manifest, page, member, artifact, rid, field))
                if not changed:
                    raise HTTPException(404, "No saved override for that field.")
            elif action in {"record", "revert-record"}:
                artifact = data.get("artifact")
                if artifact not in {"ocr_structured", "translations"}:
                    raise HTTPException(400, "Record is not part of this editor tab.")
                selected = next((record for record in self.editor_v2_effective_records(code, job_id, manifest, artifact, page)
                                 if record.get("recordId") == data.get("recordId")), None)
                if selected is None:
                    raise HTTPException(404, "Record not found.")
                changed = False
                for member in editor_v2.TEXT_STAGES:
                    for item in editor_v2.STAGE_ARTIFACTS[member]:
                        for record in self.editor_v2_effective_records(code, job_id, manifest, item, page):
                            if record.get("boxno") != selected.get("boxno"):
                                continue
                            rid = record["recordId"]
                            changed |= (editor_v2.set_record_protection(manifest, page, member, item, rid, protected)
                                        if action == "record" else editor_v2.remove_record_override(manifest, page, member, item, rid))
                if not changed:
                    raise HTTPException(404, "No saved override for that record.")
            else:
                raise HTTPException(400, "Unknown protection action.")
            after = self.editor_v2_snapshot(code, job_id, manifest, page)
            editor_v2.mark_stages_saved(manifest, page, {member for member in before if before[member] != after[member]}, before=before)
            self.save_editor_v2_manifest(code, job_id, manifest)
            self.materialize_editor_v2_page(code, job_id, manifest, page)
            return self.editor_v2_payload(code, job_id, "text", page)

    def set_editor_v2_protection(
        self,
        code: str,
        job_id: str,
        page: int,
        stage: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if stage == "text":
            return self.set_editor_v2_text_protection(code, job_id, page, data)
        try:
            stage = editor_v2.validate_stage(stage)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        self.require_editable_job(code, job_id, stage)
        with self._lock:
            manifest = self.load_editor_v2_manifest(code, job_id)
            base_revision = data.get("baseRevision")
            if base_revision != manifest["revision"]:
                raise HTTPException(status_code=409, detail="Editor revision is stale.")
            action = str(data.get("action", ""))
            if action == "freeze-stage":
                frozen = bool(data.get("protected"))
                records = {
                    artifact: self.editor_v2_effective_records(
                        code, job_id, manifest, artifact, page
                    )
                    for artifact in editor_v2.STAGE_ARTIFACTS[stage]
                }
                editor_v2.freeze_stage(manifest, page, stage, records, frozen)
            elif action == "freeze-page":
                frozen = bool(data.get("protected"))
                for page_stage in editor_v2.STAGE_ORDER:
                    records = {
                        artifact: self.editor_v2_effective_records(
                            code, job_id, manifest, artifact, page
                        )
                        for artifact in editor_v2.STAGE_ARTIFACTS[page_stage]
                    }
                    editor_v2.freeze_stage(
                        manifest, page, page_stage, records, frozen
                    )
            elif action == "field":
                artifact = str(data.get("artifact", ""))
                if artifact not in editor_v2.STAGE_ARTIFACTS[stage]:
                    raise HTTPException(status_code=400, detail="Field is not part of this stage.")
                changed = editor_v2.set_field_protection(
                    manifest,
                    page,
                    stage,
                    artifact,
                    str(data.get("recordId", "")),
                    str(data.get("field", "")),
                    bool(data.get("protected")),
                )
                if not changed:
                    raise HTTPException(status_code=404, detail="No saved override for that field.")
            elif action == "revert-field":
                artifact = str(data.get("artifact", ""))
                changed = editor_v2.remove_field_override(
                    manifest,
                    page,
                    stage,
                    artifact,
                    str(data.get("recordId", "")),
                    str(data.get("field", "")),
                )
                if not changed:
                    raise HTTPException(status_code=404, detail="No saved override for that field.")
            elif action in {"record", "revert-record"}:
                artifact = str(data.get("artifact", ""))
                if artifact not in editor_v2.STAGE_ARTIFACTS[stage]:
                    raise HTTPException(status_code=400, detail="Record is not part of this stage.")
                record_id_value = str(data.get("recordId", ""))
                changed = (
                    editor_v2.set_record_protection(
                        manifest,
                        page,
                        stage,
                        artifact,
                        record_id_value,
                        bool(data.get("protected")),
                    )
                    if action == "record"
                    else editor_v2.remove_record_override(
                        manifest, page, stage, artifact, record_id_value
                    )
                )
                if not changed:
                    raise HTTPException(status_code=404, detail="No saved override for that record.")
            else:
                raise HTTPException(status_code=400, detail="Unknown protection action.")
            manifest["revision"] += 1
            manifest["updatedAt"] = editor_v2.utc_now()
            self.save_editor_v2_manifest(code, job_id, manifest)
            self.materialize_editor_v2_page(code, job_id, manifest, page)
        return self.editor_v2_payload(code, job_id, stage, page)

    def editor_v2_revisions(self, code: str, job_id: str) -> list[dict[str, Any]]:
        self.require_editable_job(code, job_id)
        self.load_editor_v2_manifest(code, job_id)
        result: list[dict[str, Any]] = []
        for path in sorted(self.editor_v2_history_dir(code, job_id).glob("revision_*.json"), reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            result.append(
                {
                    "revision": value.get("revision"),
                    "updatedAt": value.get("updatedAt", ""),
                }
            )
        return result

    def rerun_editor_v2_changes(
        self, code: str, job_id: str
    ) -> tuple[list[int], dict[int, str]]:
        manifest = self.load_editor_v2_manifest(code, job_id)
        pending = editor_v2.pending_changes(manifest)
        if not pending:
            raise HTTPException(status_code=400, detail="No saved editor changes need regeneration.")
        pages = sorted(pending)
        resume_by_page = {
            page: editor_v2.earliest_rerun(pending[page]) for page in pages
        }
        self.rerun_completed_job_pages(
            code,
            job_id,
            pages,
            page_resume_from=resume_by_page,
        )
        return pages, resume_by_page

    def rerun_editor_v2_page(self, code: str, job_id: str, stage: str, page: int) -> str:
        try:
            editor_v2.editor_stage_members(stage)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        with self._lock:
            self.require_editable_job(code, job_id, stage)
            self.original_page_path(code, job_id, page)
            pending = editor_v2.pending_changes(self.load_editor_v2_manifest(code, job_id)).get(page)
            if not pending:
                raise HTTPException(400, "No saved editor changes need regeneration on this page.")
            resume_from = editor_v2.earliest_rerun(pending)
            self.rerun_completed_job_pages(code, job_id, [page], resume_from)
            return resume_from

    def retranslate_editor_v2_page(
        self,
        code: str,
        job_id: str,
        page: int,
        record_id_value: str | None = None,
        boxno: int | None = None,
    ) -> None:
        with self._lock:
            manifest = self.load_editor_v2_manifest(code, job_id)
            if editor_v2.pending_changes(manifest).get(page, set()).intersection({"ocr", "structure"}):
                raise HTTPException(409, "Regenerate source changes before requesting a new translation.")
            stage_value = editor_v2.stage_override(manifest, page, "translation")
            if stage_value.get("frozen"):
                raise HTTPException(
                    status_code=409,
                    detail="Unprotect Text & Translation before requesting a VLM translation.",
                )
            if record_id_value is None:
                editor_v2.set_artifact_protection(
                    manifest,
                    page,
                    "translation",
                    "translations",
                    False,
                )
            else:
                if not isinstance(boxno, int) or isinstance(boxno, bool) or boxno < 0:
                    raise HTTPException(status_code=400, detail="Select a valid translation record.")
                records = self.editor_v2_effective_records(
                    code, job_id, manifest, "translations", page
                )
                selected = next(
                    (
                        record
                        for record in records
                        if record.get("recordId") == record_id_value
                        and record.get("boxno") == boxno
                    ),
                    None,
                )
                if selected is None:
                    raise HTTPException(status_code=404, detail="Translation record not found.")
                editor_v2.set_record_protection(
                    manifest,
                    page,
                    "translation",
                    "translations",
                    record_id_value,
                    False,
                )
            manifest["revision"] += 1
            manifest["updatedAt"] = editor_v2.utc_now()
            self.save_editor_v2_manifest(code, job_id, manifest)
        self.rerun_completed_job_pages(
            code,
            job_id,
            [page],
            "translations",
            translation_boxno=boxno if record_id_value is not None else None,
        )

    def mark_editor_v2_translation_ready_for_typesetting(
        self, code: str, job_id: str, page: int
    ) -> None:
        with self._lock:
            manifest = self.load_editor_v2_manifest(code, job_id)
            editor_v2.mark_saved(manifest, page, "translation")
            self.save_editor_v2_manifest(code, job_id, manifest)

    def restore_editor_v2_revision(
        self, code: str, job_id: str, revision: int
    ) -> dict[str, Any]:
        self.require_editable_job(code, job_id)
        path = self.editor_v2_history_dir(code, job_id) / f"revision_{revision:08d}.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Unknown editor revision.")
        try:
            restored = editor_v2.normalize_manifest(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Saved revision is invalid.") from exc
        if restored is None:
            raise HTTPException(status_code=500, detail="Saved revision is invalid.")
        current = self.load_editor_v2_manifest(code, job_id)
        restored["revision"] = current["revision"] + 1
        restored["updatedAt"] = editor_v2.utc_now()
        self.save_editor_v2_manifest(code, job_id, restored)
        for page in range(len(self.original_page_files(code, job_id))):
            self.materialize_editor_v2_page(code, job_id, restored, page)
        return {"ok": True, "revision": restored["revision"]}

    def current_editor_clean_source(
        self,
        code: str,
        job_id: str,
        page: int,
        structured_records: list[Any],
    ) -> tuple[Path, bool]:
        original = self.original_page_path(code, job_id, page)
        clean_entries = [
            record
            for record in structured_records
            if isinstance(record, dict) and not bool(record.get("openLettering"))
        ]
        try:
            # Each request builds its own mask. Concurrent drafts must not share
            # a mutable image or mask filename.
            with tempfile.TemporaryDirectory(prefix="editor-clean-mask-") as directory:
                temporary_mask = Path(directory) / "mask.png"
                clean_text_regions.build_mask(clean_entries, original, temporary_mask, clean_text_regions.DEFAULT_PADDING)
                mask_bytes = temporary_mask.read_bytes()
        except clean_text_regions.InputError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot build the current cleaning mask: {exc}",
            ) from exc

        mask_digest = hashlib.sha256(mask_bytes).hexdigest()
        output_dir = self.output_dir(code, job_id)
        pipeline_mask = output_dir / "debug" / "masks" / f"{original.stem}.png"
        pipeline_cleaned = output_dir / "pages" / "cleaned" / f"{original.stem}.png"
        if pipeline_mask.is_file() and pipeline_cleaned.is_file():
            pipeline_digest = hashlib.sha256(pipeline_mask.read_bytes()).hexdigest()
            if pipeline_digest == mask_digest:
                return pipeline_cleaned, False

        source_stat = original.stat()
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "mask": mask_digest,
                    "source": str(original.resolve()),
                    "sourceSize": source_stat.st_size,
                    "sourceMtimeNs": source_stat.st_mtime_ns,
                    "padding": clean_text_regions.DEFAULT_PADDING,
                    "cropTriggerSize": clean_text_regions.DEFAULT_CROP_TRIGGER_SIZE,
                    "cropMargin": clean_text_regions.DEFAULT_CROP_MARGIN,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        preview = self.editor_v2_clean_preview_path(code, job_id, page, cache_key)
        preview.parent.mkdir(parents=True, exist_ok=True)
        current_mask = preview.with_suffix(".mask.png")
        cache_path = preview.with_suffix(".json")
        if preview.is_file() and cache_path.is_file():
            return preview, False

        try:
            with self._editor_lama_lock:
                if preview.is_file() and cache_path.is_file():
                    return preview, False
                if clean_entries and self._editor_lama_session is None:
                    self._editor_lama_session = lama_inpaint.LaMaSession(
                        clean_text_regions.DEFAULT_DEVICE
                    )
                clean_text_regions.clean_text_regions(
                    clean_entries,
                    original,
                    preview,
                    clean_text_regions.DEFAULT_PADDING,
                    clean_text_regions.DEFAULT_DEVICE,
                    None,
                    clean_text_regions.DEFAULT_CROP_TRIGGER_SIZE,
                    clean_text_regions.DEFAULT_CROP_MARGIN,
                    current_mask,
                    self._editor_lama_session,
                )
                write_json_atomic(cache_path, {"key": cache_key})
        except (clean_text_regions.InputError, lama_inpaint.LaMaError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"LaMa could not create the current preview: {exc}",
            ) from exc
        return preview, bool(clean_entries)

    def render_editor_v2_clean_preview(self, code: str, job_id: str, page: int,
                                      records: list[Any], boxes: list[Any]) -> dict[str, Any]:
        self.require_editable_job(code, job_id, "placement")
        if len(records) > 4096:
            raise HTTPException(400, "Too many source cleaning regions.")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise HTTPException(400, "Each cleaning region must be an object.")
            try:
                clean_text_regions.validate_entry(index, record)
            except clean_text_regions.InputError as exc:
                raise HTTPException(400, str(exc)) from exc
        clean_boxes = self.normalize_clean_boxes(code, job_id, page, boxes)
        path, cleaned = self.current_editor_clean_source(code, job_id, page, [*records, *clean_boxes])
        root = f"/api/job/{code}/{job_id}/editor/v2"
        original = self.original_page_path(code, job_id, page)
        pipeline = self.output_dir(code, job_id) / "pages/cleaned" / f"{original.stem}.png"
        if path == pipeline:
            url = f"{root}/image/cleaned/{page}"
        else:
            key = path.name.split(".")[1]
            url = f"{root}/clean-preview-image/{page}/{key}"
        return {"url": url, "cleanedWithLama": cleaned}

    def editor_v2_render_entries(
        self,
        code: str,
        job_id: str,
        page: int,
        placement_records: list[Any],
        text_by_box: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        if text_by_box is None:
            manifest = self.load_editor_v2_manifest(code, job_id)
            translations = self.editor_v2_effective_records(code, job_id, manifest, "translations", page)
            text_by_box = {item.get("boxno"): item.get("englishText", "") for item in translations}
        config = translate_cbz.load_config(self.config.pipeline_config, None)
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(placement_records):
            if not isinstance(item, dict):
                continue
            record = dict(item)
            region = record.get("placementRegion", record.get("region"))
            if not isinstance(region, list) or len(region) != 4:
                continue
            boxno = record.get("boxno", index)
            try:
                font_name = translate_cbz.normalize_font_name(
                    record.get("font"), config.render_font
                )
                fill = translate_cbz.normalize_fill(
                    record.get("fill", config.render_fill)
                )
            except translate_cbz.PipelineError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            assert font_name is not None and fill is not None
            entry = {
                "page": page,
                "boxno": boxno,
                "region": region,
                "englishText": str(
                    record.get("manualLineBreaks") or text_by_box.get(boxno, "")
                ),
                "font": translate_cbz.local_font_path(font_name, config.render_font),
                "fill": fill,
                "stroke": str(record.get("stroke", translate_cbz.outline_for_fill(fill))),
                "strokeWidth": record.get("strokeWidth", translate_cbz.DEFAULT_RENDER_STROKE_WIDTH),
                "gravity": str(record.get("gravity", config.render_gravity)),
            }
            font_path = Path(entry["font"])
            if font_path.is_file() and not any(font_path.resolve().is_relative_to(directory.resolve())
                    for directory in (translate_cbz.FONT_DIR, translate_cbz.BUNDLED_FONT_DIR)):
                raise HTTPException(400, "Font files must be inside a configured font directory.")
            for field in ("fontSizeWidthPercent", "minPointSize", "maxPointSize", "renderBleed",
                          "rotation", "skewX", "skewY"):
                if field in record:
                    entry[field] = record[field]
            if entry["gravity"] not in {"north", "south", "west", "east", "center", "northwest", "northeast", "southwest", "southeast"}:
                raise HTTPException(400, "Unknown text alignment.")
            if entry["stroke"] not in {"black", "white"}:
                raise HTTPException(400, "Outline colour must be black or white.")
            try:
                overlay_text.validate_entry(index, entry)
            except overlay_text.InputError as exc:
                raise HTTPException(400, str(exc)) from exc
            entries.append(entry)
        return entries

    def render_editor_v2_text_layers(self, code: str, job_id: str, page: int, records: list[dict[str, Any]]) -> list[Path]:
        self.require_editable_job(code, job_id, "placement")
        if not isinstance(records, list) or not 1 <= len(records) <= 256 or any(not isinstance(record, dict) for record in records):
            raise HTTPException(400, "A text-layer batch requires 1 to 256 placement objects.")
        original = self.original_page_path(code, job_id, page)
        with Image.open(original) as image:
            image_width, image_height = image.size
        texts = {index: record.get("englishText", "") for index, record in enumerate(records)}
        if any(not isinstance(text, str) or len(text) > 100_000 for text in texts.values()):
            raise HTTPException(400, "Preview text must be a string of at most 100000 characters per box.")
        entries = self.editor_v2_render_entries(code, job_id, page,
            [{**record, "boxno": index} for index, record in enumerate(records)], texts)
        if len(entries) != len(records):
            raise HTTPException(400, "Each text layer requires a placement region.")
        directory = self.editor_meta_dir(code, job_id) / "previews" / "text-layers"
        outputs, missing = [], {}
        for entry in entries:
            try:
                entry = overlay_text.clamp_entry_to_image(entry, image_width, image_height)
                _, _, width, height = overlay_text.rounded_region(entry["region"])
            except overlay_text.InputError as exc:
                raise HTTPException(400, str(exc)) from exc
            # Position and identity do not affect text pixels.
            entry["region"] = [0, 0, width, height]
            entry.pop("boxno", None)
            entry.pop("page", None)
            font = Path(entry["font"])
            font_stat = font.stat() if font.is_file() else None
            signature = [overlay_text.TEXT_LAYOUT_VERSION, entry, image_width, image_height,
                         [font_stat.st_size, font_stat.st_mtime_ns] if font_stat else None]
            key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            output = directory / f"{key}.png"
            outputs.append(output)
            if not output.is_file():
                missing[key] = entry
        if missing:
            directory.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.TemporaryDirectory(prefix=".layers-", dir=directory) as temporary:
                    layers = [(entry, Path(temporary) / f"{key}.png", Path(temporary) / f"{key}.txt")
                              for key, entry in missing.items()]
                    overlay_text.render_text_layers(overlay_text.ensure_magick(), layers, image_width, image_height)
                    # Publish nothing until the entire process succeeds.
                    for _entry, layer, _text in layers:
                        layer.replace(directory / layer.name)
            except overlay_text.InputError as exc:
                raise HTTPException(500, str(exc)) from exc
        return outputs

    def render_editor_v2_preview(
        self, code: str, job_id: str, page: int,
        placement_records: list[Any], structured_records: list[Any],
        clean_boxes: list[Any] | None = None,
    ) -> tuple[Path, bool, bool]:
        self.require_editable_job(code, job_id, "placement")
        boxes = self.normalize_clean_boxes(code, job_id, page, clean_boxes or [])
        entries = self.editor_v2_render_entries(code, job_id, page, placement_records)
        source, cleaned_with_lama = self.current_editor_clean_source(
            code, job_id, page, [*structured_records, *boxes],
        )
        output = self.editor_v2_preview_path(code, job_id, page)
        output.parent.mkdir(parents=True, exist_ok=True)
        source_stat = source.stat()
        font_signatures: list[dict[str, Any]] = []
        for entry in entries:
            font_path = Path(str(entry["font"]))
            if not font_path.is_file():
                continue
            font_stat = font_path.stat()
            font_signatures.append(
                {
                    "path": str(font_path.resolve()),
                    "size": font_stat.st_size,
                    "mtimeNs": font_stat.st_mtime_ns,
                }
            )
        cache_payload = {
            "layoutVersion": overlay_text.TEXT_LAYOUT_VERSION,
            "entries": entries,
            "source": str(source.resolve()),
            "sourceSize": source_stat.st_size,
            "sourceMtimeNs": source_stat.st_mtime_ns,
            "fonts": font_signatures,
        }
        cache_key = hashlib.sha256(
            json.dumps(
                cache_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cache_path = output.with_suffix(".json")
        if output.is_file() and cache_path.is_file():
            try:
                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cache_data = None
            if isinstance(cache_data, dict) and cache_data.get("key") == cache_key:
                return output, True, cleaned_with_lama
        overlay_text.overlay_text(entries, source, output)
        write_json_atomic(cache_path, {"key": cache_key})
        return output, False, cleaned_with_lama

    def set_initial_translation_notes(self, code: str, job_id: str, notes: str) -> None:
        notes = notes.strip()
        if not notes:
            return
        write_json_atomic(
            self.translation_notes_path(code, job_id),
            {"job": notes, "pages": {}},
        )

    def editor_available_stages(self, status: dict[str, Any]) -> list[str]:
        if status.get("reviewCheckpoint") == "ocr":
            return ["ocr"]
        return list(editor_v2.EDITOR_STAGE_ORDER)

    def require_editable_job(
        self,
        code: str,
        job_id: str,
        stage: str | None = None,
    ) -> dict[str, Any]:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        status = self.load_status(code, job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        with self._lock:
            is_active = (code, job_id) in self._active_processes
        editable_checkpoint = (
            status.get("status") == "paused"
            and status.get("reviewCheckpoint") == "ocr"
            and not is_active
        )
        if status.get("status") != "complete" and not editable_checkpoint:
            raise HTTPException(
                status_code=400,
                detail="Only complete jobs or OCR review checkpoints can be edited.",
            )
        if stage is not None and status.get("reviewCheckpoint") == "ocr" and stage != "ocr":
            raise HTTPException(
                status_code=409,
                detail="Only OCR & Merge is available before processing continues.",
            )
        if not self.output_dir(code, job_id).is_dir():
            raise HTTPException(status_code=400, detail="Job output directory is missing.")
        return status

    def clear_editor_changes_for_pages(
        self,
        code: str,
        job_id: str,
        pages: list[int],
        resume_from: str | None = None,
    ) -> None:
        with self._lock:
            manifest = self.load_editor_v2_manifest(code, job_id)
            for page in pages:
                editor_v2.mark_regenerated(
                    manifest, page, resume_from or "render"
                )
            manifest["revision"] += 1
            manifest["updatedAt"] = editor_v2.utc_now()
            self.save_editor_v2_manifest(code, job_id, manifest)

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
            record["region"] = self.clamp_editor_region(
                record.get("region"),
                width,
                height,
                f"rawRecords[{index}].region",
            )
            record["text"] = str(record.get("text", ""))
            normalized_raw.append(record)

        raw_by_boxno = {
            int(record["boxno"]): record
            for record in normalized_raw
            if isinstance(record.get("boxno"), int)
        }

        normalized_merged: list[dict[str, Any]] = []
        used_merged_boxnos: set[int] = set()
        next_merged_boxno = max((record.get("boxno", -1) for record in merged_records
                                 if isinstance(record, dict) and isinstance(record.get("boxno"), int)), default=-1) + 1
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
            boxno = record.get("boxno")
            if isinstance(boxno, bool) or not isinstance(boxno, int) or boxno < 0 or boxno in used_merged_boxnos:
                boxno = next_merged_boxno
                next_merged_boxno += 1
            used_merged_boxnos.add(boxno)
            record["boxno"] = boxno
            record["sourceBoxnos"] = source_boxnos
            record["sourceTexts"] = source_texts
            record["text"] = "".join(source_texts)
            record["region"] = region
            normalized_merged.append(record)

        return normalized_raw, normalized_merged

    def normalize_clean_boxes(
        self, code: str, job_id: str, page: int, records: list[Any],
    ) -> list[dict[str, Any]]:
        if len(records) > 256:
            raise HTTPException(400, "A page supports at most 256 clean boxes.")
        with Image.open(self.original_page_path(code, job_id, page)) as image:
            width, height = image.size
        result = []
        ids: set[str] = set()
        numbers: set[int] = set()
        for item in records:
            if not isinstance(item, dict):
                raise HTTPException(400, "Each clean box must be an object.")
            rid, boxno, region = item.get("recordId"), item.get("boxno"), item.get("region")
            if not isinstance(rid, str) or not rid or len(rid) > 200 or rid in ids:
                raise HTTPException(400, "Clean boxes require unique record IDs.")
            if isinstance(boxno, bool) or not isinstance(boxno, int) or boxno < 0 or boxno in numbers:
                raise HTTPException(400, "Clean boxes require unique non-negative box numbers.")
            if not isinstance(region, list) or len(region) != 4 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in region
            ):
                raise HTTPException(400, "Clean box regions require four finite numbers.")
            left, top, right, bottom = [max(0, min(limit, round(value)))
                for value, limit in zip(region, (width, height, width, height))]
            if right <= left or bottom <= top:
                raise HTTPException(400, "Clean box regions must have positive width and height inside the page.")
            result.append({"recordId": rid, "page": page, "boxno": boxno, "region": [left, top, right, bottom]})
            ids.add(rid)
            numbers.add(boxno)
        return result

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

    def run_crop_ocr(
        self,
        code: str,
        job_id: str,
        page: int,
        region: list[int],
        raw_boxno_start: int | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        self.require_editable_job(code, job_id)
        manifest = self.load_editor_v2_manifest(code, job_id)
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
            existing = self.editor_v2_effective_records(
                code, job_id, manifest, "ocr_raw", page
            )
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
