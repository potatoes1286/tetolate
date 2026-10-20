"""Independent, snapshot-based CBZ exports. These never enter the job queue."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
from typing import Any

from fastapi import HTTPException

import translate_cbz
from web_storage import write_json_atomic


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExportManagerMixin:
    def export_state_path(self, code: str, job_id: str) -> Path:
        return self.job_dir(code, job_id) / "web_meta" / "exports.json"

    def load_export_state(self, code: str, job_id: str) -> dict[str, Any]:
        path = self.export_state_path(code, job_id)
        if not path.exists():
            return {"generation": "", "variants": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("variants"), dict) or not isinstance(data.get("generation"), str):
                raise ValueError("Invalid export state")
            return data
        except (OSError, ValueError) as exc:
            raise HTTPException(500, "Could not read CBZ export state.") from exc

    def invalidate_translated_cbz_archives(self, code: str, job_id: str) -> None:
        with self._lock:
            state = self.load_export_state(code, job_id)
            state["generation"] = secrets.token_hex(16)
            state["variants"] = {
                variant: {**item, "status": "cancelled", "message": "Export discarded because its inputs changed.", "finishedAt": timestamp()}
                for variant, item in state["variants"].items() if item.get("status") == "generating"
            }
            write_json_atomic(self.export_state_path(code, job_id), state)
            for variant in translate_cbz.PACKAGE_VARIANTS:
                self.translated_cbz_variant_path(code, job_id, variant).unlink(missing_ok=True)

    def recover_interrupted_exports(self) -> None:
        for code in self.categories():
            for job_id in self.iter_job_ids(code):
                state = self.load_export_state(code, job_id)
                changed = False
                for item in state["variants"].values():
                    if item.get("status") == "generating":
                        item.update(status="failed", message="Export interrupted by server shutdown. Generate it again.", finishedAt=timestamp())
                        changed = True
                if changed:
                    write_json_atomic(self.export_state_path(code, job_id), state)

    def generate_download(self, code: str, job_id: str, variant: str, *,
                          title_mode: str = "original", alt_title_in_notes: bool = False, regenerate: bool = False) -> None:
        code = self.validate_category(code)
        job_id = self.validate_job_id(job_id)
        if variant not in translate_cbz.PACKAGE_VARIANTS:
            raise HTTPException(404, "Unknown translated CBZ variant.")
        if title_mode not in {"original", "translated"}:
            raise HTTPException(400, "Unknown CBZ title mode.")
        with self._lock:
            if self._stopping:
                raise HTTPException(503, "The server is stopping.")
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(404, "Unknown job.")
            if status.get("status") != "complete" or (code, job_id) in self._active_processes:
                raise HTTPException(409, "Only complete jobs can generate downloads.")
            if not self.input_path(code, job_id).is_file() or not self.final_page_files(code, job_id):
                raise HTTPException(400, "The original archive or rendered pages are missing.")
            state = self.load_export_state(code, job_id)
            if state["variants"].get(variant, {}).get("status") == "generating":
                raise HTTPException(409, "That CBZ is already being generated.")
            if self.translated_cbz_variant_path(code, job_id, variant).is_file() and not regenerate:
                raise HTTPException(409, "That CBZ already exists. Use the regenerate action to replace it.")
            active = any(item.get("status") == "generating" for item in state["variants"].values())
            if not active or state.get("titleMode") != title_mode or state.get("altTitleInNotes") != bool(alt_title_in_notes):
                self.invalidate_translated_cbz_archives(code, job_id)
                state = self.load_export_state(code, job_id)
            else:
                for name in translate_cbz.PACKAGE_VARIANTS:
                    self.translated_cbz_variant_path(code, job_id, name).unlink(missing_ok=True)
                state["variants"] = {name: item for name, item in state["variants"].items() if item.get("status") != "complete"}
            status.update(cbzTitleMode=title_mode, cbzAltTitleInNotes=bool(alt_title_in_notes))
            self.save_status(code, job_id, status)
            token = secrets.token_hex(16)
            state.update(titleMode=title_mode, altTitleInNotes=bool(alt_title_in_notes))
            state["variants"][variant] = {"id": token, "status": "generating", "startedAt": timestamp(),
                                           "message": f"Generating {variant.upper()} CBZ."}
            write_json_atomic(self.export_state_path(code, job_id), state)
            thread = threading.Thread(target=self.run_download_export,
                args=(code, job_id, variant, state["generation"], token, dict(status)),
                name=f"tetolate-export-{job_id}-{variant}", daemon=True)
            self._export_threads.add(thread)
            try:
                thread.start()
            except Exception:
                self._export_threads.discard(thread)
                state["variants"][variant].update(status="failed", message="Could not start CBZ export.", finishedAt=timestamp())
                write_json_atomic(self.export_state_path(code, job_id), state)
                raise

    def current_export(self, code: str, job_id: str, variant: str, generation: str, token: str) -> dict[str, Any] | None:
        if self._stopping or self.load_status(code, job_id) is None:
            return None
        state = self.load_export_state(code, job_id)
        item = state["variants"].get(variant, {})
        return state if state["generation"] == generation and item.get("id") == token and item.get("status") == "generating" else None

    def run_download_export(self, code: str, job_id: str, variant: str, generation: str, token: str, status: dict[str, Any]) -> None:
        staging: Path | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="tetolate-export-") as temporary:
                snapshot = Path(temporary)
                pages_dir = snapshot / "pages" / "final"
                pages_dir.mkdir(parents=True)
                source = snapshot / "input.cbz"
                shutil.copyfile(self.input_path(code, job_id), source)
                pages = []
                for index, path in enumerate(self.final_page_files(code, job_id)):
                    destination = pages_dir / path.name
                    shutil.copyfile(path, destination)
                    pages.append(translate_cbz.Page(index, destination))
                if not pages:
                    raise RuntimeError("Rendered pages are missing.")
                with self._lock:
                    if self.current_export(code, job_id, variant, generation, token) is None:
                        return
                title_fields = ("comicInfoTitle", "jobName", "translatedJobName", "vlmBaseUrl", "vlmModel")
                original_titles = {key: status.get(key) for key in title_fields}
                title, alternate, warning = self.resolve_cbz_export_titles(code, job_id, status)
                config = replace(translate_cbz.load_config(self.config.pipeline_config, None),
                    webp_quality=int(status.get("webpQuality") or self.config.default_webp_quality),
                    jxl_quality=int(status.get("jxlQuality") or self.config.default_jxl_quality),
                    imagemagick_workers=self.status_page_workers(status, "imagemagickWorkers", "ImageMagick workers"))
                translate_cbz.print_packaged_cbz(pages, snapshot, source, config, package_variant=variant,
                                               comicinfo_title=title, comicinfo_alt_title=alternate)
                target = self.translated_cbz_variant_path(code, job_id, variant)
                with self._lock:
                    if self.current_export(code, job_id, variant, generation, token) is None:
                        return
                    staging = translate_cbz.temporary_archive_path(target)
                shutil.copyfile(snapshot / target.name, staging)
                with self._lock:
                    state = self.current_export(code, job_id, variant, generation, token)
                    if state is None:
                        return
                    os.replace(staging, target)
                    state["variants"][variant].update(status="complete", message=f"{variant.upper()} CBZ is ready.", finishedAt=timestamp())
                    if warning:
                        state["variants"][variant]["warning"] = warning
                    write_json_atomic(self.export_state_path(code, job_id), state)
                    current = self.load_status(code, job_id)
                    if current is not None and all(current.get(key) == value for key, value in original_titles.items()):
                        for key in ("translatedJobName", "titleTranslationAttempted", "titleTranslationSkipped"):
                            if key in status:
                                current[key] = status[key]
                        if warning:
                            current["titleTranslationWarning"] = warning
                        else:
                            current.pop("titleTranslationWarning", None)
                        self.save_status(code, job_id, current)
        except Exception as exc:
            with self._lock:
                state = self.current_export(code, job_id, variant, generation, token)
                if state is not None:
                    state["variants"][variant].update(status="failed", message=f"CBZ export failed: {exc}", finishedAt=timestamp())
                    write_json_atomic(self.export_state_path(code, job_id), state)
        finally:
            if staging is not None:
                staging.unlink(missing_ok=True)
            with self._lock:
                self._export_threads.discard(threading.current_thread())
