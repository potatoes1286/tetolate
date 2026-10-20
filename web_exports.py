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
import re
import zipfile
from typing import Any

from fastapi import HTTPException

import translate_cbz
from web_storage import write_json_atomic
from web_comicinfo import metadata_archive
from web_series import chapter_label


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
            self.invalidate_category_export(code)

    def recover_interrupted_exports(self) -> None:
        for code in self.categories():
            state = self.category_export_status(code)
            if state.get("status") == "generating":
                state.update(status="failed", message="Collection export interrupted by server shutdown. Export again.")
                write_json_atomic(self.category_export_state_path(code), state)
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
            snapshot = dict(status)
            if self.is_series(code):
                snapshot["_seriesMetadata"] = self.get_series(code)
            thread = threading.Thread(target=self.run_download_export,
                args=(code, job_id, variant, state["generation"], token, snapshot),
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
                if "_seriesMetadata" in status:
                    series = status["_seriesMetadata"]
                    merged = snapshot / "metadata.cbz"
                    metadata_archive(source, merged, series["comicInfoXml"], series=series["name"],
                        volume=status["volume"], chapter=status["chapter"], title=title or status.get("jobName", ""), page_count=len(pages))
                    source = merged
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

    def category_export_state_path(self, code: str) -> Path:
        return self.category_dir(code) / "cbz_export.json"

    def category_export_path(self, code: str) -> Path:
        return self.category_dir(code) / "exported_cbz.zip"

    def category_export_status(self, code: str) -> dict[str, Any]:
        self.validate_category(code)
        path = self.category_export_state_path(code)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def public_category_export_status(self, code: str) -> dict[str, Any]:
        return {key: value for key, value in self.category_export_status(code).items() if key != "inputs"}

    def invalidate_category_export(self, code: str) -> None:
        with self._lock:
            state = self.category_export_status(code)
            if state:
                state.update(status="cancelled", message="Collection export is out of date. Export again.")
                write_json_atomic(self.category_export_state_path(code), state)
            self.category_export_path(code).unlink(missing_ok=True)

    def category_export_inputs(self, code: str) -> dict[str, Any]:
        series = self.get_series(code)
        chapters = [item for item in self.series_chapters(code) if item.get("status") == "complete"]
        return {"name": series["name"], "xml": series["comicInfoXml"], "chapters": [
            {"id": item["jobId"], "volume": item["volume"], "chapter": item["chapter"],
             "name": item.get("translatedJobName") or item.get("jobName"),
             "generation": self.load_export_state(code, item["jobId"])["generation"],
             "pages": [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in self.final_page_files(code, item["jobId"])]}
            for item in chapters]}

    def require_category_export(self, code: str) -> Path:
        with self._lock:
            self.get_series(code)
            state = self.category_export_status(code)
            if state.get("status") != "complete" or not self.category_export_path(code).is_file():
                raise HTTPException(409, "No completed CBZ collection is available.")
            if state["inputs"] != self.category_export_inputs(code):
                self.invalidate_category_export(code)
                raise HTTPException(409, "The CBZ collection is out of date. Export again.")
            return self.category_export_path(code)

    def generate_category_export(self, code: str, variant: str, grouping: str) -> None:
        with self._lock:
            series = self.get_series(code)
            if self._stopping:
                raise HTTPException(503, "The server is stopping.")
            if variant not in translate_cbz.PACKAGE_VARIANTS or grouping not in {"chapters", "volumes"}:
                raise HTTPException(400, "Choose PNG, WebP, or JXL and chapters or volumes.")
            if self.category_export_status(code).get("status") == "generating":
                raise HTTPException(409, "A CBZ collection is already being exported.")
            chapters = [item for item in self.series_chapters(code) if item.get("status") == "complete"]
            if not chapters or any(not self.final_page_files(code, item["jobId"]) for item in chapters):
                raise HTTPException(400, "Completed chapters with rendered pages are required.")
            self.category_export_path(code).unlink(missing_ok=True)
            state = {"id": secrets.token_hex(16), "status": "generating", "format": variant, "grouping": grouping,
                     "message": "Exporting CBZ collection...", "startedAt": timestamp(), "inputs": self.category_export_inputs(code)}
            write_json_atomic(self.category_export_state_path(code), state)
            thread = threading.Thread(target=self.run_category_export, args=(code, state, series, chapters), daemon=True,
                                      name=f"tetolate-export-category-{code}")
            self._export_threads.add(thread)
            try:
                thread.start()
            except Exception:
                self._export_threads.discard(thread)
                state.update(status="failed", message="Could not start collection export.")
                write_json_atomic(self.category_export_state_path(code), state)
                raise

    def run_category_export(self, code: str, state: dict[str, Any], series: dict[str, Any], chapters: list[dict[str, Any]]) -> None:
        staging = None
        def current():
            return (not self._stopping and code in self.categories()
                    and self.category_export_status(code).get("id") == state["id"]
                    and self.category_export_status(code).get("status") == "generating")
        try:
            groups = {}
            for item in chapters:
                key = ("volume", item["volume"]) if state["grouping"] == "volumes" and item["volume"] else ("chapter", item["jobId"])
                groups.setdefault(key, []).append(item)
            with tempfile.TemporaryDirectory(prefix="tetolate-series-export-") as temporary:
                work = Path(temporary)
                collection = work / "collection.zip"
                with zipfile.ZipFile(collection, "w", compression=zipfile.ZIP_STORED) as bundle:
                    for index, (key, items) in enumerate(groups.items()):
                        with self._lock:
                            if not current():
                                return
                        folder = work / str(index)
                        folder.mkdir()
                        page_directory = folder / "pages" / "final"
                        page_directory.mkdir(parents=True)
                        pages = []
                        for item in items:
                            for original in self.final_page_files(code, item["jobId"]):
                                target = page_directory / f"{len(pages):06d}.png"
                                shutil.copyfile(original, target)
                                pages.append(translate_cbz.Page(len(pages), target))
                        if not pages:
                            raise RuntimeError("Rendered pages are missing.")
                        first = items[0]
                        title = first.get("translatedJobName") or first.get("jobName") or first["jobId"]
                        name = f"{chapter_label(first)} {title}"
                        volume_export = key[0] == "volume"
                        if volume_export:
                            title = f"{series['name']} Vol. {first['volume']}"
                            name = f"Vol. {first['volume']} {series['name']}"
                        source = folder / "input.cbz"
                        metadata_archive(None if volume_export else self.input_path(code, first["jobId"]), source,
                            series["comicInfoXml"], series=series["name"], volume=first["volume"],
                            chapter="" if volume_export else first["chapter"], title=title, page_count=len(pages))
                        config = replace(translate_cbz.load_config(self.config.pipeline_config, None),
                            webp_quality=int(first.get("webpQuality") or self.config.default_webp_quality),
                            jxl_quality=int(first.get("jxlQuality") or self.config.default_jxl_quality),
                            imagemagick_workers=self.status_page_workers(first, "imagemagickWorkers", "ImageMagick workers"))
                        translate_cbz.print_packaged_cbz(pages, folder, source, config, package_variant=state["format"])
                        filename = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(". ")[:180] or first["jobId"]
                        bundle.write(folder / self.translated_cbz_variant_path(code, first["jobId"], state["format"]).name, filename + ".cbz")
                        with self._lock:
                            if not current():
                                return
                            state["message"] = f"Exported {index + 1}/{len(groups)} CBZ files."
                            write_json_atomic(self.category_export_state_path(code), state)
                with self._lock:
                    if not current():
                        return
                    staging = translate_cbz.temporary_archive_path(self.category_export_path(code))
                shutil.copyfile(collection, staging)
                with self._lock:
                    if not current():
                        return
                    if state["inputs"] != self.category_export_inputs(code):
                        self.invalidate_category_export(code)
                        return
                    os.replace(staging, self.category_export_path(code))
                    state.update(status="complete", message=f"CBZ collection ready ({len(groups)} files).", finishedAt=timestamp())
                    write_json_atomic(self.category_export_state_path(code), state)
        except Exception as exc:
            with self._lock:
                if current():
                    state.update(status="failed", message=f"Collection export failed: {exc}", finishedAt=timestamp())
                    write_json_atomic(self.category_export_state_path(code), state)
        finally:
            if staging:
                staging.unlink(missing_ok=True)
            with self._lock:
                self._export_threads.discard(threading.current_thread())
