"""Series storage, chapter dependencies, and repeatable translation context."""

from __future__ import annotations

import re
import secrets
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import HTTPException

import translate_cbz
import web_security
from web_storage import write_json_atomic


SERIES_FILENAME = ".series.json"
SERIES_CONTEXT_FILENAME = "series_context.json"


def chapter_order(value: str) -> str:
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        raise HTTPException(400, "Chapter order must be a positive number.") from None
    if not number.is_finite() or not 0 < number <= 1_000_000 or len(str(value)) > 20:
        raise HTTPException(400, "Chapter order must be between 0 and 1000000.")
    return format(number.normalize(), "f")


def chapter_filename_key(name: str) -> list[tuple[int, Any]]:
    return [(1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r"(\d+)", name)]


class SeriesManagerMixin:
    def load_series(self, code: str) -> dict[str, Any]:
        self.validate_category(code)
        path = self.category_dir(code) / SERIES_FILENAME
        return web_security.read_json_object(path) if path.exists() else {}

    def get_series(self, code: str, series_id: str) -> dict[str, Any]:
        series = self.load_series(code).get(series_id)
        if not isinstance(series, dict):
            raise HTTPException(404, "Unknown series.")
        return dict(series)

    def save_series(self, code: str, series_id: str, series: dict[str, Any]) -> None:
        data = self.load_series(code)
        data[series_id] = series
        write_json_atomic(self.category_dir(code) / SERIES_FILENAME, data)

    def create_series(self, code: str, name: str) -> str:
        name = name.strip()
        if not name or len(name) > 256:
            raise HTTPException(400, "Series name must be 1-256 characters.")
        with self._lock:
            if any(item["name"].casefold() == name.casefold() for item in self.load_series(code).values()):
                raise HTTPException(409, "That series already exists.")
            series_id = secrets.token_hex(8)
            self.save_series(code, series_id, {
                "name": name, "instructions": "", "notes": "", "revision": 0,
                "notesThroughOrder": "", "notesFromJobId": "",
            })
            return series_id

    def series_chapters(self, code: str, series_id: str) -> list[dict[str, Any]]:
        chapters = []
        for job_id in self.iter_job_ids(code):
            status = self.load_status(code, job_id) or {}
            if status.get("seriesId") == series_id:
                chapters.append({**status, "jobId": job_id})
        return sorted(chapters, key=lambda item: Decimal(item["chapterOrder"]))

    def validate_series_chapter(
        self, code: str, series_id: str, order: str, job_id: str = "",
    ) -> str:
        if not series_id:
            if order.strip():
                raise HTTPException(400, "Select a series for the chapter order.")
            return ""
        self.get_series(code, series_id)
        order = chapter_order(order)
        if any(item["jobId"] != job_id and item["chapterOrder"] == order
               for item in self.series_chapters(code, series_id)):
            raise HTTPException(409, "That chapter order is already used in this series.")
        return order

    def update_series(self, code: str, series_id: str, name: str, instructions: str, notes: str) -> None:
        name = name.strip()
        if not name or len(name) > 256:
            raise HTTPException(400, "Series name must be 1-256 characters.")
        if len(instructions) > 100_000 or len(notes) > 100_000:
            raise HTTPException(400, "Series instructions and notes must each be at most 100000 characters.")
        with self._lock:
            if any(key != series_id and item["name"].casefold() == name.casefold()
                   for key, item in self.load_series(code).items()):
                raise HTTPException(409, "That series already exists.")
            series = self.get_series(code, series_id)
            series.update(name=name, instructions=instructions.strip(), notes=notes.strip(),
                          revision=series["revision"] + 1)
            self.save_series(code, series_id, series)
            self.enqueue_ready_series_jobs()

    def assign_job_series(self, code: str, job_id: str, series_id: str, order: str) -> None:
        with self._lock:
            self.validate_category(code)
            self.validate_job_id(job_id)
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(404, "Unknown job.")
            if status.get("status") not in {"complete", "failed", "cancelled", "paused"}:
                raise HTTPException(409, "Stop the job before changing its series.")
            if (code, job_id) in self._active_processes:
                raise HTTPException(409, "Stop the active process before changing its series.")
            if self.series_context_path(code, job_id).exists():
                raise HTTPException(409, "This chapter has saved series context. Its series and order are fixed.")
            order = self.validate_series_chapter(code, series_id, order, job_id)
            status.update(seriesId=series_id, chapterOrder=order)
            if series_id:
                status["writeTranslationNotes"] = True
            self.save_status(code, job_id, status)
            self.enqueue_ready_series_jobs()

    def series_context_path(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / SERIES_CONTEXT_FILENAME

    def series_wait_reason(self, code: str, status: dict[str, Any]) -> str:
        series_id = status.get("seriesId")
        if not series_id:
            return ""
        if status.get("pendingPageReruns"):
            return ""
        series = self.get_series(code, series_id)
        order = Decimal(status["chapterOrder"])
        earlier = [item for item in self.series_chapters(code, series_id)
                   if Decimal(item["chapterOrder"]) < order]
        for item in earlier:
            if item.get("status") != "complete":
                return f"Waiting for chapter {item['chapterOrder']} ({item.get('status', 'unknown')})."
        if self.series_context_path(code, status["jobId"]).exists():
            return ""
        through = series.get("notesThroughOrder")
        if through and Decimal(through) >= order:
            return "Series notes include this chapter or a later chapter. Publish earlier chapter notes first."
        if earlier and (not through or Decimal(through) < Decimal(earlier[-1]["chapterOrder"])):
            return f"Waiting for series notes from chapter {earlier[-1]['chapterOrder']}."
        return ""

    def enqueue_ready_series_jobs(self) -> None:
        with self._lock:
            if self._stopping:
                return
            for code in self.categories():
                for series_id in self.load_series(code):
                    for status in self.series_chapters(code, series_id):
                        if status.get("status") != "queued":
                            continue
                        job_id = status["jobId"]
                        reason = self.series_wait_reason(code, status)
                        if reason:
                            status.update(phase="Waiting for series", message=reason)
                            self.save_status(code, job_id, status)
                        else:
                            self.enqueue(code, job_id)

    def prepare_series_context(self, code: str, job_id: str, status: dict[str, Any]) -> None:
        series_id = status.get("seriesId")
        if not series_id:
            return
        with self._lock:
            path = self.series_context_path(code, job_id)
            if status.get("seriesContextSaved") and self.translation_notes_path(code, job_id).exists():
                return
            if path.exists():
                snapshot = web_security.read_json_object(path)
            else:
                series = self.get_series(code, series_id)
                snapshot = {
                    "seriesId": series_id, "chapterOrder": status["chapterOrder"],
                    "revision": series["revision"], "instructions": series["instructions"],
                    "notes": series["notes"], "notesFromJobId": series["notesFromJobId"],
                    "jobNotes": translate_cbz.load_translation_notes(
                        self.translation_notes_path(code, job_id)
                        if self.translation_notes_path(code, job_id).exists() else None,
                    )["job"],
                }
                write_json_atomic(path, snapshot)
            notes_path = self.translation_notes_path(code, job_id)
            notes = translate_cbz.load_translation_notes(notes_path if notes_path.exists() else None)
            sections = []
            if snapshot["instructions"]:
                sections.append("Fixed series instructions (take priority over other notes):\n" + snapshot["instructions"])
            if snapshot["notes"]:
                sections.append("Series notes from earlier chapters:\n" + snapshot["notes"])
            if snapshot["jobNotes"]:
                sections.append("Chapter instructions:\n" + snapshot["jobNotes"])
            notes["job"] = "\n\n".join(sections)
            write_json_atomic(notes_path, notes)
            status["seriesContextSaved"] = True
            self.save_status(code, job_id, status)

    def publish_series_notes(
        self, code: str, job_id: str, *, explicit: bool = False, use_current: bool = False,
    ) -> bool:
        with self._lock:
            self.validate_category(code)
            self.validate_job_id(job_id)
            status = self.load_status(code, job_id) or {}
            if not status.get("seriesId") or status.get("status") != "complete":
                if explicit:
                    raise HTTPException(400, "Only a completed series chapter can publish notes.")
                return False
            series_id = status["seriesId"]
            series = self.get_series(code, series_id)
            through = series.get("notesThroughOrder")
            if not explicit and status.get("seriesNotesPublished"):
                return False
            if not explicit and through and Decimal(through) >= Decimal(status["chapterOrder"]):
                return False
            snapshot_path = self.series_context_path(code, job_id)
            if not explicit and snapshot_path.exists():
                snapshot = web_security.read_json_object(snapshot_path)
                if snapshot["revision"] != series["revision"]:
                    status["seriesNotesWarning"] = (
                        "Series instructions or notes changed during this chapter. "
                        "Review the notes and publish this chapter's notes explicitly."
                    )
                    self.save_status(code, job_id, status)
                    return False
            notes = series["notes"] if use_current else self.read_generated_translation_notes(code, job_id)
            if not notes and not use_current:
                if explicit:
                    raise HTTPException(400, "This chapter has no generated translation notes.")
                status["seriesNotesWarning"] = "No generated notes are available. Later chapters will wait."
                self.save_status(code, job_id, status)
                return False
            if notes in {
                "No translated records were available.",
                "No reusable translation notes were generated.",
            }:
                notes = series["notes"]
            series.update(notes=notes, revision=series["revision"] + 1,
                          notesThroughOrder=status["chapterOrder"], notesFromJobId=job_id)
            self.save_series(code, series_id, series)
            status.pop("seriesNotesWarning", None)
            status["seriesNotesPublished"] = True
            self.save_status(code, job_id, status)
            self.enqueue_ready_series_jobs()
            return True

    def recover_series_notes(self) -> None:
        """Finish a notes publication interrupted after a chapter completed."""
        with self._lock:
            for code in self.categories():
                for series_id in self.load_series(code):
                    for status in self.series_chapters(code, series_id):
                        if status.get("status") != "complete":
                            break
                        series = self.get_series(code, series_id)
                        through = series.get("notesThroughOrder")
                        if through and Decimal(through) >= Decimal(status["chapterOrder"]):
                            continue
                        if not self.publish_series_notes(code, status["jobId"]):
                            break

    def public_series(self, code: str) -> list[dict[str, Any]]:
        return sorted([{"id": key, **value} for key, value in self.load_series(code).items()],
                      key=lambda item: item["name"].casefold())
