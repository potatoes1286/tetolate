"""Category-series storage, chapter dependencies, and repeatable context."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from fastapi import HTTPException

import translate_cbz
import web_security
from web_comicinfo import parse_common_xml
from web_storage import write_json_atomic

SERIES_FILENAME = "series.json"
SERIES_CONTEXT_FILENAME = "series_context.json"


def chapter_number(value: str, label: str = "Chapter") -> str:
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, AttributeError):
        raise HTTPException(400, f"{label} must be a non-negative number.") from None
    if not number.is_finite() or not 0 <= number <= 1_000_000 or len(str(value)) > 20:
        raise HTTPException(400, f"{label} must be between 0 and 1000000.")
    return format(number.normalize(), "f")


def chapter_key(status: dict[str, Any]) -> tuple[Decimal, Decimal]:
    return Decimal(status.get("volume") or "-1"), Decimal(status["chapter"])


def chapter_label(status: dict[str, Any]) -> str:
    volume = str(status.get("volume") or "")
    chapter = str(status.get("chapter") or "")
    return (f"Vol. {volume} " if volume else "") + (f"Ch. {chapter}" if chapter else "")


def chapter_filename_key(name: str) -> list[tuple[int, Any]]:
    return [(1, int(part)) if part.isdigit() else (0, part.casefold())
            for part in re.split(r"(\d+)", name)]


class SeriesManagerMixin:
    def is_series(self, code: str) -> bool:
        return (self.category_dir(code) / SERIES_FILENAME).is_file()

    def get_series(self, code: str) -> dict[str, Any]:
        self.validate_category(code)
        path = self.category_dir(code) / SERIES_FILENAME
        if not path.is_file():
            raise HTTPException(404, "This category is not a series.")
        return web_security.read_json_object(path)

    def save_series(self, code: str, series: dict[str, Any]) -> None:
        write_json_atomic(self.category_dir(code) / SERIES_FILENAME, series)

    def initialize_series(self, code: str) -> None:
        self.save_series(code, {
            "name": code, "instructions": "", "notes": "", "revision": 0,
            "notesThrough": None, "notesFromJobId": "", "comicInfoXml": "",
        })

    def series_chapters(self, code: str) -> list[dict[str, Any]]:
        self.get_series(code)
        chapters = [{**status, "jobId": job_id} for job_id in self.iter_job_ids(code)
                    if (status := self.load_status(code, job_id)) is not None]
        return sorted(chapters, key=chapter_key)

    def validate_series_chapter(self, code: str, chapter: str, volume: str = "", job_id: str = "") -> tuple[str, str]:
        if not self.is_series(code):
            if chapter.strip() or volume.strip():
                raise HTTPException(400, "Chapter and volume require a series category.")
            return "", ""
        chapter = chapter_number(chapter)
        volume = chapter_number(volume, "Volume") if volume.strip() else ""
        if any(item["jobId"] != job_id and chapter_key(item) == chapter_key({"chapter": chapter, "volume": volume})
               for item in self.series_chapters(code)):
            raise HTTPException(409, "That chapter number is already used in this volume.")
        return chapter, volume

    def update_series(self, code: str, name: str, instructions: str, notes: str, comic_info_xml: str = "") -> None:
        name = name.strip()
        if not name or len(name) > 256:
            raise HTTPException(400, "Series name must be 1-256 characters.")
        if len(instructions) > 100_000 or len(notes) > 100_000:
            raise HTTPException(400, "Series instructions and notes must each be at most 100000 characters.")
        parse_common_xml(comic_info_xml)
        with self._lock:
            series = self.get_series(code)
            metadata_changed = series["comicInfoXml"] != comic_info_xml.strip() or series["name"] != name
            context_changed = series["instructions"] != instructions.strip() or series["notes"] != notes.strip()
            series.update(name=name, instructions=instructions.strip(), notes=notes.strip(),
                          comicInfoXml=comic_info_xml.strip(), revision=series["revision"] + int(context_changed))
            self.save_series(code, series)
            if metadata_changed:
                for job_id in self.iter_job_ids(code):
                    self.invalidate_translated_cbz_archives(code, job_id)
            self.enqueue_ready_series_jobs()

    def assign_job_chapter(self, code: str, job_id: str, chapter: str, volume: str = "") -> None:
        with self._lock:
            self.get_series(code)
            self.validate_job_id(job_id)
            status = self.load_status(code, job_id)
            if status is None:
                raise HTTPException(404, "Unknown job.")
            if status.get("status") not in {"complete", "failed", "cancelled", "paused"} or (
                (code, job_id) in self._active_processes or (code, job_id) in self._starting_jobs
            ):
                raise HTTPException(409, "Stop the job before changing chapter numbers.")
            if self.series_context_path(code, job_id).exists():
                raise HTTPException(409, "This chapter has saved context. Its chapter and volume are fixed.")
            chapter, volume = self.validate_series_chapter(code, chapter, volume, job_id)
            status.update(chapter=chapter, volume=volume, writeTranslationNotes=True)
            self.save_status(code, job_id, status)
            self.invalidate_translated_cbz_archives(code, job_id)
            self.enqueue_ready_series_jobs()

    def series_context_path(self, code: str, job_id: str) -> Path:
        return self.editor_meta_dir(code, job_id) / SERIES_CONTEXT_FILENAME

    def series_wait_reason(self, code: str, status: dict[str, Any]) -> str:
        if not self.is_series(code) or status.get("pendingPageReruns") or status.get("pendingFinishEdits"):
            return ""
        series = self.get_series(code)
        earlier = [item for item in self.series_chapters(code) if chapter_key(item) < chapter_key(status)]
        for item in earlier:
            if item.get("status") != "complete":
                return f"Waiting for {chapter_label(item)} ({item.get('status', 'unknown')})."
        if self.series_context_path(code, status["jobId"]).exists():
            return ""
        through = series["notesThrough"]
        if through and chapter_key(through) >= chapter_key(status):
            return "Series notes include this chapter or a later chapter. Publish earlier chapter notes first."
        if earlier and (not through or chapter_key(through) < chapter_key(earlier[-1])):
            return f"Waiting for series notes from {chapter_label(earlier[-1])}."
        return ""

    def enqueue_ready_series_jobs(self) -> None:
        with self._lock:
            if self._stopping:
                return
            for code in self.categories():
                if not self.is_series(code):
                    continue
                for status in self.series_chapters(code):
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
        if not self.is_series(code):
            return
        with self._lock:
            path = self.series_context_path(code, job_id)
            if status.get("seriesContextSaved") and self.translation_notes_path(code, job_id).exists():
                return
            if path.exists():
                snapshot = web_security.read_json_object(path)
            else:
                series = self.get_series(code)
                snapshot = {
                    "category": code, "chapter": status["chapter"], "volume": status["volume"],
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

    def publish_series_notes(self, code: str, job_id: str, *, explicit: bool = False, use_current: bool = False) -> bool:
        with self._lock:
            self.validate_category(code)
            self.validate_job_id(job_id)
            status = self.load_status(code, job_id) or {}
            if not self.is_series(code) or status.get("status") != "complete":
                if explicit:
                    raise HTTPException(400, "Only a completed series chapter can publish notes.")
                return False
            series = self.get_series(code)
            through = series["notesThrough"]
            if not explicit and (status.get("seriesNotesPublished") or through and chapter_key(through) >= chapter_key(status)):
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
            if notes in {"No translated records were available.", "No reusable translation notes were generated."}:
                notes = series["notes"]
            series.update(notes=notes, revision=series["revision"] + 1,
                          notesThrough={"chapter": status["chapter"], "volume": status["volume"]}, notesFromJobId=job_id)
            self.save_series(code, series)
            status.pop("seriesNotesWarning", None)
            status["seriesNotesPublished"] = True
            self.save_status(code, job_id, status)
            self.enqueue_ready_series_jobs()
            return True

    def recover_series_notes(self) -> None:
        with self._lock:
            for code in self.categories():
                if not self.is_series(code):
                    continue
                for status in self.series_chapters(code):
                    if status.get("status") != "complete":
                        break
                    series = self.get_series(code)
                    through = series["notesThrough"]
                    if through and chapter_key(through) >= chapter_key(status):
                        continue
                    if not self.publish_series_notes(code, status["jobId"]):
                        break
