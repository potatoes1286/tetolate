from __future__ import annotations

from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest import mock

from fastapi import HTTPException

from tests import test_regressions
import web_app
import web_pages


class QueueSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = test_regressions.WebAuthenticationTests().initialized_manager(Path(self.temp.name), ["default", "other"])

    def job(self, job_id="12345678", state="queued", code="default", **fields):
        self.manager.job_dir(code, job_id).mkdir(parents=True)
        status = self.manager.load_category_advanced_options(code)
        status.update(status=state, category=code, jobId=job_id, phase="Waiting for series",
                      message="Waiting for chapter 1.", createdAt="2026-10-18T00:00:00+00:00",
                      **fields)
        self.manager.save_status(code, job_id, status)
        return status

    def update(self, job_id="12345678", **changes):
        options = web_app.resolve_control_settings(self.manager, "default", changes, job_id, require_model=False)
        self.manager.update_job_advanced_options("default", job_id, **options,
            clear_vlm_auth_token=False, clear_paddleocr_vl_auth_token=False)

    def test_queued_options_are_editable_persistent_and_used_by_pipeline(self):
        original = self.job(pendingResumeFrom="translations", pendingResumePage=3)
        self.manager._queued_jobs.add(("default", "12345678"))
        self.manager._queue.put(("default", "12345678"))
        self.assertTrue(self.manager.public_status("default", "12345678", False)["canUpdateAdvancedOptions"])
        self.update(paddleocr_vl_server_url="http://127.0.0.1:8081/v1", ocr_page_workers=4,
                    lama_workers=3, imagemagick_workers=2)
        restarted = web_app.JobManager(self.manager.config)
        status = restarted.load_status("default", "12345678")
        for key in ("status", "phase", "message", "pendingResumeFrom", "pendingResumePage", "createdAt"):
            self.assertEqual(status[key], original[key])
        command = restarted.build_command("default", "12345678")
        self.assertEqual(command[command.index("--paddleocr-vl-server-url") + 1], "http://127.0.0.1:8081/v1")
        for flag, value in (("--ocr-workers", "4"), ("--lama-workers", "3"), ("--imagemagick-workers", "2")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(self.manager._queued_jobs, {("default", "12345678")})
        self.assertEqual(self.manager._queue.get_nowait(), ("default", "12345678"))

    def test_bulk_applies_all_options_only_to_queued_jobs_in_category(self):
        originals = {state: self.job(f"{index:08x}", state) for index, state in enumerate(
            ("queued", "running", "paused", "failed", "complete", "cancelled", "terminating"), 1)}
        self.job("00000008", code="other")
        self.job("00000009", seriesId="series", chapterOrder="14", pendingResumeFrom="placements")
        self.manager.set_initial_translation_notes("default", "00000009", "Keep chapter notes")
        self.manager._queued_jobs.update({("default", "00000001"), ("default", "00000009")})
        options = web_app.resolve_control_settings(self.manager, "default", {
            "thinking_budget_tokens": 0, "vlm_base_url": "http://127.0.0.1:5002/v1", "vlm_model": "new-model",
            "paddleocr_vl_server_url": "http://127.0.0.1:8081/v1", "paddleocr_vl_model": "ocr-new",
            "ocr_page_workers": 4, "lama_workers": 3, "imagemagick_workers": 2,
            "pause_after_ocr": True, "proofread_translations": False, "alt_placement_enabled": False,
            "auto_translate_comicinfo_title": False, "write_translation_notes": False, "source_language": "japanese",
        }, require_model=False)
        options.pop("vlm_auth_token")
        options.pop("paddleocr_vl_auth_token")
        self.manager.remember_category_advanced_options("default", **options, translation_notes="Future job notes")
        defaults = self.manager.category_advanced_options_path("default").read_bytes()
        self.assertEqual(set(self.manager.apply_category_options_to_queue("default")), {"00000001", "00000009"})
        updated = self.manager.load_status("default", "00000001")
        for key, field in web_app.CONTROL_OPTION_FIELDS.items():
            self.assertEqual(updated[field], options[key], field)
        for index, state in enumerate(("queued", "running", "paused", "failed", "complete", "cancelled", "terminating"), 1):
            if state != "queued":
                self.assertEqual(self.manager.load_status("default", f"{index:08x}"), originals[state])
        self.assertEqual(self.manager.load_status("other", "00000008")["ocrPageWorkers"], 1)
        series = self.manager.load_status("default", "00000009")
        self.assertEqual(series["chapterOrder"], "14")
        self.assertTrue(series["writeTranslationNotes"])
        self.assertEqual(series["pendingResumeFrom"], "placements")
        self.assertEqual(json.loads(self.manager.translation_notes_path("default", "00000009").read_text())["job"], "Keep chapter notes")
        self.assertEqual(self.manager.category_advanced_options_path("default").read_bytes(), defaults)
        self.assertEqual(self.manager._queued_jobs, {("default", "00000001"), ("default", "00000009")})

    def test_starting_or_active_jobs_reject_edits_and_are_skipped_in_bulk(self):
        for index, collection in enumerate((self.manager._starting_jobs, self.manager._active_processes), 1):
            job_id = f"{index:08x}"
            original = self.job(job_id)
            key = ("default", job_id)
            if isinstance(collection, set):
                collection.add(key)
            else:
                collection[key] = mock.Mock()
            self.assertFalse(self.manager.public_status("default", job_id, False)["canUpdateAdvancedOptions"])
            with self.assertRaises(HTTPException):
                self.update(job_id, ocr_page_workers=4)
            self.assertEqual(self.manager.load_status("default", job_id), original)
        self.assertEqual(self.manager.apply_category_options_to_queue("default"), [])

    def test_worker_claim_blocks_edits_before_reading_settings(self):
        self.job()
        entered, release = threading.Event(), threading.Event()
        def startup(*args):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test did not release the worker")
        with mock.patch.object(self.manager, "_run_job", side_effect=startup):
            worker = threading.Thread(target=self.manager.run_job, args=("default", "12345678"))
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaises(HTTPException):
                    self.update(ocr_page_workers=4)
                self.assertEqual(self.manager.apply_category_options_to_queue("default"), [])
            finally:
                release.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.manager._starting_jobs, set())
        self.update(ocr_page_workers=4)

    def test_worker_claim_is_released_after_failure(self):
        with mock.patch.object(self.manager, "_run_job", side_effect=RuntimeError("test")):
            with self.assertRaisesRegex(RuntimeError, "test"):
                self.manager.run_job("default", "12345678")
        self.assertEqual(self.manager._starting_jobs, set())

    def test_bulk_keeps_keys_for_same_endpoint_and_removes_keys_for_changed_endpoints(self):
        defaults = self.manager.load_category_advanced_options("default")
        self.job("00000001")
        self.job("00000002", vlmBaseUrl="http://old.example/v1", paddleocrVlServerUrl="http://old-ocr.example/v1")
        for job_id, token in (("00000001", "private-current-vlm"), ("00000002", "private-old-vlm")):
            self.manager.update_job_auth_tokens("default", job_id, vlm_auth_token=token, paddleocr_vl_auth_token="private-ocr")
        self.manager.apply_category_options_to_queue("default")
        self.assertEqual(self.manager.load_job_secrets("default", "00000001"), {
            "vlmAuthToken": "private-current-vlm", "paddleocrVlAuthToken": "private-ocr"})
        self.assertEqual(self.manager.load_job_secrets("default", "00000002"), {"vlmAuthToken": "private-current-vlm"})
        self.assertEqual(self.manager.saved_vlm_token("http://old.example/v1"), "private-old-vlm")
        self.assertEqual(self.manager.load_category_advanced_options("default"), defaults)

    def test_queued_job_and_category_pages_expose_settings_controls(self):
        self.job()
        job = web_pages.job_page("default", "12345678", self.manager.public_status("default", "12345678", False)).body.decode()
        self.assertIn('data-job-options="/api/v1/jobs/12345678/settings?category=default"', job)
        self.assertIn('data-use-category-options>Copy from Category Settings</button>', job)
        self.assertIn('<button type="submit">Copy to Category Settings</button>', job)
        self.assertNotIn("Save advanced options", job)
        self.assertNotIn(">Use category settings</button>", job)
        self.assertIn('<button type="button" data-use-category-options>Copy from Category Settings</button>', web_pages.advanced_options_fields(0, "http://localhost/v1", test_job_id="12345678"))
        category = web_pages.category_jobs_page("default", self.manager.public_category_jobs("default")).body.decode()
        self.assertIn("Apply saved settings to queued jobs", category)
        self.assertIn('data-queue-options-status role="status"', category)

    def test_use_category_settings_changes_only_one_job_and_keeps_notes_and_defaults(self):
        self.job("00000001", ocrPageWorkers=4)
        original = self.job("00000002", ocrPageWorkers=3)
        self.manager.set_initial_translation_notes("default", "00000001", "Keep these notes")
        notes = self.manager.translation_notes_path("default", "00000001").read_bytes()
        defaults = self.manager.load_category_advanced_options("default")
        self.manager.use_category_options_for_job("default", "00000001")
        status = self.manager.load_status("default", "00000001")
        for field in web_app.CONTROL_OPTION_FIELDS.values():
            self.assertEqual(status[field], defaults[field])
        self.assertEqual(status["status"], "queued")
        self.assertEqual(status["message"], "Waiting for chapter 1.")
        self.assertEqual(self.manager.load_status("default", "00000002"), original)
        self.assertEqual(self.manager.translation_notes_path("default", "00000001").read_bytes(), notes)
        self.assertEqual(self.manager.load_category_advanced_options("default"), defaults)

    def test_use_category_settings_is_allowed_only_for_editable_jobs(self):
        for index, state in enumerate(("queued", "complete", "failed", "cancelled", "paused", "running", "terminating")):
            with self.subTest(state=state):
                job_id = f"{index:08x}"
                original = self.job(job_id, state, ocrPageWorkers=4)
                if state in {"running", "terminating"}:
                    with self.assertRaises(HTTPException):
                        self.manager.use_category_options_for_job("default", job_id)
                    self.assertEqual(self.manager.load_status("default", job_id), original)
                else:
                    self.manager.use_category_options_for_job("default", job_id)
                    self.assertEqual(self.manager.load_status("default", job_id)["ocrPageWorkers"], 1)
