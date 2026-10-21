from __future__ import annotations

import asyncio
import io
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

import translate_cbz
import web_app
from tests import test_regressions
from web_series import chapter_number


class SeriesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helper = test_regressions.WebAuthenticationTests()
        self.manager = self.helper.initialized_manager(self.root, ["default"])
        self.manager.initialize_series("default")
        self.manager.update_series("default", "Test series", "", "")

    def chapter(self, order: str, *, enqueue: bool = False, code: str = "default", volume: str = "") -> str:
        manager = self.manager
        job_id = manager.create_job_id(code)
        manager.job_dir(code, job_id).mkdir(parents=True)
        manager.submit_job(code, job_id, f"chapter-{order}.cbz", volume=volume,
                           chapter=order, enqueue_job=enqueue, translation_notes="Keep honorifics.")
        return job_id

    def status(self, job_id: str, state: str) -> dict:
        status = self.manager.load_status("default", job_id)
        assert status is not None
        status["status"] = state
        self.manager.save_status("default", job_id, status)
        return status

    def generated_notes(self, job_id: str, text: str) -> None:
        path = self.manager.generated_translation_notes_path("default", job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_series_and_order_validation(self) -> None:
        self.assertEqual(chapter_number("01.50"), "1.5")
        for value in ("", "-1", "NaN", "Infinity", "1000001", "chapter 1"):
            with self.subTest(value=value), self.assertRaises(HTTPException):
                chapter_number(value)
        self.chapter("1.5")
        with self.assertRaises(HTTPException) as caught:
            self.chapter("1.50")
        self.assertEqual(caught.exception.status_code, 409)
        with self.assertRaises(HTTPException):
            self.manager.validate_series_chapter("default", "invalid")
        self.assertEqual(chapter_number("0"), "0")
        self.chapter("1.5", volume="2")
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        self.assertEqual(restarted.get_series("default")["name"], "Test series")

    def test_chapter_numbers_limit_normalized_length_before_expansion(self) -> None:
        self.assertEqual(chapter_number("1e-18"), "0." + "0" * 17 + "1")
        self.assertEqual(chapter_number("1.0e-18"), chapter_number("1e-18"))
        self.assertEqual(chapter_number("1e6"), "1000000")
        for value in ("1e-19", "1e-1000000", "1e-99999999999999999", "123456789012345e-19", "0." + "1" * 19):
            for label in ("Chapter", "Volume"):
                with self.subTest(value=value, label=label):
                    with self.assertRaises(HTTPException) as caught:
                        chapter_number(value, label)
                    self.assertEqual(caught.exception.status_code, 400)

    def test_extreme_exponents_are_rejected_before_formatting(self) -> None:
        with mock.patch("web_series.format", side_effect=AssertionError("Must not expand this number"), create=True):
            for value in ("1e-1000000", "1e-99999999999999999"):
                with self.subTest(value=value):
                    with self.assertRaises(HTTPException) as caught:
                        chapter_number(value)
                    self.assertEqual(caught.exception.status_code, 400)

    def test_context_is_captured_when_started_and_kept_on_retry(self) -> None:
        job_id = self.chapter("1")
        self.assertFalse(self.manager.series_context_path("default", job_id).exists())
        self.manager.update_series("default", "Test series", "Use Aki.", "Aki is the captain.")
        status = self.manager.load_status("default", job_id)
        self.manager.prepare_series_context("default", job_id, status)
        snapshot = self.manager.series_context_path("default", job_id).read_text()
        notes_path = self.manager.translation_notes_path("default", job_id)
        notes = translate_cbz.load_translation_notes(notes_path)
        self.assertIn("Use Aki.", notes["job"])
        self.assertIn("Aki is the captain.", notes["job"])
        self.assertIn("Keep honorifics.", notes["job"])
        self.manager.update_series("default", "Test series", "Changed", "New notes")
        self.manager.prepare_series_context("default", job_id, self.manager.load_status("default", job_id))
        self.assertEqual(translate_cbz.load_translation_notes(notes_path), notes)
        self.assertEqual(self.manager.series_context_path("default", job_id).read_text(), snapshot)
        self.status(job_id, "failed")
        with self.assertRaises(HTTPException):
            self.manager.assign_job_chapter("default", job_id, "", "")

    def test_failed_or_review_chapter_blocks_only_its_series(self) -> None:
        first = self.chapter("1")
        second = self.chapter("2", enqueue=True)
        self.manager.create_category("other", is_series=True)
        unrelated = self.chapter("1", enqueue=True, code="other")
        self.assertNotIn(("default", second), self.manager._queued_jobs)
        self.assertIn(("other", unrelated), self.manager._queued_jobs)
        for state in ("failed", "paused", "cancelled", "running"):
            self.status(first, state)
            self.manager.enqueue_ready_series_jobs()
            self.assertNotIn(("default", second), self.manager._queued_jobs)
        self.status(first, "complete")
        self.manager.enqueue_ready_series_jobs()
        self.assertIn("notes", self.manager.load_status("default", second)["message"])
        self.generated_notes(first, "Aki is the captain.")
        self.assertTrue(self.manager.publish_series_notes("default", first))
        self.assertIn(("default", second), self.manager._queued_jobs)
        self.assertTrue(self.manager.load_status("default", second)["writeTranslationNotes"])

    def test_renumbering_preserves_other_chapters_notes_and_rejects_conflicts(self) -> None:
        first, second = self.chapter("1"), self.chapter("2")
        self.manager.prepare_series_context("default", first, self.manager.load_status("default", first))
        self.status(first, "complete")
        self.status(second, "complete")
        self.generated_notes(second, "Notes from chapter 2")
        self.manager.publish_series_notes("default", second)
        series = self.manager.get_series("default")
        self.manager.assign_job_chapter("default", first, "1.5")
        self.assertEqual(self.manager.get_series("default"), series)
        with self.assertRaises(HTTPException) as caught:
            self.manager.assign_job_chapter("default", first, "2")
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.manager.load_status("default", first)["chapter"], "1.5")
        for state in ("queued", "running", "terminating"):
            with self.subTest(state=state):
                self.status(first, state)
                with self.assertRaises(HTTPException) as caught:
                    self.manager.assign_job_chapter("default", first, "3")
                self.assertEqual(caught.exception.status_code, 409)
        self.status(first, "complete")
        for claimed in (self.manager._starting_jobs, self.manager._active_processes):
            with self.subTest(claim=type(claimed).__name__):
                if isinstance(claimed, set):
                    claimed.add(("default", first))
                else:
                    claimed[("default", first)] = object()
                try:
                    with self.assertRaises(HTTPException) as caught:
                        self.manager.assign_job_chapter("default", first, "3")
                    self.assertEqual(caught.exception.status_code, 409)
                finally:
                    claimed.clear()

    def test_publication_replaces_notes_and_older_reruns_need_explicit_publish(self) -> None:
        first, second = self.chapter("1"), self.chapter("2")
        self.status(first, "complete")
        self.generated_notes(first, "Aki is the captain.")
        self.manager.publish_series_notes("default", first)
        self.status(second, "complete")
        self.generated_notes(second, "Aki is the captain. Ren is the healer.")
        self.manager.publish_series_notes("default", second)
        self.generated_notes(first, "Earlier rerun notes")
        self.assertFalse(self.manager.publish_series_notes("default", first))
        series = self.manager.get_series("default")
        self.assertEqual(series["notes"], "Aki is the captain. Ren is the healer.")
        self.assertEqual(series["notesThrough"]["chapter"], "2")
        self.assertTrue(self.manager.publish_series_notes("default", first, explicit=True))
        self.assertEqual(self.manager.get_series("default")["notes"], "Earlier rerun notes")

    def test_chapter_without_translated_text_keeps_existing_series_notes(self) -> None:
        self.manager.update_series("default", "Test series", "", "Captain Aki")
        job_id = self.chapter("1")
        self.status(job_id, "complete")
        self.generated_notes(job_id, "No translated records were available.")
        self.assertTrue(self.manager.publish_series_notes("default", job_id))
        series = self.manager.get_series("default")
        self.assertEqual(series["notes"], "Captain Aki")
        self.assertEqual(series["notesThrough"]["chapter"], "1")

    def test_restart_reconstructs_waiting_dependencies(self) -> None:
        first, second = self.chapter("1"), self.chapter("2")
        self.status(first, "failed")
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        restarted.requeue_interrupted_jobs()
        self.assertFalse(restarted._queued_jobs)
        self.assertIn("Ch. 1", restarted.load_status("default", second)["message"])
        restarted.save_status("default", first, {**restarted.load_status("default", first), "status": "complete"})
        self.generated_notes(first, "Saved notes")
        restarted.publish_series_notes("default", first)
        self.assertIn(("default", second), restarted._queued_jobs)

    def test_restart_finishes_interrupted_publication_without_reversing_explicit_notes(self) -> None:
        first, second = self.chapter("1"), self.chapter("2")
        self.status(first, "complete")
        self.generated_notes(first, "Chapter one notes")
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        restarted.requeue_interrupted_jobs()
        self.assertEqual(restarted.get_series("default")["notes"], "Chapter one notes")
        self.assertIn(("default", second), restarted._queued_jobs)
        self.status(second, "complete")
        self.generated_notes(second, "Chapter two notes")
        self.manager.publish_series_notes("default", second)
        self.manager.publish_series_notes("default", first, explicit=True)
        self.manager.recover_series_notes()
        self.assertEqual(self.manager.get_series("default")["notes"], "Chapter one notes")

    def test_manual_notes_changed_during_run_are_not_overwritten(self) -> None:
        first, second = self.chapter("1"), self.chapter("2")
        status = self.manager.load_status("default", first)
        self.manager.prepare_series_context("default", first, status)
        self.manager.update_series("default", "Test series", "", "Manual correction")
        self.status(first, "complete")
        self.generated_notes(first, "Older generated notes")
        self.assertFalse(self.manager.publish_series_notes("default", first))
        self.assertEqual(self.manager.get_series("default")["notes"], "Manual correction")
        self.assertIn("changed during", self.manager.load_status("default", first)["seriesNotesWarning"])
        self.assertNotIn(("default", second), self.manager._queued_jobs)
        self.manager.publish_series_notes("default", first, explicit=True, use_current=True)
        self.assertEqual(self.manager.get_series("default")["notes"], "Manual correction")
        self.assertIn(("default", second), self.manager._queued_jobs)

    def test_worker_runs_chapters_in_order_and_publishes_before_next_context(self) -> None:
        first, second = self.chapter("1"), self.chapter("2")
        seen = []
        finished = threading.Event()
        original_publish = self.manager.publish_series_notes

        def run(code, job_id, command, description):
            notes = translate_cbz.load_translation_notes(self.manager.translation_notes_path(code, job_id))
            seen.append((job_id, notes["job"]))
            self.generated_notes(job_id, "Captain Aki" if job_id == first else "Captain Aki and healer Ren")
            final = self.manager.output_dir(code, job_id) / "pages" / "final"
            final.mkdir(parents=True)
            Image.new("RGB", (5, 5)).save(final / "page_0000.png")
            return 0

        def publish(code, job_id, **kwargs):
            result = original_publish(code, job_id, **kwargs)
            if job_id == second:
                finished.set()
            return result

        with (
            mock.patch.object(self.manager, "run_pipeline_process", side_effect=run),
            mock.patch.object(self.manager, "initialize_editor_v2"),
            mock.patch.object(self.manager, "publish_series_notes", side_effect=publish),
        ):
            # Reverse requests still respect the chapter dependency.
            self.manager.enqueue("default", second)
            self.manager.enqueue("default", first)
            self.manager._worker = threading.Thread(target=self.manager.worker_loop, daemon=True)
            self.manager._worker.start()
            try:
                self.assertTrue(finished.wait(10), "Both chapters must complete")
            finally:
                self.manager.stop()
        self.assertEqual([item[0] for item in seen], [first, second])
        self.assertNotIn("Captain Aki", seen[0][1])
        self.assertIn("Captain Aki", seen[1][1])
        self.assertEqual(self.manager.get_series("default")["notes"], "Captain Aki and healer Ren")

    @staticmethod
    def archive() -> bytes:
        image = io.BytesIO()
        Image.new("RGB", (5, 5)).save(image, format="PNG")
        result = io.BytesIO()
        with zipfile.ZipFile(result, "w") as archive:
            archive.writestr("page.png", image.getvalue())
        return result.getvalue()

    def test_http_can_renumber_a_completed_chapter_with_saved_context(self) -> None:
        job = self.chapter("1")
        self.manager.prepare_series_context("default", job, self.manager.load_status("default", job))
        self.status(job, "complete")
        snapshot = self.manager.series_context_path("default", job).read_bytes()
        app = web_app.create_app(self.root / "web_config.json")
        app.state.manager = self.manager
        session = self.manager.authenticate_admin(self.helper.ADMIN_PASSWORD)

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, session)
                response = await client.post(f"/job/default/{job}/chapter", data={"chapter": "1.5", "volume": "2"})
                self.assertEqual(response.status_code, 303, response.text)
                status = self.manager.load_status("default", job)
                self.assertEqual((status["status"], status["chapter"], status["volume"]), ("complete", "1.5", "2"))
                self.assertEqual(self.manager.series_context_path("default", job).read_bytes(), snapshot)
                response = await client.post(f"/job/default/{job}/publish-series-notes")
                self.assertEqual(response.status_code, 404)

        asyncio.run(asyncio.wait_for(exercise(), timeout=15))

    def test_http_batch_orders_files_and_registers_all_before_running(self) -> None:
        app = web_app.create_app(self.root / "web_config.json")
        app.state.manager = self.manager
        session = self.manager.authenticate_admin(self.helper.ADMIN_PASSWORD)
        url = "/category/default/series"

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                # Series management has the same login requirement as jobs.
                self.assertNotEqual((await client.get("/category/default")).status_code, 200)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, session)
                response = await client.get("/category/default")
                self.assertEqual(response.status_code, 200)
                self.assertIn("Series notes", response.text)
                response = await client.post(url, data={
                    "name": "Updated series", "instructions": "Use Aki.", "notes": "Initial terms",
                })
                self.assertEqual(response.status_code, 303)
                response = await client.post(url + "/upload", data={
                    "first_chapter": "1", "vlm_model": "test-vlm-model",
                }, files=[
                    ("chapters", ("chapter10.cbz", self.archive(), "application/zip")),
                    ("chapters", ("chapter2.cbz", self.archive(), "application/zip")),
                ])
                self.assertEqual(response.status_code, 303, response.text)
                chapters = self.manager.series_chapters("default")
                self.assertEqual([item["inputFilename"] for item in chapters], ["chapter2.cbz", "chapter10.cbz"])
                self.assertEqual([item["chapter"] for item in chapters], ["1", "2"])
                self.assertEqual(self.manager._queued_jobs, {("default", chapters[0]["jobId"])})
                # A bad batch must leave no partially registered chapters.
                response = await client.post(url + "/upload", data={
                    "first_chapter": "3", "vlm_model": "test-vlm-model",
                }, files=[
                    ("chapters", ("chapter3.cbz", self.archive(), "application/zip")),
                    ("chapters", ("chapter4.cbz", b"invalid", "application/zip")),
                ])
                self.assertEqual(response.status_code, 400)
                self.assertEqual(len(self.manager.series_chapters("default")), 2)
                self.assertEqual(len(self.manager.iter_job_ids("default")), 2)

        async def direct_run_in_threadpool(function, *args):
            return function(*args)

        with mock.patch.object(web_app, "run_in_threadpool", new=direct_run_in_threadpool), mock.patch("anyio.to_thread.run_sync", new=direct_run_in_threadpool):
            asyncio.run(asyncio.wait_for(exercise(), timeout=15))


if __name__ == "__main__":
    unittest.main()
