from __future__ import annotations

import asyncio
import io
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

import translate_cbz
import web_app
import web_pages
from tests import test_regressions


class IndependentExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager = test_regressions.WebAuthenticationTests().initialized_manager(self.root, ["default"])
        self.code, self.job_id = "default", "12345678"
        self.releases: list[threading.Event] = []
        self.make_job(self.job_id)
        self.addCleanup(self.finish)

    def make_job(self, job):
        self.manager.job_dir(self.code, job).mkdir(parents=True, exist_ok=True)
        self.manager.save_status(self.code, job, {"status": "complete", "phase": "Complete", "jobName": job,
            "comicInfoTitle": "Original title", "finishedAt": "2026-10-18T00:00:00+00:00", "elapsedSeconds": 42})
        with zipfile.ZipFile(self.manager.input_path(self.code, job), "w") as archive:
            archive.writestr("ComicInfo.xml", "<ComicInfo><Title>Original title</Title></ComicInfo>")
            archive.writestr("notes.txt", "Preserved metadata")
        for index in range(2):
            for kind in ("original", "final"):
                directory = self.manager.output_dir(self.code, job) / "pages" / kind
                directory.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (80, 120), "white").save(directory / f"{index:04d}.png")

    def finish(self):
        for event in self.releases:
            event.set()
        self.join()

    def join(self):
        with self.manager._lock:
            threads = list(self.manager._export_threads)
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive(), "Export thread did not finish")

    def blocked_packager(self, expected=1, real=False):
        release, entered = threading.Event(), threading.Event()
        self.releases.append(release)
        lock, calls = threading.Lock(), []
        original = translate_cbz.print_packaged_cbz
        def package(pages, output, source, config, **options):
            with lock:
                calls.append((pages, output, source, options))
                if len(calls) >= expected:
                    entered.set()
            if not release.wait(timeout=10):
                raise RuntimeError("Synthetic export wait timed out")
            if real:
                original(pages, output, source, config, **options)
            else:
                with zipfile.ZipFile(output / web_app.TRANSLATED_CBZ_FILENAMES[options["package_variant"]], "w") as archive:
                    archive.writestr("test.txt", "Independent export")
        return package, entered, release, calls

    def state(self, variant="png", job=None):
        return self.manager.load_export_state(self.code, job or self.job_id)["variants"][variant]

    def test_category_rename_waits_for_startup_and_queue_claim_is_atomic(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        key = (self.code, self.job_id)
        state = self.manager.load_status(*key)
        state["status"] = "queued"
        self.manager.save_status(*key, state)
        self.manager._queued_jobs.add(key)
        self.manager._queue.put(key)
        self.manager._queue.put(None)
        queued_at_claim = []
        claimed_at_failure_save = []
        run_job = self.manager.run_job
        save_status = self.manager.save_status

        def claim(*args):
            queued_at_claim.append(key in self.manager._queued_jobs)
            run_job(*args)

        def title(*args):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release job startup")

        def save(code, job_id, status):
            if status.get("phase") == "Failed":
                claimed_at_failure_save.append(key in self.manager._starting_jobs)
            save_status(code, job_id, status)

        with mock.patch.object(self.manager, "run_job", side_effect=claim), \
             mock.patch.object(self.manager, "maybe_auto_translate_comicinfo_title", side_effect=title), \
             mock.patch.object(self.manager, "build_command", return_value=[]), \
             mock.patch.object(self.manager, "save_status", side_effect=save), \
             mock.patch.object(self.manager, "run_pipeline_process", side_effect=RuntimeError("Synthetic startup failure")):
            worker = threading.Thread(target=self.manager.worker_loop, daemon=True)
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertIn(key, self.manager._starting_jobs)
                self.assertNotIn(key, self.manager._queued_jobs)
                with self.assertRaises(HTTPException) as caught:
                    self.manager.rename_category(self.code, "renamed")
                self.assertEqual(caught.exception.status_code, 409)
                self.assertFalse(self.manager.category_dir("renamed").exists())
            finally:
                release.set()
                worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(queued_at_claim, [True])
        self.assertEqual(claimed_at_failure_save, [True])
        self.assertNotIn(key, self.manager._starting_jobs)
        self.assertEqual(self.manager.load_status(*key)["status"], "failed")
        self.manager.rename_category(self.code, "renamed")
        self.assertFalse(self.manager.category_dir(self.code).exists())
        self.assertEqual(self.manager.load_status("renamed", self.job_id)["status"], "failed")

    def test_category_rename_waits_for_independent_job_export(self):
        package, entered, release, _ = self.blocked_packager()
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.assertTrue(entered.wait(5))
            with self.assertRaises(HTTPException) as caught:
                self.manager.rename_category(self.code, "renamed")
            self.assertEqual(caught.exception.status_code, 409)
            release.set()
            self.join()
        self.assertEqual(self.state()["status"], "complete")
        self.manager.rename_category(self.code, "renamed")
        self.assertTrue(self.manager.translated_cbz_variant_path("renamed", self.job_id, "png").is_file())
        self.assertEqual(self.manager.load_export_state("renamed", self.job_id)["variants"]["png"]["status"], "complete")

    def test_export_starts_while_queue_paused_and_another_pipeline_is_running(self):
        package, entered, release, _ = self.blocked_packager(real=True)
        self.manager._paused = True
        self.manager._active_processes[(self.code, "otherjob")] = object()
        before = self.manager.load_status(self.code, self.job_id)
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package), \
             mock.patch.object(self.manager, "enqueue", side_effect=AssertionError("Must not queue")), \
             mock.patch.object(self.manager, "run_pipeline_process", side_effect=AssertionError("Must not run pipeline")):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.assertTrue(entered.wait(timeout=5))
            public = self.manager.public_status(self.code, self.job_id, include_log=False)
            self.assertEqual(public["status"], "complete")
            self.assertTrue(public["exportsRunning"])
            self.assertTrue(public["canEdit"])
            self.assertEqual(self.manager._queue.qsize(), 0)
            release.set(); self.join()
        self.assertEqual(self.state()["status"], "complete")
        after = self.manager.load_status(self.code, self.job_id)
        for key in ("status", "phase", "finishedAt", "elapsedSeconds"):
            self.assertEqual(after[key], before[key])
        with zipfile.ZipFile(self.manager.translated_cbz_variant_path(self.code, self.job_id, "png")) as archive:
            self.assertEqual(archive.read("notes.txt"), b"Preserved metadata")
            self.assertIn("0001.png", archive.namelist())

    def test_jobs_and_variants_generate_at_the_same_time_and_duplicates_are_rejected(self):
        other = "87654321"; self.make_job(other)
        package, entered, release, calls = self.blocked_packager(expected=3)
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.manager.generate_download(self.code, self.job_id, "webp")
            self.manager.generate_download(self.code, other, "png")
            self.assertTrue(entered.wait(timeout=5), "All three exports must start without a queue")
            with self.assertRaises(HTTPException) as error:
                self.manager.generate_download(self.code, self.job_id, "png")
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(len(calls), 3)
            release.set(); self.join()
        for job, variant in ((self.job_id, "png"), (self.job_id, "webp"), (other, "png")):
            self.assertTrue(self.manager.translated_cbz_variant_path(self.code, job, variant).is_file())

    def test_failure_does_not_fail_job_or_publish_partial_archive(self):
        def fail(pages, output, source, config, **options):
            (output / "translated.cbz").write_bytes(b"Partial zip")
            raise RuntimeError("Synthetic encoding failure")
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=fail):
            self.manager.generate_download(self.code, self.job_id, "png"); self.join()
        self.assertEqual(self.state()["status"], "failed")
        self.assertIn("Synthetic encoding failure", self.state()["message"])
        self.assertEqual(self.manager.load_status(self.code, self.job_id)["status"], "complete")
        self.assertFalse(self.manager.translated_cbz_variant_path(self.code, self.job_id, "png").exists())

    def test_rerun_invalidates_export_and_late_completion_cannot_change_job_state(self):
        package, entered, release, _ = self.blocked_packager()
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.assertTrue(entered.wait(timeout=5))
            with mock.patch.object(self.manager, "enqueue") as enqueue:
                self.manager.rerun_completed_job_pages(self.code, self.job_id, [0], resume_from="render")
                enqueue.assert_called_once()
            release.set(); self.join()
        self.assertEqual(self.state()["status"], "cancelled")
        self.assertFalse(self.manager.translated_cbz_variant_path(self.code, self.job_id, "png").exists())
        self.assertEqual(self.manager.load_status(self.code, self.job_id)["status"], "queued")

    def test_export_reads_immutable_snapshot_not_live_rendered_pages(self):
        package, entered, release, _ = self.blocked_packager(real=True)
        path = self.manager.final_page_files(self.code, self.job_id)[0]
        original = path.read_bytes()
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.assertTrue(entered.wait(timeout=5))
            Image.new("RGB", (80, 120), "black").save(path)
            release.set(); self.join()
        with zipfile.ZipFile(self.manager.translated_cbz_variant_path(self.code, self.job_id, "png")) as archive:
            self.assertEqual(archive.read(path.name), original)

    def test_deleting_job_during_export_does_not_recreate_it(self):
        package, entered, release, _ = self.blocked_packager()
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.assertTrue(entered.wait(timeout=5))
            self.manager.delete_job(self.code, self.job_id)
            release.set(); self.join()
        self.assertFalse(self.manager.job_dir(self.code, self.job_id).exists())

    def test_title_failure_falls_back_with_export_warning(self):
        with mock.patch.object(self.manager, "request_vlm_title", side_effect=RuntimeError("401 Unauthorized")):
            self.manager.generate_download(self.code, self.job_id, "png", title_mode="translated"); self.join()
        self.assertEqual(self.state()["status"], "complete")
        self.assertIn("401", self.state()["warning"])
        with zipfile.ZipFile(self.manager.translated_cbz_variant_path(self.code, self.job_id, "png")) as archive:
            self.assertIn(b"Original title", archive.read("ComicInfo.xml"))

    def test_shutdown_and_recovery_do_not_enqueue_or_fail_the_translation(self):
        package, entered, release, _ = self.blocked_packager()
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package):
            self.manager.generate_download(self.code, self.job_id, "png")
            self.assertTrue(entered.wait(timeout=5))
            self.manager._stopping = True
            release.set(); self.join()
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state(); restarted.recover_interrupted_exports()
        self.assertEqual(restarted.load_export_state(self.code, self.job_id)["variants"]["png"]["status"], "failed")
        self.assertEqual(restarted.load_status(self.code, self.job_id)["status"], "complete")
        self.assertEqual(restarted._queue.qsize(), 0)

    def test_ui_and_api_show_export_state_while_job_stays_complete(self):
        package, entered, release, _ = self.blocked_packager()
        app = web_app.create_app(self.root / "web.json"); app.state.manager = self.manager
        token = self.manager.create_control_token("exports")["token"]
        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", headers={"Authorization": "Bearer " + token}) as client:
                response = await client.post(f"/api/v1/jobs/{self.job_id}/export", json={"format": "png"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "complete")
                self.assertTrue(response.json()["exportsRunning"])
                self.assertTrue(entered.wait(timeout=5))
                markup = web_pages.download_links_html(self.code, self.job_id, response.json())
                self.assertIn("Generating PNG CBZ...", markup)
                self.assertIn("disabled", markup)
                self.assertEqual((await client.get(f"/api/v1/jobs/{self.job_id}/download/png")).status_code, 404)
                release.set(); self.join()
                self.assertEqual((await client.get(f"/api/v1/jobs/{self.job_id}/download/png")).status_code, 200)
        async def direct(function, *args, **kwargs):
            return function(*args)
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=package), \
             mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(asyncio.wait_for(exercise(), timeout=15))

    @unittest.skipUnless(shutil.which("magick"), "Requires runtime ImageMagick")
    def test_actual_webp_and_jxl_exports_need_no_vlm_or_pipeline(self):
        with mock.patch.object(self.manager, "request_vlm_title", side_effect=AssertionError("No VLM")), \
             mock.patch.object(self.manager, "run_pipeline_process", side_effect=AssertionError("No pipeline")):
            for variant, extension in (("webp", ".webp"), ("jxl", ".jxl")):
                self.manager.generate_download(self.code, self.job_id, variant); self.join()
                self.assertEqual(self.state(variant)["status"], "complete", self.state(variant))
                with zipfile.ZipFile(self.manager.translated_cbz_variant_path(self.code, self.job_id, variant)) as archive:
                    self.assertIn("0001" + extension, archive.namelist())


if __name__ == "__main__":
    unittest.main()
