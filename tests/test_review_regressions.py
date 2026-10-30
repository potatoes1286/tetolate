"""Regression checks for the final merge review. All jobs and endpoints are fixtures."""
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import copy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

from fastapi import HTTPException
from PIL import Image

import overlay_text
import paddle_ocr_image
import translate_cbz
import web_app
from tests import test_editor_reset, test_editor_visibility, test_ocr_additions, test_series


class ReviewRegressionTests(unittest.TestCase):
    def fixture(self, kind):
        fixture = kind()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_cli_ocr_key_is_owned_by_its_configured_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "vlm_config.json"
            config.write_text(json.dumps({"base_url":"http://vlm.example/v1", "model":"vision-model", "ocr":{
                "engine":"paddleocr_vl", "paddleocr_vl_server_url":"http://ocr.example/v1", "paddleocr_vl_api_key":"saved-key"}}))
            for endpoint, explicit, environment, expected in (
                (None, None, "", "saved-key"), ("http://ocr.example/v1/", None, "", "saved-key"),
                ("http://other.example/v1", None, "", None), ("http://other.example/v1", "new-key", "", "new-key"),
                ("http://other.example/v1", None, "process-key", "process-key"),
            ):
                argv = ["translate_cbz.py", str(root/"input.cbz"), str(root/"output"), "--config", str(config)]
                if endpoint is not None:
                    argv += ["--paddleocr-vl-server-url", endpoint]
                if explicit is not None:
                    argv += ["--paddleocr-vl-api-key", explicit]
                with self.subTest(endpoint=endpoint, explicit=explicit, environment=environment), \
                     mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {"TETOLATE_PADDLEOCR_VL_API_KEY":environment}), \
                     mock.patch.object(translate_cbz, "run_full_pipeline") as run, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(translate_cbz.main(), 0)
                self.assertEqual(run.call_args.args[1].ocr.paddleocr_vl_api_key, expected)

    def test_editor_ocr_never_uses_default_key_for_another_endpoint(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        config = translate_cbz.load_ocr_config({"paddleocr_vl_server_url":"http://ocr.example/v1", "paddleocr_vl_api_key":"saved-key"},
                                              translate_cbz.language_config_from_codes("jp", "en"))
        for endpoint, expected in (("http://ocr.example/v1/", "saved-key"), ("http://other.example/v1", None)):
            with self.subTest(endpoint=endpoint):
                resolved = f.manager.ocr_config_for_status(config, {"paddleocrVlServerUrl":endpoint})
                self.assertEqual(resolved.paddleocr_vl_api_key, expected)

    def test_cli_vlm_key_is_owned_by_its_configured_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "vlm_config.json"
            config.write_text(json.dumps({"base_url":"http://vlm.example/v1", "model":"vision-model", "api_key":"saved-key"}))
            for endpoint, explicit, expected in (("http://vlm.example/v1/", None, "saved-key"),
                                                  ("http://other.example/v1", None, "not-needed"),
                                                  ("http://other.example/v1", "new-key", "new-key")):
                argv = ["translate_cbz.py", str(root/"input.cbz"), str(root/"output"), "--config", str(config), "--vlm-base-url", endpoint]
                if explicit is not None:
                    argv += ["--vlm-api-key", explicit]
                with self.subTest(endpoint=endpoint, explicit=explicit), mock.patch.object(sys, "argv", argv), \
                     mock.patch.dict(os.environ, {"OPENAI_API_KEY":"unrelated-sdk-key"}, clear=True), \
                     mock.patch.object(translate_cbz, "run_full_pipeline") as run, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(translate_cbz.main(), 0)
                self.assertEqual(run.call_args.args[1].vlm.api_key, expected)

    def test_ocr_config_does_not_pair_server_environment_key_with_another_url(self):
        with mock.patch.object(paddle_ocr_image, "DEFAULT_PADDLEOCR_VL_SERVER_URL", "http://default.example/v1"), \
             mock.patch.object(paddle_ocr_image, "DEFAULT_PADDLEOCR_VL_API_KEY", "server-key"):
            for endpoint, expected in (("http://default.example/v1/", "server-key"), ("http://other.example/v1", None)):
                config = translate_cbz.load_ocr_config({"paddleocr_vl_server_url":endpoint}, translate_cbz.language_config_from_codes("jp", "en"))
                self.assertEqual(config.paddleocr_vl_api_key, expected)

    def test_worker_does_not_inherit_keys_for_other_endpoints(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m, code, job = f.manager, f.code, f.job_id
        status = m.load_status(code, job)
        status.update(vlmBaseUrl="http://other-vlm.example/v1", paddleocrVlServerUrl="http://other-ocr.example/v1")
        m.save_status(code, job, status)
        for explicit in (False, True):
            if explicit:
                m.update_job_auth_tokens(code, job, vlm_auth_token="job-vlm-key", paddleocr_vl_auth_token="job-ocr-key")
            with mock.patch.dict(os.environ, {"TETOLATE_PADDLEOCR_VL_API_KEY":"default-ocr-key", "TETOLATE_VLM_API_KEY":"default-vlm-key"}), \
                 mock.patch.object(web_app.subprocess, "Popen", side_effect=OSError("Synthetic spawn stop")) as spawn:
                m.run_pipeline_process(code, job, ["synthetic-command"], "synthetic")
            env = spawn.call_args.kwargs["env"]
            self.assertEqual(env.get("TETOLATE_PADDLEOCR_VL_API_KEY"), "job-ocr-key" if explicit else None)
            self.assertEqual(env.get("TETOLATE_VLM_API_KEY"), "job-vlm-key" if explicit else None)

    def test_vlm_probe_only_uses_default_key_for_its_own_endpoint(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m = f.manager
        pipeline = translate_cbz.load_config(m.config.pipeline_config, None)
        pipeline = replace(pipeline, vlm=replace(pipeline.vlm, base_url="http://default.example/v1", api_key="default-key"))
        for endpoint, supplied, expected in (("http://default.example/v1/", None, "default-key"),
                                              ("http://other.example/v1", None, "not-needed"),
                                              ("http://other.example/v1", "new-key", "new-key")):
            with mock.patch.object(translate_cbz, "load_config", return_value=pipeline), \
                 mock.patch.object(web_app.vlm_client, "list_vlm_model_ids", return_value=["model"]) as probe:
                m.probe_vlm_models(endpoint, supplied)
            self.assertEqual(probe.call_args.args[0].api_key, expected)

    def test_crop_ocr_reads_endpoint_and_key_in_one_locked_snapshot(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m, code, job = f.manager, f.code, f.job_id
        status = m.load_status(code, job)
        status.update(ocrEngine="paddleocr_vl", paddleocrVlServerUrl="http://old.example/v1")
        m.save_status(code, job, status)
        m.update_job_auth_tokens(code, job, vlm_auth_token=None, paddleocr_vl_auth_token="old-key")
        original = m.ocr_config_for_status
        threads, attempted, changed = [], threading.Event(), threading.Event()
        def update():
            attempted.set()
            with m._lock:
                status["paddleocrVlServerUrl"] = "http://new.example/v1"
                m.save_status(code, job, status)
                m.update_job_auth_tokens(code, job, vlm_auth_token=None, paddleocr_vl_auth_token="new-key")
                changed.set()
        def resolve(*args):
            config = original(*args)
            thread = threading.Thread(target=update)
            threads.append(thread)
            thread.start()
            self.assertTrue(attempted.wait(2))
            self.assertFalse(changed.wait(0.05))
            return config
        try:
            with mock.patch.object(m, "ocr_config_for_status", side_effect=resolve), \
                 mock.patch.object(paddle_ocr_image, "create_paddleocr_vl", side_effect=RuntimeError("Synthetic capture")) as create:
                with self.assertRaisesRegex(RuntimeError, "Synthetic capture"):
                    m.run_crop_ocr(code, job, 0, [0,0,20,20])
            self.assertEqual(create.call_args.args[1], "http://old.example/v1")
            self.assertEqual(create.call_args.kwargs["api_key"], "old-key")
        finally:
            for thread in threads:
                thread.join(3)
                self.assertFalse(thread.is_alive())

    def test_failed_vlm_probe_preserves_shared_and_job_keys(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m, code, job = f.manager, f.code, f.job_id
        endpoint = m.status_vlm_base_url(m.load_status(code, job))
        m.update_job_auth_tokens(code, job, vlm_auth_token="working-key", paddleocr_vl_auth_token=None)
        before = m.job_secrets_path(code, job).read_bytes()
        async def direct(function, *args):
            return function(*args)
        with mock.patch.object(web_app, "run_in_threadpool", new=direct), \
             mock.patch.object(m, "probe_vlm_models", side_effect=RuntimeError("HTTP 401")):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(web_app.probe_vlm_connection(m, endpoint, "invalid-key", code, job))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(m.job_secrets_path(code, job).read_bytes(), before)
        self.assertEqual(m.saved_vlm_token(endpoint), "working-key")

    def test_title_request_reads_endpoint_and_key_in_one_locked_snapshot(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m, code, job = f.manager, f.code, f.job_id
        status = m.load_status(code, job)
        status["vlmBaseUrl"] = "http://old.example/v1"
        m.save_status(code, job, status)
        m.update_job_auth_tokens(code, job, vlm_auth_token="old-key", paddleocr_vl_auth_token=None)
        original = m.status_vlm_base_url
        threads, attempted, changed = [], threading.Event(), threading.Event()
        def update():
            attempted.set()
            with m._lock:
                status["vlmBaseUrl"] = "http://new.example/v1"
                m.save_status(code, job, status)
                m.update_job_auth_tokens(code, job, vlm_auth_token="new-key", paddleocr_vl_auth_token=None)
                changed.set()
        def resolve(*args):
            endpoint = original(*args)
            thread = threading.Thread(target=update)
            threads.append(thread)
            thread.start()
            self.assertTrue(attempted.wait(2))
            self.assertFalse(changed.wait(0.05))
            return endpoint
        try:
            with mock.patch.object(m, "status_vlm_base_url", side_effect=resolve), \
                 mock.patch.object(web_app.vlm_client, "call_vlm", side_effect=RuntimeError("Synthetic capture")) as call:
                with self.assertRaisesRegex(RuntimeError, "Synthetic capture"):
                    m.request_vlm_title(code, job, "Synthetic title")
            config = call.call_args.args[0]
            self.assertEqual((config.base_url, config.api_key), ("http://old.example/v1", "old-key"))
        finally:
            for thread in threads:
                thread.join(3)
                self.assertFalse(thread.is_alive())

    def test_successful_vlm_probe_does_not_write_key_to_a_changed_job_endpoint(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m, code, job = f.manager, f.code, f.job_id
        endpoint = m.status_vlm_base_url(m.load_status(code, job))
        async def direct(function, *args):
            return function(*args)
        def probe(*_args):
            status = m.load_status(code, job)
            status["vlmBaseUrl"] = "http://other.example/v1"
            m.save_status(code, job, status)
            return ["model"]
        with mock.patch.object(web_app, "run_in_threadpool", new=direct), mock.patch.object(m, "probe_vlm_models", side_effect=probe):
            asyncio.run(web_app.probe_vlm_connection(m, endpoint, "tested-key", code, job))
        self.assertEqual(m.saved_vlm_token(endpoint), "tested-key")
        self.assertIsNone(m.saved_vlm_token("http://other.example/v1"))
        self.assertNotEqual(m.load_job_secrets(code, job).get("vlmAuthToken"), "tested-key")

    def test_pending_source_text_correction_survives_the_erase_override(self):
        f = self.fixture(test_ocr_additions.OcrAdditionTests)
        f.old_orphans()
        payload = f.payload("text")
        source = f.missed(payload["recordsByArtifact"]["ocr_structured"])[0]
        source["text"] = "Text-tab correction"
        rid, number = source["recordId"], source["boxno"]
        f.save(payload)
        for value in ("Typesetting correction", "Another correction", ""):
            payload = f.payload("placement")
            for record in payload["recordsByArtifact"]["placements"]:
                record.pop("font", None)
            saved = f.manager.save_editor_v2_update(f.code, f.job_id, 0, "placement", {
                "baseRevision":payload["revision"], "recordsByArtifact":payload["recordsByArtifact"],
                "textEdits":[{"artifact":"ocr_structured", "recordId":rid, "boxno":number, "value":value}],
            })
            for stage in ("text", "placement"):
                current = next(record for record in f.payload(stage)["recordsByArtifact"]["ocr_structured"] if record["recordId"] == rid)
                self.assertEqual(current["text"], value)
            self.assertGreater(saved["revision"], payload["revision"])

    def add_raw(self, f, payload, number):
        payload["recordsByArtifact"]["ocr_raw"].append({"page":0, "boxno":number, "recordId":f"new-raw-{number}",
            "ocrSource":"manual", "text":"New source", "region":[10,70,30,90]})
        group_number = payload["nextBoxnos"]["ocr_merged"]
        payload["recordsByArtifact"]["ocr_merged"].append({"page":0, "boxno":group_number, "recordId":f"new-group-{number}",
            "ocrSource":"manual", "sourceBoxnos":[number], "text":"New source", "region":[10,70,30,90]})

    def test_deleted_box_numbers_stay_reserved_and_unrelated_additions_stay_distinct(self):
        f = self.fixture(test_editor_visibility.EditorVisibilityTests)
        f.delete("ocr_raw", 1)
        payload = f.payload("ocr")
        invalid = copy.deepcopy(payload)
        self.add_raw(f, invalid, 1)
        with self.assertRaises(HTTPException) as error:
            f.save(invalid)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(f.payload("ocr")["revision"], payload["revision"])
        number = payload["nextBoxnos"]["ocr_raw"]
        self.assertGreater(number, 1)
        self.add_raw(f, payload, number)
        f.save(payload)
        records = f.payload("placement")["recordsByArtifact"]
        self.assertFalse(any(1 in record.get("sourceBoxnos", []) for record in records["ocr_structured"]))
        self.assertTrue(any(record.get("text") == "New source" for record in records["ocr_structured"]))
        f.delete("ocr_raw", number)
        self.assertFalse(any(record.get("text") == "New source" for record in f.payload()["recordsByArtifact"]["ocr_structured"]))
        self.assertGreater(f.payload("ocr")["nextBoxnos"]["ocr_raw"], number)

    def test_current_preview_renders_unsaved_translation_without_publishing(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        payload = f.manager.editor_v2_payload(f.code, f.job_id, "placement", 0)
        records = payload["recordsByArtifact"]
        for record in records["placements"]:
            record.pop("font", None)
        records["translations"][0]["englishText"] = "Unsaved English"
        original = f.manager.original_page_path(f.code, f.job_id, 0)
        before = f.manager.editor_v2_manifest_path(f.code, f.job_id).read_bytes()
        with mock.patch.object(f.manager, "current_editor_clean_source", return_value=(original, False)), \
             mock.patch.object(overlay_text, "overlay_text", side_effect=lambda entries, source, target: shutil.copyfile(source, target)) as render:
            f.manager.render_editor_v2_preview(f.code, f.job_id, 0, records["placements"], records["ocr_structured"], [], records["translations"])
        self.assertEqual(render.call_args.args[0][0]["englishText"], "Unsaved English")
        self.assertEqual(f.manager.editor_v2_manifest_path(f.code, f.job_id).read_bytes(), before)
        self.assertEqual(f.manager.editor_v2_payload(f.code, f.job_id, "placement", 0)["recordsByArtifact"]["translations"][0]["englishText"], "Generated")

    def test_deleting_overlapping_manual_box_keeps_original_sources(self):
        f = self.fixture(test_editor_visibility.EditorVisibilityTests)
        payload = f.payload("ocr")
        number = payload["nextBoxnos"]["ocr_raw"]
        self.add_raw(f, payload, number)
        for artifact in ("ocr_raw", "ocr_merged"):
            payload["recordsByArtifact"][artifact][-1]["region"] = [10,10,30,40]
        f.save(payload)
        f.delete("ocr_raw", number)
        f.assert_boxnos("ocr_raw", [0,1], "ocr")
        f.assert_boxnos("ocr_structured", [0,1])

    def test_reset_does_not_reuse_numbers_referenced_by_saved_text(self):
        f = self.fixture(test_editor_visibility.EditorVisibilityTests)
        payload = f.payload("ocr")
        old_number = payload["nextBoxnos"]["ocr_raw"]
        self.add_raw(f, payload, old_number)
        f.save(payload)
        payload = f.payload("text")
        old = next(record for record in payload["recordsByArtifact"]["ocr_structured"] if old_number in record.get("sourceBoxnos", []))
        old["text"] = "Keep old source correction"
        translation = next(record for record in payload["recordsByArtifact"]["translations"] if record["boxno"] == old["boxno"])
        translation["englishText"] = "Keep old translation"
        f.save(payload)
        f.reset("ocr")
        records = f.payload()["recordsByArtifact"]
        self.assertFalse(any(old_number in record.get("sourceBoxnos", []) for record in records["ocr_structured"]))
        payload = f.payload("ocr")
        self.assertGreater(payload["nextBoxnos"]["ocr_raw"], old_number)
        self.add_raw(f, payload, payload["nextBoxnos"]["ocr_raw"])
        f.save(payload)
        records = f.payload()["recordsByArtifact"]
        self.assertFalse(any(record.get("text") == "Keep old source correction" for record in records["ocr_structured"]))
        self.assertFalse(any(record.get("englishText") == "Keep old translation" for record in records["translations"]))

    def test_claimed_job_actions_fail_before_changing_status_or_outputs(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        m, code, job = f.manager, f.code, f.job_id
        actions = [("failed", lambda: m.restart_failed_job(code, job)),
                   ("complete", lambda: m.rerun_completed_job_pages(code, job, [0])),
                   ("complete", lambda: m.queue_finish_editor_edits(code, job)),
                   ("complete", lambda: m.delete_job(code, job))]
        for status_name, action in actions:
            with self.subTest(status=status_name, action=action):
                status = m.load_status(code, job)
                status["status"] = status_name
                m.save_status(code, job, status)
                before = m.status_path(code, job).read_bytes()
                m._starting_jobs.add((code,job))
                try:
                    with mock.patch.object(m, "enqueue") as enqueue, mock.patch.object(m, "invalidate_translated_cbz_archives") as invalidate:
                        with self.assertRaises(HTTPException) as error:
                            action()
                        self.assertEqual(error.exception.status_code, 409)
                        enqueue.assert_not_called()
                        invalidate.assert_not_called()
                    self.assertEqual(m.status_path(code, job).read_bytes(), before)
                finally:
                    m._starting_jobs.discard((code,job))
        m.delete_job(code, job)
        self.assertFalse(m.job_dir(code, job).exists())

    def test_translation_only_save_reserves_its_missing_raw_source_after_reset(self):
        f = self.fixture(test_editor_visibility.EditorVisibilityTests)
        payload = f.payload("ocr")
        number = payload["nextBoxnos"]["ocr_raw"]
        self.add_raw(f, payload, number)
        f.save(payload)
        payload = f.payload("text")
        record = next(record for record in payload["recordsByArtifact"]["translations"] if number in record.get("sourceBoxnos", []))
        record["englishText"] = "Old translation only"
        f.save(payload)
        f.reset("ocr")
        records = f.payload()["recordsByArtifact"]
        self.assertFalse(any(record.get("englishText") == "Old translation only" for record in records["translations"]))
        payload = f.payload("ocr")
        self.assertGreater(payload["nextBoxnos"]["ocr_raw"], number)
        self.add_raw(f, payload, payload["nextBoxnos"]["ocr_raw"])
        f.save(payload)
        self.assertFalse(any(record.get("englishText") == "Old translation only" for record in f.payload()["recordsByArtifact"]["translations"]))

    def test_current_preview_rejects_invalid_translations_before_cleaning(self):
        f = self.fixture(test_editor_reset.EditorResetTests)
        for records in ([None], [{"boxno":False,"englishText":"bad"}], [{"boxno":0,"englishText":0}],
                        [{"boxno":0,"englishText":"x" * 100001}], [{"boxno":0,"englishText":"a"}] * 2):
            with self.subTest(records=str(records)[:50]), mock.patch.object(f.manager, "current_editor_clean_source") as clean:
                with self.assertRaises(HTTPException) as error:
                    f.manager.render_editor_v2_preview(f.code, f.job_id, 0, [], [], [], records)
                self.assertEqual(error.exception.status_code, 400)
                clean.assert_not_called()

    def run_blocked_job(self, f, job, blocked_method, entered, release, failures):
        def pipeline(*_args):
            directory = f.manager.final_pages_dir("default", job)
            directory.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (5,5)).save(directory / "000.png")
            f.generated_notes(job, "Synthetic shared notes")
            return 0
        original = getattr(f.manager, blocked_method)
        def block(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release blocked worker")
            if blocked_method == "publish_series_notes":
                return original(*args, **kwargs)
        def worker():
            try:
                f.manager.run_job("default", job)
            except Exception as error:
                failures.append(error)
        return pipeline, block, worker

    def test_series_update_does_not_requeue_a_starting_chapter(self):
        f = self.fixture(test_series.SeriesTests)
        job = f.chapter("1", enqueue=True)
        self.assertEqual(f.manager._queue.get_nowait(), ("default", job))
        entered, release, failures = threading.Event(), threading.Event(), []
        pipeline, block, worker = self.run_blocked_job(f, job, "maybe_auto_translate_comicinfo_title", entered, release, failures)
        with mock.patch.object(f.manager, "maybe_auto_translate_comicinfo_title", side_effect=block), \
             mock.patch.object(f.manager, "build_command", return_value=[]), mock.patch.object(f.manager, "initialize_editor_v2"), \
             mock.patch.object(f.manager, "run_pipeline_process", side_effect=pipeline) as run:
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                f.manager.update_series("default", "Test series", "Updated instructions", "")
                self.assertTrue(f.manager._queue.empty())
            finally:
                release.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(run.call_count, 1)

    def test_category_delete_waits_for_note_publication_and_leaves_no_orphan(self):
        f = self.fixture(test_series.SeriesTests)
        job = f.chapter("1")
        entered, release, failures = threading.Event(), threading.Event(), []
        pipeline, block, worker = self.run_blocked_job(f, job, "publish_series_notes", entered, release, failures)
        with mock.patch.object(f.manager, "maybe_auto_translate_comicinfo_title"), \
             mock.patch.object(f.manager, "publish_series_notes", side_effect=block), mock.patch.object(f.manager, "build_command", return_value=[]), \
             mock.patch.object(f.manager, "initialize_editor_v2"), mock.patch.object(f.manager, "run_pipeline_process", side_effect=pipeline):
            thread = threading.Thread(target=worker)
            thread.start()
            try:
                self.assertTrue(entered.wait(10))
                with self.assertRaises(HTTPException) as error:
                    f.manager.delete_category("default", "default")
                self.assertEqual(error.exception.status_code, 409)
            finally:
                release.set()
                thread.join(10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
        f.manager.delete_category("default", "default")
        self.assertNotIn("default", f.manager.categories())
        self.assertFalse(f.manager.category_dir("default").exists())


if __name__ == "__main__":
    unittest.main()
