"""Regression checks for the pre-push audit. All jobs and models are fixtures."""
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import shutil
import unittest
from unittest import mock

from PIL import Image

import editor_v2
import overlay_text
from tests import test_control_api, test_editor_actions, test_editor_visibility, test_exports


class AuditFixTests(unittest.TestCase):
    def fixture(self, cls):
        fixture = cls()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_noop_finish_keeps_job_and_collection_exports(self):
        f = self.fixture(test_editor_actions.EditorActionTests)
        paths = f.archives()
        collection = f.manager.category_export_path(f.code)
        collection.write_bytes(b"collection")
        with mock.patch.object(f.manager, "enqueue") as enqueue:
            self.assertEqual(f.manager.queue_finish_editor_edits(f.code, f.job_id), [])
        enqueue.assert_not_called()
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(collection.read_bytes(), b"collection")

    def test_finish_rolls_back_late_publication_failures(self):
        for failure in ("final", "cleaned", "mask", "data", "manifest", "history", "status"):
            with self.subTest(failure=failure):
                f = self.fixture(test_editor_actions.EditorActionTests)
                m, code, job = f.manager, f.code, f.job_id
                f.edit(0); f.edit(1)
                with mock.patch.object(m, "enqueue"):
                    m.queue_finish_editor_edits(code, job)
                output = m.output_dir(code, job)
                source = f.root / "cached.png"
                Image.new("RGB", (100, 120), "red").save(source)
                Image.new("L", (100, 120), 255).save(source.with_suffix(".mask.png"))
                # Check replacement of old files as well as removal of new files.
                cleaned = output / "pages/cleaned/0000.png"
                cleaned.parent.mkdir(parents=True, exist_ok=True)
                cleaned.write_bytes(b"old cleaned")
                roots = [output, m.editor_v2_manifest_path(code, job).parent]
                def files():
                    return {p: p.read_bytes() for root in roots for p in root.rglob("*")
                            if p.is_file() and ".finish-" not in str(p) and p.name != "exports.json"}
                before = files()
                manifest = m.load_editor_v2_manifest(code, job)
                target = {"final": m.final_pages_dir(code, job) / "0001.png",
                    "cleaned": output / "pages/cleaned/0001.png",
                    "mask": output / "debug/masks/0001.png",
                    "data": output / "data/translations/page_0001.json",
                    "manifest": m.editor_v2_manifest_path(code, job),
                    "history": m.editor_v2_history_dir(code, job) / f"revision_{manifest['revision'] + 1:08d}.json"}.get(failure)
                replace, copyfile, save_status = os.replace, shutil.copyfile, m.save_status
                failed = False
                def fail_once(path):
                    nonlocal failed
                    if not failed and Path(path) == target:
                        failed = True
                        raise OSError("Synthetic publication failure")
                def replacing(src, dst, *args, **kwargs):
                    fail_once(dst)
                    return replace(src, dst, *args, **kwargs)
                def copying(src, dst, *args, **kwargs):
                    fail_once(dst)
                    return copyfile(src, dst, *args, **kwargs)
                def saving(c, j, status):
                    nonlocal failed
                    if failure == "status" and status["status"] == "complete":
                        failed = True
                        raise OSError("Synthetic publication failure")
                    return save_status(c, j, status)
                def render(_records, _source, destination):
                    Image.new("RGB", (100, 120), "black").save(destination)
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(m, "current_editor_clean_source", return_value=(source, True)))
                    stack.enter_context(mock.patch.object(overlay_text, "overlay_text", side_effect=render))
                    stack.enter_context(mock.patch("os.replace", side_effect=replacing))
                    stack.enter_context(mock.patch("shutil.copyfile", side_effect=copying))
                    stack.enter_context(mock.patch.object(m, "save_status", side_effect=saving))
                    with self.assertRaisesRegex(OSError, "Synthetic publication failure"):
                        m.run_finish_editor_edits(code, job)
                self.assertTrue(failed)
                after = files()
                self.assertEqual(set(before), set(after))
                for path in before:
                    self.assertEqual(before[path], after[path], str(path))
                self.assertEqual(set(editor_v2.pending_changes(m.load_editor_v2_manifest(code, job))), {0, 1})
                self.assertEqual(m.load_status(code, job)["status"], "failed")

    def test_partial_raw_deletion_rebuilds_merged_source_fields_and_reset_restores_them(self):
        f = self.fixture(test_editor_visibility.EditorVisibilityTests)
        records = f.payload("ocr")["recordsByArtifact"]
        raw = records["ocr_raw"]
        group = copy.deepcopy(records["ocr_merged"][0])
        group.update(sourceBoxnos=[0, 1], sourceTexts=["first", "second"], text="firstsecond", region=[10, 10, 70, 40])
        f.baseline("ocr_merged", [group])
        f.delete("ocr_raw", 0)
        visible = f.payload("ocr")["recordsByArtifact"]["ocr_merged"][0]
        self.assertEqual(visible["sourceBoxnos"], [1])
        self.assertEqual(visible["sourceTexts"], [raw[1]["text"]])
        self.assertEqual(visible["text"], raw[1]["text"])
        self.assertEqual(visible["region"], raw[1]["region"])
        f.reset("ocr")
        restored = f.payload("ocr")["recordsByArtifact"]["ocr_merged"][0]
        for field in ("sourceBoxnos", "sourceTexts", "text", "region"):
            self.assertEqual(restored[field], group[field])

    def test_category_paths_reject_whitespace_without_creating_orphan_directories(self):
        f = self.fixture(test_control_api.ControlAPITests)
        async def exercise(client):
            for code in ("default%20", "%20default"):
                self.assertEqual((await client.patch(f"/api/v1/categories/{code}/settings", json={"lama_workers": 2})).status_code, 404)
                self.assertEqual((await client.get(f"/api/v1/categories/{code}/settings")).status_code, 404)
            self.assertEqual((await client.patch("/api/v1/categories/default/settings", json={"lama_workers": 2})).status_code, 200)
            self.assertEqual((await client.get("/api/v1/categories/default/settings")).json()["lamaWorkers"], 2)
        f.run_http(exercise)
        self.assertFalse((f.manager.config.jobs_dir / "default ").exists())
        self.assertFalse((f.manager.config.jobs_dir / " default").exists())

    def test_title_changes_cancel_inflight_and_completed_exports(self):
        for action in ("rename", "translated", "clear", "translate", "retry", "manual-retry"):
            with self.subTest(action=action):
                f = self.fixture(test_exports.IndependentExportTests)
                m, code, job = f.manager, f.code, f.job_id
                m.set_translated_title(code, job, "Old title")
                package, entered, release, _ = f.blocked_packager()
                with mock.patch("translate_cbz.print_packaged_cbz", side_effect=package), \
                     mock.patch.object(m, "request_vlm_title", return_value=("New title", False)), \
                     mock.patch("web_app.vlm_client.call_vlm", return_value=type("Result", (), {"output": "New title"})()), \
                     mock.patch.object(m, "required_status_vlm_model", return_value="test-model"):
                    m.generate_download(code, job, "png", title_mode="translated")
                    self.assertTrue(entered.wait(5))
                    collection = m.category_export_path(code)
                    collection.write_bytes(b"old collection")
                    if action == "rename": m.rename_job(code, job, "New name")
                    elif action in {"translated", "clear"}: m.set_translated_title(code, job, "" if action == "clear" else "New title")
                    elif action == "translate": m.translate_title(code, job)
                    else: m.retry_title_translation(code, job, "New title" if action == "manual-retry" else "")
                    self.assertFalse(collection.exists())
                    release.set(); f.join()
                self.assertEqual(f.state()["status"], "cancelled")
                self.assertFalse(m.translated_cbz_variant_path(code, job, "png").exists())
                paths = [m.translated_cbz_variant_path(code, job, variant) for variant in ("png", "webp", "jxl")]
                for path in paths: path.write_bytes(b"old export")
                m.set_translated_title(code, job, "A later title")
                self.assertTrue(all(not path.exists() for path in paths))

    def test_saving_unchanged_titles_keeps_exports(self):
        f = self.fixture(test_exports.IndependentExportTests)
        m, code, job = f.manager, f.code, f.job_id
        m.set_translated_title(code, job, "Same")
        path = m.translated_cbz_variant_path(code, job, "png")
        path.write_bytes(b"current export")
        m.set_translated_title(code, job, "Same")
        m.rename_job(code, job, job)
        self.assertEqual(path.read_bytes(), b"current export")
