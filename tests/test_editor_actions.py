from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import shutil
import types
import unittest
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageChops

import editor_v2
import overlay_text
import translate_cbz
import web_app
from tests import test_editor_reset


class EditorActionTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id
        self.root = fixture.root
        self.manager.input_path(self.code, self.job_id).write_bytes(b"input")
        final = self.manager.final_pages_dir(self.code, self.job_id)
        final.mkdir(parents=True)
        for page in (0, 1):
            Image.new("RGB", (100, 120), "white").save(final / f"{page:04d}.png")
        # Use shipped fonts rather than the reset fixture's synthetic name.
        for page in (0, 1):
            path = self.manager.editor_v2_generated_path(self.code, self.job_id, "placements", page)
            records = json.loads(path.read_text())
            for record in records:
                record["font"] = "ComicNeue-Bold.ttf"
            path.write_text(json.dumps(records))

    def edit(self, page=0):
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "text", page)
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Manual final text"
        return self.manager.save_editor_v2_update(self.code, self.job_id, page, "text", {
            "baseRevision": payload["revision"], "recordsByArtifact": payload["recordsByArtifact"],
        })

    def archives(self):
        paths = [self.manager.translated_cbz_variant_path(self.code, self.job_id, variant)
                 for variant in web_app.TRANSLATED_CBZ_FILENAMES]
        for path in paths:
            path.write_bytes(b"out of date")
        return paths

    def test_stage_actions_have_explicit_pipeline_boundaries_and_invalidate_archives(self):
        for stage, phase in (("ocr", "ocr_raw"), ("text", "ocr_structured"), ("placement", "placements")):
            self.manager.save_status(self.code, self.job_id, {"status": "complete"})
            paths = self.archives()
            with mock.patch.object(self.manager, "enqueue"):
                self.assertEqual(self.manager.regenerate_editor_v2_stage(self.code, self.job_id, stage, 0), phase)
            status = self.manager.load_status(self.code, self.job_id)
            rerun = status["pendingPageReruns"][0]
            self.assertEqual((rerun["page"], rerun["resumeFrom"], rerun["editorStageOnly"]), (0, phase, stage))
            self.assertIsInstance(rerun["editorStageBase"], list)
            command = self.manager.build_command(self.code, self.job_id, phase, 0, single_page=True, editor_stage_only=stage)
            self.assertEqual(command[command.index("--editor-stage-only") + 1], stage)
            self.assertTrue(all(not path.exists() for path in paths))

    def test_partial_stage_completion_keeps_downstream_work_pending(self):
        self.manager.mark_editor_stage_complete(self.code, self.job_id, 0, "ocr")
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.assertEqual(editor_v2.pending_changes(manifest), {0: {"ocr"}})
        self.assertTrue(editor_v2.stage_status(manifest, 0, "text")["stale"])
        self.manager.mark_editor_stage_complete(self.code, self.job_id, 0, "text")
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.assertEqual(editor_v2.pending_changes(manifest), {0: {"translation"}})
        self.assertFalse(editor_v2.stage_status(manifest, 0, "text")["stale"])
        self.assertTrue(editor_v2.stage_status(manifest, 0, "placement")["stale"])
        self.manager.mark_editor_stage_complete(self.code, self.job_id, 0, "placement")
        self.assertEqual(editor_v2.pending_changes(self.manager.load_editor_v2_manifest(self.code, self.job_id)), {})

    def test_stage_worker_preserves_scope_and_reset_keeps_unapplied_results_stale(self):
        with mock.patch.object(self.manager, "enqueue"):
            self.manager.regenerate_editor_v2_stage(self.code, self.job_id, "text", 0)
        status = self.manager.load_status(self.code, self.job_id)
        reruns = status["pendingPageReruns"]
        path = self.manager.editor_v2_generated_path(self.code, self.job_id, "translations", 0)
        records = json.loads(path.read_text()); records[0]["englishText"] = "Regenerated text"
        def process(*args):
            path.write_text(json.dumps(records)); return 0
        with mock.patch.object(self.manager, "run_pipeline_process", side_effect=process), \
             mock.patch.object(self.manager, "build_command", return_value=["synthetic"]) as command:
            self.manager.run_page_rerun_batch_job(self.code, self.job_id, status, reruns)
        self.assertEqual(command.call_args.kwargs["editor_stage_only"], "text")
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": manifest["revision"]})
        self.assertEqual(editor_v2.pending_changes(self.manager.load_editor_v2_manifest(self.code, self.job_id)), {0: {"translation"}})

    def test_finish_uses_saved_text_without_vlm_and_preserves_reset_data(self):
        self.edit()
        paths = self.archives()
        before = self.manager.load_editor_v2_baseline(self.code, self.job_id, "translations", 0)
        with mock.patch.object(self.manager, "enqueue"), mock.patch.object(self.manager, "required_status_vlm_model", side_effect=AssertionError("VLM forbidden")):
            self.assertEqual(self.manager.queue_finish_editor_edits(self.code, self.job_id), [0])
        self.assertTrue(all(not path.exists() for path in paths))
        entries = []
        def render(records, source, destination):
            entries.extend(copy.deepcopy(records)); Image.new("RGB", (100, 120), "black").save(destination)
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        with mock.patch.object(self.manager, "current_editor_clean_source", return_value=(original, False)), \
             mock.patch.object(overlay_text, "overlay_text", side_effect=render), \
             mock.patch.object(self.manager, "run_pipeline_process", side_effect=AssertionError("Pipeline forbidden")):
            self.manager.run_job(self.code, self.job_id)
        self.assertEqual(entries[0]["englishText"], "Manual final text")
        self.assertEqual(self.manager.load_status(self.code, self.job_id)["status"], "complete")
        self.assertEqual(editor_v2.pending_changes(self.manager.load_editor_v2_manifest(self.code, self.job_id)), {})
        self.assertEqual(self.manager.load_editor_v2_baseline(self.code, self.job_id, "translations", 0), before)
        with Image.open(self.manager.final_pages_dir(self.code, self.job_id) / "0000.png") as image:
            self.assertEqual(image.getpixel((50, 50)), (0, 0, 0))
        self.assertEqual(self.manager.editor_v2_payload(self.code, self.job_id, "text", 0)["recordsByArtifact"]["translations"][0]["englishText"], "Manual final text")

    def test_finish_failure_keeps_previous_output_and_pending_edits(self):
        self.edit(0); self.edit(1)
        final = self.manager.final_pages_dir(self.code, self.job_id)
        before = {path.name: path.read_bytes() for path in final.iterdir()}
        with mock.patch.object(self.manager, "enqueue"):
            self.manager.queue_finish_editor_edits(self.code, self.job_id)
        def render(_records, source, destination):
            if "0001" in str(destination):
                raise overlay_text.InputError("Synthetic render failure")
            shutil.copyfile(source, destination)
        with mock.patch.object(self.manager, "current_editor_clean_source", side_effect=lambda code, job, page, records: (self.manager.original_page_path(code, job, page), False)), \
             mock.patch.object(overlay_text, "overlay_text", side_effect=render):
            with self.assertRaisesRegex(overlay_text.InputError, "Synthetic"):
                self.manager.run_finish_editor_edits(self.code, self.job_id)
        self.assertEqual(before, {path.name: path.read_bytes() for path in final.iterdir()})
        self.assertEqual(set(editor_v2.pending_changes(self.manager.load_editor_v2_manifest(self.code, self.job_id))), {0, 1})
        self.assertEqual(self.manager.load_status(self.code, self.job_id)["status"], "failed")

    def test_export_generation_removes_all_old_archives(self):
        paths = self.archives()
        with mock.patch("web_exports.threading.Thread.start"):
            self.manager.generate_download(self.code, self.job_id, "png", regenerate=True)
        self.assertTrue(all(not path.exists() for path in paths))

    @unittest.skipUnless(shutil.which("magick"), "ImageMagick is required")
    def test_finish_matches_exact_preview_and_saves_cleaned_page(self):
        self.edit()
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        records = payload["recordsByArtifact"]
        preview, _, _ = self.manager.render_editor_v2_preview(self.code, self.job_id, 0,
            records["placements"], records["ocr_structured"], records["clean_boxes"], records["translations"])
        with Image.open(preview) as image:
            expected = image.convert("RGB").copy()
        with mock.patch.object(self.manager, "enqueue"):
            self.manager.queue_finish_editor_edits(self.code, self.job_id)
        with mock.patch.object(self.manager, "required_status_vlm_model", side_effect=AssertionError("VLM forbidden")), \
             mock.patch.object(self.manager, "run_pipeline_process", side_effect=AssertionError("Pipeline forbidden")):
            self.manager.run_job(self.code, self.job_id)
        final = self.manager.final_pages_dir(self.code, self.job_id) / "0000.png"
        with Image.open(final) as image:
            self.assertIsNone(ImageChops.difference(expected, image.convert("RGB")).getbbox())
        self.assertTrue((self.manager.output_dir(self.code, self.job_id) / "pages/cleaned/0000.png").is_file())
        self.assertTrue((self.manager.output_dir(self.code, self.job_id) / "debug/masks/0000.png").is_file())

    def test_queue_routes_require_auth_and_return_the_correct_scope(self):
        app = web_app.create_app(self.root / "config/web_config.json"); app.state.manager = self.manager
        base = f"/api/job/{self.code}/{self.job_id}/editor/v2"
        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                for suffix in ("pages/0/stages/text/regenerate", "pages/0/regenerate", "regenerate-all-pages", "finish-edits"):
                    self.assertEqual((await client.post(f"{base}/{suffix}")).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                response = await client.get(f"/job/{self.code}/{self.job_id}/edit")
                for label in ("Regenerate stage", "Regenerate page", "Regenerate all pages", "Finish Edits"):
                    self.assertIn(label, response.text)
                for method, suffix, result in (("regenerate_editor_v2_stage", "pages/0/stages/text/regenerate", "ocr_structured"),
                    ("rerun_editor_v2_page", "pages/0/regenerate", "placements"),
                    ("rerun_editor_v2_changes", "regenerate-all-pages", ([0], {0: "placements"})),
                    ("queue_finish_editor_edits", "finish-edits", [0])):
                    with mock.patch.object(self.manager, method, return_value=result) as action:
                        response = await client.post(f"{base}/{suffix}")
                    self.assertEqual(response.status_code, 200, response.text); action.assert_called_once()
        async def direct(function, *args, **kwargs): return function(*args)
        with mock.patch("anyio.to_thread.run_sync", new=direct): asyncio.run(exercise())


class StageBoundaryTests(unittest.TestCase):
    def test_ocr_stage_stops_after_merge_without_structure_or_vlm(self):
        args = types.SimpleNamespace(resume_from="ocr_raw", resume_page=0, output_dir=Path("output"), editor_stage_only="ocr")
        page = translate_cbz.Page(0, Path("original.png"))
        with mock.patch.object(translate_cbz, "load_phase_records", return_value={}), \
             mock.patch.object(translate_cbz, "run_ocr", return_value={0: []}), \
             mock.patch.object(translate_cbz, "run_ocr_merge_phase") as merge, \
             mock.patch.object(translate_cbz, "run_structure_phase", side_effect=AssertionError("Structure forbidden")):
            translate_cbz.run_single_page_resume_pipeline(args, mock.sentinel.config, [page], {})
        merge.assert_called_once()

    def test_text_stage_stops_after_translation_without_placement_or_render(self):
        args = types.SimpleNamespace(resume_from="ocr_structured", resume_page=0, output_dir=Path("output"),
                                     editor_stage_only="text", fixture_dir=None, translation_boxno=None)
        page = translate_cbz.Page(0, Path("original.png"))
        with mock.patch.object(translate_cbz, "load_phase_records", return_value={0: []}), \
             mock.patch.object(translate_cbz, "load_structured_records", return_value={}), \
             mock.patch.object(translate_cbz, "run_structure_phase", return_value={0: []}), \
             mock.patch.object(translate_cbz, "run_alt_placement_phase"), \
             mock.patch.object(translate_cbz, "attach_translations", return_value={}), \
             mock.patch.object(translate_cbz, "run_translation_phase") as translate, \
             mock.patch.object(translate_cbz, "run_placement_phase", side_effect=AssertionError("Placement forbidden")), \
             mock.patch.object(translate_cbz, "render_pages", side_effect=AssertionError("Render forbidden")):
            translate_cbz.run_single_page_resume_pipeline(args, mock.sentinel.config, [page], {})
        translate.assert_called_once()
