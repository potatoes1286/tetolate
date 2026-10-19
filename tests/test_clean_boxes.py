from __future__ import annotations

import asyncio
import copy
import io
import json
import shutil
import unittest
import zipfile
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageChops, ImageDraw

import clean_text_regions
import editor_runtime
import editor_v2
import overlay_text
import translate_cbz
import web_app
from tests import test_editor_reset


class CleanBoxTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id
        self.root, self.data_dir = fixture.root, fixture.data_dir
        self.box = {"recordId": "clean-manual", "page": 0, "boxno": 0, "region": [50, 50, 80, 80]}
        self.addCleanup(editor_runtime.configure, None, None)

    def payload(self, page=0):
        return self.manager.editor_v2_payload(self.code, self.job_id, "placement", page)

    def save(self, boxes, page=0):
        payload = self.payload(page)
        return self.manager.save_editor_v2_update(self.code, self.job_id, page, "placement", {
            "baseRevision": payload["revision"], "recordsByArtifact": {"clean_boxes": boxes},
        })

    def runtime(self):
        editor_runtime.configure(self.manager.editor_v2_manifest_path(self.code, self.job_id),
                                 self.manager.editor_meta_dir(self.code, self.job_id) / "generated_v2")

    def test_save_reload_materialization_and_restart_keep_only_clean_fields(self):
        before = copy.deepcopy(self.payload()["recordsByArtifact"])
        saved = self.save([{**self.box, "englishText": "Do not typeset", "font": "missing.ttf", "page": 50}])
        self.assertEqual(saved["recordsByArtifact"]["clean_boxes"], [self.box])
        for artifact in ("ocr_structured", "translations", "placements"):
            self.assertEqual(saved["recordsByArtifact"][artifact], before[artifact])
        self.assertEqual(saved["rerunFrom"], "render")
        self.assertEqual(json.loads((self.data_dir / "clean_boxes/page_0000.json").read_text()), [self.box])
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        self.assertEqual(restarted.editor_v2_payload(self.code, self.job_id, "placement", 0)["recordsByArtifact"]["clean_boxes"], [self.box])

    def test_move_delete_restore_history_and_reset_do_not_create_a_baseline(self):
        saved = self.save([self.box])
        moved = {**self.box, "region": [60, 60, 90, 100]}
        self.save([moved])
        deleted = self.save([])
        self.assertEqual(deleted["recordsByArtifact"]["clean_boxes"], [])
        self.manager.restore_editor_v2_revision(self.code, self.job_id, saved["revision"])
        restored = self.payload()
        self.assertEqual(restored["recordsByArtifact"]["clean_boxes"], [self.box])
        self.save([{**self.box, "boxno": 1, "recordId": "clean-other-page"}], page=1)
        current = self.payload()
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "placement", {"baseRevision": current["revision"]})
        self.assertEqual(reset["recordsByArtifact"]["clean_boxes"], [])
        self.assertEqual(json.loads((self.data_dir / "clean_boxes/page_0000.json").read_text()), [])
        self.assertEqual(len(self.payload(1)["recordsByArtifact"]["clean_boxes"]), 1)

    def test_text_reset_and_model_reconciliation_keep_clean_boxes(self):
        saved = self.save([self.box])
        self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
        self.runtime()
        self.assertEqual(editor_runtime.clean_boxes(0), [self.box])
        editor_runtime.reconcile_records("placements", 0, [], "placement")
        self.assertEqual(editor_runtime.clean_boxes(0), [self.box])
        self.assertEqual(editor_runtime.clean_boxes(1), [])

    def test_regeneration_starts_at_render_without_a_model_call(self):
        self.save([self.box])
        with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
            self.assertEqual(self.manager.rerun_editor_v2_page(self.code, self.job_id, "placement", 0), "render")
        rerun.assert_called_once_with(self.code, self.job_id, [0], "render")

    def test_deleting_source_text_does_not_hide_independent_clean_boxes(self):
        self.save([self.box])
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "text", 0)
        payload["recordsByArtifact"]["ocr_structured"] = []
        self.manager.save_editor_v2_update(self.code, self.job_id, 0, "text", {
            "baseRevision": payload["revision"], "recordsByArtifact": payload["recordsByArtifact"],
        })
        later = self.payload()["recordsByArtifact"]
        self.assertEqual(later["placements"], [])
        self.assertEqual(later["clean_boxes"], [self.box])

    def test_clean_box_save_preserves_pending_upstream_work(self):
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        editor_v2.mark_stages_saved(manifest, 0, ("ocr", "translation"))
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
        saved = self.save([self.box])
        after = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.assertEqual(editor_v2.pending_changes(after)[0], {"ocr", "translation", "placement"})
        self.assertEqual(saved["rerunFrom"], "ocr_structured")

    def test_render_only_queue_and_command_work_without_a_selected_model(self):
        self.save([self.box])
        self.manager.input_path(self.code, self.job_id).write_bytes(b"input")
        status = self.manager.load_status(self.code, self.job_id)
        status["vlmModel"] = ""
        self.manager.save_status(self.code, self.job_id, status)
        with mock.patch.object(self.manager, "enqueue") as enqueue:
            pages, passes = self.manager.rerun_editor_v2_changes(self.code, self.job_id)
        self.assertEqual(pages, [0])
        self.assertEqual(passes, {0: "render"})
        enqueue.assert_called_once_with(self.code, self.job_id)
        command = self.manager.build_command(self.code, self.job_id, "render", 0)
        self.assertNotIn("--vlm-model", command)
        self.assertIn("--editor-manifest", command)

    def test_other_rerun_passes_still_require_a_selected_model(self):
        self.manager.input_path(self.code, self.job_id).write_bytes(b"input")
        status = self.manager.load_status(self.code, self.job_id)
        status["vlmModel"] = ""
        self.manager.save_status(self.code, self.job_id, status)
        with mock.patch.object(self.manager, "enqueue") as enqueue, self.assertRaises(HTTPException) as raised:
            self.manager.rerun_completed_job_pages(self.code, self.job_id, [0, 1], page_resume_from={0: "render", 1: "placements"})
        self.assertEqual(raised.exception.status_code, 400)
        enqueue.assert_not_called()

    def test_invalid_boxes_do_not_change_saved_state(self):
        before = self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes()
        invalid = [[None], [self.box, self.box], [{**self.box, "recordId": ""}],
                   [{**self.box, "boxno": True}], [{**self.box, "region": [0, 0, float("nan"), 10]}],
                   [{**self.box, "region": [0, 0, float("inf"), 10]}], [{**self.box, "region": [False, 0, 10, 10]}],
                   [{**self.box, "region": [90, 90, 80, 80]}], [{**self.box, "region": [200, 0, 300, 10]}],
                   [self.box] * 257]
        for boxes in invalid:
            with self.subTest(boxes=str(boxes)[:100]), self.assertRaises(HTTPException) as raised:
                self.save(boxes)
            self.assertEqual(raised.exception.status_code, 400)
            self.assertEqual(self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes(), before)
        clipped = self.save([{**self.box, "region": [-20, -10, 110, 130]}])
        self.assertEqual(clipped["recordsByArtifact"]["clean_boxes"][0]["region"], [0, 0, 100, 120])

    def test_preview_uses_clean_boxes_without_saving_or_typesetting_them(self):
        before = self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes()
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        with mock.patch.object(self.manager, "current_editor_clean_source", return_value=(original, True)) as clean, \
             mock.patch.object(overlay_text, "overlay_text", side_effect=lambda entries, source, target: shutil.copyfile(source, target)) as draw:
            self.manager.render_editor_v2_preview(self.code, self.job_id, 0, [], [], [self.box])
        self.assertEqual(clean.call_args.args[3], [self.box])
        self.assertEqual(draw.call_args.args[0], [])
        self.assertEqual(self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes(), before)

    def test_clean_masks_match_preview_and_pipeline_and_cache_tracks_changes(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        page = translate_cbz.Page(0, original)
        output = self.manager.output_dir(self.code, self.job_id)
        structured = [{"region": [10, 10, 20, 20], "openLettering": True}]

        def clean(entries, source, target, padding, _device, _model, _trigger, _margin, mask, _session):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            mask.parent.mkdir(parents=True, exist_ok=True)
            clean_text_regions.build_mask(entries, source, mask, padding)

        with mock.patch.object(clean_text_regions, "clean_text_regions", side_effect=clean) as cleaning, \
             mock.patch.object(self.manager, "_editor_lama_session", mock.Mock()):
            preview, _ = self.manager.current_editor_clean_source(self.code, self.job_id, 0, structured + [self.box])
            first_mask = preview.with_name("page_0000.mask.png").read_bytes()
            self.manager.current_editor_clean_source(self.code, self.job_id, 0, structured + [self.box])
            self.assertEqual(cleaning.call_count, 1)
            translate_cbz.clean_render_page(page, structured, output, None, [self.box])
            pipeline_mask = output / "debug/masks" / f"{original.stem}.png"
            self.assertEqual(pipeline_mask.read_bytes(), first_mask)
            self.manager.current_editor_clean_source(self.code, self.job_id, 0, structured + [{**self.box, "region": [60, 50, 90, 80]}])
            self.assertEqual(cleaning.call_count, 3)
            self.manager.current_editor_clean_source(self.code, self.job_id, 0, structured)
            self.assertEqual(cleaning.call_args.args[0], [])

    def test_render_and_cbz_include_clean_only_page_without_text_entries(self):
        self.save([self.box])
        self.runtime()
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        pages = [translate_cbz.Page(0, original)]
        output = self.manager.output_dir(self.code, self.job_id)
        cleaned_entries, text_entries = [], []

        def clean(entries, source, target, *_args):
            cleaned_entries.extend(entries)
            image = Image.open(source).convert("RGB")
            ImageDraw.Draw(image).rectangle(tuple(entries[0]["region"]), fill="red")
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)

        def draw(entries, source, target):
            text_entries.extend(entries)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        config = translate_cbz.load_config(self.manager.config.pipeline_config, None)
        with mock.patch.object(clean_text_regions, "clean_text_regions", side_effect=clean), \
             mock.patch.object(overlay_text, "overlay_text", side_effect=draw), \
             mock.patch.object(translate_cbz.lama_inpaint, "LaMaSession") as session:
            translate_cbz.render_pages(pages, {0: []}, output, config)
        session.assert_called_once()
        self.assertEqual(cleaned_entries, [self.box])
        self.assertEqual(text_entries, [])
        source_cbz = self.root / "source.cbz"
        with zipfile.ZipFile(source_cbz, "w") as archive:
            archive.write(original, original.name)
        result = translate_cbz.package_cbz(pages, output, source_cbz)
        with zipfile.ZipFile(result) as archive:
            image = Image.open(io.BytesIO(archive.read(original.name)))
            self.assertEqual(image.getpixel((60, 60)), (255, 0, 0))

    def test_http_preview_and_save_require_login_validate_and_preserve_revision(self):
        app = web_app.create_app(self.root / "config/web_config.json")
        app.state.manager = self.manager
        root = f"/api/job/{self.code}/{self.job_id}/editor/v2/pages/0/stages/placement"
        body = {"records": [], "cleanRecords": [], "cleanBoxes": [self.box]}
        original = self.manager.original_page_path(self.code, self.job_id, 0)

        async def direct(function, *args, **kwargs):
            return function(*args)

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.post(root + "/preview", json=body)).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                self.assertEqual((await client.post(root + "/preview", json=body, headers={"Origin": "http://other"})).status_code, 403)
                for boxes in (False, {}, [None], [{**self.box, "region": [1, 1, 0, 0]}]):
                    self.assertEqual((await client.post(root + "/preview", json={**body, "cleanBoxes": boxes})).status_code, 400)
                response = await client.post(root + "/preview", json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.payload()["revision"], 0)
                saved = await client.post(root, json={"baseRevision": 0, "recordsByArtifact": {"clean_boxes": [self.box]}})
                self.assertEqual(saved.status_code, 200)
                self.assertEqual(saved.json()["rerunFrom"], "render")
                stale = await client.post(root, json={"baseRevision": 0, "recordsByArtifact": {"clean_boxes": []}})
                self.assertEqual(stale.status_code, 409)

        with mock.patch("anyio.to_thread.run_sync", new=direct), \
             mock.patch.object(self.manager, "current_editor_clean_source", return_value=(original, True)), \
             mock.patch.object(overlay_text, "overlay_text", side_effect=lambda entries, source, target: shutil.copyfile(source, target)):
            asyncio.run(exercise())

    @unittest.skipUnless(shutil.which("magick"), "ImageMagick is not installed")
    def test_native_clean_only_render_removes_synthetic_mark_and_exports_it(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        image = Image.new("RGB", (100, 120), "white")
        ImageDraw.Draw(image).rectangle((58, 58, 70, 70), fill="black")
        image.save(original)
        self.save([self.box])
        self.runtime()
        output = self.manager.output_dir(self.code, self.job_id)
        pages = [translate_cbz.Page(0, original)]
        config = translate_cbz.load_config(self.manager.config.pipeline_config, None)
        translate_cbz.render_pages(pages, {0: []}, output, config)
        final = Image.open(translate_cbz.final_page_png_path(output, pages[0])).convert("RGB")
        self.assertIsNotNone(ImageChops.difference(image, final).getbbox())
        self.assertGreater(min(final.getpixel((64, 64))), 200)
        self.assertEqual(json.loads((output / "data/render_entries/page_0000.json").read_text()), [])
        source_cbz = self.root / "native-source.cbz"
        with zipfile.ZipFile(source_cbz, "w") as archive:
            archive.write(original, original.name)
        result = translate_cbz.package_cbz(pages, output, source_cbz)
        with zipfile.ZipFile(result) as archive:
            exported = Image.open(io.BytesIO(archive.read(original.name))).convert("RGB")
            self.assertIsNone(ImageChops.difference(exported, final).getbbox())
