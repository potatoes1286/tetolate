from __future__ import annotations

import asyncio
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageChops

import overlay_text
import web_app
from tests import test_editor_transforms


class ManualLineBreakTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_editor_transforms.EditorTransformTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.manager, self.code, self.job_id = self.fixture.manager, self.fixture.code, self.fixture.job_id
        self.text = "This is a longer caption with several words that must change wrapping."
        self.entry = {"region": [0, 0, 500, 200], "englishText": self.text,
                      "font": overlay_text.DEFAULT_FONT, "fontSizeWidthPercent": 3}

    def test_automatic_wrap_changes_with_geometry_but_fixed_breaks_do_not(self):
        wide = overlay_text.layout_entry_text(self.entry, 1000, 1200)[0]
        narrow = overlay_text.layout_entry_text({**self.entry, "region": [0, 0, 100, 300]}, 1000, 1200)[0]
        self.assertNotEqual(wide, narrow)
        chosen = "This is my chosen first line\n\nSecond line."
        for width in (60, 200, 500):
            fixed = {**self.entry, "region": [0, 0, width, 300], "manualLineBreaks": chosen}
            text, size, _ = overlay_text.layout_entry_text(fixed, 1000, 1200)
            self.assertEqual(text, chosen)
            self.assertTrue(overlay_text.lines_fit(text.splitlines(), size, width - 12, 288, overlay_text.DEFAULT_FONT))

    def test_blank_or_whitespace_restores_automatic_wrapping(self):
        expected = overlay_text.layout_entry_text(self.entry, 1000, 1200)
        for blank in ("", " \n "):
            self.assertEqual(overlay_text.layout_entry_text({**self.entry, "manualLineBreaks": blank}, 1000, 1200), expected)

    def test_rendered_hint_is_display_only_and_unrelated_saves_do_not_fix_it(self):
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        records = copy.deepcopy(payload["recordsByArtifact"]["placements"])
        self.assertTrue(records[0]["_renderedText"])
        records[0]["placementRegion"] = [12, 12, 95, 115]
        saved = self.manager.save_editor_v2_update(self.code, self.job_id, 0, "placement", {
            "baseRevision": payload["revision"], "recordsByArtifact": {"placements": records},
        })
        record = saved["recordsByArtifact"]["placements"][0]
        self.assertNotIn("manualLineBreaks", record)
        manifest = self.manager.editor_v2_manifest_path(self.code, self.job_id).read_text()
        self.assertNotIn("_renderedText", manifest)
        entry = {**record, "englishText": "Hello"}
        self.assertEqual(record["_renderedText"], self.manager.editor_v2_text_layout(self.code, self.job_id, 0, entry))

    def test_manual_edits_survive_resize_restart_and_reset_returns_to_auto(self):
        chosen = "One chosen line\nSecond line"
        self.fixture.save({"manualLineBreaks": chosen})
        saved = self.fixture.save({"placementRegion": [0, 0, 50, 115]})
        self.assertEqual(saved["recordsByArtifact"]["placements"][0]["_renderedText"], chosen)
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        record = restarted.editor_v2_payload(self.code, self.job_id, "placement", 0)["recordsByArtifact"]["placements"][0]
        self.assertEqual(record["manualLineBreaks"], chosen)
        cleared = self.fixture.save({"manualLineBreaks": ""})
        self.assertNotEqual(cleared["recordsByArtifact"]["placements"][0]["_renderedText"], chosen)
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "placement", {"baseRevision": cleared["revision"]})
        self.assertNotIn("manualLineBreaks", reset["recordsByArtifact"]["placements"][0])
        self.assertTrue(reset["recordsByArtifact"]["placements"][0]["_renderedText"])

    def test_layout_http_auth_validation_and_no_rendering_or_state_changes(self):
        app = web_app.create_app(self.fixture.root / "config/web.json")
        app.state.manager = self.manager
        url = f"/api/job/{self.code}/{self.job_id}/editor/v2/pages/0/text-layout"
        before = self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes()
        record = {"placementRegion": [0, 0, 100, 120], "englishText": self.text}
        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.post(url, json={"record": record})).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                self.assertEqual((await client.post(url, json={"record": record}, headers={"Origin": "https://elsewhere.test"})).status_code, 403)
                response = await client.post(url, json={"record": record})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["text"], self.manager.editor_v2_text_layout(self.code, self.job_id, 0, record))
                self.assertIn("no-store", response.headers["cache-control"])
                for body in ({}, {"record": {**record, "manualLineBreaks": True}}, {"record": {**record, "manualLineBreaks": "x" * 100_001}},
                             {"record": {**record, "placementRegion": [0, 0, 0.1, 0.1]}}):
                    self.assertEqual((await client.post(url, json=body)).status_code, 400)
        async def direct(function, *args, **kwargs):
            return function(*args)
        with mock.patch.object(web_app, "run_in_threadpool", new=direct), mock.patch("anyio.to_thread.run_sync", new=direct), \
             mock.patch.object(overlay_text, "ensure_magick", side_effect=AssertionError("Must not render")):
            asyncio.run(exercise())
        self.assertEqual(self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes(), before)

    @unittest.skipUnless(shutil.which("magick"), "Requires the runtime ImageMagick binary")
    def test_manual_render_uses_exact_lines_and_batch_matches_single(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chosen = "One long line of chosen text\nSecond line"
            records = [{**self.entry, "region": [0, 0, width, 300], "manualLineBreaks": chosen}
                       for width in (100, 250, 500)]
            layers = [(entry, root / f"batch-{index}.png", root / f"text-{index}.txt") for index, entry in enumerate(records)]
            overlay_text.render_text_layers(shutil.which("magick"), layers, 1000, 1200)
            for index, (entry, output, text) in enumerate(layers):
                self.assertEqual(text.read_text(), chosen)
                single = root / f"single-{index}.png"
                overlay_text.render_text_layer(shutil.which("magick"), entry, single, root / f"single-{index}.txt", 1000, 1200)
                with Image.open(output) as first, Image.open(single) as second:
                    self.assertIsNone(ImageChops.difference(first.convert("RGBA"), second.convert("RGBA")).getbbox(alpha_only=False))
                    self.assertIsNotNone(first.convert("RGBA").getchannel("A").getbbox())


if __name__ == "__main__":
    unittest.main()
