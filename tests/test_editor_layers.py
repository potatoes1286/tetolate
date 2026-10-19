from __future__ import annotations

import asyncio
import base64
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageChops

from tests import test_regressions
import overlay_text
import translate_cbz
import web_app


class EditorLayerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager, self.code, self.job_id = test_regressions.EditorV2Tests().create_job(self.root)
        self.manager.initialize_web_state()
        self.manager.initialize_editor_v2(self.code, self.job_id)
        self.fonts = self.root / "fonts"
        self.fonts.mkdir()
        patch = mock.patch.object(translate_cbz, "FONT_DIR", self.fonts)
        patch.start()
        self.addCleanup(patch.stop)
        self.record = {"boxno": 0, "placementRegion": [10, 10, 90, 110], "englishText": "A wrapped sentence"}

    def layer(self, record=None):
        return self.manager.render_editor_v2_text_layers(self.code, self.job_id, 0, [record or self.record])[0]

    def layers(self, records):
        return self.manager.render_editor_v2_text_layers(self.code, self.job_id, 0, records)

    @staticmethod
    def fake_render(_magick, layers, _width, _height):
        for _entry, output, _text in layers:
            Image.new("RGBA", (96, 116), (0, 0, 0, 0)).save(output)

    def renderer(self):
        return mock.patch.object(overlay_text, "render_text_layers", side_effect=self.fake_render)

    def test_uncached_boxes_render_together_and_duplicate_layers_are_shared(self):
        other = {**self.record, "englishText": "Other"}
        with self.renderer() as render, mock.patch.object(overlay_text, "ensure_magick", return_value="magick") as ensure:
            paths = self.layers([self.record, other, self.record])
            self.assertEqual(render.call_count, 1)
            self.assertEqual(len(render.call_args.args[1]), 2)
            self.assertEqual(paths[0], paths[2])
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(self.layers([self.record, other]), paths[:2])
            self.assertEqual(render.call_count, 1)
            self.assertEqual(ensure.call_count, 1)

    def test_mixed_batch_renders_only_missing_layers(self):
        with self.renderer() as render, mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            first = self.layer()
            paths = self.layers([self.record, {**self.record, "englishText": "Second"}, {**self.record, "font": "Bangers-Regular.ttf"}])
            self.assertEqual(first, paths[0])
            self.assertEqual(len(render.call_args.args[1]), 2)
            self.assertEqual(render.call_count, 2)

    def test_entire_batch_is_validated_before_any_render(self):
        with self.renderer() as render:
            for records in (None, [], [self.record] * 257, [None], [self.record, {**self.record, "strokeWidth": -1}]):
                with self.assertRaises(HTTPException) as raised:
                    self.layers(records)
                self.assertEqual(raised.exception.status_code, 400)
            render.assert_not_called()

    def test_failed_batch_does_not_publish_partially_rendered_layers(self):
        def fail(_magick, layers, width, height):
            self.fake_render(_magick, layers[:1], width, height)
            raise overlay_text.InputError("failed second layer")
        with mock.patch.object(overlay_text, "render_text_layers", side_effect=fail), mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            with self.assertRaises(HTTPException):
                self.layers([self.record, {**self.record, "englishText": "Second"}])
        directory = self.manager.editor_meta_dir(self.code, self.job_id) / "previews/text-layers"
        self.assertEqual(list(directory.iterdir()), [])

    def test_payload_lists_bundled_user_and_default_fonts_without_browser_font_urls(self):
        (self.fonts / "Custom.ttf").write_bytes(b"font")
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        self.assertIn("ComicNeue-Bold.ttf", payload["fonts"])
        self.assertIn("IBMPlexMono-Medium.ttf", payload["fonts"])
        self.assertIn("Custom.ttf", payload["fonts"])
        self.assertEqual(payload["defaultFont"], "ComicNeue-Bold.ttf")
        self.assertNotIn("fontUrls", payload)

    def test_shared_entries_use_user_font_precedence_and_render_settings(self):
        custom = self.fonts / "ComicNeue-Bold.ttf"
        shutil.copyfile(translate_cbz.BUNDLED_FONT_DIR / custom.name, custom)
        record = {**self.record, "manualLineBreaks": "Manual\nlines", "gravity": "southeast", "strokeWidth": 4,
                  "stroke": "white", "fill": "black", "fontSizeWidthPercent": 15,
                  "rotation": 20, "skewX": -15, "skewY": 10}
        entry = self.manager.editor_v2_render_entries(self.code, self.job_id, 0, [record], {0: "Original"})[0]
        self.assertEqual(entry["font"], str(custom))
        self.assertEqual(entry["englishText"], "Manual\nlines")
        for field in ("gravity", "strokeWidth", "stroke", "fill", "fontSizeWidthPercent", "rotation", "skewX", "skewY"):
            self.assertEqual(entry[field], record[field])

    def test_layers_cache_by_size_and_style_not_position_or_box_identity(self):
        with self.renderer() as render, mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            first = self.layer()
            moved = self.layer({**self.record, "boxno": 99, "placementRegion": [0, 0, 80, 100]})
            self.assertEqual(first, moved)
            self.assertEqual(render.call_count, 1)
            for patch in ({"gravity": "west"}, {"fontSizeWidthPercent": 10}, {"englishText": "New"},
                          {"font": "Bangers-Regular.ttf"}, {"strokeWidth": 8}, {"placementRegion": [0, 0, 70, 100]}):
                self.assertNotEqual(first, self.layer({**self.record, **patch}))
            self.assertEqual(render.call_count, 7)

    def test_transforms_invalidate_layer_cache_but_movement_does_not(self):
        with self.renderer() as render, mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            original = self.layer()
            for field in overlay_text.TEXT_TRANSFORM_LIMITS:
                record = {**self.record, field: 15}
                transformed = self.layer(record)
                self.assertNotEqual(original, transformed)
                self.assertEqual(transformed, self.layer({**record, "placementRegion": [0, 0, 80, 100]}))
                self.assertEqual(render.call_args.args[1][0][0][field], 15)
            self.assertEqual(render.call_count, 4)

    def test_changing_font_file_invalidates_cached_layer(self):
        font = self.fonts / "Custom.ttf"
        font.write_bytes(b"first font")
        record = {**self.record, "font": font.name}
        with self.renderer() as render, mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            first = self.layer(record)
            font.write_bytes(b"replacement font")
            self.assertNotEqual(first, self.layer(record))
            self.assertEqual(render.call_count, 2)

    def test_invalid_geometry_styles_text_and_font_paths_are_rejected(self):
        outside = self.root / "secret"
        outside.write_text("secret")
        (self.fonts / "escape.ttf").symlink_to(outside)
        patches = [{"placementRegion": [0, 0, float("nan"), 10]}, {"placementRegion": [1, 1, 0, 0]},
                   {"placementRegion": [False, 0, 10, 10]}, {"placementRegion": [0, 0, 0.1, 0.1]},
                   {"font": "missing.ttf"}, {"font": "/etc/passwd"}, {"font": "escape.ttf"},
                   {"gravity": "wrong"}, {"strokeWidth": -1}, {"fontSizeWidthPercent": True},
                   {"fontSizeWidthPercent": 101}, {"renderBleed": 500}, {"englishText": []},
                   {"englishText": "x" * 100_001}]
        for patch in patches:
            with self.subTest(patch=str(patch)[:100]), self.assertRaises(HTTPException) as raised:
                self.layer({**self.record, **patch})
            self.assertEqual(raised.exception.status_code, 400)

    def test_failed_render_does_not_leave_a_cached_image(self):
        with mock.patch.object(overlay_text, "ensure_magick", side_effect=overlay_text.InputError("missing magick")):
            with self.assertRaises(HTTPException) as raised:
                self.layer()
        self.assertEqual(raised.exception.status_code, 500)
        self.assertIn("missing magick", raised.exception.detail)
        self.assertEqual(list((self.manager.editor_meta_dir(self.code, self.job_id) / "previews/text-layers").iterdir()), [])

    def test_http_requires_login_same_origin_and_returns_batch_without_saving_edits(self):
        app = web_app.create_app(self.root / "config/web_config.json")
        app.state.manager = self.manager
        url = f"/api/job/{self.code}/{self.job_id}/editor/v2/pages/0/text-layers"
        body = {"records": [self.record, {**self.record, "englishText": "Second"}]}
        before = copy.deepcopy(self.manager.load_editor_v2_manifest(self.code, self.job_id))

        async def direct(function, *args, **kwargs):
            return function(*args)

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.post(url, json=body)).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                self.assertEqual((await client.post(url, json=body, headers={"Origin": "http://other"})).status_code, 403)
                self.assertEqual((await client.post(url, json=[])).status_code, 400)
                self.assertEqual((await client.post(url, content="{")).status_code, 400)
                self.assertEqual((await client.post(url.replace(self.job_id, "00000000"), json=body)).status_code, 404)
                response = await client.post(url, json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "application/json")
                self.assertEqual(response.headers["cache-control"], "private, no-store")
                self.assertEqual(len(response.json()["layers"]), 2)
                for png in response.json()["layers"]:
                    self.assertTrue(base64.b64decode(png).startswith(b"\x89PNG\r\n\x1a\n"))

        with mock.patch("anyio.to_thread.run_sync", new=direct), self.renderer(), mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            asyncio.run(exercise())
        self.assertEqual(self.manager.load_editor_v2_manifest(self.code, self.job_id), before)

    @unittest.skipUnless(shutil.which("magick"), "Requires the runtime ImageMagick binary")
    def test_preview_pixels_match_current_across_fonts_alignment_wrapping_and_size(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        Image.new("RGB", (800, 1000), "white").save(original)
        text = "UNBELIEVABLY complicated words!\nAnd more lines..."
        cases = [{"font": font, "gravity": gravity} for font in ("ComicNeue-Bold.ttf", "ComicNeue-BoldItalic.ttf")
                 for gravity in ("north", "center", "south", "west", "east", "northwest", "northeast", "southwest", "southeast")]
        cases += [{"fontSizeWidthPercent": 20}, {"strokeWidth": 8, "fill": "white", "stroke": "black"},
                  {"manualLineBreaks": "One\nTWO\nThree"}, {"placementRegion": [10, 10, 130, 700]},
                  {"placementRegion": [10, 10, 700, 130]}]
        cases += [{"rotation": angle} for angle in (-180, -35, 35, 90, 180)]
        cases += [{"skewX": -30}, {"skewY": 30}, {"rotation": 23, "skewX": 30, "skewY": -20},
                  {"rotation": -40, "skewX": 60, "skewY": 60},
                  {"rotation": 45, "placementRegion": [0, 0, 161, 103]},
                  {"rotation": -30, "skewY": -20, "placementRegion": [690, 903, 800, 1000]}]
        records = [{**self.record, "placementRegion": [40, 50, 400, 330], "englishText": text, **patch} for patch in cases]
        with mock.patch.object(overlay_text, "run_magick", wraps=overlay_text.run_magick) as run:
            layers = self.layers(records)
            self.assertEqual(run.call_count, 1, "All uncached boxes must use one Magick process")
        for patch, record, layer in zip(cases, records, layers):
            with self.subTest(patch=patch):
                region = record["placementRegion"]
                preview = Image.open(original).convert("RGBA")
                with Image.open(layer) as image:
                    preview.alpha_composite(image.convert("RGBA"),
                        (region[0] + (region[2] - region[0] - image.width) // 2,
                         region[1] + (region[3] - region[1] - image.height) // 2))
                with mock.patch.object(self.manager, "editor_v2_effective_records", return_value=[{"boxno": 0, "englishText": text}]), \
                     mock.patch.object(self.manager, "current_editor_clean_source", return_value=(original, False)):
                    current, _, _ = self.manager.render_editor_v2_preview(self.code, self.job_id, 0, [record], [])
                with Image.open(current) as image:
                    difference = ImageChops.difference(preview.convert("RGB"), image.convert("RGB"))
                    maximum_error = max(high for _low, high in difference.getextrema())
                    self.assertLessEqual(maximum_error, 1, "Text layer compositing differs from Current")
