from __future__ import annotations

import asyncio
import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import io
import os
import re
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageChops, ImageDraw

from tests import test_regressions
import overlay_text
import clean_text_regions
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

    def test_payload_exposes_saved_cleaned_background_only_when_available(self):
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        self.assertIsNone(payload["image"]["cleanedUrl"])
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        cleaned = translate_cbz.cleaned_pages_dir(self.manager.output_dir(self.code, self.job_id)) / f"{original.stem}.png"
        cleaned.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 120), "white").save(cleaned)
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        url = f"/api/job/{self.code}/{self.job_id}/editor/v2/image/cleaned/0"
        self.assertEqual(payload["image"]["cleanedUrl"], url)
        self.assertEqual(payload["image"]["url"], f"/job/{self.code}/{self.job_id}/image/0")

        app = web_app.create_app(self.root / "config/web_config.json"); app.state.manager = self.manager

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.get(url)).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                response = await client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, cleaned.read_bytes())

        async def direct(function, *args, **kwargs):
            return function(*args)

        with mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(exercise())

    def test_all_editor_module_imports_are_served_after_login(self):
        app = web_app.create_app(self.root / "config/web_config.json"); app.state.manager = self.manager
        source = (web_app.REPO_DIR / "web_editor/app.js").read_text()
        modules = set(re.findall(r'from "\./([^\"]+)"', source))
        self.assertIn("preview.js", modules)

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.get("/assets/editor-v2/preview.js")).status_code, 303)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                for module in modules:
                    response = await client.get(f"/assets/editor-v2/{module}")
                    self.assertEqual(response.status_code, 200, module)
                    self.assertIn("application/javascript", response.headers["content-type"])
                self.assertEqual((await client.get("/assets/editor-v2/missing.js")).status_code, 404)

        async def direct(function, *args, **kwargs):
            return function(*args)

        with mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(exercise())

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

    def test_layout_rule_changes_invalidate_layer_and_current_caches(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        with self.renderer() as render, mock.patch.object(overlay_text, "ensure_magick", return_value="magick"):
            first = self.layer()
            with mock.patch.object(overlay_text, "TEXT_LAYOUT_VERSION", overlay_text.TEXT_LAYOUT_VERSION + 1):
                self.assertNotEqual(first, self.layer())
            self.assertEqual(render.call_count, 2)

        def fake_overlay(_entries, _source, output):
            shutil.copyfile(original, output)

        with mock.patch.object(self.manager, "current_editor_clean_source", return_value=(original, False)), \
             mock.patch.object(overlay_text, "overlay_text", side_effect=fake_overlay) as render:
            self.manager.render_editor_v2_preview(self.code, self.job_id, 0, [self.record], [])
            self.assertTrue(self.manager.render_editor_v2_preview(self.code, self.job_id, 0, [self.record], [])[1])
            with mock.patch.object(overlay_text, "TEXT_LAYOUT_VERSION", overlay_text.TEXT_LAYOUT_VERSION + 1):
                self.assertFalse(self.manager.render_editor_v2_preview(self.code, self.job_id, 0, [self.record], [])[1])
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

    def test_background_preview_has_immutable_cache_paths_and_never_typesets_or_changes_job_data(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        output = self.manager.output_dir(self.code, self.job_id)
        final = output / "pages/final" / original.name
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_bytes(b"last pipeline render")
        manifest_before = self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes()
        records = [{"region": [20, 20, 50, 50], "openLettering": False}]

        def clean(entries, source, target, padding, _device, _model, _trigger, _margin, mask, _session):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            mask.parent.mkdir(parents=True, exist_ok=True)
            clean_text_regions.build_mask(entries, source, mask, padding)

        def render(_entries, source, target):
            shutil.copyfile(source, target)

        with mock.patch.object(self.manager, "_editor_lama_session", mock.Mock()), \
             mock.patch.object(clean_text_regions, "clean_text_regions", side_effect=clean) as cleaning, \
             mock.patch.object(overlay_text, "overlay_text", side_effect=render) as rendering:
            result = self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, records, [])
            first_path, _ = self.manager.current_editor_clean_source(self.code, self.job_id, 0, records)
            first_bytes = first_path.read_bytes()
            self.assertEqual(self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, records, [])["url"], result["url"])
            self.assertEqual(cleaning.call_count, 1)
            changed = [{**records[0], "region": [40, 40, 80, 80]}]
            other = self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, changed, [])
            self.assertNotEqual(result["url"], other["url"])
            self.assertEqual(first_path.read_bytes(), first_bytes)
            self.assertEqual(self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, records, [])["url"], result["url"])
            self.assertEqual(cleaning.call_count, 2, "returning to an older draft reuses its own cached mask")
            pipeline_cleaned = output / "pages/cleaned" / original.name
            pipeline_mask = output / "debug/masks" / original.name
            pipeline_cleaned.parent.mkdir(parents=True, exist_ok=True)
            pipeline_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, pipeline_cleaned)
            clean_text_regions.build_mask(records, original, pipeline_mask, clean_text_regions.DEFAULT_PADDING)
            matched = self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, records, [])
            self.assertEqual(cleaning.call_count, 2)
            self.assertTrue(matched["url"].endswith("/image/cleaned/0"))
            rendering.assert_not_called()
        self.assertEqual(final.read_bytes(), b"last pipeline render")
        self.assertEqual(self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes(), manifest_before)

    def test_background_preview_http_validates_regions_and_keeps_auth_checks(self):
        app = web_app.create_app(self.root / "config/web_config.json")
        app.state.manager = self.manager
        url = f"/api/job/{self.code}/{self.job_id}/editor/v2/pages/0/clean-preview"
        body = {"cleanRecords": [], "cleanBoxes": []}

        async def direct(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def exercise(render):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.post(url, json=body)).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                self.assertEqual((await client.post(url, json=body, headers={"Origin": "http://other"})).status_code, 403)
                for value in (None, "true", 1, {}):
                    self.assertEqual((await client.post(url, json={**body, "cleanBoxes": value})).status_code, 400)
                render.assert_not_called()
                response = await client.post(url, json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["url"], "cleaned.png")
                self.assertEqual(render.call_args.args[-2:], ([], []))

        with mock.patch("anyio.to_thread.run_sync", new=direct), \
             mock.patch.object(self.manager, "render_editor_v2_clean_preview", return_value={"url": "cleaned.png"}) as render:
            asyncio.run(exercise(render))

    def test_background_preview_rejects_invalid_regions_before_inpainting(self):
        invalid = [None, {"region": [False, 0, 10, 10]}, {"region": [0, 0, float("nan"), 10]},
                   {"region": [10, 10, 0, 0]}, {"region": [200, 200, 250, 250]}]
        with mock.patch.object(clean_text_regions, "clean_text_regions") as clean:
            for record in invalid:
                with self.subTest(record=str(record)), self.assertRaises(HTTPException) as raised:
                    self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, [record], [])
                self.assertEqual(raised.exception.status_code, 400)
            clean.assert_not_called()

    def test_concurrent_background_drafts_have_distinct_images_and_masks(self):
        def clean(entries, source, target, padding, _device, _model, _trigger, _margin, mask, _session):
            image = Image.open(source).convert("RGB")
            image.paste("red" if entries[0]["region"][0] == 20 else "blue", (0, 0, 100, 120))
            image.save(target)
            clean_text_regions.build_mask(entries, source, mask, padding)

        with mock.patch.object(self.manager, "_editor_lama_session", mock.Mock()), \
             mock.patch.object(clean_text_regions, "clean_text_regions", side_effect=clean) as cleaning:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.manager.render_editor_v2_clean_preview, self.code, self.job_id, 0,
                    [{"region": [left, 20, left + 20, 60]}], []) for left in (20, 50)]
                results = [future.result() for future in futures]
            self.assertNotEqual(results[0]["url"], results[1]["url"])
            for result, color in zip(results, ((255, 0, 0), (0, 0, 255))):
                key = result["url"].rsplit("/", 1)[1]
                image = self.manager.editor_v2_clean_preview_path(self.code, self.job_id, 0, key)
                with Image.open(image) as pixels:
                    self.assertEqual(pixels.getpixel((0, 0)), color)
                self.assertTrue(image.with_suffix(".mask.png").is_file())
            self.assertEqual(cleaning.call_count, 2)

    @unittest.skipUnless(os.environ.get("TETOLATE_LAMA_MODEL_PATH"), "Requires the isolated runtime LaMa model")
    def test_native_background_preview_serves_cleaned_pixels_without_magick(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        image = Image.new("RGB", (100, 120), "white")
        ImageDraw.Draw(image).rectangle((48, 48, 58, 58), fill="black")
        image.save(original)
        with mock.patch.object(overlay_text, "overlay_text", side_effect=AssertionError("Background endpoint must not typeset")):
            result = self.manager.render_editor_v2_clean_preview(self.code, self.job_id, 0, [{"region": [40, 40, 70, 70]}], [])
        if self.manager._editor_lama_session is not None:
            self.addCleanup(self.manager._editor_lama_session.close)
        app = web_app.create_app(self.root / "config/web_config.json"); app.state.manager = self.manager

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.get(result["url"])).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                response = await client.get(result["url"])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "image/png")
                self.assertEqual(response.headers["cache-control"], "no-store")
                with Image.open(io.BytesIO(response.content)) as cleaned:
                    self.assertGreater(min(cleaned.convert("RGB").getpixel((53, 53))), 200)
                invalid = result["url"].rsplit("/", 1)[0] + "/invalid"
                self.assertEqual((await client.get(invalid)).status_code, 404)

        asyncio.run(exercise())

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
        cases += [{"manualLineBreaks": "Even if you say that...", "placementRegion": [40, 50, 755, 135]}]
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
