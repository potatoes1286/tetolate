from __future__ import annotations

import copy
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from PIL import Image, ImageChops

import editor_runtime
import editor_v2
import overlay_text
import translate_cbz
import web_app
from tests import test_regressions
from web_storage import write_json_atomic


class EditorTransformTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager, self.code, self.job_id = test_regressions.EditorV2Tests().create_job(self.root)
        self.manager.initialize_web_state()
        self.output = self.manager.output_dir(self.code, self.job_id)
        self.record = {"page": 0, "boxno": 0, "region": [10, 10, 90, 110],
                       "placementRegion": [10, 10, 90, 110], "sourceBoxnos": [0],
                       "text": "Source", "englishText": "Hello", "font": "ComicNeue-Bold.ttf",
                       "fill": "black", "openLettering": False, "safeToEraseOriginal": True}
        for artifact in editor_v2.ARTIFACT_STAGES:
            record = dict(self.record)
            if artifact == "placements":
                record.pop("sourceBoxnos")
            write_json_atomic(self.output / "data" / artifact / "page_0000.json", [record])
        self.manager.initialize_editor_v2(self.code, self.job_id)
        self.transform = {"rotation": 32.5, "skewX": -20, "skewY": 15}

    def save(self, fields):
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        records = copy.deepcopy(payload["recordsByArtifact"]["placements"])
        records[0].update(fields)
        return self.manager.save_editor_v2_update(self.code, self.job_id, 0, "placement", {
            "baseRevision": payload["revision"], "recordsByArtifact": {"placements": records},
        })

    def test_angles_validate_for_save_and_preview_without_changing_the_manifest(self):
        before = self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes()
        for field, limit in overlay_text.TEXT_TRANSFORM_LIMITS.items():
            for value in (None, True, "10", float("nan"), float("inf"), -limit - 0.1, limit + 0.1):
                with self.subTest(field=field, value=value):
                    for operation in (lambda: self.save({field: value}),
                                      lambda: self.manager.editor_v2_render_entries(
                                          self.code, self.job_id, 0, [{**self.record, field: value}])):
                        with self.assertRaises(HTTPException) as raised:
                            operation()
                        self.assertEqual(raised.exception.status_code, 400)
                    self.assertEqual(self.manager.editor_v2_manifest_path(self.code, self.job_id).read_bytes(), before)
            for value in (-limit, 0, limit):
                overlay_text.validate_text_transform(0, {field: value})

    def test_save_protection_restart_reset_and_revision_restore(self):
        saved = self.save(self.transform)
        self.assertEqual(saved["rerunFrom"], "render")
        record = saved["recordsByArtifact"]["placements"][0]
        for field, value in self.transform.items():
            self.assertEqual(record[field], value)
            self.assertTrue(saved["protection"]["records"][record["recordId"]][field])
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        payload = restarted.editor_v2_payload(self.code, self.job_id, "placement", 0)
        self.assertEqual({field: payload["recordsByArtifact"]["placements"][0][field]
                          for field in self.transform}, self.transform)
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, "placements", 0)
        protected = editor_v2.effective_records(manifest, 0, "placements", baseline, protected_only=True)
        self.assertEqual({field: protected[0][field] for field in self.transform}, self.transform)
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "placement", {"baseRevision": saved["revision"]})
        for field in self.transform:
            self.assertNotIn(field, reset["recordsByArtifact"]["placements"][0])
        self.assertFalse(reset["stageStates"]["placement"]["changed"])
        self.manager.restore_editor_v2_revision(self.code, self.job_id, saved["revision"])
        restored = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
        self.assertEqual({field: restored["recordsByArtifact"]["placements"][0][field]
                          for field in self.transform}, self.transform)

    def test_zero_angles_replace_previously_saved_transforms(self):
        self.save(self.transform)
        zeros = dict.fromkeys(self.transform, 0)
        self.save(zeros)
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        record = restarted.editor_v2_payload(self.code, self.job_id, "placement", 0)["recordsByArtifact"]["placements"][0]
        self.assertEqual({field: record[field] for field in zeros}, zeros)
        entry = restarted.editor_v2_render_entries(self.code, self.job_id, 0, [record])[0]
        self.assertEqual(overlay_text.text_transform_args(entry, 96, 116), [])

    def test_cleared_font_size_stays_automatic_after_save_reload_and_render_input(self):
        original = self.manager.load_editor_v2_baseline(self.code, self.job_id, "placements", 0)
        for generated_size in (None, 2.75):
            with self.subTest(generated_size=generated_size):
                baseline = copy.deepcopy(original)
                if generated_size is not None:
                    baseline[0]["fontSizeWidthPercent"] = generated_size
                write_json_atomic(self.manager.editor_v2_generated_path(self.code, self.job_id, "placements", 0), baseline)
                self.save({"fontSizeWidthPercent": 1.65})
                payload = self.manager.editor_v2_payload(self.code, self.job_id, "placement", 0)
                records = copy.deepcopy(payload["recordsByArtifact"]["placements"])
                records[0].pop("fontSizeWidthPercent")
                self.manager.save_editor_v2_update(self.code, self.job_id, 0, "placement", {
                    "baseRevision": payload["revision"], "recordsByArtifact": {"placements": records},
                })
                restarted = web_app.JobManager(self.manager.config)
                restarted.initialize_web_state()
                record = restarted.editor_v2_payload(self.code, self.job_id, "placement", 0)["recordsByArtifact"]["placements"][0]
                self.assertNotIn("fontSizeWidthPercent", record)
                self.assertNotIn("fontSizeWidthPercent", restarted.editor_v2_render_entries(self.code, self.job_id, 0, [record])[0])
                self.assertGreater(record["_autoFontSizeWidthPercent"], 0)
                protected = editor_v2.effective_records(restarted.load_editor_v2_manifest(self.code, self.job_id),
                    0, "placements", baseline, protected_only=True)
                self.assertNotIn("fontSizeWidthPercent", protected[0])

    def test_manual_overrides_reach_final_renderer_but_model_values_are_not_accepted(self):
        original = self.manager.original_page_path(self.code, self.job_id, 0)
        page = translate_cbz.Page(0, original)
        model_values = [{**self.record, **self.transform}]
        validated = translate_cbz.validate_placements_page(page, [self.record], model_values)
        for field in self.transform:
            self.assertNotIn(field, validated[0])
        self.save(self.transform)
        editor_runtime.configure(self.manager.editor_v2_manifest_path(self.code, self.job_id),
                                 self.manager.editor_meta_dir(self.code, self.job_id) / "generated_v2")
        self.addCleanup(editor_runtime.configure, None, None)
        records = [dict(self.record)]
        translate_cbz.attach_placements([page], self.output, {0: records})
        config = translate_cbz.load_config(self.manager.config.pipeline_config, None)
        with mock.patch.object(overlay_text, "overlay_text") as render:
            translate_cbz.typeset_render_page(page, records, self.output, config)
        entry = render.call_args.args[0][0]
        self.assertEqual({field: entry[field] for field in self.transform}, self.transform)

    def test_affine_matrix_keeps_center_expands_bounds_and_uses_one_resampling(self):
        for width, height in ((96, 116), (97, 118), (117, 95)):
            for transform in (self.transform, {"rotation": 90}, {"rotation": -180},
                              {"skewX": 60, "skewY": 60}, {"rotation": 45, "skewX": -60, "skewY": 60}):
                with self.subTest(size=(width, height), transform=transform):
                    args = overlay_text.text_transform_args(transform, width, height)
                    self.assertEqual(args.count("-distort"), 1)
                    viewport = args[args.index("-define") + 1]
                    out_width, out_height = map(int, re.search(r"=(\d+)x(\d+)", viewport).groups())
                    self.assertEqual((out_width - width) % 2, 0)
                    self.assertEqual((out_height - height) % 2, 0)
                    a, b, c, d, dx, dy = map(float, args[args.index("AffineProjection") + 1].split(","))
                    self.assertAlmostEqual(a * d - b * c, 1)
                    center_x, center_y = width / 2, height / 2
                    self.assertAlmostEqual(a * center_x + c * center_y + dx, out_width / 2)
                    self.assertAlmostEqual(b * center_x + d * center_y + dy, out_height / 2)
                    for x, y in ((0, 0), (width, 0), (0, height), (width, height)):
                        self.assertTrue(2 - 1e-8 <= a * x + c * y + dx <= out_width - 2 + 1e-8)
                        self.assertTrue(2 - 1e-8 <= b * x + d * y + dy <= out_height - 2 + 1e-8)
        self.assertEqual(overlay_text.text_transform_args({}, 96, 116), [])
        self.assertEqual(overlay_text.text_transform_args(dict.fromkeys(self.transform, 0), 96, 116), [])

    @unittest.skipUnless(shutil.which("magick"), "Requires the runtime ImageMagick binary")
    def test_clockwise_rotation_matches_independent_pixel_reference(self):
        source = Image.new("RGBA", (17, 23))
        # Asymmetric opaque pixels expose an incorrect direction or half-pixel origin.
        for x in range(3, 10):
            for y in range(4, 19):
                source.putpixel((x, y), (x * 20, y * 10, 90, 255))
        path, output = self.root / "source.png", self.root / "rotated.png"
        source.save(path)
        overlay_text.run_magick([shutil.which("magick"), str(path),
            *overlay_text.text_transform_args({"rotation": 90}, *source.size), str(output)], "test rotation")
        reference = source.transpose(Image.Transpose.ROTATE_270)
        with Image.open(output) as rotated:
            left, top = (rotated.width - reference.width) // 2, (rotated.height - reference.height) // 2
            actual = rotated.convert("RGBA").crop((left, top, left + reference.width, top + reference.height))
            self.assertEqual(ImageChops.difference(actual, reference).getbbox(alpha_only=False), None)
            self.assertEqual(rotated.convert("RGBA").getpixel((0, 0))[-1], 0)

    @unittest.skipUnless(shutil.which("magick"), "Requires the runtime ImageMagick binary")
    def test_batch_and_single_layers_match_without_leaking_transform_settings(self):
        entries = [{**self.record, "font": overlay_text.DEFAULT_FONT, **patch}
                   for patch in (self.transform, {}, {"skewX": -60}, {"rotation": -45, "skewY": 60},
                                 {"rotation": 90, "englishText": ""}, dict.fromkeys(self.transform, 0))]
        layers = [(entry, self.root / f"batch-{i}.png", self.root / f"batch-{i}.txt") for i, entry in enumerate(entries)]
        overlay_text.render_text_layers(shutil.which("magick"), layers, 100, 120)
        for i, (entry, batch, _) in enumerate(layers):
            single = self.root / f"single-{i}.png"
            overlay_text.render_text_layer(shutil.which("magick"), entry, single, self.root / f"single-{i}.txt", 100, 120)
            with Image.open(single) as first, Image.open(batch) as second:
                self.assertEqual(first.size, second.size)
                self.assertEqual(ImageChops.difference(first.convert("RGBA"), second.convert("RGBA")).getbbox(alpha_only=False), None)
        with Image.open(layers[1][1]) as absent, Image.open(layers[-1][1]) as zeros:
            self.assertEqual(absent.size, zeros.size)
            self.assertEqual(absent.tobytes(), zeros.tobytes())
