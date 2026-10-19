from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import tempfile

from PIL import Image

import overlay_text as overlay


class CaptionSizingTests(unittest.TestCase):
    def layout(self, text, width=703, height=73, **fields):
        return overlay.explicit_caption_layout({"font": overlay.DEFAULT_FONT, **fields},
                                               text, width, height, 1427, 2048)

    def test_wide_single_line_caption_uses_more_height_without_exceeding_page_cap(self):
        text = "Even if you say that..."
        rendered, size, _ = self.layout(text)
        self.assertEqual(rendered, text)
        self.assertGreaterEqual(size, 40)
        self.assertLessEqual(size, 2048 * overlay.PAGE_HEIGHT_MAX_POINT_RATIO)
        self.assertTrue(overlay.lines_fit([rendered], size, 703, 73, overlay.DEFAULT_FONT))

    def test_speech_bubbles_tall_boxes_and_moderately_wide_boxes_are_unchanged(self):
        cases = [("I can't remember anything at all!", 274, 497,
                  "I can't\nremember\nanything\nat all!", 49),
                 ("We must find another way before morning.", 300, 300,
                  "We must find\nanother way\nbefore\nmorning.", 41),
                 ("Here... is a basic test~~", 262, 711, "Here...\nis a\nbasic\ntest~~", 49),
                 ("A short caption.", 280, 100, "A short caption.", 30)]
        for text, width, height, expected, size in cases:
            with self.subTest(text=text):
                self.assertEqual(self.layout(text, width, height)[:2], (expected, float(size)))

    def test_long_wide_captions_keep_existing_multiline_sizing(self):
        text = "An unusually long sentence needs enough room for every word to remain readable."
        self.assertEqual(self.layout(text)[:2],
                         ("An unusually long sentence needs enough room for every word\nto remain readable.", 21.0))

    def test_explicit_line_breaks_are_not_enlarged(self):
        self.assertEqual(self.layout("First line\nSecond line")[:2], ("First line\nSecond line", 21.0))

    def test_explicit_font_size_and_configured_maximum_are_kept(self):
        text = "Even if you say that..."
        self.assertEqual(self.layout(text, fontSizeWidthPercent=1.5)[1], 1427 * 1.5 / 100)
        self.assertLessEqual(self.layout(text, maxPointSize=25)[1], 25)

    def test_short_and_thin_captions_do_not_overflow_or_ignore_page_size(self):
        for text in ("Hi!", "A single caption."):
            for height in (35, 50, 73):
                with self.subTest(text=text, height=height):
                    rendered, size, _ = self.layout(text, height=height)
                    self.assertEqual(rendered, text)
                    self.assertTrue(overlay.lines_fit([rendered], size, 703, height, overlay.DEFAULT_FONT))
                    self.assertLessEqual(size, 2048 * overlay.PAGE_HEIGHT_MAX_POINT_RATIO)

    @unittest.skipUnless(shutil.which("magick"), "Requires the runtime ImageMagick binary")
    def test_rendered_wide_caption_is_larger_and_stays_inside_its_box(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = {"region": [0, 0, 715, 85], "englishText": "Even if you say that...",
                     "font": overlay.DEFAULT_FONT, "fill": "white", "stroke": "black", "strokeWidth": 3}
            bounds = []
            for name, fields in (("automatic", {}), ("old-size", {"fontSizeWidthPercent": 21 / 1427 * 100})):
                output = root / f"{name}.png"
                overlay.render_text_layer(shutil.which("magick"), {**entry, **fields}, output, root / f"{name}.txt", 1427, 2048)
                with Image.open(output) as layer:
                    bounds.append(layer.convert("RGBA").getchannel("A").getbbox())
            left, top, right, bottom = bounds[0]
            self.assertGreater(bottom - top, 30)
            self.assertGreater(bottom - top, (bounds[1][3] - bounds[1][1]) * 1.5)
            self.assertGreaterEqual(left, 8)
            self.assertGreaterEqual(top, 8)
            self.assertLessEqual(right, 715 + 8)
            self.assertLessEqual(bottom, 85 + 8)
