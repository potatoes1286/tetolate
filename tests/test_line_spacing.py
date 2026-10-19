from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image, ImageChops

import overlay_text


class LineSpacingTests(unittest.TestCase):
    def test_estimates_add_spacing_only_between_lines(self):
        for count in (1, 2, 3):
            lines = ["LINE"] * count
            normal = overlay_text.estimated_text_extent(lines, 20)
            for spacing in (-4, 0, 12):
                width, height = overlay_text.estimated_text_extent(lines, 20, line_spacing=spacing)
                self.assertEqual(width, normal[0])
                self.assertEqual(height, normal[1] + (count - 1) * spacing)

    def test_fit_and_fixed_size_respect_extra_spacing(self):
        self.assertTrue(overlay_text.lines_fit(["LINE"] * 3, 20, 300, 80))
        self.assertFalse(overlay_text.lines_fit(["LINE"] * 3, 20, 300, 80, line_spacing=12))
        text = "LINE\nLINE\nLINE"
        normal = overlay_text.explicit_caption_layout({"fontSizeWidthPercent": 4}, text, 300, 80, 500, 500)
        spaced = overlay_text.explicit_caption_layout({"fontSizeWidthPercent": 4, "lineSpacing": 12}, text, 300, 80, 500, 500)
        self.assertLess(spaced[1], normal[1])
        self.assertTrue(overlay_text.lines_fit(spaced[0].splitlines(), spaced[1], 300, 80, line_spacing=12))

    def test_spacing_keeps_a_single_line_at_the_same_size(self):
        for text, width, height in (("Hello", 500, 60), ("Even if you say that...", 715, 85)):
            normal = overlay_text.explicit_caption_layout({}, text, width, height, 1500, 2100)
            for spacing in (-4, 12):
                self.assertEqual(overlay_text.explicit_caption_layout({"lineSpacing": spacing}, text, width, height, 1500, 2100), normal)

    @unittest.skipUnless(shutil.which("magick"), "Requires the runtime ImageMagick binary")
    def test_pixel_gaps_batch_isolation_and_default_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = {"region": [0, 0, 300, 250], "englishText": "LINE\nLINE\nLINE",
                     "font": overlay_text.DEFAULT_FONT, "fontSizeWidthPercent": 4,
                     "strokeWidth": 0, "fill": "black", "renderBleed": 0}
            patches = ({"lineSpacing": 12}, {}, {"lineSpacing": -4}, {"lineSpacing": 0})
            layers = [(dict(entry, **patch), root / f"batch-{index}.png", root / f"text-{index}.txt")
                      for index, patch in enumerate(patches)]
            overlay_text.render_text_layers(shutil.which("magick"), layers, 500, 500)
            gaps = []
            for index, (record, batch, _) in enumerate(layers):
                single = root / f"single-{index}.png"
                overlay_text.render_text_layer(shutil.which("magick"), record, single, root / f"single-{index}.txt", 500, 500)
                with Image.open(batch) as first, Image.open(single) as second:
                    rgba = first.convert("RGBA")
                    self.assertIsNone(ImageChops.difference(rgba, second.convert("RGBA")).getbbox(alpha_only=False))
                    alpha = rgba.getchannel("A")
                    rows = [y for y in range(alpha.height) if alpha.crop((0, y, alpha.width, y + 1)).getbbox()]
                    starts = [y for i, y in enumerate(rows) if i == 0 or y > rows[i - 1] + 1]
                    self.assertEqual(len(starts), 3)
                    gaps.append([starts[1] - starts[0], starts[2] - starts[1]])
            self.assertEqual(gaps[0], [gap + 12 for gap in gaps[1]])
            self.assertEqual(gaps[2], [gap - 4 for gap in gaps[1]])
            self.assertEqual(gaps[3], gaps[1])
            with Image.open(layers[1][1]) as absent, Image.open(layers[3][1]) as zero:
                self.assertEqual(absent.tobytes(), zero.tobytes())


if __name__ == "__main__":
    unittest.main()
