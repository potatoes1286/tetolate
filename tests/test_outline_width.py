from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image, ImageChops

import overlay_text


class OutlineWidthTests(unittest.TestCase):
    def entry(self, **changes):
        return {"region": [0, 0, 320, 180], "englishText": "NO OUTLINE",
                "font": overlay_text.DEFAULT_FONT, "fontSizeWidthPercent": 4,
                "fill": "white", "stroke": "black", "strokeWidth": 0,
                "renderBleed": 0, **changes}

    def test_zero_disables_auto_outline_without_changing_other_widths(self):
        for height in (None, 120, 2100, 10000):
            with self.subTest(height=height):
                self.assertEqual(overlay_text.stroke_width({"strokeWidth": 0}, height), 0)
                self.assertEqual(overlay_text.stroke_width({"strokeWidth": 0.0}, height), 0)
                default = overlay_text.scaled_page_stroke_width(height)
                self.assertEqual(overlay_text.stroke_width({}, height), default)
                self.assertEqual(overlay_text.stroke_width({"strokeWidth": 1}, height), max(1, default))
                self.assertEqual(overlay_text.stroke_width({"strokeWidth": 20}, height), max(20, default))

    def test_zero_does_not_paint_or_composite_an_outline_layer(self):
        with tempfile.TemporaryDirectory() as temporary:
            text = Path(temporary) / "text.txt"
            args = overlay_text.text_layer_args(self.entry(), text, 800, 2100)
            self.assertNotIn("-strokewidth", args)
            self.assertNotIn("black", args)
            self.assertNotIn("-composite", args)
            self.assertEqual(args.count("-fill"), 1)
            self.assertEqual(args[args.index("-stroke") + 1], "none")
            outlined = overlay_text.text_layer_args(self.entry(strokeWidth=3), text, 800, 2100)
            self.assertIn("-strokewidth", outlined)
            self.assertIn("-composite", outlined)

    @unittest.skipUnless(shutil.which("magick"), "Requires ImageMagick")
    def test_zero_has_no_outline_pixels_and_matches_serial_in_mixed_batch(self):
        magick = shutil.which("magick")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [self.entry(strokeWidth=5), self.entry(),
                       self.entry(fill="black", stroke="white"), self.entry(strokeWidth=3), self.entry()]
            layers = [(record, root / f"batch-{i}.png", root / f"text-{i}.txt")
                      for i, record in enumerate(records)]
            overlay_text.render_text_layers(magick, layers, 800, 2100)
            for index, (record, batch, _) in enumerate(layers):
                with self.subTest(index=index):
                    single = root / f"single-{index}.png"
                    overlay_text.render_text_layer(magick, record, single, root / f"single-{index}.txt", 800, 2100)
                    with Image.open(batch) as image, Image.open(single) as reference:
                        rgba = image.convert("RGBA")
                        self.assertIsNone(ImageChops.difference(rgba, reference.convert("RGBA")).getbbox(alpha_only=False))
                        pixels = [pixel[:3] for pixel in rgba.getdata() if pixel[3] > 0]
                        self.assertTrue(pixels)
                        if record["strokeWidth"] == 0:
                            color = (255, 255, 255) if record["fill"] == "white" else (0, 0, 0)
                            self.assertEqual(set(pixels), {color})
                        else:
                            self.assertIn((0, 0, 0), pixels)


if __name__ == "__main__":
    unittest.main()
