from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest import mock

from PIL import Image, ImageChops, ImageFont

import overlay_text


class FontCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.font = self.root / "live-font.ttf"
        self.old_font = Path(overlay_text.DEFAULT_FONT).with_name("IBMPlexMono-Medium.ttf")
        self.new_font = Path(overlay_text.DEFAULT_FONT)
        # Equal-size files let the tests cover replacements that preserve size
        # and modification time. FreeType permits trailing padding in a TTF.
        size = max(self.old_font.stat().st_size, self.new_font.stat().st_size)
        self.old_bytes = self.old_font.read_bytes().ljust(size, b"\0")
        self.new_bytes = self.new_font.read_bytes().ljust(size, b"\0")
        self.font.write_bytes(self.old_bytes)

    def replace_font(self, atomic=True):
        before = self.font.stat()
        target = self.root / "replacement.ttf" if atomic else self.font
        target.write_bytes(self.new_bytes)
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        if atomic:
            target.replace(self.font)
        self.assertEqual(self.font.stat().st_size, before.st_size)
        self.assertEqual(self.font.stat().st_mtime_ns, before.st_mtime_ns)

    def test_unchanged_font_reuses_loaded_face(self):
        with mock.patch.object(ImageFont, "truetype", wraps=ImageFont.truetype) as load:
            first = overlay_text.loaded_layout_font(str(self.font), 40)
            self.assertIsNotNone(first)
            self.assertIs(overlay_text.loaded_layout_font(str(self.font), 40), first)
            self.assertEqual(load.call_count, 1)

    def test_atomic_and_inplace_replacements_reload_same_size_and_mtime(self):
        for atomic in (True, False):
            with self.subTest(atomic=atomic):
                self.font.write_bytes(self.old_bytes)
                old = overlay_text.loaded_layout_font(str(self.font), 40)
                text = "A narrow line with MMMM"
                old_width = old.getlength(text)
                signature = overlay_text.font_file_signature(str(self.font))
                self.replace_font(atomic)
                new = overlay_text.loaded_layout_font(str(self.font), 40)
                self.assertNotEqual(signature, overlay_text.font_file_signature(str(self.font)))
                self.assertIsNot(old, new)
                self.assertNotEqual(old_width, new.getlength(text))
                self.assertEqual(new.getlength(text), ImageFont.truetype(str(self.new_font), 40).getlength(text))

    def test_created_repaired_and_removed_fonts_do_not_reuse_failed_or_old_faces(self):
        self.font.unlink()
        self.assertIsNone(overlay_text.loaded_layout_font(str(self.font), 40))
        self.font.write_bytes(b"not a font")
        self.assertIsNone(overlay_text.loaded_layout_font(str(self.font), 40))
        self.font.write_bytes(self.new_bytes)
        self.assertIsNotNone(overlay_text.loaded_layout_font(str(self.font), 40))
        self.font.unlink()
        self.assertIsNone(overlay_text.loaded_layout_font(str(self.font), 40))

    def entry(self, font=None, **changes):
        return {"region": [0, 0, 1427, 231],
            "englishText": "Ah, from the bottom of my heart, I love this...!",
            "font": str(font or self.font), "fill": "white", "stroke": "black",
            "strokeWidth": 10, "gravity": "center", **changes}

    def test_hot_replacement_matches_cold_layout_across_the_reported_size_range(self):
        text = self.entry()["englishText"]
        for size in range(20, 65):
            overlay_text.estimated_text_extent([text], size, str(self.font))
        self.replace_font()
        cold = self.root / "cold-font.ttf"
        cold.write_bytes(self.new_bytes)
        for percent in (None, 2.7, 2.8, 2.85, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.55, 3.6):
            with self.subTest(percent=percent):
                fields = {} if percent is None else {"fontSizeWidthPercent": percent}
                self.assertEqual(
                    overlay_text.layout_entry_text(self.entry(**fields), 1427, 2048),
                    overlay_text.layout_entry_text(self.entry(cold, **fields), 1427, 2048))

    @unittest.skipUnless(shutil.which("magick"), "Requires ImageMagick")
    def test_hot_replacement_batch_pixels_match_cold_font_and_grow_with_requested_size(self):
        text = self.entry()["englishText"]
        for size in range(20, 65):
            overlay_text.estimated_text_extent([text], size, str(self.font))
        self.replace_font()
        cold = self.root / "cold-font.ttf"
        cold.write_bytes(self.new_bytes)
        layers = []
        for index, percent in enumerate((2.7, 2.85, 3.5, 3.55, 3.6)):
            for kind, font in (("hot", self.font), ("cold", cold)):
                layers.append((self.entry(font, fontSizeWidthPercent=percent),
                    self.root / f"{kind}-{index}.png", self.root / f"{kind}-{index}.txt"))
        overlay_text.render_text_layers(shutil.which("magick"), layers, 1427, 2048)
        widths = []
        for index in range(5):
            with Image.open(self.root / f"hot-{index}.png") as hot, Image.open(self.root / f"cold-{index}.png") as cold_image:
                hot, cold_image = hot.convert("RGBA"), cold_image.convert("RGBA")
                self.assertIsNone(ImageChops.difference(hot, cold_image).getbbox(alpha_only=False))
                left, _, right, _ = hot.getchannel("A").getbbox()
                widths.append(right - left)
        self.assertTrue(all(a < b for a, b in zip(widths, widths[1:])), widths)


if __name__ == "__main__":
    unittest.main()
