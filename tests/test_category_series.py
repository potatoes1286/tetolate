from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

from fastapi import HTTPException
from PIL import Image

from tests import test_regressions
import translate_cbz
import web_app
import web_pages
from web_comicinfo import metadata_archive, parse_common_xml


class CategorySeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = test_regressions.WebAuthenticationTests().initialized_manager(self.root, ["default"])
        self.manager.create_category("books", is_series=True)
        self.events = []
        self.addCleanup(self.finish)

    def finish(self):
        for event in self.events:
            event.set()
        for thread in list(self.manager._export_threads):
            thread.join(15)
            self.assertFalse(thread.is_alive())

    def chapter(self, chapter="1", volume="", color="red"):
        job = self.manager.create_job_id("books")
        self.manager.job_dir("books", job).mkdir(parents=True)
        image = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(image, format="PNG")
        with zipfile.ZipFile(self.manager.input_path("books", job), "w") as archive:
            archive.writestr("page.png", image.getvalue())
            archive.writestr("ComicInfo.xml", "<ComicInfo><Title>Original title</Title><Writer>Original writer</Writer><LanguageISO>ja</LanguageISO></ComicInfo>")
            archive.writestr("credits.txt", "Preserve credits")
        self.manager.submit_job("books", job, f"chapter-{chapter}.cbz", chapter=chapter, volume=volume, enqueue_job=False)
        status = self.manager.load_status("books", job)
        status.update(status="complete", jobName=f"Chapter {chapter}", autoTranslateComicInfoTitle=False)
        self.manager.save_status("books", job, status)
        directory = self.manager.final_pages_dir("books", job)
        directory.mkdir(parents=True)
        (directory / "000.png").write_bytes(image.getvalue())
        return job

    def complete_export(self, variant="png", grouping="chapters"):
        with mock.patch.object(self.manager, "request_vlm_title", side_effect=AssertionError("Export must not call VLM")):
            self.manager.generate_category_export("books", variant, grouping)
            self.finish()
        state = self.manager.category_export_status("books")
        self.assertEqual(state["status"], "complete", state)
        return self.manager.require_category_export("books")

    def test_category_flag_ui_and_decimal_volume_order(self):
        self.assertFalse(self.manager.is_series("default"))
        first = self.chapter("10", "1")
        second = self.chapter("1.5", "1")
        third = self.chapter("0", "2")
        self.assertEqual([x["jobId"] for x in self.manager.series_chapters("books")], [second, first, third])
        with self.assertRaises(HTTPException):
            self.manager.validate_series_chapter("books", "01.50", "01")
        with self.assertRaises(HTTPException):
            self.manager.validate_series_chapter("default", "1")
        markup = web_pages.admin_dashboard_page(self.manager).body.decode()
        self.assertIn('(series)</span>', markup)
        self.assertIn('name="is_series"', markup)
        self.assertIn('title="A series category', markup)
        data = self.manager.public_category_jobs("books")
        markup = web_pages.category_jobs_page("books", data).body.decode()
        for label in ("Vol. 1 Ch. 1.5 Chapter 1.5", "Common ComicInfo XML", "Export all CBZ", "Volumes", "Individual chapters"):
            self.assertIn(label, markup)
        self.assertNotIn('name="series_id"', markup)
        self.assertIn('name="volume"', markup)
        self.assertIn('name="chapter"', markup)

    def test_common_xml_validation_merging_and_metadata_invalidation(self):
        job = self.chapter("1.5", "2")
        xml = "<ComicInfo><Writer>Common writer</Writer><Title>Ignored title</Title><Number>99</Number></ComicInfo>"
        self.manager.update_series("books", "Example", "", "", xml)
        output = self.root / "metadata.cbz"
        metadata_archive(self.manager.input_path("books", job), output, xml,
            series="Example", volume="2", chapter="1.5", title="Chapter title", page_count=1)
        with zipfile.ZipFile(output) as archive:
            root = ET.fromstring(archive.read("ComicInfo.xml"))
            self.assertEqual(archive.read("credits.txt"), b"Preserve credits")
            self.assertEqual(root.findtext("Writer"), "Common writer")
            self.assertEqual(root.findtext("LanguageISO"), "ja")
            self.assertEqual(root.findtext("Title"), "Chapter title")
            self.assertEqual(root.findtext("Number"), "1.5")
            self.assertEqual(root.findtext("Volume"), "2")
            self.assertEqual(root.findtext("Series"), "Example")
        for bad in ("<Other/>", "<ComicInfo>", '<!DOCTYPE ComicInfo [<!ENTITY x "bad">]><ComicInfo>&x;</ComicInfo>', "x" * 100001):
            with self.subTest(bad=bad[:30]), self.assertRaises(HTTPException):
                parse_common_xml(bad)
        self.complete_export()
        self.manager.update_series("books", "Example", "", "", "<ComicInfo><Writer>New</Writer></ComicInfo>")
        self.assertFalse(self.manager.category_export_path("books").exists())
        self.assertEqual(self.manager.category_export_status("books")["status"], "cancelled")

    def test_volume_export_combines_ordered_pages_and_keeps_unassigned_chapters_separate(self):
        self.chapter("2", "1", "blue")
        self.chapter("1.5", "1", "red")
        self.chapter("3", "", "green")
        self.manager._paused = True
        self.manager._active_processes[("default", "otherjob")] = object()
        path = self.complete_export(grouping="volumes")
        markup = web_pages.category_jobs_page("books", self.manager.public_category_jobs("books")).body.decode()
        self.assertIn('<option value="volumes" selected>', markup)
        self.assertIn('window.location.hash === "#series-settings"', markup)
        with zipfile.ZipFile(path) as collection:
            self.assertEqual(len(collection.namelist()), 2)
            name = next(n for n in collection.namelist() if "Vol. 1" in n)
            with zipfile.ZipFile(io.BytesIO(collection.read(name))) as volume:
                pages = sorted(n for n in volume.namelist() if n.endswith(".png"))
                self.assertEqual(len(pages), 2)
                colors = [Image.open(io.BytesIO(volume.read(n))).convert("RGB").getpixel((0, 0)) for n in pages]
                self.assertEqual(colors, [(255, 0, 0), (0, 0, 255)])
                xml = ET.fromstring(volume.read("ComicInfo.xml"))
                self.assertEqual(xml.findtext("Volume"), "1")
                self.assertEqual(xml.findtext("PageCount"), "2")
                self.assertEqual(xml.findtext("Number"), "")

    def test_chapter_export_and_single_export_include_shared_xml(self):
        job = self.chapter("1.5", "2")
        self.manager.update_series("books", "Example", "", "", "<ComicInfo><Publisher>Common publisher</Publisher></ComicInfo>")
        path = self.complete_export()
        with zipfile.ZipFile(path) as collection:
            names = collection.namelist()
            self.assertEqual(len(names), 1)
            self.assertTrue(names[0].startswith("Vol. 2 Ch. 1.5"))
            with zipfile.ZipFile(io.BytesIO(collection.read(names[0]))) as archive:
                self.assertEqual(ET.fromstring(archive.read("ComicInfo.xml")).findtext("Publisher"), "Common publisher")
        self.manager.generate_download("books", job, "png")
        self.finish()
        with zipfile.ZipFile(self.manager.translated_cbz_path("books", job)) as archive:
            self.assertEqual(ET.fromstring(archive.read("ComicInfo.xml")).findtext("Publisher"), "Common publisher")

    def test_export_rejects_duplicate_and_discards_changed_inputs(self):
        job = self.chapter()
        entered, release = threading.Event(), threading.Event()
        self.events.append(release)
        original = translate_cbz.print_packaged_cbz
        def block(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release export")
            original(*args, **kwargs)
        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=block):
            self.manager.generate_category_export("books", "png", "chapters")
            self.assertTrue(entered.wait(3))
            with self.assertRaises(HTTPException):
                self.manager.generate_category_export("books", "png", "chapters")
            self.manager.invalidate_translated_cbz_archives("books", job)
            release.set()
            self.finish()
        self.assertFalse(self.manager.category_export_path("books").exists())
        self.assertEqual(self.manager.category_export_status("books")["status"], "cancelled")

    def test_category_rename_waits_for_collection_export(self):
        self.chapter()
        entered, release = threading.Event(), threading.Event()
        self.events.append(release)
        original = translate_cbz.print_packaged_cbz

        def block(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Test did not release export")
            original(*args, **kwargs)

        with mock.patch.object(translate_cbz, "print_packaged_cbz", side_effect=block):
            self.manager.generate_category_export("books", "png", "chapters")
            self.assertTrue(entered.wait(5))
            with self.assertRaises(HTTPException) as caught:
                self.manager.rename_category("books", "renamed")
            self.assertEqual(caught.exception.status_code, 409)
            self.manager.create_category("unrelated")
            self.manager.rename_category("unrelated", "other")
            release.set()
            self.finish()
        self.assertEqual(self.manager.category_export_status("books")["status"], "complete")
        self.manager.rename_category("books", "renamed")
        self.assertTrue(self.manager.require_category_export("renamed").is_file())

    def test_collection_download_checks_for_stale_pages(self):
        job = self.chapter()
        self.complete_export()
        Image.new("RGB", (10, 10)).save(self.manager.final_pages_dir("books", job) / "000.png")
        with self.assertRaises(HTTPException):
            self.manager.require_category_export("books")
        self.assertFalse(self.manager.category_export_path("books").exists())

    def test_long_series_names_keep_distinct_volume_archive_names(self):
        self.chapter("1", "1")
        self.chapter("1", "2")
        self.manager.update_series("books", "Long" * 60, "", "")
        path = self.complete_export(grouping="volumes")
        with zipfile.ZipFile(path) as collection:
            self.assertEqual(len(set(collection.namelist())), 2)
            self.assertTrue(any(name.startswith("Vol. 1 ") for name in collection.namelist()))
            self.assertTrue(any(name.startswith("Vol. 2 ") for name in collection.namelist()))

    @unittest.skipUnless(shutil.which("magick"), "ImageMagick 7 is required")
    def test_converted_collection_formats(self):
        job = self.chapter()
        Image.effect_noise((512, 512), 100).convert("RGB").save(self.manager.final_pages_dir("books", job) / "001.png")
        for variant, suffix in (("webp", ".webp"), ("jxl", ".jxl")):
            with self.subTest(variant=variant):
                path = self.complete_export(variant)
                with zipfile.ZipFile(path) as collection:
                    with zipfile.ZipFile(io.BytesIO(collection.read(collection.namelist()[0]))) as archive:
                        self.assertTrue(any(n.endswith(suffix) for n in archive.namelist()))
