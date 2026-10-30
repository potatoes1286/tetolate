from __future__ import annotations

import copy
import unittest
from unittest import mock

from fastapi import HTTPException

import editor_v2
from tests import test_editor_reset


class TextEditorTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id

    def payload(self, page=0):
        return self.manager.editor_v2_payload(self.code, self.job_id, "text", page)

    def save(self, payload):
        return self.manager.save_editor_v2_update(self.code, self.job_id, payload["page"], "text", {
            "baseRevision": payload["revision"], "recordsByArtifact": payload["recordsByArtifact"],
            "translationNotes": payload["translationNotes"],
        })

    def pending(self):
        return editor_v2.pending_changes(self.manager.load_editor_v2_manifest(self.code, self.job_id))

    def test_translation_only_starts_at_typesetting(self):
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Manual English"
        saved = self.save(payload)
        self.assertEqual(saved["availableStages"], ["ocr", "text", "placement"])
        self.assertEqual(saved["stage"], "text")
        self.assertEqual(self.pending(), {0: {"translation"}})
        self.assertEqual(saved["rerunFrom"], "placements")
        self.assertFalse(saved["stageStates"]["structure"]["changed"])
        self.assertFalse(saved["stageStates"]["erase"]["changed"])
        with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
            self.assertEqual(self.manager.rerun_editor_v2_page(self.code, self.job_id, "text", 0), "placements")
        rerun.assert_called_once_with(self.code, self.job_id, [0], "placements")

    def test_source_changes_start_at_erase_decisions(self):
        for field, value in (("text", "Corrected source"), ("sfx", True)):
            with self.subTest(field=field):
                payload = self.payload()
                payload["recordsByArtifact"]["ocr_structured"][0][field] = value
                saved = self.save(payload)
                self.assertEqual(saved["rerunFrom"], "alt_placement")
                self.assertIn("structure", self.pending()[0])

    def test_erase_changes_start_at_translation_and_keep_translation_edits(self):
        payload = self.payload()
        source = payload["recordsByArtifact"]["ocr_structured"][0]
        source.update(openLettering=True, safeToEraseOriginal=False)
        saved = self.save(payload)
        self.assertEqual(self.pending(), {0: {"erase"}})
        self.assertEqual(saved["rerunFrom"], "translations")
        self.assertEqual(saved["protection"]["recordStates"], {})
        self.assertEqual(saved["recordsByArtifact"]["alt_placement"][0]["recordId"],
                         payload["recordsByArtifact"]["alt_placement"][0]["recordId"])
        saved["recordsByArtifact"]["translations"][0]["englishText"] = "Keep this"
        saved = self.save(saved)
        self.assertEqual(self.pending(), {0: {"erase", "translation"}})
        self.assertEqual(saved["rerunFrom"], "translations")

    def test_combined_save_is_one_revision_and_keeps_all_dependencies(self):
        payload = self.payload()
        source = payload["recordsByArtifact"]["ocr_structured"][0]
        source.update(text="New source", openLettering=True, safeToEraseOriginal=False)
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "New English"
        payload["translationNotes"]["page"] = "New notes"
        saved = self.save(payload)
        self.assertEqual(saved["revision"], payload["revision"] + 1)
        self.assertEqual(self.pending(), {0: {"structure", "erase", "translation"}})
        self.assertEqual(saved["rerunFrom"], "alt_placement")
        self.assertEqual(saved["translationNotes"]["page"], "New notes")
        saved["recordsByArtifact"]["translations"][1]["englishText"] = "Later English edit"
        self.assertEqual(self.save(saved)["rerunFrom"], "alt_placement")

    def test_noop_and_notes_only_saves_do_not_schedule_pipeline_work(self):
        payload = self.payload()
        self.assertEqual(self.save(payload)["revision"], payload["revision"])
        self.assertEqual(self.pending(), {})
        payload["translationNotes"]["job"] = "Revised notes"
        saved = self.save(payload)
        self.assertEqual(saved["revision"], payload["revision"] + 1)
        self.assertIsNone(saved["rerunFrom"])
        self.assertEqual(saved["translationNotes"]["job"], "Revised notes")
        with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
            with self.assertRaises(HTTPException) as error:
                self.manager.rerun_editor_v2_page(self.code, self.job_id, "text", 0)
        self.assertEqual(error.exception.status_code, 400)
        rerun.assert_not_called()

    def test_different_pages_use_their_own_earliest_pass(self):
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "English"
        self.save(payload)
        payload = self.payload(1)
        payload["recordsByArtifact"]["ocr_structured"][0]["text"] = "Source"
        self.save(payload)
        with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
            pages, passes = self.manager.rerun_editor_v2_changes(self.code, self.job_id)
        self.assertEqual(pages, [0, 1])
        self.assertEqual(passes, {0: "placements", 1: "alt_placement"})
        rerun.assert_called_once_with(self.code, self.job_id, [0, 1], page_resume_from=passes)

    def test_reordering_keeps_record_ids_and_translation_links(self):
        payload = self.payload()
        original = copy.deepcopy(payload["recordsByArtifact"]["translations"])
        payload["recordsByArtifact"]["ocr_structured"].reverse()
        saved = self.save(payload)
        self.assertEqual([record["boxno"] for record in saved["recordsByArtifact"]["ocr_structured"]], [1, 0])
        self.assertEqual(saved["recordsByArtifact"]["translations"], original)
        self.assertEqual(saved["rerunFrom"], "alt_placement")

    def test_reject_removes_source_and_related_translations(self):
        payload = self.payload()
        payload["recordsByArtifact"]["ocr_structured"].pop(0)
        saved = self.save(payload)
        self.assertEqual([record["boxno"] for record in saved["recordsByArtifact"]["ocr_structured"]], [1])
        self.assertEqual([record["boxno"] for record in saved["recordsByArtifact"]["translations"]], [1])
        self.assertEqual(saved["rerunFrom"], "alt_placement")

    def test_protection_and_reset_cover_the_combined_tab(self):
        payload = self.payload()
        payload["recordsByArtifact"]["ocr_structured"][0]["text"] = "Source"
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "English"
        saved = self.save(payload)
        protected = self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "text", {
            "baseRevision": saved["revision"], "action": "freeze-stage", "protected": True,
        })
        self.assertTrue(protected["stageStates"]["text"]["frozen"])
        for member in editor_v2.TEXT_STAGES:
            self.assertTrue(protected["stageStates"][member]["frozen"])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": protected["revision"]})
        self.assertFalse(reset["stageStates"]["text"]["frozen"])
        self.assertFalse(reset["stageStates"]["text"]["changed"])
        self.assertEqual(reset["recordsByArtifact"]["translations"][0]["englishText"], "Generated")

    def test_translation_field_can_be_unprotected_and_reverted_in_combined_tab(self):
        payload = self.payload()
        translation = payload["recordsByArtifact"]["translations"][0]
        translation["englishText"] = "Manual"
        saved = self.save(payload)
        for action in ("field", "revert-field"):
            saved = self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "text", {
                "baseRevision": saved["revision"], "action": action, "protected": False,
                "pipelineStage": "translation", "artifact": "translations",
                "recordId": translation["recordId"], "field": "englishText",
            })
        self.assertEqual(saved["recordsByArtifact"]["translations"][0]["englishText"], "Generated")
        self.assertEqual(saved["rerunFrom"], "placements")

    def test_frozen_text_can_be_edited_without_losing_protection(self):
        payload = self.payload()
        frozen = self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "text", {
            "baseRevision": payload["revision"], "action": "freeze-stage", "protected": True,
        })
        self.assertIsNone(frozen["rerunFrom"])
        frozen["recordsByArtifact"]["translations"][0]["englishText"] = "Edited while frozen"
        saved = self.save(frozen)
        self.assertEqual(saved["recordsByArtifact"]["translations"][0]["englishText"], "Edited while frozen")
        self.assertTrue(saved["stageStates"]["text"]["frozen"])
        self.assertEqual(saved["rerunFrom"], "placements")

    def test_reset_unprocessed_translation_clears_out_of_date_flags(self):
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Edited"
        saved = self.save(payload)
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
        self.assertEqual(reset["recordsByArtifact"]["translations"][0]["englishText"], "Generated")
        self.assertEqual(self.pending(), {})
        self.assertIsNone(reset["rerunFrom"])
        for state in reset["stageStates"].values():
            self.assertFalse(state["stale"])
            self.assertFalse(state["pending"])

    def test_reset_ocr_clears_text_and_typesetting_tags(self):
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "ocr", 0)
        artifacts = payload["recordsByArtifact"]
        artifacts["ocr_raw"][0]["text"] = "Changed OCR"
        artifacts["ocr_merged"][0]["text"] = "Changed OCR"
        saved = self.manager.save_editor_v2_update(self.code, self.job_id, 0, "ocr", {
            "baseRevision": payload["revision"], "recordsByArtifact": artifacts,
        })
        self.assertTrue(saved["stageStates"]["text"]["stale"])
        self.assertTrue(saved["stageStates"]["placement"]["stale"])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "ocr", {"baseRevision": saved["revision"]})
        self.assertEqual(self.pending(), {})
        self.assertFalse(reset["stageStates"]["text"]["stale"])
        self.assertFalse(reset["stageStates"]["placement"]["stale"])

    def test_reset_combined_text_clears_its_internal_stale_flags(self):
        payload = self.payload()
        payload["recordsByArtifact"]["ocr_structured"][0].update(text="Source", openLettering=True, safeToEraseOriginal=False)
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "English"
        saved = self.save(payload)
        self.assertTrue(saved["stageStates"]["text"]["stale"])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
        self.assertEqual(self.pending(), {})
        self.assertTrue(all(not state["stale"] for state in reset["stageStates"].values()))

    def test_reset_text_does_not_clear_pending_ocr_or_other_pages(self):
        payload = self.payload(1)
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Other page"
        self.save(payload)
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        editor_v2.mark_saved(manifest, 0, "ocr")
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "This page"
        saved = self.save(payload)
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
        self.assertEqual(self.pending(), {0: {"ocr"}, 1: {"translation"}})
        self.assertTrue(reset["stageStates"]["text"]["stale"])
        self.assertTrue(reset["stageStates"]["placement"]["stale"])

    def test_reset_clears_stuck_flags_when_overrides_are_already_gone(self):
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        editor_v2.mark_stages_saved(manifest, 0, editor_v2.TEXT_STAGES)
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": manifest["revision"]})
        self.assertEqual(self.pending(), {})
        self.assertTrue(all(not state["stale"] for state in reset["stageStates"].values()))

    def test_reset_processed_edits_still_requires_output_regeneration(self):
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Already rendered"
        self.save(payload)
        self.manager.clear_editor_changes_for_pages(self.code, self.job_id, [0], "placements")
        payload = self.payload()
        self.assertEqual(self.pending(), {})
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": payload["revision"]})
        self.assertEqual(self.pending(), {0: {"translation"}})
        self.assertTrue(reset["stageStates"]["placement"]["stale"])

    def test_reset_after_reediting_processed_data_keeps_real_staleness(self):
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Already rendered"
        self.save(payload)
        self.manager.clear_editor_changes_for_pages(self.code, self.job_id, [0], "placements")
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Unprocessed edit"
        saved = self.save(payload)
        for _ in range(2):
            saved = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
            self.assertEqual(self.pending(), {0: {"translation"}})
            self.assertTrue(saved["stageStates"]["placement"]["stale"])

    def test_invalid_and_stale_saves_leave_the_manifest_unchanged(self):
        path = self.manager.editor_v2_manifest_path(self.code, self.job_id)
        original = path.read_bytes()
        for body, code in (({}, 400), ({"baseRevision": 99}, 409),
                           ({"baseRevision": 0, "recordsByArtifact": {"ocr_structured": [], "translations": [None]}}, 400)):
            with self.assertRaises(HTTPException) as error:
                self.manager.save_editor_v2_update(self.code, self.job_id, 0, "text", body)
            self.assertEqual(error.exception.status_code, code)
            self.assertEqual(path.read_bytes(), original)

    def test_pending_ocr_placeholders_are_not_saved_as_manual_edits(self):
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        raw = self.manager.load_editor_v2_baseline(self.code, self.job_id, "ocr_raw", 0)
        editor_v2.update_artifact_override(manifest, 0, "ocr", "ocr_raw", raw,
            [*raw, {"page": 0, "boxno": 2, "text": "New OCR", "region": [50, 50, 70, 70]}])
        baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, "ocr_merged", 0)
        merged = [*baseline, {"page": 0, "boxno": 2, "sourceBoxnos": [2], "text": "New OCR",
                             "region": [50, 50, 70, 70]}]
        editor_v2.update_artifact_override(manifest, 0, "ocr", "ocr_merged", baseline, merged)
        editor_v2.mark_saved(manifest, 0, "ocr")
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
        payload = self.payload()
        self.assertEqual(len(payload["recordsByArtifact"]["ocr_structured"]), 3)
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Only this edit"
        saved = self.save(payload)
        self.assertEqual(self.pending(), {0: {"ocr", "translation"}})
        self.assertFalse(saved["stageStates"]["structure"]["changed"])
        self.assertFalse(saved["stageStates"]["erase"]["changed"])
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        actual = self.manager.editor_v2_effective_records(self.code, self.job_id, manifest, "translations", 0)
        self.assertEqual([record["boxno"] for record in actual], [0, 1])

    def test_retranslation_cannot_skip_pending_source_changes(self):
        payload = self.payload()
        payload["recordsByArtifact"]["ocr_structured"][0]["text"] = "New source"
        self.save(payload)
        with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
            with self.assertRaises(HTTPException) as error:
                self.manager.retranslate_editor_v2_page(self.code, self.job_id, 0)
        self.assertEqual(error.exception.status_code, 409)
        rerun.assert_not_called()


if __name__ == "__main__":
    unittest.main()
