from __future__ import annotations

import copy
import unittest

from fastapi import HTTPException

import editor_v2
import translate_cbz
from tests import test_editor_reset
from web_storage import write_json_atomic


class TypesettingTextTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id
        config = translate_cbz.load_config(self.manager.config.pipeline_config, None)
        for page in (0, 1):
            records = self.manager.normalize_saved_placements(self.code, self.job_id, page,
                self.manager.load_editor_v2_baseline(self.code, self.job_id, "placements", page))
            for record in records:
                record.pop("text", None)
                record.pop("englishText", None)
                record.update(font=config.render_font, fill=config.render_fill,
                              stroke=translate_cbz.outline_for_fill(config.render_fill), gravity=config.render_gravity)
            write_json_atomic(self.manager.editor_v2_generated_path(self.code, self.job_id, "placements", page), records)
            write_json_atomic(fixture.data_dir / "placements" / f"page_{page:04d}.json", records)

    def payload(self, stage="placement"):
        return self.manager.editor_v2_payload(self.code, self.job_id, stage, 0)

    def edit(self, payload, artifact, value, index=0):
        record = payload["recordsByArtifact"][artifact][index]
        return {"artifact": artifact, "recordId": record["recordId"], "boxno": record["boxno"], "value": value}

    def save(self, payload, edits):
        return self.manager.save_editor_v2_update(self.code, self.job_id, 0, "placement", {
            "baseRevision": payload["revision"], "textEdits": edits,
            "recordsByArtifact": {key: payload["recordsByArtifact"][key] for key in ("placements", "clean_boxes")},
        })

    def pending(self):
        return editor_v2.pending_changes(self.manager.load_editor_v2_manifest(self.code, self.job_id))

    def test_translation_edit_is_shared_and_only_needs_typesetting(self):
        payload = self.payload()
        original_source = copy.deepcopy(payload["recordsByArtifact"]["ocr_structured"])
        saved = self.save(payload, [self.edit(payload, "translations", "New translation")])
        self.assertEqual(saved["revision"], payload["revision"] + 1)
        self.assertEqual(self.pending(), {0: {"translation"}})
        self.assertEqual(saved["rerunFrom"], "placements")
        self.assertEqual(self.payload("text")["recordsByArtifact"]["translations"][0]["englishText"], "New translation")
        self.assertEqual(saved["recordsByArtifact"]["ocr_structured"], original_source)
        self.assertNotIn("englishText", saved["recordsByArtifact"]["placements"][0])

    def test_source_translation_and_style_save_in_one_revision(self):
        payload = self.payload()
        payload["recordsByArtifact"]["placements"][0]["rotation"] = 10
        saved = self.save(payload, [self.edit(payload, "ocr_structured", "New source"),
                                    self.edit(payload, "translations", "New translation")])
        self.assertEqual(saved["revision"], payload["revision"] + 1)
        self.assertEqual(self.pending(), {0: {"structure", "translation", "placement"}})
        self.assertEqual(saved["rerunFrom"], "alt_placement")
        self.assertEqual(self.payload("text")["recordsByArtifact"]["ocr_structured"][0]["text"], "New source")
        self.assertEqual(saved["recordsByArtifact"]["placements"][0]["rotation"], 10)
        for artifact, field in (("placements", "rotation"), ("ocr_structured", "text"), ("translations", "englishText")):
            rid = saved["recordsByArtifact"][artifact][0]["recordId"]
            self.assertTrue(saved["protection"]["records"][rid][field])

    def test_field_protection_and_revert_keep_the_placement_tab_and_other_edits(self):
        payload = self.payload()
        edits = [self.edit(payload, "ocr_structured", "New source"), self.edit(payload, "translations", "New translation")]
        saved = self.save(payload, edits)
        for edit, member, field in ((edits[0], "structure", "text"), (edits[1], "translation", "englishText")):
            self.assertTrue(saved["protection"]["records"][edit["recordId"]][field])
            for action in ("field", "revert-field"):
                saved = self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "placement", {
                    "baseRevision": saved["revision"], "action": action, "protected": False,
                    "pipelineStage": member, "artifact": edit["artifact"], "recordId": edit["recordId"], "field": field,
                })
                self.assertEqual(saved["stage"], "placement")
            self.assertEqual(saved["recordsByArtifact"][edit["artifact"]][0][field], "Generated")

    def test_reset_text_restores_shared_text_but_not_typesetting_style(self):
        payload = self.payload()
        payload["recordsByArtifact"]["placements"][0]["rotation"] = 10
        saved = self.save(payload, [self.edit(payload, "translations", "New translation")])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
        self.assertEqual(reset["recordsByArtifact"]["translations"][0]["englishText"], "Generated")
        self.assertEqual(reset["recordsByArtifact"]["placements"][0]["rotation"], 10)

    def test_invalid_patch_does_not_save_other_changes(self):
        payload = self.payload()
        valid = self.edit(payload, "ocr_structured", "New source")
        for invalid in (None, {}, [valid, valid], [{**valid, "recordId": "wrong"}], [{**valid, "value": 0}],
                        [{**valid, "value": "x" * 100001}], [{**valid, "boxno": True}]):
            with self.subTest(invalid=str(invalid)[:80]):
                with self.assertRaises(HTTPException) as error:
                    self.save(payload, invalid)
                self.assertEqual(error.exception.status_code, 400)
                self.assertEqual(self.payload()["revision"], payload["revision"])
                self.assertEqual(self.payload()["recordsByArtifact"]["ocr_structured"][0]["text"], "Generated")

    def test_stale_revision_rejects_text_and_style_together(self):
        payload = self.payload()
        self.save(payload, [self.edit(payload, "translations", "First")])
        payload["recordsByArtifact"]["placements"][0]["rotation"] = 10
        with self.assertRaises(HTTPException) as error:
            self.save(payload, [self.edit(payload, "translations", "Stale")])
        self.assertEqual(error.exception.status_code, 409)
        current = self.payload()
        self.assertEqual(current["recordsByArtifact"]["translations"][0]["englishText"], "First")
        self.assertFalse(current["recordsByArtifact"]["placements"][0].get("rotation"))

    def test_missing_translation_can_be_added_and_keeps_a_stable_id(self):
        records = self.manager.load_editor_v2_baseline(self.code, self.job_id, "translations", 0)[1:]
        write_json_atomic(self.manager.editor_v2_generated_path(self.code, self.job_id, "translations", 0), records)
        payload = self.payload()
        edit = {"artifact": "translations", "recordId": "manual-translation", "boxno": 0, "value": "Added"}
        saved = self.save(payload, [edit])
        translation = next(record for record in saved["recordsByArtifact"]["translations"] if record["boxno"] == 0)
        self.assertEqual(translation["recordId"], "manual-translation")
        saved = self.save(saved, [{**edit, "value": "Revised"}])
        self.assertEqual([(record["recordId"], record["englishText"]) for record in saved["recordsByArtifact"]["translations"]
                         if record["boxno"] == 0], [("manual-translation", "Revised")])

    def test_frozen_upstream_text_accepts_edits_without_losing_protection(self):
        payload = self.payload()
        self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "text", {
            "baseRevision": payload["revision"], "action": "freeze-stage", "protected": True,
        })
        payload = self.payload()
        saved = self.save(payload, [self.edit(payload, "ocr_structured", "Source"), self.edit(payload, "translations", "English")])
        self.assertEqual(saved["recordsByArtifact"]["ocr_structured"][0]["text"], "Source")
        self.assertEqual(saved["recordsByArtifact"]["translations"][0]["englishText"], "English")
        self.assertTrue(saved["stageStates"]["text"]["frozen"])

    def test_clean_box_only_save_does_not_promote_upstream_display_records(self):
        payload = self.payload()
        payload["recordsByArtifact"]["clean_boxes"] = [{"recordId": "clean", "boxno": 0, "region": [40, 50, 80, 90]}]
        saved = self.save(payload, [])
        self.assertEqual(self.pending(), {0: {"placement"}})
        self.assertEqual(saved["recordsByArtifact"]["ocr_structured"], payload["recordsByArtifact"]["ocr_structured"])
        self.assertEqual(saved["recordsByArtifact"]["translations"], payload["recordsByArtifact"]["translations"])
