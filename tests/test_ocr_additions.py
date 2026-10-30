from __future__ import annotations

import copy
import json
import unittest
from unittest import mock

import editor_v2
import web_app
from tests import test_editor_reset
from web_storage import write_json_atomic


class OcrAdditionTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id
        self.data_dir = fixture.data_dir

    def payload(self, stage="ocr"):
        return self.manager.editor_v2_payload(self.code, self.job_id, stage, 0)

    def save(self, payload, deleted_groups=None):
        return self.manager.save_editor_v2_update(self.code, self.job_id, 0, payload["stage"], {
            "baseRevision": payload["revision"], "recordsByArtifact": payload["recordsByArtifact"],
            "deletedOcrGroupIds": deleted_groups or [],
        })

    def old_orphans(self):
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, "ocr_raw", 0)
        raw = [*baseline, *[{"page": 0, "boxno": box, "recordId": f"raw-missed-{box}",
                            "ocrSource": "manual", "text": f"Missed {box}", "region": [50, 60, 80, 90]}
                           for box in (8, 9)]]
        editor_v2.update_artifact_override(manifest, 0, "ocr", "ocr_raw", baseline, raw)
        editor_v2.mark_saved(manifest, 0, "ocr")
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
        write_json_atomic(self.data_dir / "ocr_raw/page_0000.json", raw)

    def save_placements(self, payload):
        # The shared fixture uses a made-up generated font. Use the configured
        # backup font so this test exercises normal placement validation.
        for record in payload["recordsByArtifact"]["placements"]:
            record.pop("font", None)
        return self.save(payload)

    def test_moving_pending_typesetting_boxes_does_not_create_more_boxes_on_tab_changes(self):
        self.old_orphans()
        original = self.payload("placement")
        expected_sources = {tuple(record["sourceBoxnos"]): (record["boxno"], record["recordId"])
                            for record in self.missed(original["recordsByArtifact"]["ocr_structured"])}
        expected_placements = {record["boxno"]: record["recordId"]
                               for record in original["recordsByArtifact"]["placements"]}
        for iteration in range(4):
            payload = self.payload("placement")
            for source in self.missed(payload["recordsByArtifact"]["ocr_structured"]):
                placement = next(record for record in payload["recordsByArtifact"]["placements"]
                                 if record["boxno"] == source["boxno"])
                placement["placementRegion"] = [5 + iteration, 10, 25 + iteration, 35]
            saved = self.save_placements(payload)
            for stage in ("ocr", "text", "placement"):
                records = self.payload(stage)["recordsByArtifact"]
                self.assertEqual(len(records["placements"]), 4)
                self.assertEqual(len(records["ocr_structured"]), 4)
                self.assertEqual({tuple(record["sourceBoxnos"]): (record["boxno"], record["recordId"])
                                  for record in self.missed(records["ocr_structured"])}, expected_sources)
                self.assertEqual({record["boxno"]: record["recordId"] for record in records["placements"]}, expected_placements)
                for record in records["placements"]:
                    if record["boxno"] in {value[0] for value in expected_sources.values()}:
                        self.assertEqual(record["placementRegion"], [5 + iteration, 10, 25 + iteration, 35])
            self.assertEqual(saved["rerunFrom"], "ocr_structured")
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.assertEqual(len(self.manager.editor_v2_effective_records(self.code, self.job_id, manifest, "ocr_structured", 0)), 2)
        self.assertEqual(editor_v2.pending_changes(manifest)[0], {"ocr", "placement"})

    def test_editing_only_a_pending_translation_keeps_its_source_identity(self):
        self.old_orphans()
        payload = self.payload("text")
        source = self.missed(payload["recordsByArtifact"]["ocr_structured"])[0]
        translation = next(record for record in payload["recordsByArtifact"]["translations"] if record["boxno"] == source["boxno"])
        translation["englishText"] = "Manual translation"
        self.save(payload)
        for stage in ("text", "placement", "ocr", "placement"):
            records = self.payload(stage)["recordsByArtifact"]
            repeated = next(record for record in self.missed(records["ocr_structured"])
                            if record["sourceBoxnos"] == source["sourceBoxnos"])
            self.assertEqual((repeated["boxno"], repeated["recordId"]), (source["boxno"], source["recordId"]))
            self.assertEqual(next(record for record in records["translations"] if record["boxno"] == source["boxno"])["englishText"], "Manual translation")
            self.assertEqual(len(records["placements"]), 4)

    def test_newly_drawn_box_keeps_moved_placement_after_reload_and_reset(self):
        payload = self.payload()
        payload["recordsByArtifact"]["ocr_raw"].append({"page": 0, "boxno": 2, "recordId": "raw-drawn",
            "ocrSource": "manual", "region": [50, 60, 80, 90], "text": ""})
        payload["recordsByArtifact"]["ocr_merged"].append({"page": 0, "boxno": 2, "recordId": "group-drawn",
            "ocrSource": "manual", "region": [50, 60, 80, 90], "sourceBoxnos": [2], "sourceTexts": [""], "text": ""})
        self.save(payload)
        payload = self.payload("placement")
        source = next(record for record in payload["recordsByArtifact"]["ocr_structured"] if record["sourceBoxnos"] == [2])
        placement = next(record for record in payload["recordsByArtifact"]["placements"] if record["boxno"] == source["boxno"])
        rid = placement["recordId"]
        placement["placementRegion"] = [5, 5, 20, 25]
        saved = self.save_placements(payload)
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        for stage in ("ocr", "text", "placement"):
            records = restarted.editor_v2_payload(self.code, self.job_id, stage, 0)["recordsByArtifact"]
            self.assertEqual(len(records["placements"]), 3)
            moved = next(record for record in records["placements"] if record["recordId"] == rid)
            self.assertEqual(moved["placementRegion"], [5, 5, 20, 25])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "placement", {"baseRevision": saved["revision"]})
        self.assertEqual(len(reset["recordsByArtifact"]["placements"]), 3)
        restored = next(record for record in reset["recordsByArtifact"]["placements"] if record["recordId"] == rid)
        self.assertEqual(restored["placementRegion"], [50, 60, 80, 90])

    def test_pending_box_identity_survives_frozen_typesetting_and_collision_with_generated_box(self):
        baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, "placements", 0)
        extra = {"page": 0, "boxno": 2, "placementRegion": [30, 30, 45, 50], "recordId": "generated-extra"}
        write_json_atomic(self.manager.editor_v2_generated_path(self.code, self.job_id, "placements", 0), [*baseline, extra])
        self.old_orphans()
        payload = self.payload("placement")
        source = self.missed(payload["recordsByArtifact"]["ocr_structured"])[0]
        self.assertNotEqual(source["boxno"], 2)
        moved = next(record for record in payload["recordsByArtifact"]["placements"] if record["boxno"] == source["boxno"])
        moved["placementRegion"] = [5, 5, 20, 25]
        saved = self.save_placements(payload)
        frozen = self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "placement", {
            "baseRevision": saved["revision"], "action": "freeze-stage", "protected": True})
        for stage in ("ocr", "text", "placement"):
            records = self.payload(stage)["recordsByArtifact"]
            self.assertEqual(len(records["placements"]), 5)
            current = next(record for record in records["placements"] if record["recordId"] == moved["recordId"])
            self.assertEqual(current["placementRegion"], [5, 5, 20, 25])
            self.assertEqual(next(record for record in records["placements"] if record["recordId"] == "generated-extra")["placementRegion"], [30, 30, 45, 50])
        self.assertTrue(frozen["stageStates"]["placement"]["frozen"])

    def test_loading_moved_pending_boxes_is_read_only_and_keeps_clean_boxes(self):
        self.old_orphans()
        payload = self.payload("placement")
        source = self.missed(payload["recordsByArtifact"]["ocr_structured"])[0]
        next(record for record in payload["recordsByArtifact"]["placements"] if record["boxno"] == source["boxno"])["placementRegion"] = [5, 5, 20, 25]
        payload["recordsByArtifact"]["clean_boxes"] = [{"page": 0, "boxno": 0, "recordId": "clean", "region": [70, 80, 90, 100]}]
        saved = self.save_placements(payload)
        paths = [self.manager.editor_v2_manifest_path(self.code, self.job_id),
                 *(self.data_dir / artifact / "page_0000.json" for artifact in editor_v2.ARTIFACT_STAGES)]
        before = {path: path.read_bytes() for path in paths}
        for stage in ("ocr", "text", "placement", "placement"):
            records = self.payload(stage)["recordsByArtifact"]
            self.assertEqual(len(records["placements"]), 4)
            self.assertEqual(records["clean_boxes"], saved["recordsByArtifact"]["clean_boxes"])
        self.assertEqual({path: path.read_bytes() for path in paths}, before)

    @staticmethod
    def missed(records):
        return [record for record in records if any(source in (8, 9) for source in record.get("sourceBoxnos", []))]

    def test_already_saved_orphans_appear_in_both_later_tabs_without_file_changes(self):
        self.old_orphans()
        paths = [self.manager.editor_v2_manifest_path(self.code, self.job_id),
                 *(self.data_dir / artifact / "page_0000.json" for artifact in editor_v2.ARTIFACT_STAGES)]
        before = {path: path.read_bytes() for path in paths}
        ids = None
        for stage in ("ocr", "text", "placement", "text"):
            payload = self.payload(stage)
            artifacts = payload["recordsByArtifact"]
            groups = self.missed(artifacts["ocr_merged"])
            structured = self.missed(artifacts["ocr_structured"])
            self.assertEqual(len(groups), 2)
            self.assertEqual({record["text"] for record in structured}, {"Missed 8", "Missed 9"})
            self.assertEqual(len(structured), 2)
            for source in structured:
                self.assertTrue(any(record["boxno"] == source["boxno"] for record in artifacts["placements"]))
                translation = next(record for record in artifacts["translations"] if record["boxno"] == source["boxno"])
                self.assertEqual(translation["englishText"], "")
            current_ids = [record["recordId"] for record in groups]
            if ids is not None:
                self.assertEqual(current_ids, ids)
            ids = current_ids
        self.assertEqual({path: path.read_bytes() for path in paths}, before)

    def test_saved_groups_feed_regeneration_and_preserve_manual_translation(self):
        self.old_orphans()
        payload = self.payload("text")
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Keep this manual edit"
        saved = self.save(payload)
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.assertEqual(editor_v2.pending_changes(manifest)[0], {"ocr", "translation"})
        merged = json.loads((self.data_dir / "ocr_merged/page_0000.json").read_text())
        self.assertEqual(len(self.missed(merged)), 2)
        with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
            self.assertEqual(self.manager.rerun_editor_v2_page(self.code, self.job_id, "text", 0), "ocr_structured")
        rerun.assert_called_once_with(self.code, self.job_id, [0], "ocr_structured")
        self.assertEqual(saved["recordsByArtifact"]["translations"][0]["englishText"], "Keep this manual edit")
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        repeated = restarted.editor_v2_payload(self.code, self.job_id, "text", 0)
        self.assertEqual(len(self.missed(repeated["recordsByArtifact"]["ocr_structured"])), 2)

    def test_deleting_a_derived_group_does_not_recreate_it(self):
        self.old_orphans()
        payload = self.payload()
        deleted = [group["recordId"] for group in payload["recordsByArtifact"]["ocr_merged"] if 8 in group.get("sourceBoxnos", [])]
        payload["recordsByArtifact"]["ocr_merged"] = [group for group in payload["recordsByArtifact"]["ocr_merged"]
            if 8 not in group.get("sourceBoxnos", [])]
        self.save(payload, deleted)
        for stage in ("ocr", "text", "placement"):
            artifacts = self.payload(stage)["recordsByArtifact"]
            self.assertIn(8, [record["boxno"] for record in artifacts["ocr_raw"]])
            self.assertEqual([record["sourceBoxnos"] for record in self.missed(artifacts["ocr_merged"])], [[9]])
            self.assertEqual([record["sourceBoxnos"] for record in self.missed(artifacts["ocr_structured"])], [[9]])

    def test_reject_and_reset_restore_pending_manual_descendants(self):
        self.old_orphans()
        payload = self.payload("text")
        payload["recordsByArtifact"]["ocr_structured"] = [record for record in payload["recordsByArtifact"]["ocr_structured"]
            if 8 not in record.get("sourceBoxnos", [])]
        saved = self.save(payload)
        self.assertEqual([record["sourceBoxnos"] for record in self.missed(saved["recordsByArtifact"]["ocr_structured"])], [[9]])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "text", {"baseRevision": saved["revision"]})
        self.assertEqual(len(self.missed(reset["recordsByArtifact"]["ocr_structured"])), 2)
        self.assertEqual(len(self.missed(reset["recordsByArtifact"]["ocr_merged"])), 2)

    def test_deleted_baseline_manual_group_returns_only_after_reset(self):
        baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, "ocr_raw", 0)
        manual = {"page": 0, "boxno": 8, "recordId": "raw-missed-8", "ocrSource": "manual",
                  "text": "Missed 8", "region": [50, 60, 80, 90]}
        write_json_atomic(self.manager.editor_v2_generated_path(self.code, self.job_id, "ocr_raw", 0), [*baseline, manual])
        payload = self.payload()
        deleted = [group["recordId"] for group in payload["recordsByArtifact"]["ocr_merged"] if 8 in group.get("sourceBoxnos", [])]
        payload["recordsByArtifact"]["ocr_merged"] = [record for record in payload["recordsByArtifact"]["ocr_merged"]
            if 8 not in record.get("sourceBoxnos", [])]
        saved = self.save(payload, deleted)
        self.assertEqual(self.missed(saved["recordsByArtifact"]["ocr_merged"]), [])
        reset = self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, "ocr", {"baseRevision": saved["revision"]})
        self.assertEqual(len(self.missed(reset["recordsByArtifact"]["ocr_structured"])), 1)

    def test_singleton_creation_keeps_existing_groups_and_ignores_non_manual_orphans(self):
        payload = self.payload()
        groups_before = copy.deepcopy(payload["recordsByArtifact"]["ocr_merged"])
        payload["recordsByArtifact"]["ocr_raw"] += [
            {"boxno": 8, "recordId": "raw-manual-8", "region": [50, 60, 80, 90], "text": "Manual", "ocrSource": "manual"},
            {"boxno": 9, "recordId": "raw-other-9", "region": [50, 60, 80, 90], "text": "Other"}]
        saved = self.save(payload)
        groups = saved["recordsByArtifact"]["ocr_merged"]
        for original, group in zip(groups_before, groups):
            self.assertEqual(original["recordId"], group["recordId"])
            self.assertEqual(original["sourceBoxnos"], group["sourceBoxnos"])
        self.assertEqual([record["sourceBoxnos"] for record in self.missed(groups)], [[8]])

    def test_old_browser_draft_without_new_singletons_does_not_delete_saved_manual_boxes(self):
        self.old_orphans()
        payload = self.payload()
        payload["recordsByArtifact"]["ocr_merged"] = [group for group in payload["recordsByArtifact"]["ocr_merged"]
                                                      if not set(group.get("sourceBoxnos", [])).intersection((8, 9))]
        saved = self.save(payload)
        self.assertEqual(len(self.missed(saved["recordsByArtifact"]["ocr_merged"])), 2)
        self.assertEqual(len(self.missed(saved["recordsByArtifact"]["ocr_structured"])), 2)

    def test_new_text_grouped_with_existing_text_is_not_lost_or_duplicated(self):
        self.old_orphans()
        payload = self.payload()
        groups = payload["recordsByArtifact"]["ocr_merged"]
        groups[0]["sourceBoxnos"].append(8)
        payload["recordsByArtifact"]["ocr_merged"] = [record for record in groups if record.get("sourceBoxnos") != [8]]
        saved = self.save(payload)
        structured = saved["recordsByArtifact"]["ocr_structured"]
        self.assertEqual([record["sourceBoxnos"] for record in self.missed(structured)], [[8], [9]])
        self.assertEqual(sum(0 in record["sourceBoxnos"] for record in structured), 1)
        self.assertEqual([record["text"] for record in self.missed(structured)], ["Missed 8", "Missed 9"])
