from __future__ import annotations

import copy
import asyncio
import json
import unittest
from unittest import mock

from httpx import ASGITransport, AsyncClient

import editor_v2
import web_app
from tests import test_editor_reset
from web_storage import write_json_atomic


class EditorVisibilityTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id
        self.data_dir = fixture.data_dir
        # Keep distinct OCR detections apart: overlapping detections are suppressed together.
        for artifact in editor_v2.ARTIFACT_STAGES:
            records = self.manager.load_editor_v2_baseline(self.code, self.job_id, artifact, 0)
            records[1]["region"] = records[1]["placementRegion"] = [50, 10, 70, 40]
            self.baseline(artifact, records)

    def payload(self, stage="text", page=0):
        return self.manager.editor_v2_payload(self.code, self.job_id, stage, page)

    def save(self, payload):
        return self.manager.save_editor_v2_update(self.code, self.job_id, payload["page"], payload["stage"], {
            "baseRevision": payload["revision"], "recordsByArtifact": payload["recordsByArtifact"],
        })

    def reset(self, stage):
        return self.manager.reset_editor_v2_stage(self.code, self.job_id, 0, stage, {"baseRevision": self.payload(stage)["revision"]})

    def baseline(self, artifact, records):
        write_json_atomic(self.manager.editor_v2_generated_path(self.code, self.job_id, artifact, 0), records)
        write_json_atomic(self.data_dir / artifact / "page_0000.json", records)

    def delete(self, artifact, boxno):
        payload = self.payload("ocr" if artifact in {"ocr_raw", "ocr_merged"} else "text")
        payload["recordsByArtifact"][artifact] = [record for record in payload["recordsByArtifact"][artifact] if record["boxno"] != boxno]
        return self.save(payload)

    def assert_boxnos(self, artifact, expected, stage="placement", page=0):
        self.assertEqual([record["boxno"] for record in self.payload(stage, page)["recordsByArtifact"][artifact]], expected, artifact)

    def manual_downstream(self):
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Saved English"
        self.save(payload)
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, "placements", 0)
        edited = copy.deepcopy(baseline)
        edited[0]["fontSizeWidthPercent"] = 3.5
        editor_v2.update_artifact_override(manifest, 0, "placement", "placements", baseline, edited)
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)

    def test_text_rejection_hides_every_descendant_but_not_ocr(self):
        self.delete("ocr_structured", 0)
        for artifact in ("ocr_structured", "translations", "alt_placement", "placements"):
            self.assert_boxnos(artifact, [1])
        self.assert_boxnos("ocr_merged", [0, 1], "ocr")
        self.assert_boxnos("ocr_raw", [0, 1], "ocr")
        self.assert_boxnos("placements", [0, 1], page=1)
        self.reset("text")
        for artifact in ("ocr_structured", "translations", "alt_placement", "placements"):
            self.assert_boxnos(artifact, [0, 1])

    def test_raw_deletion_hides_descendants_and_reset_restores_manual_edits(self):
        self.manual_downstream()
        self.delete("ocr_raw", 0)
        for artifact in ("ocr_raw", "ocr_merged", "ocr_structured", "translations", "alt_placement", "placements"):
            self.assert_boxnos(artifact, [1])
        self.reset("ocr")
        restored = self.payload()["recordsByArtifact"]
        self.assertEqual(restored["translations"][0]["englishText"], "Saved English")
        self.assertEqual(restored["placements"][0]["fontSizeWidthPercent"], 3.5)

    def test_merged_group_deletion_keeps_raw_boxes_and_does_not_renumber_siblings(self):
        self.delete("ocr_merged", 0)
        self.assert_boxnos("ocr_raw", [0, 1], "ocr")
        for artifact in ("ocr_merged", "ocr_structured", "translations", "alt_placement", "placements"):
            self.assert_boxnos(artifact, [1])
        self.reset("ocr")
        self.assert_boxnos("placements", [0, 1])

    def test_deleting_every_raw_box_hides_all_later_records(self):
        payload = self.payload("ocr")
        payload["recordsByArtifact"]["ocr_raw"] = []
        self.save(payload)
        for artifact in ("ocr_merged", "ocr_structured", "translations", "alt_placement", "placements"):
            self.assert_boxnos(artifact, [])
        self.reset("ocr")
        self.assert_boxnos("placements", [0, 1])

    def test_partial_raw_deletion_keeps_the_group_while_any_source_remains(self):
        group = copy.deepcopy(self.payload("ocr")["recordsByArtifact"]["ocr_merged"][0])
        group["sourceBoxnos"] = [0, 1]
        self.baseline("ocr_merged", [group])
        source = copy.deepcopy(self.payload()["recordsByArtifact"]["ocr_structured"][0])
        source["sourceBoxnos"] = [0, 1]
        self.baseline("ocr_structured", [source])
        self.delete("ocr_raw", 0)
        self.assert_boxnos("ocr_merged", [0])
        self.assert_boxnos("ocr_structured", [0])
        self.assertIn(0, [record["boxno"] for record in self.payload()["recordsByArtifact"]["placements"]])

    def test_lineage_does_not_assume_merged_and_structured_box_numbers_match(self):
        sources = self.payload()["recordsByArtifact"]["ocr_structured"]
        sources[0]["sourceBoxnos"], sources[1]["sourceBoxnos"] = [1], [0]
        self.baseline("ocr_structured", sources)
        self.delete("ocr_merged", 0)
        self.assert_boxnos("ocr_structured", [0])
        self.assert_boxnos("placements", [0])
        self.reset("ocr")
        self.assert_boxnos("placements", [0, 1])

    def test_rejected_reordered_source_is_not_recreated_as_an_ocr_placeholder(self):
        sources = self.payload()["recordsByArtifact"]["ocr_structured"]
        sources[0]["sourceBoxnos"], sources[1]["sourceBoxnos"] = [1], [0]
        self.baseline("ocr_structured", sources)
        self.delete("ocr_structured", 1)
        self.assert_boxnos("ocr_structured", [0])
        self.assert_boxnos("placements", [0])

    def test_saving_another_text_record_does_not_delete_hidden_edits(self):
        self.manual_downstream()
        self.delete("ocr_raw", 0)
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Visible English"
        self.save(payload)
        self.reset("ocr")
        records = self.payload()["recordsByArtifact"]
        self.assertEqual(records["translations"][0]["englishText"], "Saved English")
        self.assertEqual(records["translations"][1]["englishText"], "Visible English")
        self.assertEqual(records["placements"][0]["fontSizeWidthPercent"], 3.5)

    def test_saving_visible_typesetting_keeps_hidden_placement_edits(self):
        self.manual_downstream()
        self.delete("ocr_structured", 0)
        payload = self.payload("placement")
        for record in payload["recordsByArtifact"]["placements"]:
            record.pop("font", None)
            record["fontSizeWidthPercent"] = 4.0
        self.save(payload)
        self.reset("text")
        placements = self.payload()["recordsByArtifact"]["placements"]
        self.assertEqual([(record["boxno"], record["fontSizeWidthPercent"]) for record in placements], [(0, 3.5), (1, 4.0)])

    def test_noop_save_while_a_box_is_hidden_does_not_create_more_overrides(self):
        self.delete("ocr_raw", 0)
        payload = self.payload()
        saved = self.save(payload)
        self.assertEqual(saved["revision"], payload["revision"])
        self.assertFalse(saved["stageStates"]["structure"]["changed"])
        self.assertFalse(saved["stageStates"]["translation"]["changed"])

    def test_resetting_later_tabs_does_not_undo_the_upstream_deletion(self):
        self.delete("ocr_raw", 0)
        self.reset("text")
        self.reset("placement")
        self.assert_boxnos("placements", [1])
        self.reset("ocr")
        self.assert_boxnos("placements", [0, 1])

    def test_backing_artifacts_are_not_erased_when_a_record_is_hidden(self):
        self.manual_downstream()
        self.delete("ocr_structured", 0)
        for artifact in ("translations", "alt_placement", "placements"):
            records = json.loads((self.data_dir / artifact / "page_0000.json").read_text())
            self.assertEqual([record["boxno"] for record in records], [0, 1])
        self.assertEqual(json.loads((self.data_dir / "translations" / "page_0000.json").read_text())[0]["englishText"], "Saved English")

    def test_pending_ocr_box_can_be_rejected_before_structure_is_generated(self):
        payload = self.payload("ocr")
        payload["recordsByArtifact"]["ocr_raw"].append({"page": 0, "boxno": 2, "recordId": "new-raw",
            "text": "New OCR", "region": [10, 60, 30, 80]})
        payload["recordsByArtifact"]["ocr_merged"].append({"page": 0, "boxno": 2, "recordId": "new-group",
            "sourceBoxnos": [2], "region": [10, 60, 30, 80]})
        self.save(payload)
        self.assert_boxnos("ocr_structured", [0, 1, 2])
        self.delete("ocr_structured", 2)
        self.assert_boxnos("ocr_structured", [0, 1])
        self.assert_boxnos("placements", [0, 1])
        payload = self.payload()
        payload["recordsByArtifact"]["translations"][0]["englishText"] = "Another edit"
        self.save(payload)
        self.assert_boxnos("placements", [0, 1])
        self.reset("text")
        self.assert_boxnos("ocr_structured", [0, 1, 2])
        self.assert_boxnos("placements", [0, 1, 2])

    def test_ocr_deletion_hides_even_frozen_downstream_records(self):
        payload = self.payload()
        self.manager.set_editor_v2_protection(self.code, self.job_id, 0, "text", {
            "baseRevision": payload["revision"], "action": "freeze-page", "protected": True,
        })
        self.delete("ocr_raw", 0)
        self.assert_boxnos("placements", [1])
        self.reset("ocr")
        self.assert_boxnos("placements", [0, 1])

    def test_rejection_follows_source_lineage_after_generated_box_numbers_change(self):
        self.delete("ocr_structured", 0)
        sources = self.manager.load_editor_v2_baseline(self.code, self.job_id, "ocr_structured", 0)
        source = sources[0]
        source.pop("recordId", None)
        source["boxno"] = 8
        self.baseline("ocr_structured", [source, sources[1]])
        placements = self.manager.load_editor_v2_baseline(self.code, self.job_id, "placements", 0)
        placement = placements[0]
        placement.pop("recordId", None)
        placement["boxno"] = 8
        self.baseline("placements", [placement, placements[1]])
        self.assert_boxnos("ocr_structured", [1])  # The other merged group is still visible.
        self.assertNotIn(8, [record["boxno"] for record in self.payload()["recordsByArtifact"]["placements"]])

    def test_http_tabs_hide_deleted_boxes_and_reset_restores_them(self):
        app = web_app.create_app(self.manager.config.pipeline_config.parent / "web_config.json")
        app.state.manager = self.manager
        session = self.manager.create_admin_session()
        root = f"/api/job/{self.code}/{self.job_id}/editor/v2/pages/0/stages"

        async def direct(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, session)
                payload = (await client.get(root + "/ocr")).json()
                payload["recordsByArtifact"]["ocr_raw"].pop(0)
                response = await client.post(root + "/ocr", json={
                    "baseRevision": payload["revision"], "recordsByArtifact": payload["recordsByArtifact"],
                })
                self.assertEqual(response.status_code, 200)
                for stage in ("text", "placement"):
                    response = await client.get(root + "/" + stage)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual([record["boxno"] for record in response.json()["recordsByArtifact"]["placements"]], [1])
                response = await client.post(root + "/ocr/reset", json={"baseRevision": response.json()["revision"]})
                self.assertEqual(response.status_code, 200)
                response = await client.get(root + "/placement")
                self.assertEqual([record["boxno"] for record in response.json()["recordsByArtifact"]["placements"]], [0, 1])

        with mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(asyncio.wait_for(exercise(), timeout=15))


if __name__ == "__main__":
    unittest.main()
