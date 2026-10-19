from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

import editor_v2
import web_app
import web_pages
from tests import test_regressions
from web_storage import write_json_atomic


class EditorResetTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager, self.code, self.job_id = test_regressions.EditorV2Tests().create_job(self.root)
        self.manager.initialize_web_state()
        Image.new("RGB", (100, 120), "white").save(
            self.manager.original_pages_dir(self.code, self.job_id) / "0001.png"
        )
        self.data_dir = self.manager.output_dir(self.code, self.job_id) / "data"
        for page in (0, 1):
            for artifact in editor_v2.ARTIFACT_STAGES:
                write_json_atomic(self.data_dir / artifact / f"page_{page:04d}.json", [
                    {"page": page, "boxno": box, "region": [10, 10, 30, 40],
                     "sourceBoxnos": [box], "text": "Generated", "englishText": "Generated",
                     "placementRegion": [10, 10, 30, 40], "font": "Original font",
                     "openLettering": False, "safeToEraseOriginal": True}
                    for box in (0, 1)
                ])
        self.notes = {"job": "Saved job notes", "pages": {"0": "Saved page notes"}}
        write_json_atomic(self.manager.translation_notes_path(self.code, self.job_id), self.notes)
        self.manager.initialize_editor_v2(self.code, self.job_id)

    def reset(self, stage: str, revision: int = 0, page: int = 0):
        return self.manager.reset_editor_v2_stage(
            self.code, self.job_id, page, stage, {"baseRevision": revision}
        )

    def test_all_stages_reset_fields_additions_deletions_order_and_protection(self) -> None:
        for stage in editor_v2.STAGE_ORDER:
            with self.subTest(stage=stage):
                manifest = editor_v2.default_manifest()
                effective = {}
                for artifact in editor_v2.STAGE_ARTIFACTS[stage]:
                    baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, artifact, 0)
                    edited = copy.deepcopy(baseline[:1])
                    edited[0].update(text="Edited", englishText="Edited", font="Edited font",
                                     openLettering=True, safeToEraseOriginal=False)
                    edited.insert(0, {**baseline[1], "recordId": "manual", "boxno": 9})
                    editor_v2.update_artifact_override(manifest, 0, stage, artifact, baseline, edited)
                    effective[artifact] = edited
                editor_v2.freeze_stage(manifest, 0, stage, effective, True)
                editor_v2.mark_saved(manifest, 0, stage)
                self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
                result = self.reset(stage, manifest["revision"])
                for artifact in editor_v2.STAGE_ARTIFACTS[stage]:
                    expected = self.manager.load_editor_v2_baseline(self.code, self.job_id, artifact, 0)
                    actual = result["recordsByArtifact"][artifact]
                    # The payload can include display hints; the persisted records must be exact.
                    persisted = json.loads((self.data_dir / artifact / "page_0000.json").read_text())
                    self.assertEqual(persisted, expected)
                    self.assertEqual([r["recordId"] for r in actual], [r["recordId"] for r in expected])
                self.assertFalse(result["stageStates"][stage]["changed"])
                self.assertFalse(result["stageStates"][stage]["frozen"])
                self.assertEqual(result["protection"]["records"], {})
                self.assertEqual(result["protection"]["recordStates"], {})

    def test_reset_keeps_other_pages_stages_notes_and_upstream_pending_work(self) -> None:
        manifest = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        manifest["translationNotes"] = copy.deepcopy(self.notes)
        for page, stage, artifact, field, value in (
            (0, "structure", "ocr_structured", "text", "Manual structure"),
            (0, "erase", "ocr_structured", "openLettering", True),
            (0, "translation", "translations", "englishText", "Manual translation"),
            (1, "translation", "translations", "englishText", "Other page"),
        ):
            baseline = self.manager.load_editor_v2_baseline(self.code, self.job_id, artifact, page)
            edited = copy.deepcopy(baseline)
            edited[0][field] = value
            editor_v2.update_artifact_override(manifest, page, stage, artifact, baseline, edited)
        editor_v2.mark_saved(manifest, 0, "ocr")
        before = copy.deepcopy(manifest)
        self.manager.save_editor_v2_manifest(self.code, self.job_id, manifest)
        result = self.reset("erase", manifest["revision"])
        after = self.manager.load_editor_v2_manifest(self.code, self.job_id)
        self.assertEqual(after["pages"]["1"], before["pages"]["1"])
        for stage in ("structure", "translation"):
            self.assertEqual(after["pages"]["0"]["stages"][stage], before["pages"]["0"]["stages"][stage])
        self.assertEqual(result["recordsByArtifact"]["ocr_structured"][0]["text"], "Manual structure")
        self.assertFalse(result["recordsByArtifact"]["ocr_structured"][0]["openLettering"])
        self.assertEqual(result["recordsByArtifact"]["translations"][0]["englishText"], "Manual translation")
        self.assertEqual(editor_v2.pending_changes(after), {0: {"ocr", "erase"}})
        self.assertTrue(result["stageStates"]["erase"]["stale"])
        self.assertTrue(result["stageStates"]["placement"]["stale"])
        self.assertEqual(after["translationNotes"], self.notes)
        self.assertEqual(json.loads(self.manager.translation_notes_path(self.code, self.job_id).read_text()), self.notes)

    def test_reset_is_persistent_and_saved_revision_can_restore_the_edits(self) -> None:
        payload = self.manager.editor_v2_payload(self.code, self.job_id, "translation", 0)
        records = payload["recordsByArtifact"]["translations"]
        records[0]["englishText"] = "Manual"
        saved = self.manager.save_editor_v2_update(self.code, self.job_id, 0, "translation", {
            "baseRevision": payload["revision"], "recordsByArtifact": {"translations": records},
        })
        result = self.reset("translation", saved["revision"])
        self.assertEqual(result["revision"], saved["revision"] + 1)
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        self.assertEqual(restarted.editor_v2_payload(self.code, self.job_id, "translation", 0)["recordsByArtifact"]["translations"][0]["englishText"], "Generated")
        self.manager.restore_editor_v2_revision(self.code, self.job_id, saved["revision"])
        restored = self.manager.editor_v2_payload(self.code, self.job_id, "translation", 0)
        self.assertEqual(restored["recordsByArtifact"]["translations"][0]["englishText"], "Manual")

    def test_invalid_or_stale_revisions_do_not_change_files(self) -> None:
        path = self.manager.editor_v2_manifest_path(self.code, self.job_id)
        before = path.read_bytes()
        for revision, status in ((None, 400), (False, 400), ("0", 400), (-1, 409), (10, 409)):
            with self.subTest(revision=revision), self.assertRaises(HTTPException) as error:
                self.reset("translation", revision)
            self.assertEqual(error.exception.status_code, status)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_target_or_running_job_is_rejected(self) -> None:
        for stage, page, status in (("unknown", 0, 404), ("translation", 99, 404)):
            with self.subTest(stage=stage, page=page), self.assertRaises(HTTPException) as error:
                self.reset(stage, page=page)
            self.assertEqual(error.exception.status_code, status)
        self.manager.save_status(self.code, self.job_id, {"status": "running"})
        with self.assertRaises(HTTPException) as error:
            self.reset("ocr")
        self.assertEqual(error.exception.status_code, 400)

    def test_ocr_checkpoint_allows_only_ocr_reset(self) -> None:
        self.manager.save_status(self.code, self.job_id, {"status": "paused", "reviewCheckpoint": "ocr"})
        self.assertEqual(self.reset("ocr")["availableStages"], ["ocr"])
        with self.assertRaises(HTTPException) as error:
            self.reset("translation", 1)
        self.assertEqual(error.exception.status_code, 409)

    def test_reset_endpoint_requires_login_and_same_origin(self) -> None:
        app = web_app.create_app(self.root / "config" / "web_config.json")
        app.state.manager = self.manager
        session = self.manager.create_admin_session()
        url = f"/api/job/{self.code}/{self.job_id}/editor/v2/pages/0/stages/text/reset"

        async def direct(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                self.assertEqual((await client.post(url, json={"baseRevision": 0})).status_code, 401)
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, session)
                self.assertEqual((await client.post(url, json={"baseRevision": 0}, headers={"Origin": "http://elsewhere"})).status_code, 403)
                self.assertEqual((await client.post(url, json=[])).status_code, 400)
                response = await client.post(url, json={"baseRevision": 0})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["revision"], 1)
                self.assertEqual((await client.post(url, json={"baseRevision": 0})).status_code, 409)

        with mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(asyncio.wait_for(exercise(), timeout=15))

    def test_editor_has_reset_control(self) -> None:
        page = web_pages.editor_page(self.manager, self.code, self.job_id).body.decode()
        self.assertIn('id="reset-stage"', page)
        self.assertIn('>Reset stage</button>', page)
        self.assertIn("Other stages, pages, and saved translation notes are kept.", page)


if __name__ == "__main__":
    unittest.main()
