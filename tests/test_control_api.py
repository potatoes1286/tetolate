from __future__ import annotations

import asyncio
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from httpx import ASGITransport, AsyncClient
from PIL import Image

from tests import test_regressions
import web_app
import web_control_api
from web_control_auth import CONTROL_TOKENS_FILENAME, LOCAL_CLI_TOKEN_FILENAME
import tetolate_cli


def archive_bytes() -> bytes:
    image = io.BytesIO()
    Image.new("RGB", (5, 5)).save(image, format="PNG")
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("page.png", image.getvalue())
    return result.getvalue()


class ControlAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helper = test_regressions.WebAuthenticationTests()
        self.manager = self.helper.initialized_manager(self.root, ["default"])
        self.app = web_app.create_app(self.root / "web_config.json")
        self.app.state.manager = self.manager
        self.token = self.manager.create_control_token("tests")["token"]

    def run_http(self, exercise):
        async def direct(function, *args, **kwargs):
            return function(*args)

        async def run():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://testserver") as client:
                client.headers["Authorization"] = "Bearer " + self.token
                await exercise(client)

        with (
            mock.patch.object(web_control_api, "run_in_threadpool", new=direct),
            mock.patch.object(web_app, "run_in_threadpool", new=direct),
            mock.patch("anyio.to_thread.run_sync", new=direct),
        ):
            asyncio.run(asyncio.wait_for(run(), timeout=15))

    def test_tokens_are_private_hashed_persistent_and_revocable(self):
        path = self.manager.config.jobs_dir / CONTROL_TOKENS_FILENAME
        self.assertNotIn(self.token, path.read_text())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        self.assertIsNotNone(restarted.authenticate_control_token(self.token))
        records = restarted.list_control_tokens()
        self.assertNotIn("hash", json.dumps(records))
        self.assertNotIn(self.token, json.dumps(records))
        restarted.revoke_control_token(records[0]["id"])
        self.assertIsNone(self.manager.authenticate_control_token(self.token))

    def test_local_cli_credential_is_private_persistent_and_instance_specific(self):
        path = self.manager.config.jobs_dir / LOCAL_CLI_TOKEN_FILENAME
        token = tetolate_cli.local_control_token("http://127.0.0.1:8088", self.manager.config.jobs_dir)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertTrue(self.manager.authenticate_control_token(token)["local"])
        self.assertNotIn(token, json.dumps(self.manager.list_control_tokens()))
        restarted = web_app.JobManager(self.manager.config)
        restarted.initialize_web_state()
        self.assertIsNotNone(restarted.authenticate_control_token(token))
        other_root = self.root / "other"
        other_root.mkdir()
        other = self.helper.initialized_manager(other_root)
        self.assertIsNone(other.authenticate_control_token(token))
        self.assertIsNone(self.manager.authenticate_control_token("not-a-token"))
        self.assertIsNone(self.manager.authenticate_control_token("non-ascii-\u00e9"))

    def test_local_cli_access_does_not_make_http_or_browser_access_anonymous(self):
        token = tetolate_cli.local_control_token("http://localhost", self.manager.config.jobs_dir)
        async def exercise(client):
            client.headers.pop("Authorization")
            response = await client.get("/api/v1/status", headers={"X-Forwarded-For": "127.0.0.1", "X-Tetolate-CLI": "true"})
            self.assertEqual(response.status_code, 401)
            client.headers["Authorization"] = "Bearer " + token
            response = await client.get("/api/v1/status")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(token, response.text)
            self.assertEqual((await client.post("/api/v1/pause")).status_code, 200)
            self.assertTrue(self.manager._paused)
            self.assertEqual((await client.post("/api/v1/resume")).status_code, 200)
            self.assertFalse(self.manager._paused)
            self.assertEqual((await client.post("/admin/pause")).status_code, 303)
            self.assertFalse(self.manager._paused)
            response = await client.post("/api/v1/pause", headers={"Origin": "https://other.example"})
            self.assertEqual(response.status_code, 403)
            self.assertFalse(self.manager._paused)
            response = await client.post("/api/v1/tokens", json={"name": "agent", "read_only": True})
            self.assertEqual(response.status_code, 201)
            self.assertTrue(response.json()["readOnly"])
            self.assertNotEqual(response.json()["token"], token)
            client.headers["Authorization"] = "Bearer " + response.json()["token"]
            self.assertEqual((await client.post("/api/v1/pause")).status_code, 403)
        self.run_http(exercise)

    def test_local_cli_credential_rejects_symlinks_and_invalid_content(self):
        path = self.manager.config.jobs_dir / LOCAL_CLI_TOKEN_FILENAME
        path.write_text(json.dumps({"token": "do-not-echo-this-secret"}))
        with self.assertRaises(RuntimeError) as error:
            self.manager.initialize_web_state()
        self.assertNotIn("do-not-echo-this-secret", str(error.exception))
        path.unlink()
        target = self.root / "target.json"
        target.write_text("{}")
        path.symlink_to(target)
        with self.assertRaises(RuntimeError):
            self.manager.initialize_web_state()

    def test_login_and_read_only_authentication_preserve_browser_boundary(self):
        async def exercise(client):
            client.headers.pop("Authorization")
            response = await client.get("/api/v1/status")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json()["error"]["code"], "unauthorized")
            response = await client.post("/api/v1/auth/login", json={"password": "wrong-password"})
            self.assertEqual(response.status_code, 401)
            self.assertNotIn("wrong-password", response.text)
            response = await client.post("/api/v1/auth/login", json={
                "password": self.helper.ADMIN_PASSWORD, "name": "reader", "read_only": True,
            })
            self.assertEqual(response.status_code, 201)
            token = response.json()["token"]
            token_id = response.json()["id"]
            client.headers["Authorization"] = "Bearer " + token
            self.assertEqual((await client.get("/api/v1/status")).status_code, 200)
            response = await client.post("/api/v1/pause")
            self.assertEqual(response.status_code, 403)
            self.assertFalse(self.manager._paused)
            # A control token does not authenticate browser forms.
            self.assertEqual((await client.post("/admin/pause")).status_code, 303)
            response = await client.delete(f"/api/v1/tokens/{token_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual((await client.get("/api/v1/status")).status_code, 401)
            response = await client.post("/api/v1/auth/login", json={"password": self.helper.ADMIN_PASSWORD}, headers={"Origin": "https://other.example"})
            self.assertEqual(response.status_code, 403)

        self.run_http(exercise)

    def test_token_login_is_rate_limited_and_write_token_cannot_mint_tokens(self):
        async def exercise(client):
            response = await client.post("/api/v1/tokens", json={"name": "child"})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(len(self.manager.list_control_tokens()), 1)
            for _ in range(web_app.ADMIN_LOGIN_FAILURE_LIMIT):
                response = await client.post("/api/v1/auth/login", json={"password": "wrong"})
                self.assertEqual(response.status_code, 401)
            response = await client.post("/api/v1/auth/login", json={"password": self.helper.ADMIN_PASSWORD})
            self.assertEqual(response.status_code, 429)
            self.assertIn("retry-after", response.headers)

        self.run_http(exercise)

    def test_browser_can_create_and_revoke_token_without_exposing_stored_keys(self):
        async def exercise(client):
            client.headers.pop("Authorization")
            session = self.manager.authenticate_admin(self.helper.ADMIN_PASSWORD)
            client.cookies.set(web_app.ADMIN_COOKIE_NAME, session)
            response = await client.post("/admin/control-tokens", data={"name": "browser"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("only shown once", response.text)
            self.assertIn("no-store", response.headers["cache-control"])
            response = await client.get("/admin")
            self.assertIn("Terminal access", response.text)
            self.assertNotIn(self.token, response.text)
            token_id = next(item["id"] for item in self.manager.list_control_tokens() if item["name"] == "browser")
            response = await client.post(f"/admin/control-tokens/{token_id}/revoke")
            self.assertEqual(response.status_code, 303)
            self.assertEqual(len(self.manager.list_control_tokens()), 1)

        self.run_http(exercise)

    def test_categories_and_series_json_actions(self):
        async def exercise(client):
            self.assertEqual((await client.get("/api/v1/status")).json()["apiVersion"], 1)
            response = await client.post("/api/v1/categories", json={"name": "books"})
            self.assertEqual(response.status_code, 201)
            response = await client.post("/api/v1/categories/books/series", json={"name": "Example"})
            self.assertEqual(response.status_code, 201)
            series_id = response.json()["id"]
            response = await client.patch(f"/api/v1/categories/books/series/{series_id}", json={"notes": "Aki is the captain.", "instructions": "Keep names."})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["notes"], "Aki is the captain.")
            response = await client.get("/api/v1/series")
            self.assertEqual(response.json()["series"][0]["category"], "books")
            response = await client.patch("/api/v1/categories/books", json={"name": "manga"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual((await client.get(f"/api/v1/categories/manga/series/{series_id}")).status_code, 200)
            response = await client.delete("/api/v1/categories/manga", follow_redirects=False)
            self.assertEqual(response.status_code, 422)
            response = await client.request("DELETE", "/api/v1/categories/manga", json={"confirmation": "manga"})
            self.assertEqual(response.status_code, 200)

        self.run_http(exercise)

    def test_submit_settings_jobs_and_actions_share_manager_state(self):
        async def exercise(client):
            response = await client.patch("/api/v1/categories/default/settings", json={"vlm_model": "chosen", "vlm_auth_token": "private-vlm-key"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("private-vlm-key", response.text)
            response = await client.post("/api/v1/jobs", data={"category": "default"}, files={"file": ("input.cbz", archive_bytes(), "application/zip")})
            self.assertEqual(response.status_code, 201, response.text)
            job_id = response.json()["jobId"]
            self.assertNotIn("private-vlm-key", response.text)
            self.assertEqual(response.json()["vlmModel"], "chosen")
            self.assertEqual(self.manager.load_job_secrets("default", job_id)["vlmAuthToken"], "private-vlm-key")
            response = await client.get("/api/v1/jobs?category=default&state=queued")
            self.assertEqual(len(response.json()["jobs"]), 1)
            response = await client.get(f"/api/v1/jobs/{job_id}")
            self.assertEqual(response.json()["category"], "default")
            status = self.manager.load_status("default", job_id)
            status["status"] = "failed"
            self.manager.save_status("default", job_id, status)
            response = await client.patch(f"/api/v1/jobs/{job_id}/settings", json={"thinking_budget_tokens": 512, "pause_after_ocr": True})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["thinkingBudgetTokens"], 512)
            self.assertEqual(response.json()["vlmModel"], "chosen")
            response = await client.patch(f"/api/v1/jobs/{job_id}/name", json={"name": "Chapter one"})
            self.assertEqual(response.json()["jobName"], "Chapter one")
            response = await client.patch(f"/api/v1/jobs/{job_id}/name", json={"name": "../invalid"})
            self.assertEqual(response.status_code, 400)
            response = await client.post(f"/api/v1/jobs/{job_id}/restart")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "queued")
            await client.post("/api/v1/pause")
            self.assertTrue(self.manager._paused)
            await client.post("/api/v1/resume")
            self.assertFalse(self.manager._paused)
            schema = (await client.get("/api/v1/openapi.json")).json()
            self.assertIn("/api/v1/jobs/{job_id}/rerun", schema["paths"])

        self.run_http(exercise)

    def test_series_membership_and_publication_actions(self):
        async def exercise(client):
            response = await client.post("/api/v1/categories/default/series", json={"name": "Example"})
            series_id = response.json()["id"]
            response = await client.post("/api/v1/jobs", data={"series_id": series_id, "chapter": "1.5"}, files={"file": ("input.cbz", archive_bytes(), "application/zip")})
            self.assertEqual(response.status_code, 201)
            job_id = response.json()["jobId"]
            self.assertEqual(response.json()["chapterOrder"], "1.5")
            status = self.manager.load_status("default", job_id)
            status["status"] = "complete"
            self.manager.save_status("default", job_id, status)
            response = await client.post(f"/api/v1/jobs/{job_id}/series-notes", json={"use_current": True})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.manager.get_series("default", series_id)["notesThroughOrder"], "1.5")
            response = await client.get(f"/api/v1/categories/default/series/{series_id}")
            self.assertEqual(len(response.json()["chapters"]), 1)

        self.run_http(exercise)

    def test_logs_cursor_keeps_repeated_lines_and_resets_on_rotation(self):
        job_id = "12345678"
        self.manager.save_status("default", job_id, {"status": "running"})
        path = self.manager.log_path("default", job_id)
        text = "a" * 1023 + "🌍\nsame\nsame\n"
        path.write_text(text, encoding="utf-8")

        async def exercise(client):
            response = await client.get(f"/api/v1/jobs/{job_id}/logs?limit=1024")
            first = response.json()
            self.assertTrue(first["hasMore"])
            second = (await client.get(f"/api/v1/jobs/{job_id}/logs", params={"cursor": first["cursor"]})).json()
            self.assertEqual(first["text"] + second["text"], text)
            empty = (await client.get(f"/api/v1/jobs/{job_id}/logs", params={"cursor": second["cursor"]})).json()
            self.assertEqual(empty["text"], "")
            replacement = path.with_suffix(".new")
            replacement.write_text("new log\n")
            replacement.replace(path)
            result = (await client.get(f"/api/v1/jobs/{job_id}/logs", params={"cursor": second["cursor"]})).json()
            self.assertTrue(result["reset"])
            self.assertEqual(result["text"], "new log\n")
            response = await client.get(f"/api/v1/jobs/{job_id}/logs?cursor=broken")
            self.assertEqual(response.status_code, 400)

        self.run_http(exercise)

    def test_downloads_export_and_rerun_use_specific_routes(self):
        job_id = "12345678"
        self.manager.save_status("default", job_id, {"status": "complete", "jobName": "Book"})
        output = self.manager.output_dir("default", job_id)
        output.mkdir()
        self.manager.input_path("default", job_id).write_bytes(archive_bytes())
        archive = self.manager.translated_cbz_variant_path("default", job_id, "webp")
        archive.write_bytes(b"test archive")

        async def exercise(client):
            response = await client.post(f"/api/v1/jobs/{job_id}/export", json={"format": "webp"})
            self.assertEqual(response.status_code, 200, response.text)
            response = await client.get(f"/api/v1/jobs/{job_id}/download/webp")
            self.assertEqual(response.content, b"test archive")
            self.assertIn("Book.cbz", response.headers["content-disposition"])
            response = await client.get(f"/api/v1/jobs/{job_id}/download/original")
            self.assertEqual(response.content, self.manager.input_path("default", job_id).read_bytes())
            with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
                response = await client.post(f"/api/v1/jobs/{job_id}/rerun", json={"stage": "translations", "pages": [0]})
                self.assertEqual(response.status_code, 200, response.text)
                rerun.assert_called_once_with("default", job_id, [0], resume_from="translations")
            with mock.patch.object(self.manager, "terminate_active_job") as terminate:
                response = await client.post(f"/api/v1/jobs/{job_id}/terminate")
                self.assertEqual(response.status_code, 200)
                terminate.assert_called_once_with("default", job_id)

        self.run_http(exercise)

    def test_bad_or_oversized_uploads_do_not_create_jobs_and_errors_hide_secrets(self):
        async def exercise(client):
            response = await client.post("/api/v1/jobs", files={"file": ("bad.cbz", b"not a zip")})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(self.manager.iter_job_ids("default"), [])
            response = await client.patch("/api/v1/categories/default/settings", json={"vlm_auth_token": {"secret": "do-not-echo"}})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("do-not-echo", response.text)
            self.manager.config = replace(self.manager.config, max_upload_bytes=5)
            response = await client.post("/api/v1/jobs", files={"file": ("big.cbz", archive_bytes())})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["error"]["code"], "too_large")
            self.assertEqual(self.manager.iter_job_ids("default"), [])
            async def chunks():
                yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="big.cbz"\r\n\r\n'
                yield archive_bytes() + b'\r\n--test--\r\n'
            response = await client.post("/api/v1/jobs", content=chunks(), headers={
                "Content-Type": "multipart/form-data; boundary=test",
            })
            self.assertEqual(response.status_code, 413, response.text)
            self.assertEqual(response.json()["error"]["code"], "too_large")

        self.run_http(exercise)

    def stopped_job(self):
        job_id = self.manager.create_job_id("default")
        self.manager.job_dir("default", job_id).mkdir(parents=True)
        self.manager.input_path("default", job_id).write_bytes(archive_bytes())
        self.manager.submit_job("default", job_id, "original.cbz", enqueue_job=False)
        status = self.manager.load_status("default", job_id)
        status["status"] = "complete"
        self.manager.save_status("default", job_id, status)
        return job_id

    def test_title_set_clear_translate_retry_and_refusal(self):
        job_id = self.stopped_job()
        base = f"/api/v1/jobs/{job_id}"

        async def exercise(client):
            response = await client.patch(base + "/translated-title", json={"name": "New title"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["translatedJobName"], "New title")
            response = await client.patch(base + "/translated-title", json={"name": ""})
            self.assertEqual(response.json()["translatedJobName"], "")
            response = await client.patch(base + "/translated-title", json={"name": "../invalid"})
            self.assertEqual(response.status_code, 400)
            with mock.patch.object(web_app.vlm_client, "call_vlm", return_value=mock.Mock(output="English title")):
                response = await client.post(base + "/translate-title")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["translatedJobName"], "English title")
            response = await client.post(base + "/retry-title-translation", json={"manual_title": "Manual title"})
            self.assertEqual(response.json()["translatedJobName"], "Manual title")
            with mock.patch.object(self.manager, "request_vlm_title", return_value=("", True)):
                response = await client.post(base + "/retry-title-translation", json={})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["titleTranslationSkipped"])
            with mock.patch.object(self.manager, "translate_title", side_effect=RuntimeError("secret-key")):
                response = await client.post(base + "/translate-title")
            self.assertEqual(response.status_code, 502)
            self.assertNotIn("secret-key", response.text)
            status = self.manager.load_status("default", job_id)
            status["status"] = "queued"
            self.manager.save_status("default", job_id, status)
            self.assertEqual((await client.post(base + "/translate-title")).status_code, 409)

        self.run_http(exercise)

    def test_vlm_probe_uses_job_or_category_keys_and_never_echoes_them(self):
        job_id = self.stopped_job()

        async def exercise(client):
            status = self.manager.load_status("default", job_id)
            endpoint = status["vlmBaseUrl"]
            self.manager.update_job_auth_tokens("default", job_id, vlm_auth_token="saved-job-key", paddleocr_vl_auth_token=None)
            with mock.patch.object(self.manager, "probe_vlm_models", return_value=["model-a", "model-b"]) as probe:
                response = await client.post("/api/v1/vlm/test", json={"job_id": job_id})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["models"], ["model-a", "model-b"])
                probe.assert_called_once_with(endpoint, "saved-job-key")
                self.assertNotIn("saved-job-key", response.text)
            with mock.patch.object(self.manager, "probe_vlm_models", return_value=["model-a"]) as probe:
                response = await client.post("/api/v1/vlm/test", json={"job_id": job_id, "vlm_auth_token": "new-job-key"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.manager.saved_vlm_token(endpoint), "new-job-key")
                self.assertEqual(self.manager.load_job_secrets("default", job_id)["vlmAuthToken"], "new-job-key")
                response = await client.post("/api/v1/vlm/test", json={"category": "default"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(probe.call_args.args[1], "new-job-key")
            with mock.patch.object(self.manager, "probe_vlm_models", side_effect=RuntimeError("Connection refused new-job-key")):
                response = await client.post("/api/v1/vlm/test", json={"job_id": job_id})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["error"]["code"], "endpoint_error")
            self.assertNotIn("new-job-key", response.text)
            self.assertIn("\n", response.json()["error"]["message"])
            response = await client.post("/api/v1/vlm/test", json={"vlm_auth_token": {"bad": "do-not-echo"}})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("do-not-echo", response.text)

        self.run_http(exercise)

    def test_images_upload_preserves_selection_order_and_rejects_mixed_or_bad_inputs(self):
        def png(color):
            output = io.BytesIO()
            Image.new("RGB", (5, 5), color).save(output, format="PNG")
            return output.getvalue()

        async def exercise(client):
            response = await client.post("/api/v1/jobs", files=[
                ("page_images", ("z.png", png("red"), "image/png")),
                ("page_images", ("a.png", png("blue"), "image/png")),
            ])
            self.assertEqual(response.status_code, 201, response.text)
            job_id = response.json()["jobId"]
            with zipfile.ZipFile(self.manager.input_path("default", job_id)) as archive:
                self.assertEqual(archive.namelist(), ["0000.png", "0001.png"])
                for filename, color in (("0000.png", (255, 0, 0)), ("0001.png", (0, 0, 255))):
                    with Image.open(io.BytesIO(archive.read(filename))) as image:
                        self.assertEqual(image.convert("RGB").getpixel((0, 0)), color)
            for files in ([('file', ('book.cbz', archive_bytes())), ('page_images', ('page.png', png('red')))],
                          [('page_images', ('bad.png', b'not an image'))],
                          [('page_images', ('bad.txt', b'not an image'))]):
                response = await client.post("/api/v1/jobs", files=files)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(len(self.manager.iter_job_ids("default")), 1)
            self.assertEqual((await client.post("/api/v1/jobs")).status_code, 400)

        self.run_http(exercise)

    def test_batch_upload_natural_order_and_atomic_rollback(self):
        series_id = self.manager.create_series("default", "Books")
        url = f"/api/v1/categories/default/series/{series_id}/upload"

        async def exercise(client):
            files = [("chapters", (name, archive_bytes())) for name in ("ch10.cbz", "ch2.zip")]
            response = await client.post(url, data={"first_chapter": "1.5"}, files=files)
            self.assertEqual(response.status_code, 201, response.text)
            jobs = response.json()["jobs"]
            self.assertEqual([job["inputFilename"] for job in jobs], ["ch2.zip", "ch10.cbz"])
            self.assertEqual([job["chapterOrder"] for job in jobs], ["1.5", "2.5"])
            self.assertEqual(self.manager._queued_jobs, {("default", jobs[0]["jobId"])})
            before = set(self.manager.iter_job_ids("default"))
            for data, uploads, expected in (({"first_chapter": "3"}, [("chapters", ("3.cbz", archive_bytes())), ("chapters", ("4.cbz", b"bad"))], 400),
                                             ({"first_chapter": "1.5"}, files, 409),
                                             ({"first_chapter": "1000000"}, files, 400)):
                response = await client.post(url, data=data, files=uploads)
                self.assertEqual(response.status_code, expected, response.text)
                self.assertEqual(set(self.manager.iter_job_ids("default")), before)
            submit = self.manager.submit_job
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise web_app.HTTPException(400, "Registration failed")
                return submit(*args, **kwargs)

            with mock.patch.object(self.manager, "submit_job", side_effect=fail_second):
                response = await client.post(url, data={"first_chapter": "3"}, files=files)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(set(self.manager.iter_job_ids("default")), before)
            self.assertEqual(self.manager._queued_jobs, {("default", jobs[0]["jobId"])})
            self.manager.config = replace(self.manager.config, max_upload_bytes=5)
            self.assertEqual((await client.post(url, files=files)).status_code, 413)

        self.run_http(exercise)

    def test_page_reads_and_ocr_merge_rerun(self):
        job_id = self.stopped_job()
        for kind in ("original", "final"):
            path = self.manager.output_dir("default", job_id) / "pages" / kind / "0000.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (5, 5)).save(path)

        async def exercise(client):
            for kind in ("original", "final"):
                url = f"/api/v1/jobs/{job_id}/pages"
                response = await client.get(url, params={"kind": kind})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["pages"][0]["page"], 0)
                response = await client.get(url + "/0", params={"kind": kind})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.content.startswith(b"\x89PNG"))
                self.assertEqual((await client.get(url + "/-1")).status_code, 404)
            with mock.patch.object(self.manager, "rerun_completed_job_pages") as rerun:
                response = await client.post(f"/api/v1/jobs/{job_id}/rerun", json={"stage": "ocr_merge", "pages": [0]})
                self.assertEqual(response.status_code, 200)
                rerun.assert_called_once_with("default", job_id, [0], resume_from="ocr_merge")

        self.run_http(exercise)

    def test_password_change_checks_old_password_and_signs_out_browser_sessions(self):
        session = self.manager.authenticate_admin(self.helper.ADMIN_PASSWORD)

        async def exercise(client):
            data = {"current_password": "wrong-private-password", "new_password": "new-private-password", "confirm_password": "new-private-password"}
            response = await client.post("/api/v1/auth/password", json=data)
            self.assertEqual(response.status_code, 401)
            self.assertNotIn("wrong-private-password", response.text)
            data["current_password"] = self.helper.ADMIN_PASSWORD
            data["confirm_password"] = "different-private-password"
            response = await client.post("/api/v1/auth/password", json=data)
            self.assertEqual(response.status_code, 400)
            data["confirm_password"] = data["new_password"]
            response = await client.post("/api/v1/auth/password", json=data)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(self.manager.admin_password_matches("new-private-password"))
            self.assertFalse(self.manager.admin_password_matches(self.helper.ADMIN_PASSWORD))
            self.assertFalse(self.manager.admin_session_is_valid(session))
            self.assertEqual(len(self.manager._admin_sessions), 0)
            self.assertEqual((await client.get("/api/v1/status")).status_code, 200)
            self.assertNotIn("new-private-password", response.text)

        self.run_http(exercise)

    def test_read_only_tokens_cannot_use_new_mutations(self):
        job_id = self.stopped_job()
        series_id = self.manager.create_series("default", "Example")

        async def exercise(client):
            client.headers["Authorization"] = "Bearer " + self.manager.create_control_token("reader", True)["token"]
            for method, path in (("POST", "/auth/password"), ("POST", "/vlm/test"),
                                 ("PATCH", f"/jobs/{job_id}/translated-title"),
                                 ("POST", f"/jobs/{job_id}/translate-title"),
                                 ("POST", f"/jobs/{job_id}/retry-title-translation"),
                                 ("POST", f"/categories/default/series/{series_id}/upload"), ("POST", "/jobs")):
                response = await client.request(method, "/api/v1" + path)
                self.assertEqual(response.status_code, 403, path)
            self.assertEqual((await client.get(f"/api/v1/jobs/{job_id}/pages?kind=original")).status_code, 200)

        self.run_http(exercise)


if __name__ == "__main__":
    unittest.main()
