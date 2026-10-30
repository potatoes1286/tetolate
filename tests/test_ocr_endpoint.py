from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import socket
import types
import unittest
from unittest import mock
import urllib.error
import urllib.request

from httpx import ASGITransport, AsyncClient

import paddle_ocr_image as ocr
import paddle_ocr_server
import translate_cbz
import web_app
from tests import test_editor_reset


class OcrEndpointTests(unittest.TestCase):
    endpoint = "http://ocr.example:8081/v1"
    model = "PaddlePaddle/PaddleOCR-VL-1.6"

    def response(self, body=None):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            json.dumps({"data": [{"id": self.model}]}).encode() if body is None else body
        )
        return response

    def test_preflight_rejects_redirects_without_forwarding_auth(self):
        handler = ocr.NoEndpointRedirect()
        request = urllib.request.Request(self.endpoint + "/models", headers={"Authorization": "Bearer test-secret"})
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                self.assertIsNone(handler.redirect_request(request, None, code, "Redirect", {}, "http://other-host.invalid/models"))
                error = urllib.error.HTTPError(request.full_url, code, "Redirect", {}, None)
                with mock.patch.object(ocr.urllib.request.OpenerDirector, "open", side_effect=error):
                    with self.assertRaisesRegex(ocr.InputError, "final API address") as caught:
                        ocr.check_paddleocr_vl_endpoint(self.endpoint, self.model, api_key="test-secret")
                    self.assertNotIn("test-secret", str(caught.exception))

    def test_preflight_checks_models_with_saved_auth_and_short_timeout(self):
        with mock.patch.object(ocr.urllib.request.OpenerDirector, "open", return_value=self.response()) as request:
            ocr.check_paddleocr_vl_endpoint(self.endpoint, self.model, api_key="test-secret")
        self.assertEqual(request.call_args.args[0].full_url, self.endpoint + "/models")
        self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(request.call_args.kwargs["timeout"], 5)

    def test_network_errors_have_actionable_messages(self):
        for reason, expected in (
            (socket.gaierror(-2, "Name or service not known"), "Container names"),
            (ConnectionRefusedError(), "port is reachable"),
            (TimeoutError(), "within 5 seconds"),
        ):
            with self.subTest(reason=reason), mock.patch.object(ocr.urllib.request.OpenerDirector, "open", side_effect=urllib.error.URLError(reason)):
                with self.assertRaises(ocr.InputError) as error:
                    ocr.check_paddleocr_vl_endpoint(self.endpoint, self.model)
                self.assertIn(expected, str(error.exception))
                self.assertIn(self.endpoint, str(error.exception))
        with mock.patch.object(ocr.urllib.request.OpenerDirector, "open", side_effect=ConnectionRefusedError()):
            with self.assertRaisesRegex(ocr.InputError, "Did you forget to set the port"):
                ocr.check_paddleocr_vl_endpoint("http://host.docker.internal/v1", self.model)

    def test_http_errors_do_not_echo_keys_or_response_bodies(self):
        for status, expected in ((401, "OCR API key"), (403, "OCR API key"), (404, "ending in /v1"), (400, "another service"), (503, "returned an error")):
            error = urllib.error.HTTPError(self.endpoint, status, "test-secret", {}, io.BytesIO(b"test-secret"))
            with self.subTest(status=status), mock.patch.object(ocr.urllib.request.OpenerDirector, "open", side_effect=error):
                with self.assertRaises(ocr.InputError) as caught:
                    ocr.check_paddleocr_vl_endpoint(self.endpoint, self.model, api_key="test-secret")
                self.assertIn(expected, str(caught.exception))
                self.assertIn(f"Endpoint error: {status}", str(caught.exception))
                self.assertNotIn("test-secret", str(caught.exception))

    def test_wrong_service_malformed_json_and_wrong_model_fail(self):
        for body in (b"LanguageTool API", b"<html>Error</html>", b"{}", b"null", b'{"data":[null]}', b"x" * (1024 * 1024 + 1)):
            with self.subTest(body=body[:20]), mock.patch.object(ocr.urllib.request.OpenerDirector, "open", return_value=self.response(body)):
                with self.assertRaisesRegex(ocr.InputError, "different service"):
                    ocr.check_paddleocr_vl_endpoint(self.endpoint, self.model)
        with mock.patch.object(ocr.urllib.request.OpenerDirector, "open", return_value=self.response(b'{"data":[{"id":"other"}]}')):
            with self.assertRaisesRegex(ocr.InputError, "does not list model"):
                ocr.check_paddleocr_vl_endpoint(self.endpoint, self.model)

    def test_embedded_credentials_are_rejected_without_a_request(self):
        with mock.patch.object(ocr.urllib.request.OpenerDirector, "open") as request:
            with self.assertRaisesRegex(ocr.InputError, "auth token field"):
                ocr.check_paddleocr_vl_endpoint("http://user:secret@ocr/v1", self.model)
            request.assert_not_called()

    def test_endpoint_is_checked_before_local_or_worker_model_setup(self):
        for worker in (False, True):
            with self.subTest(worker=worker), mock.patch.object(ocr, "check_paddleocr_vl_endpoint", side_effect=ocr.InputError("Endpoint unavailable")), mock.patch.object(ocr, "configure_paddlex_runtime") as setup:
                with self.assertRaisesRegex(ocr.InputError, "Endpoint unavailable"):
                    if worker:
                        paddle_ocr_server.create_local_engine("paddleocr_vl", {"vl_rec_server_url": self.endpoint})
                    else:
                        ocr.create_paddleocr_vl("cpu", self.endpoint, self.model)
                setup.assert_not_called()

    def test_pipeline_checks_before_output_changes_or_extraction(self):
        config = types.SimpleNamespace(ocr=types.SimpleNamespace(
            engine="paddleocr_vl", service_url=None, paddleocr_vl_server_url=self.endpoint,
            paddleocr_vl_model=self.model, paddleocr_vl_api_key=None,
        ))
        with mock.patch.object(ocr, "check_paddleocr_vl_endpoint", side_effect=ocr.InputError("Endpoint unavailable")), mock.patch.object(translate_cbz, "prepare_output_dir") as prepare, mock.patch.object(translate_cbz, "extract_cbz") as extract:
            with self.assertRaisesRegex(ocr.InputError, "Endpoint unavailable"):
                translate_cbz.run_full_pipeline(types.SimpleNamespace(), config, {})
            prepare.assert_not_called()
            extract.assert_not_called()

    def test_regular_ocr_and_remote_worker_do_not_probe_from_wrong_host(self):
        with mock.patch.object(ocr, "check_paddleocr_vl_endpoint") as check:
            translate_cbz.check_ocr_endpoint(types.SimpleNamespace(engine="paddle"))
            translate_cbz.check_ocr_endpoint(types.SimpleNamespace(engine="paddleocr_vl", service_url="http://worker"))
            ocr.create_paddleocr_vl("cpu", service_url="http://worker")
            check.assert_not_called()

    def test_inference_failure_after_preflight_is_human_readable(self):
        for message, expected in (("Exception from the 'vlm' worker: Connection error.", "Could not connect"), ("401 Unauthorized", "OCR API key"), ("private-prompt", "could not process")):
            engine = mock.Mock(); engine.predict.side_effect = RuntimeError(message)
            with self.subTest(message=message), self.assertRaisesRegex(ocr.InputError, expected) as error:
                ocr.run_paddleocr_vl(engine, Path("crop.png"))
            self.assertNotIn("private-prompt", str(error.exception))


class OcrEditorSettingsTests(unittest.TestCase):
    def setUp(self):
        fixture = test_editor_reset.EditorResetTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.manager, self.code, self.job_id = fixture.manager, fixture.code, fixture.job_id
        self.app = web_app.create_app(fixture.root / "config/web_config.json")
        self.app.state.manager = self.manager
        self.job = f"/api/v1/jobs/{self.job_id}"
        self.crop = f"/api/job/{self.code}/{self.job_id}/editor/v2/ocr-crop"

    def run_http(self, exercise):
        async def direct(function, *args, **kwargs):
            return function(*args)

        async def run():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://testserver") as client:
                client.cookies.set(web_app.ADMIN_COOKIE_NAME, self.manager.authenticate_admin("changeme"))
                await exercise(client)

        with mock.patch.object(web_app, "run_in_threadpool", new=direct), mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(run())

    def test_saved_endpoint_workers_and_key_reach_editor_ocr_after_reload(self):
        async def exercise(client):
            endpoint = "http://127.0.0.1:18081/v1"
            status = self.manager.load_status(self.code, self.job_id)
            status["vlmModel"] = ""  # OCR settings do not require a translation model.
            self.manager.save_status(self.code, self.job_id, status)
            response = await client.patch(self.job + f"/settings?category={self.code}", json={
                "paddleocr_vl_server_url": endpoint, "ocr_page_workers": 3,
                "paddleocr_vl_auth_token": "private-ocr-key", "ocr_engine": "paddleocr_vl",
            })
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("private-ocr-key", response.text)
            restarted = web_app.JobManager(self.manager.config)
            restarted.initialize_web_state()
            self.app.state.manager = restarted
            client.cookies.set(web_app.ADMIN_COOKIE_NAME, restarted.authenticate_admin("changeme"))
            response = await client.get(self.job + f"?category={self.code}")
            self.assertEqual(response.json()["paddleocrVlServerUrl"], endpoint)
            self.assertEqual(response.json()["ocrPageWorkers"], 3)
            self.assertEqual(response.json()["vlmModel"], "")
            with mock.patch.object(ocr, "create_paddleocr_vl", side_effect=ocr.InputError("Could not connect to the PaddleOCR-VL endpoint.")) as create:
                response = await client.post(self.crop, json={"page": 0, "region": [0, 0, 20, 20]})
            self.assertEqual(response.status_code, 500)
            self.assertIn("Could not connect", response.json()["detail"])
            self.assertEqual(create.call_args.args[1], endpoint)
            self.assertEqual(create.call_args.kwargs["api_key"], "private-ocr-key")

        self.run_http(exercise)

    def test_late_connection_error_is_json_not_plain_server_error(self):
        async def exercise(client):
            engine = mock.Mock()
            engine.predict.side_effect = RuntimeError("Exception from the 'vlm' worker: Connection error.")
            with mock.patch.object(ocr, "create_paddleocr_vl", return_value=engine):
                response = await client.post(self.crop, json={"page": 0, "region": [0, 0, 20, 20]})
            self.assertEqual(response.status_code, 500)
            self.assertIn("Could not connect", response.json()["detail"])
            engine.close.assert_called_once()

        self.run_http(exercise)

    def test_job_page_connects_initial_and_polled_forms_to_autosave(self):
        async def exercise(client):
            response = await client.get(f"/job/{self.code}/{self.job_id}")
            self.assertEqual(response.status_code, 200)
            self.assertIn(f'data-job-options="{self.job}/settings?category={self.code}"', response.text)
            self.assertIn("Options save automatically.", response.text)
            self.assertIn("function initJobOptions()", response.text)
            self.assertGreaterEqual(response.text.count("initJobOptions();"), 2)

        self.run_http(exercise)


if __name__ == "__main__":
    unittest.main()
