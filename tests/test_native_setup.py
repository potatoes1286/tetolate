from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import urllib.error

import native_setup
import paddle_ocr_image
import translate_cbz
import web_app


class NativeSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.web = self.root / "config/web_config.json"
        self.pipeline = self.web.with_name("vlm_config.json")

    def run_setup(self, answers=(), *, no_start=True, model_error=None, extra=()):
        output = io.StringIO()
        args = ["native_setup.py", "--config", str(self.web), *extra, *(["--no-start"] if no_start else [])]
        def models(endpoint, key, label):
            if label == model_error:
                raise native_setup.SetupError("Synthetic endpoint failure")
            return ["vision-model"] if label == "VLM" else ["ocr-model"]
        with mock.patch.object(sys, "argv", args), mock.patch.object(native_setup, "check_magick"), \
             mock.patch("builtins.input", side_effect=answers), mock.patch.object(native_setup.getpass, "getpass", return_value="fixture-key"), \
             mock.patch.object(native_setup, "endpoint_models", side_effect=models) as probe, \
             mock.patch.object(native_setup.os, "execv") as start, redirect_stdout(output), redirect_stderr(output):
            result = native_setup.main()
        return result, output.getvalue(), start, probe

    def test_new_setup_creates_private_native_configs_used_by_pipeline_and_web(self):
        result, output, start, probe = self.run_setup(["", "", "", "http://127.0.0.1:18081/v1", ""])
        self.assertEqual(result, 0, output)
        start.assert_not_called()
        self.assertNotIn("fixture-key", output)
        self.assertEqual(probe.call_count, 2)
        for path in (self.web, self.pipeline):
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        pipeline = translate_cbz.load_config(self.pipeline, None)
        web = web_app.load_web_config(self.web)
        self.assertEqual(pipeline.vlm.base_url, "http://127.0.0.1:8080/v1")
        self.assertEqual(pipeline.ocr.paddleocr_vl_server_url, "http://127.0.0.1:18081/v1")
        self.assertEqual(pipeline.ocr.paddleocr_vl_api_key, "fixture-key")
        self.assertEqual(web.default_paddleocr_vl_server_url, pipeline.ocr.paddleocr_vl_server_url)
        self.assertEqual(web.default_paddleocr_vl_model, "ocr-model")
        self.assertEqual(web.listen_host, "127.0.0.1")

    def test_builtin_ocr_does_not_probe_or_require_an_ocr_service(self):
        result, output, start, probe = self.run_setup(["", "", "paddle"], no_start=False)
        self.assertEqual(result, 0, output)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(translate_cbz.load_config(self.pipeline, None).ocr.engine, "paddle")
        self.assertEqual(web_app.load_web_config(self.web).default_ocr_engine, "paddle")
        self.assertIn("127.0.0.1", start.call_args.args[1])

    def test_existing_configs_are_byte_identical_unless_user_chooses_update(self):
        self.web.parent.mkdir(parents=True)
        self.web.write_text('{"listen": "0.0.0.0:9000", "jobs_dir":"../jobs"}\n')
        self.pipeline.write_text('{"base_url":"http://vlm.example/v1", "model":"vision-model", "api_key":"fixture-key", "ocr":{"engine":"paddle"}}\n')
        before = {path: path.read_bytes() for path in (self.web, self.pipeline)}
        result, output, start, _ = self.run_setup([""], no_start=False, extra=["--listen", "--port", "9001"])
        self.assertEqual(result, 0, output)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(start.call_args.args[1][-4:], ["--address", "0.0.0.0", "--port", "9001"])
        result, output, _, _ = self.run_setup(["y", "http://new.example/v1", "", "paddle"])
        self.assertEqual(result, 0, output)
        self.assertEqual(json.loads(self.pipeline.read_text())["base_url"], "http://new.example/v1")
        self.assertEqual(self.web.read_bytes(), before[self.web])

    def test_endpoint_failure_or_cancellation_does_not_write_config_or_start(self):
        for service in ("VLM", "PaddleOCR-VL"):
            with self.subTest(service=service):
                result, output, start, _ = self.run_setup(["", "", "", ""], no_start=False, model_error=service)
                self.assertEqual(result, 1, output)
                self.assertFalse(self.pipeline.exists())
                self.assertFalse(self.web.exists())
                start.assert_not_called()
        result, output, _, _ = self.run_setup([EOFError()])
        self.assertEqual(result, 1, output)
        self.assertFalse(self.pipeline.exists())

    def test_saved_model_not_in_service_is_an_error_without_rewriting(self):
        self.web.parent.mkdir(parents=True)
        self.web.write_text("{}")
        self.pipeline.write_text(json.dumps({"base_url":"http://vlm.example/v1", "model":"missing-model"}))
        before = self.pipeline.read_bytes()
        result, output, start, _ = self.run_setup([""])
        self.assertEqual(result, 1)
        self.assertIn("saved VLM model", output)
        self.assertEqual(self.pipeline.read_bytes(), before)
        start.assert_not_called()

    def test_complete_configuration_is_validated_before_replacing_existing_files(self):
        self.web.parent.mkdir(parents=True)
        self.web.write_text("{}")
        self.pipeline.write_text(json.dumps({"base_url":"http://vlm.example/v1", "model":"vision-model",
                                             "output":{"webp_quality":"invalid"}, "ocr":{"engine":"paddle"}}))
        before = self.pipeline.read_bytes()
        result, output, start, _ = self.run_setup(["y", "http://new.example/v1", "", "paddle"])
        self.assertEqual(result, 1, output)
        self.assertEqual(self.pipeline.read_bytes(), before)
        start.assert_not_called()

    def test_api_keys_are_kept_only_for_the_same_endpoint(self):
        for changed in (False, True):
            pipeline = {"base_url":"http://vlm.example/v1", "api_key":"old-vlm-key", "model":"model",
                "ocr":{"engine":"paddleocr_vl", "paddleocr_vl_server_url":"http://ocr.example/v1",
                       "paddleocr_vl_api_key":"old-ocr-key", "paddleocr_vl_model":"model"}}
            answers = ["http://new-vlm.example/v1" if changed else "", "", "",
                       "http://new-ocr.example/v1" if changed else "", ""]
            with self.subTest(changed=changed), mock.patch("builtins.input", side_effect=answers), \
                 mock.patch.object(native_setup.getpass, "getpass", return_value=""), \
                 mock.patch.object(native_setup, "endpoint_models", return_value=["model"]) as probe, redirect_stdout(io.StringIO()):
                native_setup.configure(pipeline, edit=True)
                self.assertEqual(probe.call_args_list[0].args[1], "not-needed" if changed else "old-vlm-key")
                self.assertEqual(probe.call_args_list[1].args[1], "" if changed else "old-ocr-key")

    def test_server_address_defaults_and_container_binding_are_explicit(self):
        parser = web_app.server_parser()
        self.assertEqual(parser.parse_args([]).address, "127.0.0.1")
        self.assertEqual(parser.parse_args(["--listen"]).address, "0.0.0.0")
        self.assertEqual(parser.parse_args(["--address", "::1"]).address, "::1")
        for args, address in (([], "127.0.0.1"), (["--listen"], "0.0.0.0"), (["--address", "192.0.2.10"], "192.0.2.10")):
            with mock.patch.object(sys, "argv", ["web_app.py", *args]), \
                 mock.patch.object(web_app, "load_web_config", return_value=types.SimpleNamespace(listen_host="0.0.0.0", listen_port=9000)), \
                 mock.patch.object(web_app, "create_app"), mock.patch("uvicorn.run") as run:
                web_app.main()
            self.assertEqual(run.call_args.kwargs["host"], address)
            self.assertEqual(run.call_args.kwargs["port"], 9000)
        dockerfile = (native_setup.ROOT / "Dockerfile").read_text()
        self.assertIn('"/data/config/web_config.json", "--listen"]', dockerfile)

    def test_magick_checks_version_formats_and_font_rendering(self):
        for version, formats, expected in (
            ("ImageMagick 6.9", "", "ImageMagick 7"),
            ("ImageMagick 7.1", " PNG* PNG rw-\n WEBP* WEBP rw+\n", "JXL"),
            ("ImageMagick 7.1", " PNG* PNG rw-\n WEBP* WEBP rw+\n JXL* JXL r--\n", "JXL"),
            ("ImageMagick 7.1", " PNG* PNG rw-\n WEBP* WEBP rw+\n JXL* JXL rw+\n", None),
        ):
            with self.subTest(version=version, expected=expected), mock.patch.object(native_setup.shutil, "which", return_value="/bin/magick"), \
                 mock.patch.object(native_setup.subprocess, "run", side_effect=[types.SimpleNamespace(stdout=version), types.SimpleNamespace(stdout=formats), types.SimpleNamespace()]) as run, \
                 redirect_stdout(io.StringIO()):
                if expected:
                    with self.assertRaisesRegex(native_setup.SetupError, expected):
                        native_setup.check_magick()
                else:
                    native_setup.check_magick()
                    self.assertEqual(run.call_count, 3)

    def test_model_probe_is_bounded_and_does_not_print_credentials_or_error_bodies(self):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"data":[{"id":"model"}]}'
        with mock.patch.object(native_setup.urllib.request.OpenerDirector, "open", return_value=response) as request:
            self.assertEqual(native_setup.endpoint_models("http://example.test/v1/", "fixture-key", "VLM"), ["model"])
            self.assertEqual(request.call_args.kwargs["timeout"], 5)
            self.assertEqual(request.call_args.args[0].get_header("Authorization"), "Bearer fixture-key")
        for status in (302, 401, 404, 503):
            error = urllib.error.HTTPError("http://example.test", status, "fixture-key", {}, io.BytesIO(b"private-response"))
            with self.subTest(status=status), mock.patch.object(native_setup.urllib.request.OpenerDirector, "open", side_effect=error):
                with self.assertRaises(native_setup.SetupError) as caught:
                    native_setup.endpoint_models("http://example.test/v1", "fixture-key", "VLM")
                self.assertNotIn("fixture-key", str(caught.exception))
                self.assertNotIn("private-response", str(caught.exception))
        for body in (b"{}", b"null", b"html", b'{"data":[]}', b"x" * (1024 * 1024 + 1)):
            response.read.return_value = body
            with self.subTest(body=body[:20]), mock.patch.object(native_setup.urllib.request.OpenerDirector, "open", return_value=response):
                with self.assertRaises(native_setup.SetupError):
                    native_setup.endpoint_models("http://example.test/v1", "", "VLM")

    def test_invalid_endpoint_and_key_are_rejected_before_any_request(self):
        for endpoint, key in (("http://user:secret@example.test/v1", ""), ("http://example.test/v1?token=secret", ""),
                              ("file:///tmp/models", ""), ("http://example.test/v1", "key\nheader"), ("http://example.test/v1", "key\0header")):
            with self.subTest(endpoint=endpoint), mock.patch.object(native_setup.urllib.request.OpenerDirector, "open") as request:
                with self.assertRaises(native_setup.SetupError):
                    native_setup.endpoint_models(endpoint, key, "VLM")
                request.assert_not_called()

    def test_shell_setup_uses_managed_python_and_locked_dependencies(self):
        launcher = self.root / "setup.sh"
        shutil.copyfile(native_setup.ROOT / "setup.sh", launcher)
        for name in ("uv", "magick"):
            stub = self.root / name
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
            stub.chmod(0o755)
        result = subprocess.run(["bash", str(launcher), "--no-start"], env={**os.environ, "PATH":f"{self.root}:{os.environ['PATH']}"},
                                text=True, capture_output=True, check=True)
        commands = result.stdout.splitlines()
        self.assertEqual(commands[0], "python install 3.11")
        self.assertIn("sync --locked --managed-python --python 3.11 --extra inpaint --extra ocr", commands[1])
        self.assertIn("--no-start", commands[2])


if __name__ == "__main__":
    unittest.main()
