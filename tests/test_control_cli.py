from __future__ import annotations

import argparse
import contextlib
import io
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.error import HTTPError

import tetolate_cli as cli


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cli_runs_with_only_python_standard_library(self):
        result = subprocess.run([sys.executable, "-S", str(Path(cli.__file__)), "jobs", "submit", "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("--settings-file", result.stdout)

    def test_global_options_work_before_and_after_subcommands(self):
        for arguments in (["--json", "--server", "http://localhost:9999", "jobs", "list"],
                          ["jobs", "list", "--json", "--server", "http://localhost:9999"]):
            args = cli.parser().parse_args(arguments)
            self.assertTrue(args.json)
            self.assertEqual(args.server, "http://localhost:9999")

    def test_invalid_server_and_tokens_fail_without_echoing_credentials(self):
        for url in ("file:///tmp/file", "http://user:secret@host", "http://localhost:bad", "http://localhost?token=secret"):
            with self.subTest(url=url), self.assertRaises(cli.ClientError) as error:
                cli.Client(url)
            self.assertNotIn("secret", str(error.exception))
        with self.assertRaises(cli.ClientError) as error:
            cli.Client("http://localhost", "secret\nheader")
        self.assertNotIn("secret", str(error.exception))

    def test_settings_parse_types_and_credentials_require_a_file(self):
        args = SimpleNamespace(settings_file=None, set=["vlm_model=123", "pause_after_ocr=true", "ocr_page_workers=3"])
        self.assertEqual(cli.settings_changes(args), {"vlm_model": "123", "pause_after_ocr": True, "ocr_page_workers": 3})
        args.set = ["vlm_auth_token=private-secret"]
        with self.assertRaises(cli.ClientError) as error:
            cli.settings_changes(args)
        self.assertNotIn("private-secret", str(error.exception))
        args.settings_file = self.root / "missing.json"
        with self.assertRaises(cli.ClientError):
            cli.settings_changes(args)

    def test_saved_token_is_not_sent_when_server_changes(self):
        path = self.root / "cli.json"
        cli.save_config(path, {"server": "http://127.0.0.1:8088", "token": "saved-private-token"})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {"TETOLATE_CONTROL_TOKEN": ""}), mock.patch.object(cli, "Client") as factory:
            factory.return_value.token = None
            with contextlib.redirect_stderr(stderr):
                result = cli.main(["--config", str(path), "--server", "http://other.example", "--json", "status"])
            self.assertEqual(result, cli.EXIT_AUTH)
            self.assertEqual(factory.call_args.args[1], None)
        self.assertNotIn("saved-private-token", stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue())["error"]["code"], "unauthorized")

    def test_multipart_streams_archive_and_sets_exact_length(self):
        path = self.root / "book.cbz"
        path.write_bytes(b"file data" * 200000)
        body = cli.Multipart(path, {"category": "default", "settings": "{}"})
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Do not buffer uploads")):
            chunks = list(body)
        self.assertEqual(sum(len(chunk) for chunk in chunks), body.length)
        self.assertGreater(len(chunks), 3)
        self.assertIn(b'name="file"; filename="book.cbz"', chunks[0])
        self.assertEqual(b"".join(chunks[1:-1]), path.read_bytes())

    def test_wait_returns_terminal_failure_pause_and_timeout_codes(self):
        args = SimpleNamespace(job_id="12345678", category=None, timeout=1, interval=0.1)
        client = mock.Mock(timeout=30)
        for state, code in (("failed", 1), ("cancelled", 1), ("paused", 1)):
            client.request.return_value = {"status": state, "message": "stopped"}
            with self.subTest(state=state), self.assertRaises(cli.ClientError) as error:
                cli.wait_job(client, args)
            self.assertEqual(error.exception.exit_code, code)
        client.request.return_value = {"status": "complete", "jobId": "12345678"}
        self.assertEqual(cli.wait_job(client, args)["jobId"], "12345678")
        client.request.return_value = {"status": "queued"}
        with mock.patch.object(cli.time, "monotonic", side_effect=[0, 0.1, 1.1, 1.2]), mock.patch.object(cli.time, "sleep"):
            with self.assertRaises(cli.ClientError) as error:
                cli.wait_job(client, args)
        self.assertEqual(error.exception.exit_code, cli.EXIT_TIMEOUT)

    def test_log_follow_is_cursor_based_and_emits_json_lines(self):
        args = SimpleNamespace(job_id="12345678", category="default", timeout=10, interval=0.01, json=True, follow=True)
        client = mock.Mock(timeout=30)
        client.request.side_effect = [
            {"text": "same\n", "cursor": "42:5", "hasMore": True, "status": "running", "reset": False},
            {"text": "same\n", "cursor": "42:10", "hasMore": False, "status": "complete", "reset": False},
        ]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli.read_logs(client, args)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual("".join(item["text"] for item in events), "same\nsame\n")
        self.assertIn("cursor=42%3A5", client.request.call_args_list[1].args[1])

    def test_download_failure_preserves_existing_output(self):
        client = cli.Client("http://localhost:8088", "test-token")
        output = self.root / "book.cbz"
        output.write_bytes(b"old archive")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {}
        response.read.side_effect = [b"partial", OSError("connection closed")]
        with mock.patch.object(client, "open", return_value=response):
            with self.assertRaises(cli.ClientError) as error:
                client.download("/jobs/12345678/download/png", output)
            self.assertEqual(error.exception.exit_code, cli.EXIT_CONNECTION)
        self.assertEqual(output.read_bytes(), b"old archive")
        self.assertEqual(list(self.root.glob(".tetolate-download-*")), [])

    def test_http_auth_failure_has_a_stable_exit_code(self):
        client = cli.Client("http://localhost:8088", "test-token")
        error = HTTPError("http://localhost", 401, "Unauthorized", {}, io.BytesIO(b'{"error":{"code":"unauthorized","message":"Revoked token"}}'))
        with mock.patch.object(client.opener, "open", side_effect=error):
            with self.assertRaises(cli.ClientError) as caught:
                client.request("GET", "/status")
        self.assertEqual(caught.exception.exit_code, cli.EXIT_AUTH)
        self.assertEqual(caught.exception.code, "unauthorized")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.side_effect = IncompleteRead(b'{"unfinished":')
        with mock.patch.object(client, "open", return_value=response):
            with self.assertRaises(cli.ClientError) as caught:
                client.request("GET", "/status")
        self.assertEqual(caught.exception.exit_code, cli.EXIT_CONNECTION)

    def execute(self, arguments, client=None):
        args = cli.parser().parse_args(arguments)
        client = client or mock.Mock(token="test-token", server="http://localhost:8088", timeout=30)
        result = cli.execute(client, args, self.root / "cli.json", {})
        return client, result

    def test_title_page_and_rerun_commands_send_correct_requests(self):
        for arguments, method, suffix, data in (
            (["set-translated-title", "12345678", "Manual"], "PATCH", "/translated-title", {"name": "Manual"}),
            (["set-translated-title", "12345678", ""], "PATCH", "/translated-title", {"name": ""}),
            (["translate-title", "12345678"], "POST", "/translate-title", None),
            (["retry-title-translation", "12345678", "--manual-title", "Manual"], "POST", "/retry-title-translation", {"manual_title": "Manual"}),
            (["rerun", "12345678", "--from", "ocr_merge", "--pages", "0-2,2,5"], "POST", "/rerun", {"stage": "ocr_merge", "pages": [0, 1, 2, 5]}),
        ):
            client, _ = self.execute(["jobs", *arguments])
            self.assertEqual(client.request.call_args.args[:2], (method, "/jobs/12345678" + suffix))
            if data is not None:
                self.assertEqual(client.request.call_args.args[2], data)
        client, _ = self.execute(["jobs", "pages", "12345678", "--kind", "original", "--category", "default"])
        client.request.assert_called_once_with("GET", "/jobs/12345678/pages?category=default&kind=original")
        output = self.root / "page.png"
        client, _ = self.execute(["jobs", "page", "12345678", "0", "--output", str(output)])
        client.download.assert_called_once_with("/jobs/12345678/pages/0?kind=final", output)
        for bad in ("-1", "2-1", "1-2-3", "0-100001", "1,a"):
            with self.subTest(bad=bad), self.assertRaises(cli.ClientError):
                cli.page_selection(bad)

    def test_multiple_files_stream_with_exact_length_and_page_order(self):
        first, second = self.root / 'z".png', self.root / "a.png"
        first.write_bytes(b"first" * 200000)
        second.write_bytes(b"second")
        body = cli.Multipart([("page_images", first), ("page_images", second)], {"settings": "{}"})
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Do not buffer uploads")):
            chunks = list(body)
        content = b"".join(chunks)
        self.assertEqual(len(content), body.length)
        self.assertEqual(content.count(b'name="page_images"'), 2)
        self.assertLess(content.index(b'filename="z_.png"'), content.index(b'filename="a.png"'))
        self.assertIn(b"second\r\n--tetolate-", content)

    def test_image_and_series_upload_commands(self):
        first, second = self.root / "2.png", self.root / "1.png"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        client, _ = self.execute(["jobs", "submit", "--images", str(first), str(second)])
        body = client.request.call_args.kwargs["multipart"]
        self.assertEqual(client.request.call_args.args, ("POST", "/jobs"))
        self.assertEqual([path for _, path in body.parts], [first, second])
        with self.assertRaises(cli.ClientError):
            self.execute(["jobs", "submit"])
        with self.assertRaises(cli.ClientError):
            self.execute(["jobs", "submit", str(first), "--images", str(second)])
        client, _ = self.execute(["series", "upload", "abc123", str(first), str(second), "--category", "default", "--first-chapter", "2.5"])
        self.assertEqual(client.request.call_args.args, ("POST", "/categories/default/series/abc123/upload"))
        body = client.request.call_args.kwargs["multipart"]
        self.assertIn(b"2.5", b"".join(body))
        self.assertEqual([path for _, path in body.parts], [first, second])

    def test_vlm_test_reads_credentials_from_file(self):
        path = self.root / "settings.json"
        path.write_text(json.dumps({"vlm_auth_token": "private-key"}))
        client, _ = self.execute(["vlm", "test", "--endpoint", "http://localhost:5001/v1", "--job", "12345678", "--settings-file", str(path)])
        client.request.assert_called_once_with("POST", "/vlm/test", {
            "vlm_auth_token": "private-key", "vlm_base_url": "http://localhost:5001/v1", "job_id": "12345678",
        })

    def test_password_change_reads_three_lines_without_saving_or_printing_them(self):
        with mock.patch.object(cli.sys, "stdin", io.StringIO("old-secret\nnew-secret\nnew-secret\n")):
            client, _ = self.execute(["password", "--password-stdin"])
        client.request.assert_called_once_with("POST", "/auth/password", {
            "current_password": "old-secret", "new_password": "new-secret", "confirm_password": "new-secret",
        })
        self.assertFalse((self.root / "cli.json").exists())
        with mock.patch.object(cli.sys, "stdin", io.StringIO("old-secret\n")):
            with self.assertRaises(cli.ClientError) as error:
                self.execute(["password", "--password-stdin"])
            self.assertNotIn("old-secret", str(error.exception))

    def test_create_token_does_not_replace_saved_login(self):
        path = self.root / "cli.json"
        cli.save_config(path, {"server": "http://localhost:8088", "token": "existing-token"})
        args = cli.parser().parse_args(["tokens", "create", "agent", "--read-only", "--password-stdin"])
        client = cli.Client("http://localhost:8088", "existing-token")
        with mock.patch.object(cli.sys, "stdin", io.StringIO("admin-secret\n")), mock.patch.object(cli.Client, "request", return_value={"token": "new-token"}) as request:
            result = cli.execute(client, args, path, cli.load_config(path))
        self.assertEqual(result["token"], "new-token")
        request.assert_called_once_with("POST", "/auth/login", {"password": "admin-secret", "name": "agent", "read_only": True})
        self.assertEqual(cli.load_config(path)["token"], "existing-token")


if __name__ == "__main__":
    unittest.main()
