from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import types
import unittest
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

import clean_text_regions
import editor_runtime
import editor_v2
import lama_inpaint
import merge_text_json
import overlay_text
import ocr_merge
import paddle_ocr_image
import paddle_ocr_server
import placement_detection
import prompt_templates
import translate_cbz
import vlm_client
import web_app
import web_pages
import web_security


class RepositoryHygieneTests(unittest.TestCase):
    def test_checked_in_examples_do_not_contain_credentials(self) -> None:
        repo_dir = Path(__file__).resolve().parents[1]
        config_dir = repo_dir / "data" / "config"
        vlm_config = json.loads(
            (config_dir / "vlm_config.example.json").read_text(encoding="utf-8")
        )
        web_config = json.loads(
            (config_dir / "web_config.example.json").read_text(encoding="utf-8")
        )

        self.assertIn(vlm_config.get("api_key"), {None, "", "not-needed"})
        self.assertEqual(vlm_config["temperature"], 0.7)
        self.assertEqual(vlm_config["timeout"], 1000)
        self.assertEqual(vlm_config["output"], {"webp_quality": 70, "jxl_quality": 65})
        self.assertNotIn("lama", vlm_config)
        self.assertNotIn("render", vlm_config)
        self.assertNotIn("language", vlm_config)
        self.assertEqual(set(vlm_config["backup_font"]), {"font", "fill", "gravity"})
        self.assertEqual(
            set(vlm_config["default_language"]),
            {"source", "target"},
        )
        self.assertEqual(set(vlm_config["ocr"]), {"min_score"})
        self.assertEqual(vlm_config["ocr"]["min_score"], 0.75)
        self.assertEqual(
            set(web_config),
            {
                "listen",
                "jobs_dir",
                "max_upload_bytes",
            },
        )

    def test_web_server_defaults_to_loopback(self) -> None:
        self.assertEqual(web_app.DEFAULT_LISTEN, "127.0.0.1:8088")

    def test_web_listen_supports_hostname_and_bracketed_ipv6(self) -> None:
        self.assertEqual(web_app.parse_listen("localhost:9000"), ("localhost", 9000))
        self.assertEqual(web_app.parse_listen("[::1]:9000"), ("::1", 9000))

    def test_pipeline_runtime_defaults_match_the_public_example(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "vlm_config.json"
            config_path.write_text(
                json.dumps({"base_url": "http://127.0.0.1:8080/v1"}),
                encoding="utf-8",
            )
            config = translate_cbz.load_config(config_path, None)

        assert config.vlm is not None
        self.assertEqual(config.vlm.temperature, 0.7)
        self.assertEqual(config.vlm.timeout, 1000)
        self.assertEqual(config.webp_quality, 70)
        self.assertEqual(config.jxl_quality, 65)
        self.assertEqual(config.ocr.min_score, 0.75)
        self.assertEqual(config.ocr.engine, paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL)
        self.assertEqual(config.render_font, "DejaVu-Sans")

    def test_hidden_ocr_options_are_not_read_from_vlm_config(self) -> None:
        language = translate_cbz.language_config_from_codes("jp", "en")
        config = translate_cbz.load_ocr_config(
            {
                "min_score": 0.8,
                "engine": "paddle",
                "service_url": "http://ignored.invalid",
                "tile_enabled": True,
            },
            language,
        )

        self.assertEqual(config.min_score, 0.8)
        self.assertEqual(config.engine, paddle_ocr_image.DEFAULT_OCR_ENGINE)
        self.assertEqual(config.service_url, paddle_ocr_image.DEFAULT_OCR_SERVICE_URL)
        self.assertEqual(config.tile_enabled, paddle_ocr_image.DEFAULT_TILE_ENABLED)

    def test_missing_user_font_uses_backup_font(self) -> None:
        self.assertEqual(
            translate_cbz.normalize_font_name(None, "DejaVu-Sans"),
            "DejaVu-Sans",
        )

    def test_quality_config_rejects_fractional_values(self) -> None:
        with self.assertRaises(translate_cbz.PipelineError):
            translate_cbz.config_quality(
                {"quality": 70.9}, "quality", 65, "output"
            )
        with self.assertRaises(translate_cbz.PipelineError):
            translate_cbz.config_quality(
                {"quality": "70.9"}, "quality", 65, "output"
            )

    def test_quality_config_accepts_integral_float_values(self) -> None:
        self.assertEqual(
            translate_cbz.config_quality({"quality": 70.0}, "quality", 65, "output"),
            70,
        )


class VLMModelDiscoveryTests(unittest.TestCase):
    @staticmethod
    def vlm_config(model: str | None = None) -> translate_cbz.VLMConfig:
        return translate_cbz.VLMConfig(
            base_url="http://vlm.example/v1",
            api_key="not-needed",
            model=model,
            temperature=0.7,
            max_tokens=1024,
            thinking_budget_tokens=0,
            timeout=30,
            provider=None,
        )

    def test_list_vlm_model_ids_deduplicates_and_closes_client(self) -> None:
        client = mock.Mock()
        client.models.list.return_value = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(id="first"),
                {"id": "second"},
                types.SimpleNamespace(id="first"),
                types.SimpleNamespace(id="  third  "),
                types.SimpleNamespace(id=""),
            ]
        )
        with mock.patch.object(vlm_client, "openai_client", return_value=client):
            result = vlm_client.list_vlm_model_ids(self.vlm_config())

        self.assertEqual(result, ["first", "second", "third"])
        client.models.list.assert_called_once_with()
        client.close.assert_called_once_with()

    def test_list_vlm_model_ids_raises_safe_error_and_closes_client(self) -> None:
        client = mock.Mock()
        client.models.list.side_effect = RuntimeError("connection refused: secret-token")
        with mock.patch.object(vlm_client, "openai_client", return_value=client):
            with self.assertRaisesRegex(vlm_client.PipelineError, "Could not list VLM models") as raised:
                vlm_client.list_vlm_model_ids(self.vlm_config())

        self.assertNotIn("secret-token", str(raised.exception))
        client.close.assert_called_once_with()

    def test_list_vlm_model_ids_rejects_empty_results(self) -> None:
        client = mock.Mock()
        client.models.list.return_value = types.SimpleNamespace(data=[])
        with mock.patch.object(vlm_client, "openai_client", return_value=client):
            with self.assertRaisesRegex(vlm_client.PipelineError, "No VLM models"):
                vlm_client.list_vlm_model_ids(self.vlm_config())

        client.close.assert_called_once_with()

    def test_vlm_model_override_avoids_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "vlm_config.json"
            config_path.write_text(
                json.dumps({"base_url": "http://vlm.example/v1"}), encoding="utf-8"
            )
            config = translate_cbz.load_config(config_path, None)

        overridden = translate_cbz.apply_vlm_model_override(config, " selected-model ")
        assert overridden.vlm is not None
        with mock.patch.object(vlm_client, "list_vlm_model_ids") as list_models:
            self.assertEqual(vlm_client.resolve_vlm_model(overridden.vlm), "selected-model")

        list_models.assert_not_called()

    def test_vlm_model_override_requires_a_nonempty_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "vlm_config.json"
            config_path.write_text(
                json.dumps({"base_url": "http://vlm.example/v1"}), encoding="utf-8"
            )
            config = translate_cbz.load_config(config_path, None)

        with self.assertRaisesRegex(translate_cbz.PipelineError, "non-empty model ID"):
            translate_cbz.apply_vlm_model_override(config, "   ")


class PromptTemplateTests(unittest.TestCase):
    def test_prompt_files_are_valid_and_all_builders_render(self) -> None:
        language = translate_cbz.language_config_from_codes("jp", "en")
        page = translate_cbz.Page(index=0, image_path=Path("unused.png"))
        record = {
            "page": 0,
            "boxno": 0,
            "sourceBoxnos": [3],
            "region": [10, 20, 40, 80],
            "text": "source",
            "englishText": "translation",
            "sfx": False,
            "openLettering": True,
        }
        by_page = {0: [record]}
        notes = {"job": "Keep names consistent.", "pages": {}}
        placements = [
            {
                "page": 0,
                "boxno": 0,
                "box_2d": [10, 20, 40, 80],
                "placementRegion": [2, 3, 8, 10],
            }
        ]

        rendered = (
            translate_cbz.structure_prompt(page, by_page, language),
            translate_cbz.alt_placement_prompt(page, [record], language),
            translate_cbz.translation_prompt(page, by_page, notes, language),
            translate_cbz.proofreading_prompt(by_page, notes, language),
            translate_cbz.generated_translation_notes_prompt(
                by_page, notes, language
            ),
            translate_cbz.placement_open_prompt(page, [record], language),
            translate_cbz.placement_style_prompt(
                page, [record], placements, language, "DejaVu-Sans"
            ),
            translate_cbz.prompt_with_validation_feedback(
                "original", translate_cbz.PipelineError("invalid output")
            ),
        )

        self.assertTrue(all(item.strip() for item in rendered))
        self.assertFalse(any("${" in item for item in rendered))

    def test_prompt_files_reload_without_process_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            "os.environ", {"TETOLATE_PROMPTS_DIR": temp_dir}
        ):
            path = Path(temp_dir) / "test.txt"
            path.write_text("First ${value}", encoding="utf-8")
            self.assertEqual(
                prompt_templates.load_prompt("test.txt", value="result"),
                "First result",
            )
            path.write_text("Second ${value}", encoding="utf-8")
            self.assertEqual(
                prompt_templates.load_prompt("test.txt", value="result"),
                "Second result",
            )

    def test_missing_prompt_value_is_a_pipeline_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            "os.environ", {"TETOLATE_PROMPTS_DIR": temp_dir}
        ):
            (Path(temp_dir) / "test.txt").write_text(
                "Missing ${required}", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                translate_cbz.PipelineError, "requires missing value"
            ):
                prompt_templates.load_prompt("test.txt")

    def test_proofreading_batches_keep_pages_together(self) -> None:
        records = [
            {"page": 0, "boxno": 0, "text": "a", "englishText": "A"},
            {"page": 0, "boxno": 1, "text": "b", "englishText": "B"},
            {"page": 1, "boxno": 0, "text": "c", "englishText": "C"},
            {"page": 2, "boxno": 0, "text": "d", "englishText": "D"},
        ]

        batches = translate_cbz.proofreading_record_batches(
            records,
            max_records=2,
            max_characters=10_000,
        )

        self.assertEqual(
            [[record["page"] for record in batch] for batch in batches],
            [[0, 0], [1, 2]],
        )

    def test_proofreading_batch_validator_applies_changed_rows_only(self) -> None:
        records = [
            {"page": 3, "boxno": 4, "text": "a", "englishText": "A"},
            {"page": 3, "boxno": 5, "text": "b", "englishText": "B"},
        ]

        corrected = translate_cbz.validate_proofread_record_batch(
            records,
            "0\tCorrected\n",
        )

        self.assertEqual(corrected[0]["englishText"], "Corrected")
        self.assertEqual(corrected[1]["englishText"], "B")
        unchanged = translate_cbz.validate_proofread_record_batch(
            records,
            "<NO_CHANGES>\n",
        )
        self.assertEqual([item["englishText"] for item in unchanged], ["A", "B"])
        with self.assertRaisesRegex(translate_cbz.PipelineError, "row<TAB>"):
            translate_cbz.validate_proofread_record_batch(records, "Only one\n")
        with self.assertRaisesRegex(translate_cbz.PipelineError, "must not remove"):
            translate_cbz.validate_proofread_record_batch(records, "1\t<EMPTY>\n")


class WebAuthenticationTests(unittest.TestCase):
    ADMIN_PASSWORD = "test-admin-password"

    def test_job_runtime_accumulates_active_segments(self) -> None:
        first_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        status: dict[str, object] = {"elapsedSeconds": 10.0}

        web_app.start_active_runtime(status, first_start)
        web_app.stop_active_runtime(status, first_start + timedelta(seconds=50))
        self.assertEqual(status["elapsedSeconds"], 60.0)
        self.assertNotIn("activeStartedAt", status)

        second_start = first_start + timedelta(minutes=10)
        web_app.start_active_runtime(status, second_start)
        timing = web_app.job_timing(
            status,
            second_start + timedelta(seconds=25),
        )
        self.assertEqual(timing.elapsed_seconds, 85.0)

        web_app.stop_active_runtime(status, second_start + timedelta(seconds=40))
        self.assertEqual(status["elapsedSeconds"], 100.0)

    def test_recent_log_keeps_one_thousand_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "job.log"
            log_path.write_text(
                "".join(f"line {index}\n" for index in range(1205)),
                encoding="utf-8",
            )

            lines = web_app.safe_log_lines(log_path)

        self.assertEqual(len(lines), 1000)
        self.assertEqual(lines[0], "line 205")
        self.assertEqual(lines[-1], "line 1204")

    def test_job_page_uses_wide_tall_log(self) -> None:
        response = web_pages.job_page(
            "default",
            "12345678",
            {"status": "running", "recentLog": ["first", "second"]},
        )
        body = response.body.decode("utf-8")

        self.assertIn('<body class="wide-page">', body)
        self.assertIn('<pre id="log" class="job-log">', body)
        self.assertIn("height: clamp(32rem, 68vh, 64rem)", body)

    def write_web_config(self, root: Path) -> Path:
        repo_dir = Path(__file__).resolve().parents[1]
        data: dict[str, object] = {
            "listen": "127.0.0.1:8088",
            "jobs_dir": str(root / "jobs"),
            "max_upload_bytes": 1024 * 1024,
        }
        pipeline_data = json.loads(
            (repo_dir / "data" / "config" / "vlm_config.example.json").read_text(
                encoding="utf-8"
            )
        )
        pipeline_data["model"] = "test-vlm-model"
        (root / "vlm_config.json").write_text(
            json.dumps(pipeline_data),
            encoding="utf-8",
        )
        path = root / "web_config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def initialized_manager(
        self,
        root: Path,
        categories: list[str] | None = None,
    ) -> web_app.JobManager:
        config = web_app.load_web_config(self.write_web_config(root))
        manager = web_app.JobManager(config)
        config.jobs_dir.mkdir(parents=True)
        web_security.write_private_json_atomic(
            manager.web_state_path(),
            {
                "version": 1,
                "adminPasswordHash": web_security.hash_password(self.ADMIN_PASSWORD),
                "categories": categories or [],
            },
        )
        manager.initialize_web_state()
        return manager

    def test_unknown_web_config_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self.write_web_config(root)
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data["unused_setting"] = True
            config_path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "unused_setting"):
                web_app.load_web_config(config_path)

    def test_delete_job_reports_directory_deletion_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.initialized_manager(Path(temp_dir), ["default"])
            job_id = "12345678"
            manager.job_dir("default", job_id).mkdir(parents=True)
            manager.save_status("default", job_id, {"status": "failed"})

            with mock.patch(
                "web_app.shutil.rmtree",
                side_effect=PermissionError("permission denied"),
            ):
                with self.assertRaisesRegex(HTTPException, "Could not delete job") as raised:
                    manager.delete_job("default", job_id)

            self.assertEqual(raised.exception.status_code, 500)

    def test_upload_removes_job_directory_when_submission_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self.write_web_config(root)
            manager = self.initialized_manager(root, ["default"])
            app = web_app.create_app(config_path)
            app.state.manager = manager

            archive_bytes = io.BytesIO()
            with zipfile.ZipFile(archive_bytes, "w") as archive:
                image = io.BytesIO()
                Image.new("RGB", (2, 2), "white").save(image, format="PNG")
                archive.writestr("001.png", image.getvalue())

            async def exercise_upload() -> None:
                upload_endpoint = next(
                    route.endpoint for route in app.routes if route.path == "/upload"
                )
                request = web_app.Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/upload",
                        "raw_path": b"/upload",
                        "scheme": "http",
                        "query_string": b"",
                        "headers": [],
                        "server": ("testserver", 80),
                        "client": ("testclient", 123),
                        "root_path": "",
                        "app": app,
                    }
                )
                class FakeUpload:
                    filename = "comic.cbz"

                    def __init__(self, payload: bytes) -> None:
                        self.payload = payload

                    async def read(self, _size: int) -> bytes:
                        payload, self.payload = self.payload, b""
                        return payload

                    async def close(self) -> None:
                        return None

                cbz = FakeUpload(archive_bytes.getvalue())
                with mock.patch.object(
                    manager,
                    "submit_job",
                    side_effect=RuntimeError("queue unavailable"),
                ):
                    async def direct_run_in_threadpool(function: object, *args: object) -> object:
                        return function(*args)  # type: ignore[operator]

                    with mock.patch(
                        "web_app.run_in_threadpool",
                        side_effect=direct_run_in_threadpool,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "queue unavailable"):
                            await upload_endpoint(
                                request,
                                category="default",
                                cbz=cbz,
                                page_images=None,
                                translation_notes="",
                                thinking_budget_tokens="",
                                vlm_base_url="",
                                vlm_model="vision-model",
                                pause_after_ocr=None,
                                enable_alt_placement=None,
                                enable_proofreading=None,
                                enable_translation_notes=None,
                                source_language="",
                                ocr_engine="",
                                paddleocr_vl_server_url="",
                                paddleocr_vl_model="",
                                ocr_page_workers="1",
                                lama_workers="1",
                                imagemagick_workers="1",
                                vlm_auth_token="",
                                paddleocr_vl_auth_token="",
                            )

            asyncio.run(exercise_upload())
            self.assertEqual(list(manager.jobs_dir("default").iterdir()), [])

    def test_upload_request_size_limit_rejects_declared_oversize_before_parsing(self) -> None:
        downstream_called = False
        sent_messages: list[dict[str, object]] = []

        class Config:
            max_upload_bytes = 3

        class Manager:
            config = Config()

        async def downstream(scope: dict[str, object], receive: object, send: object) -> None:
            nonlocal downstream_called
            downstream_called = True

        downstream.state = types.SimpleNamespace(manager=Manager())  # type: ignore[attr-defined]

        async def receive() -> dict[str, object]:
            raise AssertionError("the request body should not be read")

        async def send(message: dict[str, object]) -> None:
            sent_messages.append(message)

        async def exercise() -> None:
            middleware = web_app.UploadSizeLimitMiddleware(downstream)
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/upload",
                    "headers": [(b"content-length", b"4")],
                },
                receive,
                send,
            )

        asyncio.run(exercise())
        self.assertFalse(downstream_called)
        self.assertEqual(sent_messages[0]["status"], 413)

    def test_upload_request_size_limit_rejects_oversize_stream(self) -> None:
        sent_messages: list[dict[str, object]] = []
        receive_calls = 0

        class Config:
            max_upload_bytes = 3

        class Manager:
            config = Config()

        async def downstream(scope: dict[str, object], receive: object, send: object) -> None:
            await receive()  # type: ignore[misc]
            await receive()  # type: ignore[misc]

        downstream.state = types.SimpleNamespace(manager=Manager())  # type: ignore[attr-defined]

        async def receive() -> dict[str, object]:
            nonlocal receive_calls
            receive_calls += 1
            return {"type": "http.request", "body": b"12", "more_body": True}

        async def send(message: dict[str, object]) -> None:
            sent_messages.append(message)

        async def exercise() -> None:
            middleware = web_app.UploadSizeLimitMiddleware(downstream)
            await middleware(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/upload",
                    "headers": [],
                },
                receive,
                send,
            )

        asyncio.run(exercise())
        self.assertEqual(receive_calls, 2)
        self.assertEqual(sent_messages[0]["status"], 413)

    def test_password_hash_round_trip_and_strict_parsing(self) -> None:
        encoded = web_security.hash_password("a sufficiently long password")

        self.assertTrue(
            web_security.verify_password("a sufficiently long password", encoded)
        )
        self.assertFalse(web_security.verify_password("wrong password", encoded))
        with self.assertRaises(ValueError):
            web_security.parse_password_hash("not-a-password-hash")

    def test_first_start_generates_password_and_private_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = web_app.load_web_config(self.write_web_config(root))
            manager = web_app.JobManager(config)
            config.jobs_dir.mkdir(parents=True)
            output = io.StringIO()

            with mock.patch("sys.stderr", output):
                manager.initialize_web_state()

            generated = output.getvalue().split("shown once): ", 1)[1].strip()
            state_path = manager.web_state_path()
            state = web_security.read_json_object(state_path)
            self.assertTrue(
                web_security.verify_password(generated, state["adminPasswordHash"])
            )
            self.assertEqual(state_path.stat().st_mode & 0o777, 0o600)

    def test_generated_password_is_printed_before_state_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = web_app.load_web_config(self.write_web_config(root))
            manager = web_app.JobManager(config)
            config.jobs_dir.mkdir(parents=True)
            output = io.StringIO()

            with (
                mock.patch("sys.stderr", output),
                mock.patch.object(
                    manager,
                    "save_web_state",
                    side_effect=OSError("simulated write failure"),
                ),
                self.assertRaisesRegex(OSError, "simulated write failure"),
            ):
                manager.initialize_web_state()

            self.assertIn("generated admin password (shown once):", output.getvalue())

    def test_sessions_are_individual_and_revocable(self) -> None:
        manager = object.__new__(web_app.JobManager)
        manager._lock = threading.RLock()
        manager._admin_sessions = {}

        first = manager.create_admin_session()
        second = manager.create_admin_session()
        self.assertTrue(manager.admin_session_is_valid(first))
        self.assertTrue(manager.admin_session_is_valid(second))

        manager.revoke_admin_session(first)
        self.assertFalse(manager.admin_session_is_valid(first))
        self.assertTrue(manager.admin_session_is_valid(second))

    def test_password_change_invalidates_existing_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self.initialized_manager(root)
            old_session = manager.create_admin_session()

            new_session = manager.change_admin_password(
                self.ADMIN_PASSWORD,
                "x",
                "x",
            )

            self.assertFalse(manager.admin_session_is_valid(old_session))
            self.assertTrue(manager.admin_session_is_valid(new_session))
            self.assertFalse(manager.admin_password_matches(self.ADMIN_PASSWORD))
            self.assertTrue(manager.admin_password_matches("x"))

    def test_categories_can_be_created_and_cascade_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self.initialized_manager(root)

            manager.create_category("volume-one")
            job_id = "12345678"
            manager.job_dir("volume-one", job_id).mkdir(parents=True)
            manager.save_status("volume-one", job_id, {"status": "complete"})
            manager.delete_category("volume-one", "volume-one")

            self.assertNotIn("volume-one", manager.categories())
            self.assertFalse(manager.category_dir("volume-one").exists())

    def test_vlm_endpoint_is_remembered_and_passed_to_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self.initialized_manager(root, ["default"])
            manager.category_dir("default").mkdir(parents=True, exist_ok=True)
            endpoint = "http://vlm.internal:9000/v1"
            manager.remember_category_advanced_options(
                "default",
                thinking_budget_tokens=256,
                vlm_base_url=endpoint,
                vlm_model="vision-model",
                pause_after_ocr=False,
                proofread_translations=True,
                write_translation_notes=True,
                alt_placement_enabled=True,
                source_language="jp",
                ocr_engine=paddle_ocr_image.OCR_ENGINE_PADDLEOCR_VL,
                paddleocr_vl_server_url="http://ocr.internal:8081/v1",
                paddleocr_vl_model="test-ocr-model",
            )
            options = manager.load_category_advanced_options("default")

            job_id = "12345678"
            manager.job_dir("default", job_id).mkdir(parents=True)
            manager.save_status(
                "default",
                job_id,
                {
                    "status": "failed",
                    "vlmBaseUrl": endpoint,
                    "vlmModel": "vision-model",
                },
            )
            command = manager.build_command("default", job_id)

        self.assertEqual(options["vlmBaseUrl"], endpoint)
        endpoint_index = command.index("--vlm-base-url")
        self.assertEqual(command[endpoint_index + 1], endpoint)

    def test_job_auth_tokens_are_private_and_passed_by_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self.initialized_manager(root, ["default"])
            code, job_id = "default", "12345678"
            manager.job_dir(code, job_id).mkdir(parents=True)
            manager.save_status(code, job_id, {"status": "failed"})
            manager.update_job_auth_tokens(
                code,
                job_id,
                vlm_auth_token="test-vlm-token",
                paddleocr_vl_auth_token="test-ocr-token",
                preserve_existing=False,
            )

            secret_path = manager.job_secrets_path(code, job_id)
            self.assertEqual(secret_path.stat().st_mode & 0o777, 0o600)
            public = manager.public_status(code, job_id, include_log=False)
            self.assertTrue(public["hasVlmAuthToken"])
            self.assertTrue(public["hasPaddleocrVlAuthToken"])
            self.assertNotIn("test-vlm-token", json.dumps(public))
            self.assertNotIn("test-ocr-token", json.dumps(public))
            status_text = manager.status_path(code, job_id).read_text(encoding="utf-8")
            self.assertNotIn("test-vlm-token", status_text)
            self.assertNotIn("test-ocr-token", status_text)

            process = mock.Mock()
            process.pid = 1234
            process.stdout = io.StringIO("")
            process.wait.return_value = 0
            command = ["translate", "input.cbz"]
            with mock.patch("web_app.subprocess.Popen", return_value=process) as popen:
                self.assertEqual(
                    manager.run_pipeline_process(code, job_id, command, "test"),
                    0,
                )
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["TETOLATE_VLM_API_KEY"], "test-vlm-token")
            self.assertEqual(
                environment["TETOLATE_PADDLEOCR_VL_API_KEY"], "test-ocr-token"
            )
            log = manager.log_path(code, job_id).read_text(encoding="utf-8")
            self.assertNotIn("test-vlm-token", log)
            self.assertNotIn("test-ocr-token", log)

            manager.update_job_auth_tokens(
                code,
                job_id,
                vlm_auth_token=None,
                paddleocr_vl_auth_token=None,
                clear_vlm_auth_token=True,
                clear_paddleocr_vl_auth_token=True,
            )
            self.assertFalse(secret_path.exists())

    def test_advanced_options_separate_ocr_and_vlm_tokens(self) -> None:
        markup = web_pages.advanced_options_fields(
            2048,
            "http://127.0.0.1:8080/v1",
            ocr_page_workers=2,
            lama_workers=3,
            imagemagick_workers=4,
            has_vlm_auth_token=True,
            has_paddleocr_vl_auth_token=True,
        )

        self.assertIn("<legend>Workflow</legend>", markup)
        self.assertIn("<legend>OCR</legend>", markup)
        self.assertIn("<legend>Parallel work</legend>", markup)
        self.assertIn("<legend>Translation VLM</legend>", markup)
        self.assertIn('name="ocr_page_workers" type="number" min="1" max="32" step="1" value="2"', markup)
        self.assertIn('name="lama_workers" type="number" min="1" max="32" step="1" value="3"', markup)
        self.assertIn('name="imagemagick_workers" type="number" min="1" max="32" step="1" value="4"', markup)
        self.assertIn('name="paddleocr_vl_auth_token" type="password"', markup)
        self.assertIn('name="vlm_auth_token" type="password"', markup)
        self.assertIn('name="clear_paddleocr_vl_auth_token"', markup)
        self.assertIn('name="clear_vlm_auth_token"', markup)
        self.assertNotIn("test-vlm-token", markup)
        self.assertNotIn("test-ocr-token", markup)

    def test_advanced_options_include_vlm_model_selection_and_connection_test(self) -> None:
        markup = web_pages.advanced_options_fields(
            2048,
            "http://127.0.0.1:8080/v1",
            vlm_model="example-vlm",
            test_category="manga",
            test_job_id="12345678",
        )

        self.assertIn('<select name="vlm_model" required>', markup)
        self.assertIn('<option value="example-vlm" selected>example-vlm</option>', markup)
        self.assertIn('class="vlm-test-button" type="button"', markup)
        self.assertIn('data-vlm-test-category="manga"', markup)
        self.assertIn('data-vlm-test-job-id="12345678"', markup)
        self.assertIn("Test connection and load models", markup)
        self.assertIn("data-vlm-test-status", markup)

        category_body = web_pages.category_jobs_page(
            "manga",
            {"jobs": [], "defaultVlmModel": "example-vlm"},
        ).body.decode("utf-8")
        self.assertIn('<option value="example-vlm" selected>example-vlm</option>', category_body)
        self.assertIn('data-vlm-test-category="manga"', category_body)
        self.assertIn('fetch("/api/vlm/test"', category_body)

        job_body = web_pages.job_page(
            "manga",
            "12345678",
            {
                "status": "failed",
                "recentLog": [],
                "canUpdateAdvancedOptions": True,
                "vlmModel": "example-vlm",
            },
        ).body.decode("utf-8")
        self.assertIn('data-vlm-test-job-id="12345678"', job_body)

    def test_job_page_dynamic_advanced_options_include_vlm_connection_test(self) -> None:
        body = web_pages.job_page(
            "manga",
            "12345678",
            {"status": "failed", "recentLog": []},
        ).body.decode("utf-8")

        self.assertIn("function advancedOptionsMarkup(data)", body)
        self.assertIn("data.vlmModel || data.defaultVlmModel", body)
        self.assertIn('<select name="vlm_model" required>${vlmModelOptions(vlmModel)}</select>', body)
        self.assertIn('fetch("/api/vlm/test"', body)
        self.assertIn("vlmBaseUrl: endpoint.value", body)
        self.assertIn("payload.vlmAuthToken = token.value", body)
        self.assertIn("payload.category = category", body)
        self.assertIn("payload.jobId = testJobId", body)
        self.assertIn("Connected. Found ${loadedVlmModels.length} model(s).", body)
        self.assertIn("status.textContent = error instanceof Error", body)
        self.assertIn('advancedOptions.querySelector("form")', body)

    def test_vlm_model_is_saved_and_passed_to_the_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.initialized_manager(Path(temp_dir), ["default"])
            code, job_id = "default", "12345678"
            manager.job_dir(code, job_id).mkdir(parents=True)
            manager.save_status(
                code,
                job_id,
                {
                    "status": "failed",
                    "vlmModel": " selected-model ",
                },
            )

            payload = manager.public_status(code, job_id, include_log=False)
            command = manager.build_command(code, job_id)

        self.assertEqual(payload["vlmModel"], "selected-model")
        model_index = command.index("--vlm-model")
        self.assertEqual(command[model_index + 1], "selected-model")

    def test_vlm_probe_uses_supplied_or_saved_private_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self.write_web_config(root)
            manager = self.initialized_manager(root, ["default"])
            code, job_id = "default", "12345678"
            manager.job_dir(code, job_id).mkdir(parents=True)
            manager.save_status(code, job_id, {"status": "failed", "vlmModel": "model"})
            manager.update_job_auth_tokens(
                code,
                job_id,
                vlm_auth_token="saved-token",
                paddleocr_vl_auth_token=None,
                preserve_existing=False,
            )
            app = web_app.create_app(config_path)
            app.state.manager = manager
            session = manager.authenticate_admin(self.ADMIN_PASSWORD)
            assert session is not None

            async def direct_run_in_threadpool(function: object, *args: object) -> object:
                return function(*args)  # type: ignore[operator]

            async def exercise() -> None:
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    client.cookies.set(web_app.ADMIN_COOKIE_NAME, session)
                    with mock.patch.object(
                        manager,
                        "probe_vlm_models",
                        return_value=["first", "second"],
                    ) as probe, mock.patch.object(
                        web_app,
                        "run_in_threadpool",
                        new=direct_run_in_threadpool,
                    ):
                        response = await client.post(
                            "/api/vlm/test",
                            json={
                                "vlmBaseUrl": "http://vlm.example/v1",
                                "vlmAuthToken": "supplied-token",
                                "category": code,
                                "jobId": job_id,
                            },
                        )
                        category_response = await client.post(
                            "/api/vlm/test",
                            json={
                                "vlmBaseUrl": "http://vlm.example/v1",
                                "category": code,
                            },
                        )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(category_response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {
                        "ok": True,
                        "models": ["first", "second"],
                        "message": "Connected. Found 2 model(s).",
                    },
                )
                self.assertNotIn("supplied-token", response.text)
                self.assertNotIn("saved-token", response.text)
                self.assertEqual(
                    probe.call_args_list,
                    [
                        mock.call("http://vlm.example/v1", "supplied-token"),
                        mock.call("http://vlm.example/v1", None),
                    ],
                )

            asyncio.run(exercise())

    def test_vlm_probe_failure_adds_docker_note_only_for_localhost(self) -> None:
        self.assertIn(
            "host.docker.internal",
            web_app.vlm_probe_failure_detail("http://127.0.0.1:8080/v1"),
        )
        self.assertIn(
            "host.docker.internal",
            web_app.vlm_probe_failure_detail("http://[::1]:8080/v1"),
        )
        self.assertNotIn(
            "host.docker.internal",
            web_app.vlm_probe_failure_detail("https://vlm.example/v1"),
        )

    def test_page_worker_limits_are_validated(self) -> None:
        self.assertEqual(web_app.parse_page_workers_form("1", "OCR workers"), 1)
        self.assertEqual(web_app.parse_page_workers_form("32", "OCR workers"), 32)
        for value in ("0", "33", "1.5", "many"):
            with self.subTest(value=value), self.assertRaises(HTTPException):
                web_app.parse_page_workers_form(value, "OCR workers")

    def test_pipeline_command_contains_saved_worker_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self.initialized_manager(root, ["default"])
            job_id = "12345678"
            manager.job_dir("default", job_id).mkdir(parents=True)
            manager.save_status(
                "default",
                job_id,
                {
                    "status": "failed",
                    "ocrPageWorkers": 2,
                    "lamaWorkers": 3,
                    "imagemagickWorkers": 4,
                },
            )

            command = manager.build_command("default", job_id)

        self.assertEqual(command[command.index("--ocr-workers") + 1], "2")
        self.assertEqual(command[command.index("--lama-workers") + 1], "3")
        self.assertEqual(command[command.index("--imagemagick-workers") + 1], "4")

    def test_web_pipeline_skips_archives_and_package_generation_selects_one_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = self.initialized_manager(root, ["default"])
            job_id = "12345678"
            manager.job_dir("default", job_id).mkdir(parents=True)
            manager.save_status("default", job_id, {"status": "failed"})

            normal_command = manager.build_command("default", job_id)
            package_command = manager.build_command(
                "default", job_id, "package", 0, package_variant="webp"
            )

        self.assertIn("--skip-package", normal_command)
        self.assertNotIn("--package-variant", normal_command)
        self.assertNotIn("--skip-package", package_command)
        self.assertEqual(
            package_command[package_command.index("--package-variant") + 1],
            "webp",
        )

    def test_case_colliding_saved_categories_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = web_app.load_web_config(self.write_web_config(root))
            manager = web_app.JobManager(config)
            config.jobs_dir.mkdir(parents=True)
            web_security.write_private_json_atomic(
                manager.web_state_path(),
                {
                    "version": 1,
                    "adminPasswordHash": web_security.hash_password(self.ADMIN_PASSWORD),
                    "categories": ["Manga", "manga"],
                },
            )

            with self.assertRaisesRegex(RuntimeError, "differ only by letter case"):
                manager.initialize_web_state()

    def test_only_one_manager_can_lock_a_jobs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self.write_web_config(root)
            first = web_app.JobManager(web_app.load_web_config(config_path))
            second = web_app.JobManager(web_app.load_web_config(config_path))
            first.config.jobs_dir.mkdir(parents=True)

            first.acquire_instance_lock()
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "Another tetolate web server"
                ):
                    second.acquire_instance_lock()
            finally:
                first.release_instance_lock()

            second.acquire_instance_lock()
            second.release_instance_lock()

    def test_admin_login_failures_are_rate_limited_and_clearable(self) -> None:
        manager = object.__new__(web_app.JobManager)
        manager._lock = threading.RLock()
        manager._admin_login_failures = {}

        for _ in range(web_app.ADMIN_LOGIN_FAILURE_LIMIT):
            manager.record_admin_login_failure("127.0.0.1")

        self.assertGreater(manager.admin_login_retry_after("127.0.0.1"), 0)
        manager.clear_admin_login_failures("127.0.0.1")
        self.assertEqual(manager.admin_login_retry_after("127.0.0.1"), 0)

    def test_admin_state_describes_actual_worker_and_job_activity(self) -> None:
        self.assertEqual(
            web_pages.admin_state_label(
                {"paused": False, "workerRunning": True, "queuedCount": 0, "active": []}
            ),
            "Idle",
        )
        self.assertEqual(
            web_pages.admin_state_label(
                {"paused": False, "workerRunning": True, "queuedCount": 1, "active": []}
            ),
            "Queued",
        )
        self.assertEqual(
            web_pages.admin_state_label(
                {
                    "paused": False,
                    "workerRunning": True,
                    "queuedCount": 0,
                    "active": [{"jobId": "12345678"}],
                }
            ),
            "Processing",
        )
        self.assertEqual(
            web_pages.admin_state_label(
                {"paused": True, "workerRunning": True, "queuedCount": 0, "active": []}
            ),
            "Paused",
        )
        self.assertEqual(
            web_pages.admin_state_label(
                {"paused": False, "workerRunning": False, "queuedCount": 0, "active": []}
            ),
            "Stopped",
        )

    def test_all_non_login_routes_require_admin_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = self.write_web_config(root)
            manager = self.initialized_manager(root, ["manga"])
            app = web_app.create_app(config_path)
            app.state.manager = manager

            async def direct_run_in_threadpool(function: object, *args: object) -> object:
                return function(*args)  # type: ignore[operator]

            async def exercise_routes() -> None:
                transport = ASGITransport(app=app)
                async with AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    follow_redirects=False,
                ) as client:
                    response = await client.get("/")
                    self.assertEqual(response.status_code, 303)
                    self.assertEqual(response.headers["location"], "/admin")

                    response = await client.get("/api/job/manga/12345678")
                    self.assertEqual(response.status_code, 401)

                    response = await client.get("/assets/editor-v2/app.js")
                    self.assertEqual(response.status_code, 303)

                    response = await client.get("/docs")
                    self.assertEqual(response.status_code, 303)

                    response = await client.post(
                        "/admin/login",
                        data={"password": self.ADMIN_PASSWORD},
                    )
                    self.assertEqual(response.status_code, 303)
                    self.assertIn(web_app.ADMIN_COOKIE_NAME, client.cookies)
                    cookie_header = response.headers["set-cookie"].lower()
                    self.assertIn("httponly", cookie_header)
                    self.assertIn("samesite=strict", cookie_header)

                    self.assertIn("max-age=", cookie_header)

                    response = await client.get("/admin")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("Job Categories", response.text)
                    self.assertIn("manga", response.text)

                    response = await client.get("/category/manga")
                    self.assertEqual(response.status_code, 200)

                    response = await client.get("/job/manga")
                    self.assertEqual(response.status_code, 404)

                    response = await client.post(
                        "/admin/categories",
                        data={"category": "cross-site-category"},
                        headers={"Origin": "https://attacker.example"},
                    )
                    self.assertEqual(response.status_code, 403)
                    self.assertNotIn("cross-site-category", manager.categories())

                    response = await client.post(
                        "/admin/categories",
                        data={"category": "second-volume"},
                    )
                    self.assertEqual(response.status_code, 303)
                    self.assertIn("second-volume", manager.categories())

                    response = await client.get(
                        "/admin/categories/second-volume/delete"
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("permanently removes", response.text)

                    response = await client.post(
                        "/admin/categories/second-volume/delete",
                        data={"confirmation": "second-volume"},
                    )
                    self.assertEqual(response.status_code, 303)
                    self.assertNotIn("second-volume", manager.categories())

                    response = await client.post("/admin/logout")
                    self.assertEqual(response.status_code, 303)
                    response = await client.get("/category/manga")
                    self.assertEqual(response.status_code, 303)
                    self.assertEqual(response.headers["location"], "/admin")

            with mock.patch.object(
                web_app,
                "run_in_threadpool",
                direct_run_in_threadpool,
            ):
                asyncio.run(exercise_routes())


class PageSelectionTests(unittest.TestCase):
    def test_huge_range_is_rejected_before_materialization(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            web_app.parse_page_selection("0-1000000000")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("cannot contain more than", str(raised.exception.detail))


class VlmCancellationTests(unittest.TestCase):
    def test_validated_array_does_not_retry_pipeline_cancellation(self) -> None:
        page = translate_cbz.Page(index=3, image_path=Path("unused.png"))
        validator = mock.Mock()

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            translate_cbz,
            "get_vlm_array",
            side_effect=translate_cbz.PipelineCancelled("cancelled"),
        ) as get_vlm_array:
            with self.assertRaises(translate_cbz.PipelineCancelled):
                translate_cbz.get_validated_vlm_array(
                    "ocr_structured",
                    page,
                    "prompt",
                    Path(temp_dir),
                    mock.sentinel.config,
                    None,
                    validator,
                )

        get_vlm_array.assert_called_once()
        validator.assert_not_called()

    def test_validation_error_is_added_to_retry_prompt(self) -> None:
        page = translate_cbz.Page(index=3, image_path=Path("unused.png"))
        validator = mock.Mock(
            side_effect=[
                translate_cbz.PipelineError(
                    "structured page 3 item 0 must include boolean `sfx`."
                ),
                "validated",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            translate_cbz,
            "get_vlm_array",
            return_value=[],
        ) as get_vlm_array, mock.patch("sys.stderr"):
            result = translate_cbz.get_validated_vlm_array(
                "ocr_structured",
                page,
                "ORIGINAL PROMPT",
                Path(temp_dir),
                types.SimpleNamespace(model="test-model"),
                None,
                validator,
            )

        self.assertEqual(result, "validated")
        self.assertEqual(get_vlm_array.call_count, 2)
        first_prompt = get_vlm_array.call_args_list[0].args[2]
        retry_prompt = get_vlm_array.call_args_list[1].args[2]
        self.assertEqual(first_prompt, "ORIGINAL PROMPT")
        self.assertTrue(retry_prompt.startswith("ORIGINAL PROMPT"))
        self.assertIn("VALIDATION RETRY", retry_prompt)
        self.assertIn("must include boolean `sfx`", retry_prompt)


class OcrMergeTests(unittest.TestCase):
    def test_source_text_order_supports_korean_ltr_and_east_asian_rtl(self) -> None:
        page = translate_cbz.Page(index=0, image_path=Path("unused.png"))
        raw_records = [
            {"page": 0, "boxno": 0, "region": [10, 10, 20, 30], "text": "L"},
            {"page": 0, "boxno": 1, "region": [30, 10, 40, 30], "text": "R"},
        ]

        with mock.patch.object(
            ocr_merge,
            "should_merge_ocr_records",
            return_value=True,
        ), mock.patch.object(
            ocr_merge,
            "union_regions",
            return_value=[10, 10, 40, 30],
        ):
            korean = ocr_merge.merge_ocr_records_for_page(
                page,
                raw_records,
                right_to_left=False,
            )
            rtl = ocr_merge.merge_ocr_records_for_page(
                page,
                raw_records,
                right_to_left=True,
            )

        self.assertEqual(korean[0]["sourceBoxnos"], [0, 1])
        self.assertEqual(korean[0]["text"], "L\nR")
        self.assertEqual(rtl[0]["sourceBoxnos"], [1, 0])
        self.assertEqual(rtl[0]["text"], "RL")


class AltPlacementHydrationTests(unittest.TestCase):
    def test_explicit_open_lettering_survives_default_hydration(self) -> None:
        page = translate_cbz.Page(index=0, image_path=Path("unused.png"))
        records = {
            0: [
                {
                    "page": 0,
                    "boxno": 7,
                    "region": [1, 2, 3, 4],
                    "text": "text",
                    "openLettering": True,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            translate_cbz.hydrate_alt_placement_fields(
                [page],
                records,
                Path(temp_dir),
            )

        record = records[0][0]
        self.assertIs(record["openLettering"], True)
        self.assertIs(record["safeToEraseOriginal"], False)
        self.assertEqual(record["altPlacementReason"], "unclear")

    def test_alt_placement_prompt_uses_numeric_reason_codes(self) -> None:
        page = translate_cbz.Page(index=0, image_path=Path("unused.png"))
        prompt = translate_cbz.alt_placement_prompt(
            page,
            [
                {
                    "page": 0,
                    "boxno": 0,
                    "region": [1, 2, 3, 4],
                    "text": "text",
                    "sfx": False,
                }
            ],
            translate_cbz.language_config_from_codes("jp", "en"),
        )

        self.assertIn("[boxno,safeToEraseOriginal,reasonCode]", prompt)
        self.assertIn("All three values are integers", prompt)
        self.assertIn("[[0,1,0]]", prompt)

    def test_single_flat_alt_placement_row_is_normalized(self) -> None:
        page = translate_cbz.Page(index=1, image_path=Path("unused.png"))
        records = [
            {
                "page": 1,
                "boxno": 0,
                "region": [1, 2, 3, 4],
                "text": "text",
            }
        ]

        placements = translate_cbz.validate_alt_placement_page(
            page,
            records,
            [0, 0, 6],
        )

        self.assertEqual(len(placements), 1)
        self.assertIs(placements[0]["safeToEraseOriginal"], False)
        self.assertIs(placements[0]["openLettering"], True)
        self.assertEqual(placements[0]["altPlacementReason"], "over_face_body")


class PaddleOcrTilingTests(unittest.TestCase):
    def test_remote_worker_receives_engine_and_tiling_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            Image.new("RGB", (20, 10), "white").save(image_path)
            client = paddle_ocr_image.create_paddle_ocr(
                "japan",
                "cpu",
                None,
                None,
                False,
                False,
                ocr_version="PP-OCRv6",
                service_url="http://paddleocr:8090/",
            )
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {"records": [{"page": 3, "boxno": 0, "region": [1, 2, 3, 4], "text": "x"}]}
            ).encode("utf-8")
            with mock.patch.object(
                paddle_ocr_image.urllib.request,
                "urlopen",
                return_value=response,
            ) as urlopen:
                records = paddle_ocr_image.extract_image_records(
                    client,
                    image_path,
                    page=3,
                    min_score=0.6,
                    tile_enabled=True,
                    tile_width=10,
                    tile_height=10,
                    tile_overlap=2,
                )

        self.assertEqual(records[0]["text"], "x")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://paddleocr:8090/ocr")
        self.assertEqual(payload["engine"], "paddle")
        self.assertEqual(payload["page"], 3)
        self.assertEqual(payload["options"]["ocr_version"], "PP-OCRv6")
        self.assertEqual(payload["options"]["min_score"], 0.6)
        self.assertIs(payload["options"]["tile_enabled"], True)

    def test_worker_rejects_invalid_base64(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            paddle_ocr_server.decode_image("not base64")
        self.assertEqual(raised.exception.status_code, 400)

    def test_current_paddleocr_result_schema_is_normalized(self) -> None:
        records = paddle_ocr_image.extract_records(
            [
                {
                    "rec_texts": ["text"],
                    "rec_scores": [0.95],
                    "rec_polys": [[[10, 20], [30, 20], [30, 40], [10, 40]]],
                    "rec_boxes": [[10, 20, 30, 40]],
                }
            ],
            page=2,
            min_score=0.5,
            image_size=(100, 100),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["page"], 2)
        self.assertEqual(records[0]["text"], "text")
        self.assertEqual(records[0]["region"], [10, 20, 30, 40])

    def test_sequence_result_schema_is_rejected(self) -> None:
        result = [
            [
                [
                    [[10, 20], [30, 20], [30, 40], [10, 40]],
                    ["text", 0.95],
                ]
            ]
        ]

        with self.assertRaisesRegex(
            paddle_ocr_image.InputError,
            "unsupported result structure",
        ):
            paddle_ocr_image.extract_records(
                result,
                page=0,
                min_score=0.5,
                image_size=(100, 100),
            )

    def test_disabling_full_image_pass_only_ocr_scans_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "source.png"
            Image.new("RGB", (20, 10), "white").save(image_path)
            scanned_paths: list[Path] = []

            def record_scan(_ocr: object, path: Path) -> list[object]:
                scanned_paths.append(Path(path))
                return []

            with mock.patch.object(
                paddle_ocr_image,
                "run_paddle_ocr",
                side_effect=record_scan,
            ):
                records = paddle_ocr_image.extract_image_records(
                    mock.sentinel.ocr,
                    image_path,
                    page=0,
                    min_score=0.0,
                    tile_enabled=True,
                    tile_width=10,
                    tile_height=10,
                    tile_overlap=0,
                    tile_include_full_image=False,
                )

        self.assertEqual(records, [])
        self.assertEqual(len(scanned_paths), 2)
        self.assertNotIn(image_path, scanned_paths)
        self.assertTrue(all(path.name.startswith("tile_") for path in scanned_paths))

    def test_unknown_nonempty_result_schema_is_not_treated_as_empty(self) -> None:
        with self.assertRaisesRegex(paddle_ocr_image.InputError, "unsupported result structure"):
            paddle_ocr_image.extract_records(
                {"unexpected": [1, 2, 3]},
                page=0,
                min_score=0.0,
                image_size=(100, 100),
            )

    def test_document_preprocessing_is_rejected_for_box_output(self) -> None:
        with self.assertRaisesRegex(paddle_ocr_image.InputError, "transformed image coordinates"):
            paddle_ocr_image.create_paddle_ocr(
                "japan",
                "cpu",
                None,
                None,
                True,
                False,
            )


class MergeTextJsonTests(unittest.TestCase):
    def test_changed_ocr_text_removes_stale_translation(self) -> None:
        master = {
            (0, 1): {
                "page": 0,
                "boxno": 1,
                "region": [0, 0, 10, 10],
                "sfx": False,
                "openLettering": False,
                "text": "old",
                "englishText": "stale",
            }
        }
        excerpt = [
            {
                "page": 0,
                "boxno": 1,
                "region": [1, 1, 11, 11],
                "sfx": True,
                "openLettering": True,
                "text": "new",
            }
        ]

        merge_text_json.merge_ocr(master, excerpt)

        self.assertEqual(master[(0, 1)]["text"], "new")
        self.assertNotIn("englishText", master[(0, 1)])

    def test_translation_source_mismatch_is_rejected(self) -> None:
        master = {
            (0, 1): {
                "page": 0,
                "boxno": 1,
                "text": "source",
            }
        }
        excerpt = [
            {
                "page": 0,
                "boxno": 1,
                "text": "different source",
                "englishText": "translation",
            }
        ]

        with self.assertRaisesRegex(merge_text_json.InputError, "text mismatch"):
            merge_text_json.merge_translation(master, excerpt)

        self.assertNotIn("englishText", master[(0, 1)])


class CleanTextRegionTests(unittest.TestCase):
    def test_retained_mask_cannot_overwrite_input_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_json = root / "text.json"
            input_image = root / "input.png"
            text_json.write_text("[]\n", encoding="utf-8")
            input_image.write_bytes(b"image")

            with self.assertRaisesRegex(
                clean_text_regions.InputError,
                "Retained mask must be different from the input image",
            ):
                clean_text_regions.check_paths(
                    text_json,
                    input_image,
                    root / "output.png",
                    None,
                    input_image,
                    require_lama=False,
                )

    def test_connected_lama_regions_are_merged_without_joining_distant_boxes(self) -> None:
        self.assertCountEqual(
            lama_inpaint.merge_connected_boxes(
                [(10, 10, 20, 20), (20, 12, 30, 18), (100, 100, 110, 110)]
            ),
            [(10, 10, 30, 20), (100, 100, 110, 110)],
        )

    def test_lama_inference_only_replaces_masked_pixels(self) -> None:
        import numpy as np
        import torch

        class WhiteModel:
            def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return torch.ones_like(image)

        image = np.full((9, 10, 3), 40, dtype=np.uint8)
        mask = np.zeros((9, 10), dtype=np.uint8)
        mask[2:7, 3:8] = 255

        result = lama_inpaint.infer_crop(
            WhiteModel(),
            torch.device("cpu"),
            image,
            mask,
        )

        self.assertEqual(result.shape, image.shape)
        self.assertTrue(np.all(result[2:7, 3:8] == 255))
        self.assertTrue(np.all(result[0:2] == 40))
        self.assertTrue(np.all(result[:, 0:3] == 40))

    def test_invalid_configured_lama_model_is_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "custom.pt"
            model_path.write_bytes(b"not a model")

            with self.assertRaisesRegex(lama_inpaint.LaMaError, "SHA-256"):
                lama_inpaint.ensure_model(model_path)

            self.assertEqual(model_path.read_bytes(), b"not a model")


class AtomicPackagingTests(unittest.TestCase):
    def test_failed_packaging_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            final_dir = translate_cbz.final_pages_dir(output_dir)
            final_dir.mkdir(parents=True)
            page = translate_cbz.Page(index=0, image_path=Path("page.png"))
            translate_cbz.final_page_png_path(output_dir, page).write_bytes(b"page data")

            destination = translate_cbz.translated_cbz_path(output_dir)
            original_contents = b"existing archive"
            destination.write_bytes(original_contents)
            invalid_input_cbz = root / "invalid.cbz"
            invalid_input_cbz.write_bytes(b"not a zip archive")

            with self.assertRaises(translate_cbz.PipelineError):
                translate_cbz.package_cbz([page], output_dir, invalid_input_cbz)

            self.assertEqual(destination.read_bytes(), original_contents)
            self.assertEqual(list(output_dir.glob(f".{destination.name}.*.tmp")), [])

    def test_packaging_converts_pages_in_parallel_and_validates_source_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            final_dir = translate_cbz.final_pages_dir(output_dir)
            final_dir.mkdir(parents=True)
            pages = []
            for index in range(4):
                page = translate_cbz.Page(index=index, image_path=Path(f"page-{index}.png"))
                pages.append(page)
                translate_cbz.final_page_png_path(output_dir, page).write_bytes(
                    f"page {index}".encode()
                )

            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("original.png", b"source")
                archive.writestr("ComicInfo.xml", b"<ComicInfo />")

            fixture_dir = root / "fixtures"
            fixture_dir.mkdir()
            config = replace(
                translate_cbz.load_config(None, fixture_dir),
                imagemagick_workers=2,
            )
            barrier = threading.Barrier(2)

            def convert(source: Path, destination: Path, _quality: int) -> None:
                barrier.wait(timeout=2)
                destination.write_bytes(source.read_bytes())

            with (
                mock.patch.object(
                    translate_cbz,
                    "convert_final_page_with_magick",
                    side_effect=convert,
                ),
                mock.patch.object(
                    translate_cbz,
                    "validate_cbz_members",
                    wraps=translate_cbz.validate_cbz_members,
                ) as validate_members,
                mock.patch("sys.stderr"),
            ):
                translate_cbz.print_packaged_cbz(pages, output_dir, input_cbz, config)

            validate_members.assert_called_once()
            for archive_path in (
                translate_cbz.translated_cbz_path(output_dir),
                translate_cbz.translated_webp_cbz_path(output_dir),
                translate_cbz.translated_jxl_cbz_path(output_dir),
            ):
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertEqual(archive.read("ComicInfo.xml"), b"<ComicInfo />")
                    self.assertEqual(
                        archive.namelist()[:4],
                        ["page-0.jpg", "page-1.webp", "page-2.webp", "page-3.webp"]
                        if archive_path == translate_cbz.translated_webp_cbz_path(output_dir)
                        else (
                            ["page-0.jpg", "page-1.jxl", "page-2.jxl", "page-3.jxl"]
                            if archive_path == translate_cbz.translated_jxl_cbz_path(output_dir)
                            else ["page-0.png", "page-1.png", "page-2.png", "page-3.png"]
                        ),
                    )

    def test_package_variant_generates_only_requested_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translate_cbz.final_pages_dir(output_dir).mkdir(parents=True)
            page = translate_cbz.Page(index=0, image_path=Path("page.png"))
            translate_cbz.final_page_png_path(output_dir, page).write_bytes(b"page")
            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("original.png", b"source")

            config = translate_cbz.load_config(None, root / "fixtures")

            def convert(source: Path, destination: Path, _quality: int) -> None:
                destination.write_bytes(source.read_bytes())

            with mock.patch.object(
                translate_cbz,
                "convert_final_page_with_magick",
                side_effect=convert,
            ), mock.patch("sys.stderr"):
                translate_cbz.print_packaged_cbz(
                    [page],
                    output_dir,
                    input_cbz,
                    config,
                    package_variant="webp",
                )

            self.assertFalse(translate_cbz.translated_cbz_path(output_dir).exists())
            self.assertTrue(translate_cbz.translated_webp_cbz_path(output_dir).exists())
            self.assertFalse(translate_cbz.translated_jxl_cbz_path(output_dir).exists())

    def test_package_variant_png_does_not_generate_alternate_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translate_cbz.final_pages_dir(output_dir).mkdir(parents=True)
            page = translate_cbz.Page(index=0, image_path=Path("page.png"))
            translate_cbz.final_page_png_path(output_dir, page).write_bytes(b"page")
            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("original.png", b"source")

            config = translate_cbz.load_config(None, root / "fixtures")
            with mock.patch.object(
                translate_cbz,
                "package_converted_cbz",
            ) as alternate_package, mock.patch("sys.stderr"):
                translate_cbz.print_packaged_cbz(
                    [page],
                    output_dir,
                    input_cbz,
                    config,
                    package_variant="png",
                )

            alternate_package.assert_not_called()
            self.assertTrue(translate_cbz.translated_cbz_path(output_dir).exists())

    def test_requested_variant_failure_preserves_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            translate_cbz.final_pages_dir(output_dir).mkdir(parents=True)
            page = translate_cbz.Page(index=0, image_path=Path("page.png"))
            translate_cbz.final_page_png_path(output_dir, page).write_bytes(b"page")
            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("original.png", b"source")
            destination = translate_cbz.translated_webp_cbz_path(output_dir)
            destination.write_bytes(b"previous archive")
            config = translate_cbz.load_config(None, root / "fixtures")

            with mock.patch.object(
                translate_cbz,
                "convert_final_page_with_magick",
                side_effect=translate_cbz.PipelineError("conversion failed"),
            ), mock.patch("sys.stderr"):
                with self.assertRaises(translate_cbz.PipelineError):
                    translate_cbz.print_packaged_cbz(
                        [page],
                        output_dir,
                        input_cbz,
                        config,
                        package_variant="webp",
                    )

            self.assertEqual(destination.read_bytes(), b"previous archive")
            self.assertEqual(list(output_dir.glob(f".{destination.name}.*.tmp")), [])


class ArchiveSafetyTests(unittest.TestCase):
    def test_web_upload_accepts_zip_with_images_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "comic.zip"
            image_data = io.BytesIO()
            Image.new("RGB", (2, 2), "white").save(image_data, format="PNG")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("pages/001.png", image_data.getvalue())
                archive.writestr("ComicInfo.xml", "<ComicInfo />")

            web_app.validate_uploaded_comic_archive(archive_path, archive_path.name)

    def test_web_upload_rejects_zip_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "metadata.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("ComicInfo.xml", "<ComicInfo />")

            with self.assertRaisesRegex(HTTPException, "at least one supported image"):
                web_app.validate_uploaded_comic_archive(archive_path, archive_path.name)

    def test_web_upload_rejects_zip_with_invalid_image_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "invalid-image.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("001.png", b"not an image")

            with self.assertRaisesRegex(HTTPException, "not a readable image"):
                web_app.validate_uploaded_comic_archive(archive_path, archive_path.name)

    def test_web_file_selector_accepts_cbz_and_zip(self) -> None:
        self.assertIn(".cbz", web_pages.UPLOAD_COMIC_ARCHIVE_ACCEPT)
        self.assertIn(".zip", web_pages.UPLOAD_COMIC_ARCHIVE_ACCEPT)

    def test_extraction_rejects_unsafe_member_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("../page.png", b"not an image")

            with self.assertRaisesRegex(translate_cbz.PipelineError, "unsafe entry name"):
                translate_cbz.extract_cbz(input_cbz, root / "output")

    def test_extraction_rejects_oversized_declared_archive_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("ComicInfo.xml", b"1234")

            with mock.patch.object(translate_cbz, "MAX_ARCHIVE_TOTAL_BYTES", 3):
                with self.assertRaisesRegex(translate_cbz.PipelineError, "archive limit"):
                    translate_cbz.extract_cbz(input_cbz, root / "output")

    def test_extraction_rejects_oversized_decoded_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_cbz = root / "input.cbz"
            image_data = io.BytesIO()
            Image.new("RGB", (2, 2), "white").save(image_data, format="PNG")
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("page.png", image_data.getvalue())

            with mock.patch.object(translate_cbz, "MAX_PAGE_PIXELS", 3):
                with self.assertRaisesRegex(translate_cbz.PipelineError, "decodes to 4 pixels"):
                    translate_cbz.extract_cbz(input_cbz, root / "output")


class PipelineSmokeTests(unittest.TestCase):
    def test_paddleocr_vl_pages_use_configured_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixtures"
            fixture_dir.mkdir()
            pages = [
                translate_cbz.Page(index, root / f"page_{index:04d}.png")
                for index in range(3)
            ]
            for page in pages:
                Image.new("RGB", (20, 20), "white").save(page.image_path)
            config = replace(
                translate_cbz.load_config(None, fixture_dir),
                ocr_page_workers=2,
            )
            barrier = threading.Barrier(2)

            def extract(_ocr, _path, page, _min_score):
                if page < 2:
                    barrier.wait(timeout=2)
                return [
                    {
                        "page": page,
                        "boxno": 0,
                        "region": [1, 1, 5, 5],
                        "text": str(page),
                        "score": 1.0,
                    }
                ]

            with (
                mock.patch.object(
                    paddle_ocr_image,
                    "create_paddleocr_vl",
                    side_effect=[mock.sentinel.ocr_1, mock.sentinel.ocr_2],
                ) as create,
                mock.patch.object(
                    paddle_ocr_image,
                    "extract_paddleocr_vl_image_records",
                    side_effect=extract,
                ),
                mock.patch.object(paddle_ocr_image, "close_ocr_engine") as close,
                mock.patch.object(paddle_ocr_image, "draw_boxes"),
                mock.patch("sys.stderr"),
            ):
                records = translate_cbz.run_ocr(pages, root / "output", config)

            self.assertEqual(create.call_count, 2)
            self.assertEqual(close.call_count, 2)
            self.assertEqual([records[index][0]["text"] for index in range(3)], ["0", "1", "2"])

    def test_render_pages_reuses_one_lama_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixtures"
            fixture_dir.mkdir()
            pages: list[translate_cbz.Page] = []
            records: dict[int, list[dict[str, object]]] = {}
            for index in range(3):
                image_path = root / f"page_{index:04d}.png"
                Image.new("RGB", (20, 20), "white").save(image_path)
                pages.append(translate_cbz.Page(index, image_path))
                records[index] = [
                    {
                        "page": index,
                        "boxno": 0,
                        "region": [2, 2, 12, 12],
                        "placementRegion": [2, 2, 12, 12],
                        "englishText": "Text",
                        "openLettering": False,
                        "fill": "black",
                    }
                ]
            config = replace(
                translate_cbz.load_config(None, fixture_dir),
                lama_workers=3,
                imagemagick_workers=2,
            )
            session = mock.Mock()
            seen_sessions: list[object] = []

            def clean(_entries, source, destination, *_args):
                seen_sessions.append(_args[-1])
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.open(source).save(destination)

            def overlay(_entries, source, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.open(source).save(destination)

            with (
                mock.patch.object(
                    translate_cbz.lama_inpaint,
                    "LaMaSession",
                    return_value=session,
                ) as session_factory,
                mock.patch.object(
                    translate_cbz.clean_text_regions,
                    "clean_text_regions",
                    side_effect=clean,
                ),
                mock.patch.object(
                    translate_cbz.overlay_text,
                    "overlay_text",
                    side_effect=overlay,
                ),
                mock.patch("sys.stderr"),
            ):
                translate_cbz.render_pages(pages, records, root / "output", config)

            session_factory.assert_called_once_with(clean_text_regions.DEFAULT_DEVICE)
            session.close.assert_called_once_with()
            self.assertEqual(seen_sessions, [session, session, session])
            self.assertTrue(all((root / "output" / "pages" / "final" / page.image_path.name).is_file() for page in pages))

    def test_single_page_render_ignores_unselected_empty_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pages = []
            records = {}
            for index, english_text in enumerate(("Visible", "")):
                image_path = root / f"page_{index:04d}.png"
                Image.new("RGB", (20, 20), "white").save(image_path)
                pages.append(translate_cbz.Page(index, image_path))
                records[index] = [
                    {
                        "page": index,
                        "boxno": 0,
                        "region": [2, 2, 12, 12],
                        "placementRegion": [2, 2, 12, 12],
                        "englishText": english_text,
                        "openLettering": True,
                        "fill": "black",
                    }
                ]
            config = translate_cbz.load_config(None, root)

            with mock.patch.object(
                translate_cbz,
                "clean_render_page",
            ) as clean, mock.patch.object(
                translate_cbz,
                "typeset_render_page",
            ) as typeset:
                translate_cbz.render_pages(
                    pages,
                    records,
                    root / "output",
                    config,
                    start_page=0,
                    end_page=0,
                )

            self.assertEqual(clean.call_count, 1)
            self.assertEqual(typeset.call_count, 1)
            self.assertEqual(clean.call_args.args[0].index, 0)
            self.assertEqual(typeset.call_args.args[0].index, 0)

    def test_stop_after_ocr_writes_raw_and_merged_review_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_cbz = root / "input.cbz"
            output_dir = root / "output"
            fixture_dir = root / "fixtures"
            fixture_dir.mkdir()

            image_data = io.BytesIO()
            Image.new("RGB", (16, 20), "white").save(image_data, format="PNG")
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("001.png", image_data.getvalue())

            args = types.SimpleNamespace(
                input_cbz=input_cbz,
                output_dir=output_dir,
                overwrite=False,
                stop_after="ocr_merged",
                fixture_dir=fixture_dir,
            )
            config = translate_cbz.load_config(None, fixture_dir)
            with mock.patch.object(
                paddle_ocr_image,
                "create_paddleocr_vl",
                return_value=mock.sentinel.ocr,
            ), mock.patch.object(
                paddle_ocr_image,
                "extract_paddleocr_vl_image_records",
                return_value=[
                    {
                        "page": 0,
                        "boxno": 0,
                        "region": [2, 3, 8, 15],
                        "text": "テスト",
                        "score": 0.99,
                    }
                ],
            ), mock.patch.object(
                paddle_ocr_image,
                "close_ocr_engine",
            ), mock.patch("sys.stderr"):
                translate_cbz.run_full_pipeline(args, config, {})

            raw = json.loads(
                (output_dir / "data" / "ocr_raw" / "page_0000.json").read_text(
                    encoding="utf-8"
                )
            )
            merged = json.loads(
                (output_dir / "data" / "ocr_merged" / "page_0000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(raw[0]["text"], "テスト")
            self.assertEqual(merged[0]["text"], "テスト")
            self.assertFalse(
                (output_dir / "data" / "ocr_structured" / "page_0000.json").exists()
            )
            self.assertFalse(translate_cbz.translated_cbz_path(output_dir).exists())

    def test_empty_ocr_book_runs_to_packaged_cbz(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_cbz = root / "input.cbz"
            output_dir = root / "output"
            fixture_dir = root / "fixtures"
            fixture_dir.mkdir()

            image_data = io.BytesIO()
            Image.new("RGB", (16, 20), "white").save(image_data, format="PNG")
            comic_info = b"<ComicInfo><Title>Smoke test</Title></ComicInfo>"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("001.png", image_data.getvalue())
                archive.writestr("ComicInfo.xml", comic_info)

            args = types.SimpleNamespace(
                input_cbz=input_cbz,
                output_dir=output_dir,
                overwrite=False,
                stop_after=None,
                fixture_dir=fixture_dir,
            )
            config = translate_cbz.load_config(None, fixture_dir)
            with mock.patch.object(
                paddle_ocr_image,
                "create_paddleocr_vl",
                return_value=mock.sentinel.ocr,
            ), mock.patch.object(
                paddle_ocr_image,
                "extract_paddleocr_vl_image_records",
                return_value=[],
            ), mock.patch.object(
                paddle_ocr_image,
                "close_ocr_engine",
            ), mock.patch("sys.stderr"):
                translate_cbz.run_full_pipeline(args, config, {})

            output_cbz = translate_cbz.translated_cbz_path(output_dir)
            self.assertTrue(output_cbz.is_file())
            with zipfile.ZipFile(output_cbz) as archive:
                self.assertEqual(archive.read("ComicInfo.xml"), comic_info)
                with Image.open(archive.open("0000.png")) as rendered:
                    self.assertEqual(rendered.size, (16, 20))
            for phase in (
                "ocr_raw",
                "ocr_merged",
                "ocr_structured",
                "alt_placement",
                "translations",
                "placements",
                "render_entries",
            ):
                self.assertEqual(
                    translate_cbz.load_json(
                        translate_cbz.data_page_path(
                            output_dir,
                            phase,
                            translate_cbz.Page(0, Path("0000.png")),
                        ),
                        phase,
                    ),
                    [],
                )


class ResumeManifestTests(unittest.TestCase):
    def test_resume_rejects_a_different_input_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            original_dir = translate_cbz.original_pages_dir(output_dir)
            original_dir.mkdir(parents=True)
            (original_dir / "0000.png").write_bytes(b"page")
            input_cbz = root / "input.cbz"
            with zipfile.ZipFile(input_cbz, "w") as archive:
                archive.writestr("page.png", b"different page")
            translate_cbz.write_json(
                translate_cbz.input_manifest_path(output_dir),
                {"version": 1, "sha256": "0" * 64, "sizeBytes": 1, "pageCount": 1},
            )

            with self.assertRaisesRegex(translate_cbz.PipelineError, "different input CBZ"):
                translate_cbz.verify_resume_input(input_cbz, output_dir)

    def test_resume_rewinds_when_an_earlier_phase_artifact_is_missing(self) -> None:
        pages = [
            translate_cbz.Page(index=index, image_path=Path(f"{index:04d}.png"))
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            translate_cbz.write_json(
                translate_cbz.data_page_path(output_dir, "alt_placement", pages[1]),
                [],
            )
            with mock.patch("sys.stderr"):
                start_page = translate_cbz.rewind_resume_page_for_missing_artifacts(
                    output_dir,
                    "alt_placement",
                    pages,
                    requested_page=2,
                )

        self.assertEqual(start_page, 0)


class StructurePromptTests(unittest.TestCase):
    def test_structure_prompt_uses_only_current_page_context_and_defines_regions(self) -> None:
        page = translate_cbz.Page(index=1, image_path=Path("unused.png"))
        records = {
            0: [{"page": 0, "boxno": 0, "text": "UNRELATED_PAGE_TEXT"}],
            1: [
                {
                    "page": 1,
                    "boxno": 0,
                    "sourceBoxnos": [4],
                    "region": [10, 20, 30, 40],
                    "text": "current",
                }
            ],
        }

        prompt = translate_cbz.structure_prompt(
            page,
            records,
            translate_cbz.language_config_from_codes("jp", "en"),
        )

        self.assertNotIn("UNRELATED_PAGE_TEXT", prompt)
        self.assertIn("[left,top,right,bottom]", prompt)
        self.assertIn("do not re-derive or debate", prompt)
        self.assertIn('[mergedBoxno,classification]', prompt)
        self.assertIn('"text", "sfx", or "reject"', prompt)
        self.assertIn("already written in English", prompt)
        self.assertIn("`22%`", prompt)
        self.assertIn("There is no output-order field", prompt)
        self.assertIn("Return exactly 1 rows", prompt)

    def test_named_classifications_define_reading_order_and_flags(self) -> None:
        page = translate_cbz.Page(index=0, image_path=Path("unused.png"))
        merged = [
            {
                "page": 0,
                "boxno": boxno,
                "sourceBoxnos": [10 + boxno],
                "sourceTexts": [text],
                "region": [boxno * 10, 0, boxno * 10 + 5, 20],
                "text": text,
            }
            for boxno, text in enumerate(("A", "B", "C"))
        ]

        structured = translate_cbz.validate_ordered_merged_page(
            page,
            merged,
            [
                [2, "sfx"],
                [0, "text", "A corrected"],
                [1, "reject"],
            ],
        )

        self.assertEqual(len(structured), 2)
        self.assertEqual(structured[0]["sourceBoxnos"], [12])
        self.assertTrue(structured[0]["sfx"])
        self.assertEqual(structured[0]["text"], "C")
        self.assertEqual(structured[1]["sourceBoxnos"], [10])
        self.assertFalse(structured[1]["sfx"])
        self.assertEqual(structured[1]["text"], "A corrected")

    def test_translation_free_numbers_symbols_and_english_are_skipped(self) -> None:
        language = translate_cbz.language_config_from_codes("jp", "en")

        self.assertFalse(translate_cbz.text_needs_translation("22%", language))
        self.assertFalse(translate_cbz.text_needs_translation("!? @#$", language))
        self.assertFalse(
            translate_cbz.text_needs_translation("LEVEL 22", language)
        )
        self.assertTrue(translate_cbz.text_needs_translation("レベル22", language))
        self.assertTrue(translate_cbz.text_needs_translation("自己紹介", language))

    def test_kept_sfx_requires_visible_translation(self) -> None:
        page = translate_cbz.Page(index=0, image_path=Path("unused.png"))
        records = [
            {
                "page": 0,
                "boxno": 0,
                "region": [10, 20, 30, 40],
                "text": "source effect",
                "sfx": True,
                "openLettering": False,
            }
        ]

        with self.assertRaisesRegex(
            translate_cbz.PipelineError,
            "empty translation for kept SFX",
        ):
            translate_cbz.validate_translation_page(page, records, [[0, ""]])

        translated = translate_cbz.validate_translation_page(
            page,
            records,
            [[0, "VISIBLE EFFECT"]],
        )
        self.assertEqual(translated[0]["englishText"], "VISIBLE EFFECT")
        prompt = translate_cbz.translation_prompt(
            page,
            {0: records},
            {"job": "", "pages": {}},
            translate_cbz.language_config_from_codes("jp", "en"),
        )
        self.assertIn("Never return an empty translation for SFX", prompt)


class VlmJsonParsingTests(unittest.TestCase):
    def test_missing_outer_array_around_rows_is_recovered(self) -> None:
        with mock.patch("sys.stderr"):
            parsed = translate_cbz.parse_json_array(
                '[0,"First"],[1,"Second"]',
                Path("response.txt"),
            )

        self.assertEqual(parsed, [[0, "First"], [1, "Second"]])

    def test_missing_outer_opening_bracket_is_recovered(self) -> None:
        with mock.patch("sys.stderr"):
            parsed = translate_cbz.parse_json_array(
                "[0,1,0],[1,1,2]]",
                Path("response.txt"),
            )

        self.assertEqual(parsed, [[0, 1, 0], [1, 1, 2]])

    def test_unrelated_invalid_json_is_not_recovered(self) -> None:
        with self.assertRaisesRegex(translate_cbz.PipelineError, "not valid JSON"):
            translate_cbz.parse_json_array(
                '[0,"First"] trailing text',
                Path("response.txt"),
            )


class DebugAnnotationTests(unittest.TestCase):
    def test_debug_annotations_scale_with_page_resolution(self) -> None:
        image = mock.MagicMock()
        image.size = (4000, 6000)
        with mock.patch.object(translate_cbz.Image, "open") as open_image:
            open_image.return_value.__enter__.return_value = image
            box_width, font_size = translate_cbz.debug_annotation_size(
                Path("page.png")
            )

        self.assertEqual(box_width, 12)
        self.assertEqual(font_size, 100)

    def test_debug_annotations_keep_readable_minimums(self) -> None:
        image = mock.MagicMock()
        image.size = (500, 800)
        with mock.patch.object(translate_cbz.Image, "open") as open_image:
            open_image.return_value.__enter__.return_value = image
            box_width, font_size = translate_cbz.debug_annotation_size(
                Path("page.png")
            )

        self.assertEqual(box_width, 3)
        self.assertEqual(font_size, 18)


class OpenPlacementLabelTests(unittest.TestCase):
    def test_sparse_box_numbers_use_contiguous_request_labels(self) -> None:
        records = [
            {
                "page": 35,
                "boxno": 0,
                "region": [10, 10, 20, 20],
                "text": "first",
                "englishText": "First",
                "sfx": True,
                "openLettering": True,
            },
            {
                "page": 35,
                "boxno": 1,
                "region": [30, 30, 40, 40],
                "text": "closed",
                "englishText": "Closed",
                "sfx": False,
                "openLettering": False,
            },
            {
                "page": 35,
                "boxno": 2,
                "region": [50, 50, 60, 60],
                "text": "second",
                "englishText": "Second",
                "sfx": True,
                "openLettering": True,
            },
        ]
        language = translate_cbz.language_config_from_codes("jp", "en")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            page = translate_cbz.Page(35, image_path)
            prompt = translate_cbz.placement_open_prompt(page, records, language)
            placements = translate_cbz.validate_open_placements_page(
                page,
                records,
                [
                    [0, [100, 100, 200, 200]],
                    [1, [300, 300, 400, 400]],
                ],
            )

        self.assertIn('"cols":["label","region"', prompt)
        self.assertNotIn('"cols":["boxno"', prompt)
        self.assertEqual([item["boxno"] for item in placements], [0, 2])

    def test_numeric_string_labels_are_accepted(self) -> None:
        records = [
            {
                "page": 4,
                "boxno": 7,
                "region": [10, 10, 20, 20],
                "text": "sample",
                "englishText": "Sample",
                "sfx": True,
                "openLettering": True,
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            placements = translate_cbz.validate_open_placements_page(
                translate_cbz.Page(4, image_path),
                records,
                [["0", [100, 200, 300, 400]]],
            )

        self.assertEqual(placements[0]["boxno"], 7)
        self.assertEqual(placements[0]["box_2d"], [100, 200, 300, 400])

    def test_expansion_placement_region_is_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (100, 80), "white").save(image_path)
            expansions = translate_cbz.validate_expansions_page(
                translate_cbz.Page(0, image_path),
                [{"boxno": 3, "openLettering": False}],
                [{"label": 3, "placementRegion": [-10, -5, 120, 100]}],
            )

        self.assertEqual(expansions[0]["placementRegion"], [0, 0, 100, 80])


class PlacementRayTests(unittest.TestCase):
    def test_ray_stopping_at_the_actual_image_edge_is_a_boundary(self) -> None:
        with Image.new("L", (20, 20), 255) as image:
            endpoint = placement_detection.cast_placement_ray(
                image.load(),
                (8, 10),
                "left",
                -1.0,
                0.0,
                [7, 8, 10, 12],
                [],
                [0, 0, 20, 20],
                255,
                20,
                20,
            )

        self.assertTrue(endpoint["hitBoundary"])
        self.assertEqual(endpoint["stopReason"], "image_bounds")
        self.assertEqual(endpoint["x"], 0)

    def test_ray_stopping_at_an_internal_search_edge_is_not_a_boundary(self) -> None:
        with Image.new("L", (20, 20), 255) as image:
            endpoint = placement_detection.cast_placement_ray(
                image.load(),
                (8, 10),
                "left",
                -1.0,
                0.0,
                [7, 8, 10, 12],
                [],
                [3, 0, 17, 20],
                255,
                20,
                20,
            )

        self.assertFalse(endpoint["hitBoundary"])
        self.assertEqual(endpoint["stopReason"], "search_bounds")

    def test_diagonal_rays_constrain_horizontal_region(self) -> None:
        endpoints = [
            {"direction": "left", "x": 1021, "y": 1678, "hitBoundary": True},
            {"direction": "right", "x": 1416, "y": 1678, "hitBoundary": True},
            {"direction": "up", "x": 1153, "y": 1364, "hitBoundary": True},
            {"direction": "down", "x": 1153, "y": 1973, "hitBoundary": True},
            {"direction": "up_left", "x": 1024, "y": 1549, "hitBoundary": True},
            {"direction": "up_right", "x": 1321, "y": 1510, "hitBoundary": True},
            {"direction": "down_left", "x": 1029, "y": 1802, "hitBoundary": True},
            {"direction": "down_right", "x": 1287, "y": 1812, "hitBoundary": True},
        ]
        with mock.patch.object(
            placement_detection,
            "cast_placement_ray",
            side_effect=endpoints,
        ):
            component = placement_detection.ray_cast_component(
                None,
                (1153, 1678),
                [1072, 1464, 1236, 1894],
                [],
                [427, 174, 1446, 2048],
                255,
                1446,
                2048,
            )

        self.assertEqual(component["region"], [1029, 1364, 1288, 1974])
        self.assertTrue(
            next(
                endpoint["usedForRegion"]
                for endpoint in component["rayEndpoints"]
                if endpoint["direction"] == "down_right"
            )
        )

    def test_separate_source_boxes_split_overlapping_expansions(self) -> None:
        expansions = [
            {"boxno": 0, "placementRegion": [10, 10, 100, 180]},
            {"boxno": 1, "placementRegion": [20, 60, 110, 230]},
        ]
        records = [
            {"boxno": 0, "region": [30, 30, 70, 80]},
            {"boxno": 1, "region": [35, 120, 75, 170]},
        ]

        placement_detection.resolve_overlapping_expansions(expansions, records)

        self.assertEqual(expansions[0]["placementRegion"], [10, 10, 100, 100])
        self.assertEqual(expansions[1]["placementRegion"], [20, 100, 110, 230])
        self.assertTrue(expansions[0]["overlapAdjusted"])
        self.assertTrue(expansions[1]["overlapAdjusted"])


class ResumeFlowTests(unittest.TestCase):
    def test_structured_resume_starts_later_phases_from_page_zero(self) -> None:
        pages = [
            translate_cbz.Page(index=index, image_path=Path(f"{index:04d}.png"))
            for index in range(2)
        ]
        structured = {0: [], 1: []}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            translate_cbz.write_json(
                translate_cbz.data_page_path(output_dir, "ocr_structured", pages[0]),
                [],
            )
            args = types.SimpleNamespace(
                overwrite=False,
                resume_page=1,
                output_dir=output_dir,
                resume_from="ocr_structured",
                single_page=False,
                input_cbz=Path("input.cbz"),
                fixture_dir=None,
            )
            with mock.patch.object(translate_cbz, "verify_resume_input"), mock.patch.object(
                translate_cbz,
                "load_extracted_pages",
                return_value=pages,
            ), mock.patch.object(
                translate_cbz,
                "load_phase_records",
                return_value={0: [], 1: []},
            ), mock.patch.object(
                translate_cbz,
                "load_structured_records",
                return_value={0: []},
            ), mock.patch.object(
                translate_cbz,
                "run_structure_phase",
                return_value=structured,
            ), mock.patch.object(
                translate_cbz,
                "run_alt_placement_phase",
            ) as run_alt, mock.patch.object(
                translate_cbz,
                "run_translation_phase",
            ) as run_translation, mock.patch.object(
                translate_cbz,
                "run_post_translation_phases",
            ), mock.patch.object(
                translate_cbz,
                "run_placement_phase",
            ), mock.patch.object(
                translate_cbz,
                "render_pages",
            ), mock.patch.object(
                translate_cbz,
                "print_packaged_cbz",
            ):
                translate_cbz.run_resume_pipeline(args, mock.sentinel.config, {})

        self.assertEqual(run_alt.call_args.kwargs["start_page"], 0)
        self.assertEqual(run_translation.call_args.kwargs["start_page"], 0)
        self.assertEqual(run_translation.call_args.kwargs["existing_by_page"], {})

    def test_structured_resume_preserves_complete_earlier_downstream_pages(self) -> None:
        pages = [
            translate_cbz.Page(index=index, image_path=Path(f"{index:04d}.png"))
            for index in range(2)
        ]
        structured = {0: [], 1: []}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for phase in ("ocr_structured", "alt_placement", "translations"):
                translate_cbz.write_json(
                    translate_cbz.data_page_path(output_dir, phase, pages[0]),
                    [],
                )
            args = types.SimpleNamespace(
                overwrite=False,
                resume_page=1,
                output_dir=output_dir,
                resume_from="ocr_structured",
                single_page=False,
                input_cbz=Path("input.cbz"),
                fixture_dir=None,
            )
            with mock.patch.object(translate_cbz, "verify_resume_input"), mock.patch.object(
                translate_cbz,
                "load_extracted_pages",
                return_value=pages,
            ), mock.patch.object(
                translate_cbz,
                "load_phase_records",
                return_value={0: [], 1: []},
            ), mock.patch.object(
                translate_cbz,
                "load_structured_records",
                return_value={0: []},
            ), mock.patch.object(
                translate_cbz,
                "run_structure_phase",
                return_value=structured,
            ), mock.patch.object(
                translate_cbz,
                "run_alt_placement_phase",
            ) as run_alt, mock.patch.object(
                translate_cbz,
                "run_translation_phase",
            ) as run_translation, mock.patch.object(
                translate_cbz,
                "run_post_translation_phases",
            ), mock.patch.object(
                translate_cbz,
                "run_placement_phase",
            ), mock.patch.object(
                translate_cbz,
                "render_pages",
            ), mock.patch.object(
                translate_cbz,
                "print_packaged_cbz",
            ):
                translate_cbz.run_resume_pipeline(args, mock.sentinel.config, {})

        self.assertEqual(run_alt.call_args.kwargs["start_page"], 1)
        self.assertEqual(run_translation.call_args.kwargs["start_page"], 1)
        self.assertEqual(run_translation.call_args.kwargs["existing_by_page"], {0: []})


class StrictInputTests(unittest.TestCase):
    def test_fractional_integer_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(translate_cbz.PipelineError, "must be an integer"):
            translate_cbz.config_int({"value": 1.5}, "value", 1, "test")

    def test_overlay_regions_are_clamped_to_page_bounds(self) -> None:
        entry = {
            "boxno": 2,
            "region": [-10, -20, 50, 60],
            "englishText": "text",
        }
        with mock.patch("sys.stderr"):
            clamped = overlay_text.clamp_entry_to_image(entry, 40, 30)

        self.assertEqual(clamped["region"], [0.0, 0.0, 40.0, 30.0])
        self.assertEqual(entry["region"], [-10, -20, 50, 60])

    def test_web_ocr_endpoint_accepts_any_valid_http_endpoint(self) -> None:
        self.assertEqual(
            web_app.validate_paddleocr_vl_server_url(
                "http://ocr.internal:8081/v1",
                "http://127.0.0.1:8081/v1",
            ),
            "http://ocr.internal:8081/v1",
        )

    def test_web_ocr_endpoint_rejects_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            web_app.validate_paddleocr_vl_server_url(
                "http://user:password@ocr.internal:8081/v1",
                "http://127.0.0.1:8081/v1",
            )

    def test_web_vlm_endpoint_accepts_http_and_rejects_credentials(self) -> None:
        self.assertEqual(
            web_app.validate_vlm_base_url(
                "https://vlm.example/v1/",
                "http://127.0.0.1:8080/v1",
            ),
            "https://vlm.example/v1",
        )
        with self.assertRaisesRegex(ValueError, "credentials"):
            web_app.validate_vlm_base_url(
                "http://user:password@vlm.example/v1",
                "http://127.0.0.1:8080/v1",
            )


class OriginalInputUiTests(unittest.TestCase):
    def test_complete_job_generates_missing_translated_archives(self) -> None:
        markup = web_pages.download_links_html(
            "manga",
            "abc12345",
            {
                "status": "complete",
                "downloads": {
                    "png": {"available": True, "size": "1 MB", "downloadToken": "png-1"},
                    "webp": {"available": False},
                    "jxl": {"available": False},
                },
                "canView": True,
            },
        )

        self.assertIn("Download PNG CBZ (1 MB)", markup)
        self.assertIn("/download/png?v=png-1", markup)
        self.assertIn("Generate WebP CBZ", markup)
        self.assertIn("Generate JXL CBZ", markup)
        self.assertIn(
            'action="/job/manga/abc12345/generate-download/webp" method="post"',
            markup,
        )
        self.assertIn(
            'action="/job/manga/abc12345/generate-download/jxl" method="post"',
            markup,
        )

    def test_incomplete_job_does_not_show_translated_archive_controls(self) -> None:
        markup = web_pages.download_links_html(
            "manga",
            "abc12345",
            {"status": "running", "downloads": {}},
        )

        self.assertNotIn("Generate PNG CBZ", markup)
        self.assertNotIn("Download PNG CBZ", markup)

    def test_job_download_links_include_original_view_and_cbz(self) -> None:
        markup = web_pages.download_links_html(
            "manga",
            "abc12345",
            {
                "downloads": {},
                "inputSize": "12.3 MB",
                "inputDownloadToken": "123-456",
                "hasOriginalDownload": True,
                "originalDownloadUrl": "/job/manga/abc12345/download-original",
                "canViewOriginal": True,
                "originalViewUrl": "/job/manga/abc12345/view-original",
                "originalPageCount": 7,
            },
        )

        self.assertIn("View original (7 pages)", markup)
        self.assertIn("Download original CBZ (12.3 MB)", markup)
        self.assertIn("download-original?v=123-456", markup)

    def test_original_zip_download_uses_zip_label(self) -> None:
        markup = web_pages.download_links_html(
            "category",
            "abc12345",
            {
                "downloads": {},
                "inputFilename": "comic.zip",
                "hasOriginalDownload": True,
                "originalDownloadUrl": "/job/category/abc12345/download-original",
            },
        )

        self.assertIn("Download original ZIP", markup)

    def test_original_viewer_uses_original_image_routes(self) -> None:
        response = web_app.job_viewer_page(
            "manga",
            "abc12345",
            {"inputFilename": "book.cbz"},
            [{"index": 0, "token": "1-2"}],
            original=True,
        )
        body = response.body.decode("utf-8")

        self.assertIn("Original Browser View", body)
        self.assertIn("/view-original/image/0?v=1-2", body)
        self.assertIn('alt="Original page 0"', body)


class EditorV2Tests(unittest.TestCase):
    def create_job(self, root: Path) -> tuple[web_app.JobManager, str, str]:
        repo_dir = Path(__file__).resolve().parents[1]
        config_dir = root / "config"
        config_dir.mkdir()
        pipeline_config = config_dir / "vlm_config.json"
        pipeline_data = json.loads(
            (repo_dir / "data/config/vlm_config.example.json").read_text(
                encoding="utf-8"
            )
        )
        pipeline_data["model"] = "test-vlm-model"
        pipeline_config.write_text(json.dumps(pipeline_data), encoding="utf-8")
        web_config = config_dir / "web_config.json"
        web_config.write_text(
            json.dumps(
                {
                    "listen": "127.0.0.1:8088",
                    "jobs_dir": str(root / "jobs"),
                    "max_upload_bytes": 1024 * 1024,
                }
            ),
            encoding="utf-8",
        )
        manager = web_app.JobManager(web_app.load_web_config(web_config))
        manager.config.jobs_dir.mkdir(parents=True)
        manager._categories = {"default"}
        code, job_id = "default", "87654321"
        manager.job_dir(code, job_id).mkdir(parents=True)
        manager.save_status(code, job_id, {"status": "complete"})
        original_dir = manager.original_pages_dir(code, job_id)
        original_dir.mkdir(parents=True)
        Image.new("RGB", (100, 120), "white").save(original_dir / "0000.png")
        return manager, code, job_id

    def test_page_rerun_invalidates_all_translated_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.input_path(code, job_id).write_bytes(b"input")
            output_dir = manager.output_dir(code, job_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            for variant in web_app.TRANSLATED_CBZ_FILENAMES:
                manager.translated_cbz_variant_path(code, job_id, variant).write_bytes(
                    b"stale archive"
                )

            with mock.patch.object(manager, "enqueue"):
                manager.rerun_completed_job_pages(code, job_id, [0], "render")

            for variant in web_app.TRANSLATED_CBZ_FILENAMES:
                self.assertFalse(
                    manager.translated_cbz_variant_path(code, job_id, variant).exists()
                )

    def test_changed_fields_are_protected_and_unprotected_fields_are_rebased(self) -> None:
        manifest = editor_v2.default_manifest()
        baseline = editor_v2.hydrate_ids(
            "translations", 0, [{"page": 0, "boxno": 2, "englishText": "Generated"}]
        )
        edited = [dict(baseline[0], englishText="Manual")]

        editor_v2.update_artifact_override(
            manifest, 0, "translation", "translations", baseline, edited
        )
        record_id = baseline[0]["recordId"]
        field = editor_v2.artifact_override(
            manifest, 0, "translation", "translations"
        )["fields"][record_id]["englishText"]
        self.assertTrue(field["protected"])
        self.assertEqual(
            editor_v2.effective_records(manifest, 0, "translations", baseline)[0][
                "englishText"
            ],
            "Manual",
        )

        self.assertTrue(
            editor_v2.set_field_protection(
                manifest,
                0,
                "translation",
                "translations",
                record_id,
                "englishText",
                False,
            )
        )
        editor_v2.discard_unprotected(
            manifest, 0, "translation", "translations"
        )
        self.assertEqual(
            editor_v2.effective_records(manifest, 0, "translations", baseline)[0][
                "englishText"
            ],
            "Generated",
        )

    def test_saved_ocr_addition_appears_in_downstream_editor_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            data_dir = manager.output_dir(code, job_id) / "data"
            records_by_artifact = {
                "ocr_raw": [
                    {
                        "page": 0,
                        "boxno": 0,
                        "region": [10, 10, 30, 40],
                        "text": "既存",
                    }
                ],
                "ocr_merged": [
                    {
                        "page": 0,
                        "boxno": 0,
                        "sourceBoxnos": [0],
                        "sourceTexts": ["既存"],
                        "region": [10, 10, 30, 40],
                        "text": "既存",
                    }
                ],
                "ocr_structured": [
                    {
                        "page": 0,
                        "boxno": 0,
                        "sourceBoxnos": [0],
                        "sourceTexts": ["既存"],
                        "region": [10, 10, 30, 40],
                        "text": "既存",
                        "sfx": False,
                        "openLettering": False,
                    }
                ],
                "alt_placement": [],
                "translations": [
                    {
                        "page": 0,
                        "boxno": 0,
                        "text": "既存",
                        "englishText": "Existing",
                    }
                ],
                "placements": [
                    {
                        "page": 0,
                        "boxno": 0,
                        "placementRegion": [10, 10, 30, 40],
                    }
                ],
            }
            for artifact, records in records_by_artifact.items():
                path = data_dir / artifact / "page_0000.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(records), encoding="utf-8")
            manager.initialize_editor_v2(code, job_id)
            payload = manager.editor_v2_payload(code, job_id, "ocr", 0)
            raw_records = payload["recordsByArtifact"]["ocr_raw"] + [
                {
                    "page": 0,
                    "boxno": 1,
                    "recordId": "raw_manual_test",
                    "region": [50, 50, 80, 90],
                    "text": "テスト",
                }
            ]
            merged_records = payload["recordsByArtifact"]["ocr_merged"] + [
                {
                    "page": 0,
                    "boxno": 1,
                    "recordId": "group_manual_test",
                    "sourceBoxnos": [1],
                    "sourceTexts": ["テスト"],
                    "region": [50, 50, 80, 90],
                    "text": "テスト",
                }
            ]

            saved = manager.save_editor_v2_update(
                code,
                job_id,
                0,
                "ocr",
                {
                    "baseRevision": payload["revision"],
                    "recordsByArtifact": {
                        "ocr_raw": raw_records,
                        "ocr_merged": merged_records,
                    },
                },
            )

            downstream = saved["recordsByArtifact"]
            structured_test = next(
                record
                for record in downstream["ocr_structured"]
                if record.get("text") == "テスト"
            )
            self.assertEqual(structured_test["region"], [50, 50, 80, 90])
            translation_test = next(
                record
                for record in downstream["translations"]
                if record.get("boxno") == structured_test["boxno"]
            )
            self.assertEqual(translation_test["englishText"], "")
            placement_test = next(
                record
                for record in downstream["placements"]
                if record.get("boxno") == structured_test["boxno"]
            )
            self.assertEqual(placement_test["placementRegion"], [50, 50, 80, 90])
            self.assertTrue(saved["stageStates"]["structure"]["stale"])

            erase_payload = manager.editor_v2_payload(code, job_id, "erase", 0)
            structured_records = erase_payload["recordsByArtifact"]["ocr_structured"]
            structured_test = next(
                record for record in structured_records if record.get("text") == "テスト"
            )
            structured_test["altPlacementReason"] = "inside a clear text region"
            structured_test["openLettering"] = False
            erase_saved = manager.save_editor_v2_update(
                code,
                job_id,
                0,
                "erase",
                {
                    "baseRevision": erase_payload["revision"],
                    "recordsByArtifact": {
                        "ocr_structured": structured_records,
                        "alt_placement": erase_payload["recordsByArtifact"][
                            "alt_placement"
                        ],
                    },
                },
            )
            self.assertFalse(erase_saved["stageStates"]["structure"]["stale"])
            self.assertFalse(erase_saved["stageStates"]["erase"]["stale"])

            translation_payload = manager.editor_v2_payload(
                code, job_id, "translation", 0
            )
            translation_records = translation_payload["recordsByArtifact"][
                "translations"
            ]
            translation_test = next(
                record
                for record in translation_records
                if record.get("boxno") == structured_test["boxno"]
            )
            translation_test["englishText"] = "Test"
            translated = manager.save_editor_v2_update(
                code,
                job_id,
                0,
                "translation",
                {
                    "baseRevision": translation_payload["revision"],
                    "recordsByArtifact": {"translations": translation_records},
                },
            )

            for stage in ("structure", "erase", "translation"):
                self.assertFalse(translated["stageStates"][stage]["stale"])
            self.assertTrue(translated["stageStates"]["placement"]["stale"])
            manifest = manager.load_editor_v2_manifest(code, job_id)
            pending = editor_v2.pending_changes(manifest)
            self.assertEqual(pending, {0: {"translation"}})
            self.assertEqual(
                editor_v2.earliest_rerun(set().union(*pending.values())),
                "placements",
            )
            stored_translations = manager.editor_v2_effective_records(
                code, job_id, manifest, "translations", 0
            )
            stored_test = next(
                record
                for record in stored_translations
                if record.get("boxno") == structured_test["boxno"]
            )
            self.assertEqual(stored_test["englishText"], "Test")

    def test_saving_a_later_stage_accepts_current_upstream_values(self) -> None:
        manifest = editor_v2.default_manifest()
        editor_v2.mark_saved(manifest, 0, "ocr")

        editor_v2.mark_saved(manifest, 0, "translation")

        for stage in ("ocr", "structure", "erase", "translation"):
            self.assertFalse(editor_v2.stage_status(manifest, 0, stage)["stale"])
        self.assertTrue(editor_v2.stage_status(manifest, 0, "placement")["stale"])
        self.assertEqual(editor_v2.pending_changes(manifest), {0: {"translation"}})

    def test_width_relative_font_size_is_used_exactly(self) -> None:
        rendered, point_size, _technical = overlay_text.explicit_caption_layout(
            {"fontSizeWidthPercent": 2.5},
            "A short line",
            500,
            300,
            1000,
        )
        self.assertEqual(rendered, "A short line")
        self.assertEqual(point_size, 25.0)

    def test_width_relative_font_size_is_reduced_when_it_cannot_fit(self) -> None:
        rendered, point_size, technical = overlay_text.explicit_caption_layout(
            {"fontSizeWidthPercent": 10},
            "A generic line that must fit",
            120,
            180,
            1000,
        )

        self.assertLess(point_size, 100.0)
        self.assertEqual(point_size, technical)
        self.assertTrue(
            overlay_text.lines_fit(rendered.splitlines(), point_size, 120, 180)
        )

    def test_failed_page_batch_keeps_remaining_pages_for_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.input_path(code, job_id).write_bytes(b"input")
            status = manager.load_status(code, job_id)
            self.assertIsNotNone(status)
            status.update(
                {
                    "status": "running",
                    "pendingPageReruns": [
                        {"page": 2, "resumeFrom": "placements"},
                        {"page": 4, "resumeFrom": "translations"},
                        {"page": 6, "resumeFrom": "render"},
                    ],
                    "pendingResumeFrom": "placements",
                    "pendingResumePage": 2,
                    "lastResumeFrom": "translations",
                    "lastResumePage": 2,
                }
            )
            manager.save_status(code, job_id, status)

            with mock.patch.object(
                manager,
                "run_pipeline_process",
                side_effect=[0, 1],
            ) as run_process:
                manager.run_page_rerun_batch_job(
                    code,
                    job_id,
                    status,
                    status["pendingPageReruns"],
                )

            failed = manager.load_status(code, job_id)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(
                failed["pendingPageReruns"],
                [
                    {"page": 4, "resumeFrom": "translations"},
                    {"page": 6, "resumeFrom": "render"},
                ],
            )
            self.assertEqual(failed["pendingResumePage"], 4)
            self.assertNotIn("pendingPackageOnly", failed)
            for call in run_process.call_args_list:
                command = call.args[2]
                self.assertIn("--single-page", command)
                self.assertIn("--skip-package", command)
            self.assertEqual(
                run_process.call_args_list[0].args[2][
                    run_process.call_args_list[0].args[2].index("--resume-from") + 1
                ],
                "placements",
            )
            self.assertEqual(
                run_process.call_args_list[1].args[2][
                    run_process.call_args_list[1].args[2].index("--resume-from") + 1
                ],
                "translations",
            )

            with mock.patch.object(manager, "enqueue"):
                manager.restart_failed_job(code, job_id)
            queued = manager.load_status(code, job_id)
            self.assertEqual(queued["pendingPageReruns"], failed["pendingPageReruns"])

            with mock.patch.object(manager, "run_page_rerun_batch_job") as rerun:
                manager.run_job(code, job_id)
            self.assertEqual(rerun.call_args.args[3], failed["pendingPageReruns"])

    def test_caption_layout_keeps_trailing_punctuation_with_its_word(self) -> None:
        normalized = overlay_text.normalize_caption_text("A generic sample .")
        self.assertEqual(normalized, "A generic sample.")
        self.assertEqual(
            overlay_text.split_word_for_capacity("sample.", 6),
            ["sample."],
        )

        rendered, _point_size, _technical = overlay_text.explicit_caption_layout(
            {},
            normalized,
            176,
            442,
            2079,
            2929,
        )
        lines = rendered.splitlines()
        self.assertIn("sample.", lines)
        self.assertFalse(
            any(overlay_text.punctuation_only(line.strip()) for line in lines)
        )

    def test_tall_caption_uses_width_before_fragmenting_words(self) -> None:
        rendered, point_size, _technical = overlay_text.explicit_caption_layout(
            {},
            "Sample text is readable.",
            329,
            798,
            2079,
            2929,
        )

        self.assertEqual(rendered.splitlines(), ["Sample", "text is", "readable."])
        self.assertGreaterEqual(point_size, 40.0)

    def test_tall_caption_reduces_size_instead_of_excessive_breaks(self) -> None:
        rendered, _point_size, _technical = overlay_text.explicit_caption_layout(
            {},
            "Here... is a basic test~~",
            262,
            711,
            2079,
            2929,
        )

        self.assertEqual(
            rendered.splitlines(),
            ["Here...", "is a", "basic", "test~~"],
        )

    def test_placement_fill_is_corrected_for_background_legibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            white_page = root / "white.png"
            black_page = root / "black.png"
            Image.new("L", (100, 100), 255).save(white_page)
            Image.new("L", (100, 100), 0).save(black_page)

            white_result = translate_cbz.correct_placement_fills_for_legibility(
                translate_cbz.Page(index=0, image_path=white_page),
                [{"boxno": 0, "placementRegion": [10, 10, 90, 90], "fill": "white"}],
            )
            black_result = translate_cbz.correct_placement_fills_for_legibility(
                translate_cbz.Page(index=1, image_path=black_page),
                [{"boxno": 0, "placementRegion": [10, 10, 90, 90], "fill": "black"}],
            )

            self.assertEqual(white_result[0]["fill"], "black")
            self.assertEqual(black_result[0]["fill"], "white")

    def test_protected_ocr_deletion_suppresses_shifted_detection(self) -> None:
        manifest = editor_v2.default_manifest()
        baseline = editor_v2.hydrate_ids(
            "ocr_raw",
            0,
            [{"page": 0, "boxno": 0, "region": [10, 10, 50, 80], "text": "noise"}],
        )
        editor_v2.update_artifact_override(
            manifest, 0, "ocr", "ocr_raw", baseline, []
        )
        shifted = [
            {"page": 0, "boxno": 7, "region": [12, 12, 52, 82], "text": "noise"}
        ]
        self.assertEqual(
            editor_v2.effective_records(
                manifest, 0, "ocr_raw", shifted, protected_only=True
            ),
            [],
        )

    def test_stage_freeze_restores_complete_snapshot(self) -> None:
        manifest = editor_v2.default_manifest()
        snapshot = [{"page": 0, "boxno": 0, "englishText": "Keep"}]
        editor_v2.freeze_stage(
            manifest, 0, "translation", {"translations": snapshot}, True
        )
        effective = editor_v2.effective_records(
            manifest,
            0,
            "translations",
            [{"page": 0, "boxno": 0, "englishText": "Replace"}],
            protected_only=True,
        )
        self.assertEqual(effective[0]["englishText"], "Keep")

    def test_pending_changes_choose_earliest_required_pass(self) -> None:
        manifest = editor_v2.default_manifest()
        editor_v2.mark_saved(manifest, 3, "placement")
        editor_v2.mark_saved(manifest, 1, "structure")
        pending = editor_v2.pending_changes(manifest)
        self.assertEqual(pending, {1: {"structure"}, 3: {"placement"}})
        self.assertEqual(
            editor_v2.earliest_rerun(set().union(*pending.values())),
            "alt_placement",
        )
        editor_v2.mark_regenerated(manifest, 1, "translations")
        self.assertEqual(editor_v2.pending_changes(manifest)[1], {"structure"})
        self.assertTrue(editor_v2.stage_status(manifest, 1, "translation")["stale"])

    def test_all_changed_pages_keep_individual_resume_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.input_path(code, job_id).write_bytes(b"input")
            Image.new("RGB", (100, 120), "white").save(
                manager.original_pages_dir(code, job_id) / "0001.png"
            )
            manager.initialize_editor_v2(code, job_id)
            manifest = manager.load_editor_v2_manifest(code, job_id)
            editor_v2.mark_saved(manifest, 0, "placement")
            editor_v2.mark_saved(manifest, 1, "ocr")
            manager.save_editor_v2_manifest(code, job_id, manifest)

            with mock.patch.object(manager, "enqueue"):
                pages, resume_by_page = manager.rerun_editor_v2_changes(
                    code, job_id
                )

            self.assertEqual(pages, [0, 1])
            self.assertEqual(
                resume_by_page,
                {0: "render", 1: "ocr_structured"},
            )
            status = manager.load_status(code, job_id)
            self.assertEqual(
                status["pendingPageReruns"],
                [
                    {"page": 0, "resumeFrom": "render"},
                    {"page": 1, "resumeFrom": "ocr_structured"},
                ],
            )

    def test_editor_regeneration_starts_after_the_selected_stage(self) -> None:
        self.assertEqual(
            editor_v2.RERUN_FROM,
            {
                "ocr": "ocr_structured",
                "structure": "alt_placement",
                "erase": "translations",
                "translation": "placements",
                "placement": "render",
            },
        )

    def test_page_rerun_materializes_editor_state_before_queueing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            translation_path = (
                manager.output_dir(code, job_id)
                / "data"
                / "translations"
                / "page_0000.json"
            )
            translation_path.parent.mkdir(parents=True)
            translation_path.write_text(
                json.dumps(
                    [{"page": 0, "boxno": 4, "englishText": "Generated"}]
                ),
                encoding="utf-8",
            )
            manager.input_path(code, job_id).write_bytes(b"input")
            manager.initialize_editor_v2(code, job_id)
            payload = manager.editor_v2_payload(code, job_id, "translation", 0)
            record = payload["recordsByArtifact"]["translations"][0]
            record["englishText"] = "Manual translation"
            manager.save_editor_v2_update(
                code,
                job_id,
                0,
                "translation",
                {
                    "baseRevision": payload["revision"],
                    "recordsByArtifact": {"translations": [record]},
                },
            )

            translation_path.write_text(
                json.dumps(
                    [{"page": 0, "boxno": 4, "englishText": "Generated translation"}]
                ),
                encoding="utf-8",
            )
            with mock.patch.object(manager, "enqueue"):
                manager.rerun_completed_job_pages(
                    code,
                    job_id,
                    [0],
                    editor_v2.RERUN_FROM["translation"],
                )

            materialized = json.loads(translation_path.read_text(encoding="utf-8"))
            self.assertEqual(materialized[0]["englishText"], "Manual translation")
            status = manager.load_status(code, job_id)
            self.assertEqual(
                status["pendingPageReruns"],
                [{"page": 0, "resumeFrom": "placements"}],
            )

    def test_selected_retranslation_queues_only_selected_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.input_path(code, job_id).write_bytes(b"input")
            translation_path = (
                manager.output_dir(code, job_id)
                / "data"
                / "translations"
                / "page_0000.json"
            )
            translation_path.parent.mkdir(parents=True)
            translation_path.write_text(
                json.dumps(
                    [
                        {"page": 0, "boxno": 0, "englishText": "Generated first"},
                        {"page": 0, "boxno": 1, "englishText": "Generated second"},
                    ]
                ),
                encoding="utf-8",
            )
            manager.initialize_editor_v2(code, job_id)
            payload = manager.editor_v2_payload(code, job_id, "translation", 0)
            records = payload["recordsByArtifact"]["translations"]
            records[0]["englishText"] = "Manual first"
            records[1]["englishText"] = "Manual second"
            saved = manager.save_editor_v2_update(
                code,
                job_id,
                0,
                "translation",
                {
                    "baseRevision": payload["revision"],
                    "recordsByArtifact": {"translations": records},
                },
            )
            selected = saved["recordsByArtifact"]["translations"][1]

            with mock.patch.object(manager, "enqueue"):
                manager.retranslate_editor_v2_page(
                    code,
                    job_id,
                    0,
                    selected["recordId"],
                    selected["boxno"],
                )

            status = manager.load_status(code, job_id)
            self.assertEqual(
                status["pendingPageReruns"],
                [{"page": 0, "resumeFrom": "translations"}],
            )
            self.assertEqual(status["pendingTranslationBoxno"], 1)
            command = manager.build_command(
                code,
                job_id,
                "translations",
                0,
                single_page=True,
                translation_boxno=status["pendingTranslationBoxno"],
            )
            self.assertIn("--translation-boxno", command)
            self.assertEqual(command[command.index("--translation-boxno") + 1], "1")

            manifest = manager.load_editor_v2_manifest(code, job_id)
            protection = editor_v2.protection_payload(
                manifest, 0, "translation"
            )["records"]
            self.assertTrue(protection[records[0]["recordId"]]["englishText"])
            self.assertFalse(protection[records[1]["recordId"]]["englishText"])

    def test_selected_retranslation_job_finishes_without_placement_or_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.initialize_editor_v2(code, job_id)
            status = manager.load_status(code, job_id)
            status.update(
                {
                    "status": "queued",
                    "pendingPageReruns": [
                        {"page": 0, "resumeFrom": "translations"}
                    ],
                    "pendingResumeFrom": "translations",
                    "pendingResumePage": 0,
                    "pendingTranslationBoxno": 4,
                }
            )
            manager.save_status(code, job_id, status)

            with mock.patch.object(
                manager, "run_pipeline_process", return_value=0
            ) as run_process:
                manager.run_page_rerun_batch_job(
                    code, job_id, status, status["pendingPageReruns"]
                )

            self.assertEqual(run_process.call_count, 1)
            command = run_process.call_args.args[2]
            self.assertIn("--translation-boxno", command)
            self.assertNotIn("--resume-from package", " ".join(command))
            completed = manager.load_status(code, job_id)
            self.assertEqual(completed["status"], "complete")
            self.assertIn("Typesetting is out of date", completed["message"])
            self.assertNotIn("pendingTranslationBoxno", completed)
            manifest = manager.load_editor_v2_manifest(code, job_id)
            self.assertEqual(
                editor_v2.pending_changes(manifest),
                {0: {"translation"}},
            )
            self.assertTrue(
                editor_v2.stage_status(manifest, 0, "placement")["stale"]
            )

    def test_runtime_keeps_protected_edit_and_updates_generated_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "editor_v2.json"
            baseline_dir = root / "generated"
            manifest = editor_v2.default_manifest()
            original = editor_v2.hydrate_ids(
                "translations",
                0,
                [{"page": 0, "boxno": 0, "englishText": "Original"}],
            )
            editor_v2.update_artifact_override(
                manifest,
                0,
                "translation",
                "translations",
                original,
                [dict(original[0], englishText="Protected")],
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            editor_runtime.configure(manifest_path, baseline_dir)
            try:
                effective = editor_runtime.reconcile_records(
                    "translations",
                    0,
                    [{"page": 0, "boxno": 0, "englishText": "New model text"}],
                    "translation",
                )
            finally:
                editor_runtime.configure(None, None)
            self.assertEqual(effective[0]["englishText"], "Protected")
            generated = json.loads(
                (baseline_dir / "translations" / "page_0000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(generated[0]["englishText"], "New model text")

    def test_runtime_applies_protected_edit_without_replacing_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "editor_v2.json"
            baseline_dir = root / "generated"
            baseline_path = baseline_dir / "translations" / "page_0000.json"
            original = editor_v2.hydrate_ids(
                "translations",
                0,
                [{"page": 0, "boxno": 0, "englishText": "Original"}],
            )
            manifest = editor_v2.default_manifest()
            editor_v2.update_artifact_override(
                manifest,
                0,
                "translation",
                "translations",
                original,
                [dict(original[0], englishText="Protected")],
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            baseline_path.parent.mkdir(parents=True)
            baseline_path.write_text(json.dumps(original), encoding="utf-8")

            editor_runtime.configure(manifest_path, baseline_dir)
            try:
                effective = editor_runtime.apply_protected_records(
                    "translations",
                    0,
                    [{"page": 0, "boxno": 0, "englishText": "Saved output"}],
                    "translation",
                )
            finally:
                editor_runtime.configure(None, None)

            self.assertEqual(effective[0]["englishText"], "Protected")
            stored_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_baseline[0]["englishText"], "Original")

    def test_translation_pass_applies_protected_edit_before_downstream_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            manifest_path = root / "editor_v2.json"
            baseline_dir = root / "generated"
            original = editor_v2.hydrate_ids(
                "translations",
                0,
                [
                    {
                        "page": 0,
                        "boxno": 4,
                        "text": "source",
                        "englishText": "",
                    }
                ],
            )
            manifest = editor_v2.default_manifest()
            editor_v2.update_artifact_override(
                manifest,
                0,
                "translation",
                "translations",
                original,
                [dict(original[0], englishText="Manual translation")],
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            structured = {
                0: [{"page": 0, "boxno": 4, "text": "source"}]
            }
            page = translate_cbz.Page(index=0, image_path=root / "page.png")
            generated = [
                {
                    "page": 0,
                    "boxno": 4,
                    "text": "source",
                    "englishText": "Generated translation",
                }
            ]
            config = types.SimpleNamespace(
                language=mock.sentinel.language,
                vlm=mock.sentinel.vlm,
            )

            editor_runtime.configure(manifest_path, baseline_dir)
            try:
                with mock.patch.object(
                    translate_cbz,
                    "translation_prompt",
                    return_value="prompt",
                ), mock.patch.object(
                    translate_cbz,
                    "ensure_structured_debug_image",
                    return_value=root / "debug.png",
                ), mock.patch.object(
                    translate_cbz,
                    "get_validated_vlm_array",
                    return_value=generated,
                ):
                    result = translate_cbz.run_translation_phase(
                        [page],
                        structured,
                        output_dir,
                        config,
                        None,
                        {},
                    )
            finally:
                editor_runtime.configure(None, None)

            self.assertEqual(result[0][0]["englishText"], "Manual translation")
            self.assertEqual(structured[0][0]["englishText"], "Manual translation")
            written = json.loads(
                (
                    output_dir / "data" / "translations" / "page_0000.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(written[0]["englishText"], "Manual translation")
            generated_baseline = json.loads(
                (
                    baseline_dir / "translations" / "page_0000.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                generated_baseline[0]["englishText"], "Generated translation"
            )

    def test_selected_translation_replaces_only_requested_box(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            page = translate_cbz.Page(index=0, image_path=root / "page.png")
            structured = {
                0: [
                    {"page": 0, "boxno": 0, "text": "first"},
                    {"page": 0, "boxno": 1, "text": "second"},
                ]
            }
            translation_path = (
                output_dir / "data" / "translations" / "page_0000.json"
            )
            translation_path.parent.mkdir(parents=True)
            translation_path.write_text(
                json.dumps(
                    [
                        {"page": 0, "boxno": 0, "text": "first", "englishText": "Keep"},
                        {"page": 0, "boxno": 1, "text": "second", "englishText": "Old"},
                    ]
                ),
                encoding="utf-8",
            )
            config = types.SimpleNamespace(
                language=translate_cbz.language_config_from_codes("jp", "en"),
                vlm=mock.sentinel.vlm,
            )
            generated = [
                {"page": 0, "boxno": 1, "text": "second", "englishText": "New"}
            ]

            with mock.patch.object(
                translate_cbz,
                "translation_prompt",
                return_value="prompt",
            ) as prompt, mock.patch.object(
                translate_cbz,
                "ensure_structured_debug_image",
                return_value=root / "debug.png",
            ), mock.patch.object(
                translate_cbz,
                "get_validated_vlm_array",
                return_value=generated,
            ):
                result = translate_cbz.run_translation_phase(
                    [page],
                    structured,
                    output_dir,
                    config,
                    None,
                    {},
                    selected_boxno=1,
                )

            self.assertEqual(
                [record["englishText"] for record in result[0]],
                ["Keep", "New"],
            )
            self.assertEqual(prompt.call_args.args[4], [structured[0][1]])

    def test_selected_translation_resume_stops_before_typesetting(self) -> None:
        page = translate_cbz.Page(index=0, image_path=Path("page.png"))
        args = types.SimpleNamespace(
            resume_from="translations",
            resume_page=0,
            output_dir=Path("output"),
            fixture_dir=None,
            translation_boxno=4,
            skip_package=True,
        )
        structured = {0: [{"page": 0, "boxno": 4, "text": "source"}]}

        with mock.patch.object(
            translate_cbz, "load_phase_records", return_value={}
        ), mock.patch.object(
            translate_cbz, "load_structured_records", return_value=structured
        ), mock.patch.object(
            translate_cbz, "attach_translations", return_value={}
        ), mock.patch.object(
            translate_cbz, "run_translation_phase", return_value={0: []}
        ) as translate, mock.patch.object(
            translate_cbz, "run_placement_phase"
        ) as placement, mock.patch.object(
            translate_cbz, "render_pages"
        ) as render, mock.patch.object(
            translate_cbz, "print_packaged_cbz"
        ) as package:
            translate_cbz.run_single_page_resume_pipeline(
                args,
                mock.sentinel.config,
                [page],
                {},
            )

        self.assertEqual(translate.call_args.kwargs["selected_boxno"], 4)
        placement.assert_not_called()
        render.assert_not_called()
        package.assert_not_called()

    def test_subset_reconciliation_preserves_other_generated_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "editor_v2.json"
            baseline_dir = root / "generated"
            baseline = editor_v2.hydrate_ids(
                "translations",
                0,
                [
                    {"page": 0, "boxno": 0, "englishText": "Generated first"},
                    {"page": 0, "boxno": 1, "englishText": "Generated second"},
                ],
            )
            manifest = editor_v2.default_manifest()
            edited = [
                dict(baseline[0], englishText="Manual first"),
                dict(baseline[1], englishText="Manual second"),
            ]
            editor_v2.update_artifact_override(
                manifest,
                0,
                "translation",
                "translations",
                baseline,
                edited,
            )
            editor_v2.set_record_protection(
                manifest,
                0,
                "translation",
                "translations",
                baseline[1]["recordId"],
                False,
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            baseline_path = baseline_dir / "translations" / "page_0000.json"
            baseline_path.parent.mkdir(parents=True)
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            editor_runtime.configure(manifest_path, baseline_dir)
            try:
                result = editor_runtime.reconcile_record_subset(
                    "translations",
                    0,
                [{"page": 0, "boxno": 1, "englishText": "New second"}],
                    edited,
                    "translation",
                    "boxno",
                )
            finally:
                editor_runtime.configure(None, None)

            self.assertEqual(
                [record["englishText"] for record in result],
                ["Manual first", "New second"],
            )
            stored_baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [record["englishText"] for record in stored_baseline],
                ["Generated first", "New second"],
            )

    def test_manager_initializes_and_saves_versioned_editor_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            translations_path = (
                manager.output_dir(code, job_id)
                / "data"
                / "translations"
                / "page_0000.json"
            )
            translations_path.parent.mkdir(parents=True)
            translations_path.write_text(
                json.dumps(
                    [{"page": 0, "boxno": 0, "englishText": "Generated"}]
                ),
                encoding="utf-8",
            )
            manager.initialize_editor_v2(code, job_id)
            payload = manager.editor_v2_payload(code, job_id, "translation", 0)
            record = payload["recordsByArtifact"]["translations"][0]
            record["englishText"] = "Manual"
            saved = manager.save_editor_v2_update(
                code,
                job_id,
                0,
                "translation",
                {
                    "baseRevision": payload["revision"],
                    "recordsByArtifact": {"translations": [record]},
                },
            )
            self.assertEqual(
                saved["recordsByArtifact"]["translations"][0]["englishText"],
                "Manual",
            )
            self.assertTrue(
                saved["protection"]["records"][record["recordId"]]["englishText"]
            )

    def test_web_app_exposes_only_versioned_editor_mutation_routes(self) -> None:
        paths = {route.path for route in web_app.create_app().routes}
        self.assertIn(
            "/api/job/{code}/{job_id}/editor/v2/pages/{page}/stages/{stage}",
            paths,
        )
        self.assertIn("/api/job/{code}/{job_id}/editor/v2/ocr-crop", paths)
        self.assertIn(
            "/api/job/{code}/{job_id}/editor/v2/pages/{page}/retranslate",
            paths,
        )
        self.assertNotIn("/api/job/{code}/{job_id}/edit/{stage}/{page}", paths)
        self.assertNotIn("/api/job/{code}/{job_id}/edit/regenerate", paths)

    def test_editor_page_uses_five_stage_module_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.initialize_editor_v2(code, job_id)
            body = web_pages.editor_page(manager, code, job_id).body.decode("utf-8")
            for label in (
                "OCR &amp; Merge",
                "Structure",
                "Erase &amp; Alternate Placement",
                "Translation",
                "Typesetting",
            ):
                self.assertIn(label, body)
            self.assertEqual(body.count('class="stage-tab"'), 5)
            self.assertNotIn('data-stage="preview"', body)
            self.assertIn('/assets/editor-v2/app.js?v=8', body)
            self.assertIn('Previous [A]', body)
            self.assertIn('Next [D]', body)
            self.assertIn('id="record-list"', body)
            self.assertIn('id="record-form"', body)
            self.assertIn('id="record-resizer"', body)
            self.assertIn('id="inspector-resizer"', body)
            self.assertIn(
                "The current stage is used as input and is not regenerated.",
                body,
            )
            self.assertIn(
                "Rerun every page with saved editor changes",
                body,
            )
            self.assertIn('id="editor-tooltip"', body)

            editor_script = (
                Path(__file__).resolve().parents[1] / "web_editor" / "app.js"
            ).read_text(encoding="utf-8")
            self.assertIn(
                'new Set(["current", "original", "preview", "render"])',
                editor_script,
            )
            self.assertIn('let typesettingView = localStorage.getItem(typesettingViewKey) || "current"', editor_script)
            self.assertIn('"Retranslate page"', editor_script)
            self.assertIn('"Retranslate selected"', editor_script)
            self.assertIn('"Queueing regeneration..."', editor_script)

    def test_ocr_review_checkpoint_exposes_only_ocr_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.initialize_editor_v2(code, job_id)
            manager.input_path(code, job_id).write_bytes(b"input")
            status = manager.load_status(code, job_id)
            self.assertIsNotNone(status)
            status.update(
                {
                    "status": "paused",
                    "phase": "OCR review",
                    "pauseAfterOcr": True,
                    "reviewCheckpoint": "ocr",
                    "pendingResumeFrom": "ocr_structured",
                    "pendingResumePage": 0,
                }
            )
            manager.save_status(code, job_id, status)

            public = manager.public_status(code, job_id, include_log=False)
            self.assertTrue(public["canEdit"])
            self.assertTrue(public["canRestart"])
            self.assertTrue(public["ocrReviewCheckpoint"])
            body = web_pages.editor_page(manager, code, job_id).body.decode("utf-8")
            self.assertEqual(body.count('class="stage-tab"'), 1)
            self.assertIn("OCR review checkpoint", body)
            self.assertIn('id="continue-processing"', body)
            self.assertIn('availableStages: ["ocr"]', body)

            payload = manager.editor_v2_payload(code, job_id, "ocr", 0)
            self.assertEqual(payload["availableStages"], ["ocr"])
            with self.assertRaises(HTTPException) as raised:
                manager.editor_v2_payload(code, job_id, "structure", 0)
            self.assertEqual(raised.exception.status_code, 409)

    def test_ocr_review_continue_queues_structure_with_editor_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.initialize_editor_v2(code, job_id)
            manager.input_path(code, job_id).write_bytes(b"input")
            status = manager.load_status(code, job_id)
            self.assertIsNotNone(status)
            status.update(
                {
                    "status": "paused",
                    "pauseAfterOcr": True,
                    "reviewCheckpoint": "ocr",
                    "pendingResumeFrom": "ocr_structured",
                    "pendingResumePage": 0,
                }
            )
            manager.save_status(code, job_id, status)

            with mock.patch.object(manager, "enqueue") as enqueue:
                manager.restart_failed_job(code, job_id)

            queued = manager.load_status(code, job_id)
            self.assertEqual(queued["status"], "queued")
            self.assertEqual(queued["pendingResumeFrom"], "ocr_structured")
            self.assertEqual(queued["pendingResumePage"], 0)
            self.assertNotIn("reviewCheckpoint", queued)
            command = manager.build_command(
                code,
                job_id,
                queued["pendingResumeFrom"],
                queued["pendingResumePage"],
            )
            self.assertIn("--editor-manifest", command)
            self.assertIn("--editor-baseline-dir", command)
            self.assertNotIn("--stop-after", command)
            enqueue.assert_called_once_with(code, job_id)

    def test_successful_ocr_stop_creates_review_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.input_path(code, job_id).write_bytes(b"input")
            status = manager.load_status(code, job_id)
            self.assertIsNotNone(status)
            status.update({"status": "queued", "pauseAfterOcr": True})
            manager.save_status(code, job_id, status)

            with mock.patch.object(
                manager,
                "run_pipeline_process",
                return_value=0,
            ) as run_process:
                manager.run_job(code, job_id)

            checkpoint = manager.load_status(code, job_id)
            self.assertEqual(checkpoint["status"], "paused")
            self.assertEqual(checkpoint["phase"], "OCR review")
            self.assertEqual(checkpoint["reviewCheckpoint"], "ocr")
            self.assertEqual(checkpoint["pendingResumeFrom"], "ocr_structured")
            self.assertTrue(manager.has_editor_v2(code, job_id))
            command = run_process.call_args.args[2]
            self.assertEqual(command[command.index("--stop-after") + 1], "ocr_merged")

    def test_editor_rejects_job_without_editor_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            with self.assertRaises(HTTPException) as raised:
                web_pages.editor_page(manager, code, job_id)
            self.assertEqual(raised.exception.status_code, 409)
            self.assertEqual(
                raised.exception.detail,
                "Editor data is unavailable for this job.",
            )

    def test_exact_preview_reuses_matching_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            manager.initialize_editor_v2(code, job_id)
            placements = [
                {
                    "page": 0,
                    "boxno": 0,
                    "placementRegion": [10, 10, 80, 90],
                    "fill": "black",
                }
            ]

            def fake_overlay(_entries, _source, output):
                Image.new("RGB", (100, 120), "white").save(output)

            with mock.patch.object(
                overlay_text, "overlay_text", side_effect=fake_overlay
            ) as render:
                _path, first_cached, _first_cleaned = manager.render_editor_v2_preview(
                    code, job_id, 0, placements, []
                )
                _path, second_cached, _second_cleaned = manager.render_editor_v2_preview(
                    code, job_id, 0, placements, []
                )
                changed = [{**placements[0], "placementRegion": [20, 10, 90, 90]}]
                _path, changed_cached, _changed_cleaned = manager.render_editor_v2_preview(
                    code, job_id, 0, changed, []
                )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertFalse(changed_cached)
            self.assertEqual(render.call_count, 2)

    def test_current_preview_cleans_only_when_erase_mask_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            original = manager.original_page_path(code, job_id, 0)
            output_dir = manager.output_dir(code, job_id)
            pipeline_cleaned = output_dir / "pages" / "cleaned" / "0000.png"
            pipeline_mask = output_dir / "debug" / "masks" / "0000.png"
            pipeline_cleaned.parent.mkdir(parents=True)
            pipeline_mask.parent.mkdir(parents=True)
            Image.new("RGB", (100, 120), "white").save(pipeline_cleaned)
            initial_records = [
                {
                    "page": 0,
                    "boxno": 0,
                    "region": [10, 10, 30, 40],
                    "openLettering": False,
                }
            ]
            clean_text_regions.build_mask(
                initial_records,
                original,
                pipeline_mask,
                clean_text_regions.DEFAULT_PADDING,
            )

            def fake_clean(
                _entries,
                source,
                destination,
                _padding,
                _device,
                _model_path,
                _crop_trigger_size,
                _crop_margin,
                _keep_mask,
                _lama_session,
            ):
                Image.open(source).save(destination)

            with mock.patch.object(
                clean_text_regions,
                "clean_text_regions",
                side_effect=fake_clean,
            ) as clean:
                source, cleaned = manager.current_editor_clean_source(
                    code, job_id, 0, initial_records
                )
                changed_records = [
                    {**initial_records[0], "region": [50, 50, 80, 90]}
                ]
                changed_source, changed_cleaned = manager.current_editor_clean_source(
                    code, job_id, 0, changed_records
                )
                cached_source, cached_cleaned = manager.current_editor_clean_source(
                    code, job_id, 0, changed_records
                )

            self.assertEqual(source, pipeline_cleaned)
            self.assertFalse(cleaned)
            self.assertEqual(changed_source, cached_source)
            self.assertTrue(changed_cleaned)
            self.assertFalse(cached_cleaned)
            self.assertEqual(clean.call_count, 1)

    def test_editor_reports_automatic_width_relative_font_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, code, job_id = self.create_job(Path(temp_dir))
            output_data = manager.output_dir(code, job_id) / "data"
            translations_path = output_data / "translations" / "page_0000.json"
            placements_path = output_data / "placements" / "page_0000.json"
            translations_path.parent.mkdir(parents=True)
            placements_path.parent.mkdir(parents=True)
            translations_path.write_text(
                json.dumps([{"page": 0, "boxno": 0, "englishText": "A wrapped sentence"}]),
                encoding="utf-8",
            )
            placements_path.write_text(
                json.dumps(
                    [
                        {
                            "page": 0,
                            "boxno": 0,
                            "placementRegion": [10, 10, 90, 110],
                            "fill": "black",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            manager.initialize_editor_v2(code, job_id)

            payload = manager.editor_v2_payload(code, job_id, "placement", 0)
            placement = payload["recordsByArtifact"]["placements"][0]

            self.assertGreater(placement["_autoFontSizeWidthPercent"], 0)
            self.assertGreater(placement["_roughPointSize"], 0)
            self.assertTrue(placement["_roughText"])


if __name__ == "__main__":
    unittest.main()
