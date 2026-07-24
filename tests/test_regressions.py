from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from PIL import Image

import clean_text_regions
import lama_inpaint
import merge_text_json
import overlay_text
import paddle_ocr_image
import paddle_ocr_server
import translate_cbz
import web_app
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

class WebAuthenticationTests(unittest.TestCase):
    ADMIN_PASSWORD = "current admin password"

    def write_web_config(self, root: Path) -> Path:
        repo_dir = Path(__file__).resolve().parents[1]
        data: dict[str, object] = {
            "listen": "127.0.0.1:8088",
            "jobs_dir": str(root / "jobs"),
            "max_upload_bytes": 1024 * 1024,
        }
        (root / "vlm_config.json").write_text(
            (
                repo_dir / "data" / "config" / "vlm_config.example.json"
            ).read_text(encoding="utf-8"),
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
                },
            )
            command = manager.build_command("default", job_id)

        self.assertEqual(options["vlmBaseUrl"], endpoint)
        endpoint_index = command.index("--vlm-base-url")
        self.assertEqual(command[endpoint_index + 1], endpoint)

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
            web_app.admin_state_label(
                {"paused": False, "workerRunning": True, "queuedCount": 0, "active": []}
            ),
            "Idle",
        )
        self.assertEqual(
            web_app.admin_state_label(
                {"paused": False, "workerRunning": True, "queuedCount": 1, "active": []}
            ),
            "Queued",
        )
        self.assertEqual(
            web_app.admin_state_label(
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
            web_app.admin_state_label(
                {"paused": True, "workerRunning": True, "queuedCount": 0, "active": []}
            ),
            "Paused",
        )
        self.assertEqual(
            web_app.admin_state_label(
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

                    response = await client.get("/assets/editor.js")
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
            translate_cbz,
            "should_merge_ocr_records",
            return_value=True,
        ), mock.patch.object(
            translate_cbz,
            "union_regions",
            return_value=[10, 10, 40, 30],
        ):
            korean = translate_cbz.merge_ocr_records_for_page(
                page,
                raw_records,
                right_to_left=False,
            )
            rtl = translate_cbz.merge_ocr_records_for_page(
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


class ArchiveSafetyTests(unittest.TestCase):
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
            0: [{"page": 0, "boxno": 0, "text": "OTHER_PAGE_SECRET"}],
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

        self.assertNotIn("OTHER_PAGE_SECRET", prompt)
        self.assertIn("[left,top,right,bottom]", prompt)
        self.assertIn("do not re-derive or debate", prompt)
        self.assertIn('[mergedBoxno,classification]', prompt)
        self.assertIn('"text", "sfx", or "reject"', prompt)
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


class PlacementRayTests(unittest.TestCase):
    def test_ray_stopping_at_the_actual_image_edge_is_a_boundary(self) -> None:
        with Image.new("L", (20, 20), 255) as image:
            endpoint = translate_cbz.cast_placement_ray(
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
            endpoint = translate_cbz.cast_placement_ray(
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
            translate_cbz,
            "cast_placement_ray",
            side_effect=endpoints,
        ):
            component = translate_cbz.ray_cast_component(
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
    def test_job_download_links_include_original_view_and_cbz(self) -> None:
        markup = web_app.download_links_html(
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


if __name__ == "__main__":
    unittest.main()
