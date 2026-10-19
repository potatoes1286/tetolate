from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from httpx import ASGITransport, AsyncClient

import web_app
import web_security
from tests import test_regressions


class RememberSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.helper = test_regressions.WebAuthenticationTests()
        self.password = self.helper.ADMIN_PASSWORD
        self.manager = self.helper.initialized_manager(self.root)

    def restart(self):
        manager = web_app.JobManager(self.manager.config)
        manager.initialize_web_state()
        return manager

    def test_remembered_session_is_hashed_private_and_survives_restart(self):
        token = self.manager.authenticate_admin(self.password, remember_session=True)
        regular = self.manager.authenticate_admin(self.password)
        state = self.manager.web_state_path()
        self.assertNotIn(token, state.read_text())
        self.assertNotIn(regular, state.read_text())
        self.assertEqual(state.stat().st_mode & 0o777, 0o600)
        self.assertEqual(set(json.loads(state.read_text())["rememberedAdminSessions"]), {self.manager.session_digest(token)})
        restarted = self.restart()
        self.assertTrue(restarted.admin_session_is_valid(token))
        self.assertFalse(restarted.admin_session_is_valid(regular))

    def test_remembered_session_expires_after_thirty_days(self):
        now = time.time()
        with mock.patch.object(web_app.time, "time", return_value=now):
            token = self.manager.create_admin_session(remember_session=True)
        with mock.patch.object(web_app.time, "time", return_value=now + web_app.REMEMBERED_ADMIN_SESSION_SECONDS - 1):
            self.assertTrue(self.restart().admin_session_is_valid(token))
        with mock.patch.object(web_app.time, "time", return_value=now + web_app.REMEMBERED_ADMIN_SESSION_SECONDS):
            self.assertFalse(self.manager.admin_session_is_valid(token))
            self.assertFalse(self.restart().admin_session_is_valid(token))
        self.assertEqual(json.loads(self.manager.web_state_path().read_text())["rememberedAdminSessions"], {})

    def test_normal_session_retains_twelve_hour_server_limit(self):
        with mock.patch.object(web_app.time, "monotonic", return_value=100):
            token = self.manager.create_admin_session()
        with mock.patch.object(web_app.time, "monotonic", return_value=100 + web_app.ADMIN_SESSION_SECONDS):
            self.assertFalse(self.manager.admin_session_is_valid(token))

    def test_logout_revokes_only_that_remembered_session_durably(self):
        first = self.manager.create_admin_session(remember_session=True)
        second = self.manager.create_admin_session(remember_session=True)
        self.manager.revoke_admin_session(first)
        restarted = self.restart()
        self.assertFalse(restarted.admin_session_is_valid(first))
        self.assertTrue(restarted.admin_session_is_valid(second))

    def test_password_change_revokes_all_old_session_types_after_restart(self):
        remembered = self.manager.create_admin_session(remember_session=True)
        regular = self.manager.create_admin_session()
        new = self.manager.change_admin_password(self.password, "new-password", "new-password")
        self.assertFalse(self.manager.admin_session_is_valid(remembered))
        self.assertFalse(self.manager.admin_session_is_valid(regular))
        self.assertTrue(self.manager.admin_session_is_valid(new))
        self.assertFalse(self.restart().admin_session_is_valid(remembered))

    def test_failed_password_write_keeps_password_and_sessions(self):
        token = self.manager.create_admin_session(remember_session=True)
        with mock.patch.object(self.manager, "save_web_state", side_effect=OSError("Write failed")):
            with self.assertRaises(OSError):
                self.manager.change_admin_password(self.password, "new-password", "new-password")
        self.assertTrue(self.manager.admin_password_matches(self.password))
        self.assertTrue(self.manager.admin_session_is_valid(token))
        self.assertTrue(self.restart().admin_session_is_valid(token))

    def test_failed_session_write_does_not_leave_an_unreturned_valid_token(self):
        with mock.patch.object(self.manager, "save_web_state", side_effect=OSError("Write failed")):
            with self.assertRaises(OSError):
                self.manager.create_admin_session(remember_session=True)
        self.assertEqual(self.manager._remembered_admin_sessions, {})

    def test_failed_logout_write_is_not_reported_as_success(self):
        token = self.manager.create_admin_session(remember_session=True)
        with mock.patch.object(self.manager, "save_web_state", side_effect=OSError("Write failed")):
            with self.assertRaises(OSError):
                self.manager.revoke_admin_session(token)
        self.assertTrue(self.manager.admin_session_is_valid(token))

    def test_invalid_persisted_sessions_are_rejected(self):
        original = json.loads(self.manager.web_state_path().read_text())
        for value in ([], {"raw-token": time.time() + 100}, {"0" * 64: True}, {"0" * 64: "expiry"}):
            with self.subTest(value=value):
                web_security.write_private_json_atomic(self.manager.web_state_path(), {**original, "rememberedAdminSessions": value})
                with self.assertRaisesRegex(RuntimeError, "rememberedAdminSessions"):
                    self.restart()

    def test_http_login_cookie_security_persistence_and_logout(self):
        app = web_app.create_app(self.root / "web_config.json"); app.state.manager = self.manager
        async def exercise():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="https://testserver") as client:
                response = await client.get("/admin")
                self.assertIn('name="remember_session"', response.text)
                self.assertIn("Remember session (30 days)", response.text)
                self.assertNotIn('value="1" checked', response.text)
                response = await client.post("/admin/login", data={"password": self.password})
                cookie = response.headers["set-cookie"].lower()
                self.assertNotIn("max-age", cookie); self.assertNotIn("expires", cookie)
                for attribute in ("secure", "httponly", "samesite=strict", "path=/"):
                    self.assertIn(attribute, cookie)
                response = await client.post("/admin/login", data={"password": self.password, "remember_session": "1"})
                self.assertEqual(response.status_code, 303)
                self.assertIn(f"max-age={web_app.REMEMBERED_ADMIN_SESSION_SECONDS}", response.headers["set-cookie"].lower())
                token = client.cookies[web_app.ADMIN_COOKIE_NAME]
                app.state.manager = self.restart()
                self.assertIn("Job Categories", (await client.get("/admin")).text)
                self.assertEqual((await client.post("/admin/logout")).status_code, 303)
                self.assertNotIn(web_app.ADMIN_COOKIE_NAME, client.cookies)
                self.assertFalse(self.restart().admin_session_is_valid(token))
        async def direct(function, *args, **kwargs): return function(*args)
        with mock.patch.object(web_app, "run_in_threadpool", new=direct), mock.patch("anyio.to_thread.run_sync", new=direct):
            asyncio.run(exercise())

    def test_bad_password_never_creates_a_remembered_session(self):
        self.assertIsNone(self.manager.authenticate_admin("wrong-password", remember_session=True))
        self.assertEqual(self.manager._remembered_admin_sessions, {})


if __name__ == "__main__":
    unittest.main()
