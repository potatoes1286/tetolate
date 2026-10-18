"""Persistent control tokens, separate from browser sessions and VLM keys."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

import web_security


CONTROL_TOKENS_FILENAME = ".control-tokens.json"


class ControlAuthMixin:
    def load_control_tokens(self) -> dict[str, Any]:
        path = self.config.jobs_dir / CONTROL_TOKENS_FILENAME
        return web_security.read_json_object(path) if path.exists() else {}

    def list_control_tokens(self) -> list[dict[str, Any]]:
        return [{key: value for key, value in record.items() if key != "hash"}
                for record in self.load_control_tokens().values()]

    def create_control_token(self, name: str, read_only: bool = False) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 100:
            raise HTTPException(400, "Token name must be 1-100 characters.")
        token_id = secrets.token_hex(8)
        token = f"tetolate_{token_id}_{secrets.token_urlsafe(32)}"
        record = {
            "id": token_id, "name": name, "readOnly": read_only,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "hash": hashlib.sha256(token.encode()).hexdigest(),
        }
        with self._lock:
            records = self.load_control_tokens()
            records[token_id] = record
            web_security.write_private_json_atomic(
                self.config.jobs_dir / CONTROL_TOKENS_FILENAME, records,
            )
        return {"token": token, **{key: value for key, value in record.items() if key != "hash"}}

    def authenticate_control_token(self, token: str) -> dict[str, Any] | None:
        parts = token.split("_", 2)
        if len(parts) != 3 or parts[0] != "tetolate":
            return None
        record = self.load_control_tokens().get(parts[1])
        if not record or not secrets.compare_digest(
            record["hash"], hashlib.sha256(token.encode()).hexdigest(),
        ):
            return None
        return {key: value for key, value in record.items() if key != "hash"}

    def revoke_control_token(self, token_id: str) -> None:
        with self._lock:
            records = self.load_control_tokens()
            if token_id not in records:
                raise HTTPException(404, "Unknown control token.")
            del records[token_id]
            web_security.write_private_json_atomic(
                self.config.jobs_dir / CONTROL_TOKENS_FILENAME, records,
            )
