"""Small security helpers for passwords and private JSON files."""

from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
import threading
from typing import Any


_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 600_000
_SALT_BYTES = 16
_DIGEST_BYTES = 32
_MAX_PASSWORD_BYTES = 1_024
_JSON_WRITE_LOCK = threading.Lock()


def _password_bytes(password: str) -> bytes:
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    encoded = password.encode("utf-8")
    if not encoded:
        raise ValueError("password must not be empty")
    if len(encoded) > _MAX_PASSWORD_BYTES:
        raise ValueError("password must not exceed 1024 UTF-8 bytes")
    return encoded


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str, expected_bytes: int) -> bytes:
    expected_chars = 4 * ((expected_bytes + 2) // 3)
    if len(value) != expected_chars:
        raise ValueError("invalid encoded field length")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 field") from exc
    if len(decoded) != expected_bytes or _b64encode(decoded) != value:
        raise ValueError("invalid encoded field")
    return decoded


def hash_password(password: str) -> str:
    """Return a self-describing PBKDF2-HMAC-SHA256 password hash."""
    password_bytes = _password_bytes(password)
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password_bytes, salt, _ITERATIONS, dklen=_DIGEST_BYTES
    )
    return f"{_ALGORITHM}${_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def parse_password_hash(encoded: str) -> tuple[bytes, bytes]:
    """Strictly parse an encoded password hash or raise ValueError."""
    if not isinstance(encoded, str):
        raise ValueError("password hash must be a string")
    parts = encoded.split("$")
    if len(parts) != 4:
        raise ValueError("invalid password hash field count")
    algorithm, iterations, salt_text, digest_text = parts
    if algorithm != _ALGORITHM:
        raise ValueError("unsupported password hash algorithm")
    if iterations != str(_ITERATIONS):
        raise ValueError("unsupported password hash iteration count")
    return _b64decode(salt_text, _SALT_BYTES), _b64decode(
        digest_text,
        _DIGEST_BYTES,
    )


def verify_password(password: str, encoded: str) -> bool:
    """Return whether password matches a strictly parsed encoded hash."""
    try:
        password_bytes = _password_bytes(password)
    except (TypeError, UnicodeError, ValueError):
        return False
    try:
        salt, expected = parse_password_hash(encoded)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", password_bytes, salt, _ITERATIONS, dklen=_DIGEST_BYTES
    )
    return hmac.compare_digest(actual, expected)


def read_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a UTF-8 JSON object, translating read and parse errors to ValueError."""
    try:
        with open(path, encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON object from {os.fspath(path)!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON value in {os.fspath(path)!r} is not an object")
    return value


def _refuse_symlink(path: str | os.PathLike[str]) -> None:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise ValueError(f"refusing to replace symlink destination {os.fspath(path)!r}")


def _fsync_parent(parent: str) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    unsupported = {
        errno.EBADF,
        errno.EINVAL,
        errno.EPERM,
        errno.EROFS,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
    try:
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno in unsupported:
            return
        raise
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        if exc.errno not in unsupported:
            raise
    finally:
        os.close(directory_fd)


def write_private_json_atomic(
    path: str | os.PathLike[str], data: dict[str, Any]
) -> None:
    """Atomically write a JSON object with mode 0600 and durable flushes."""
    destination = os.fspath(path)
    parent = os.path.dirname(destination) or "."
    prefix = f".{os.path.basename(destination)}."
    temporary: str | None = None
    fd = -1

    with _JSON_WRITE_LOCK:
        _refuse_symlink(destination)
        try:
            fd, temporary = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=".tmp")
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                fd = -1
                json.dump(
                    data,
                    output,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            _refuse_symlink(destination)
            os.replace(temporary, destination)
            temporary = None
            _fsync_parent(parent)
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
