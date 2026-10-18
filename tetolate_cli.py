#!/usr/bin/env python3
"""Terminal client for a running Tetolate server. Python standard library only."""

from __future__ import annotations

import argparse
import getpass
from http.client import HTTPException as HTTPTransportError
import json
import math
import os
from pathlib import Path
import secrets
import socket
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


API = "/api/v1"
EXIT_AUTH = 3
EXIT_TIMEOUT = 4
EXIT_CONNECTION = 5
FINISHED = {"complete", "failed", "cancelled"}


class ClientError(Exception):
    def __init__(self, message: str, exit_code: int = 1, code: str = "error"):
        super().__init__(message)
        self.exit_code = exit_code
        self.code = code


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def server_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ClientError("Server must be an HTTP(S) URL without credentials, query, or fragment.", 2, "invalid_server")
    try:
        parsed.port
    except ValueError:
        raise ClientError("Invalid server port.", 2, "invalid_server") from None
    return value.rstrip("/")


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ClientError("CLI configuration must be a JSON object.", 2)
    return data


def save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ClientError("CLI configuration cannot be a symbolic link.", 2)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".tetolate-cli-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Multipart:
    """A streamed request body with a known length; files are not loaded into RAM."""
    def __init__(self, files: Path | list[tuple[str, Path]], fields: dict):
        files = [("file", files)] if isinstance(files, Path) else files
        boundary = "tetolate-" + secrets.token_hex(16)
        self.content_type = "multipart/form-data; boundary=" + boundary
        parts = []
        for name, value in fields.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        self.parts = []
        for index, (field, path) in enumerate(files):
            if not path.is_file():
                raise ClientError(f"Upload file does not exist: {path}", 2)
            filename = path.name.replace('"', "_").replace("\\", "_").replace("\r", "_").replace("\n", "_")
            header = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
            self.parts.append(((b"".join(parts) if index == 0 else b"\r\n") + header, path))
        self.suffix = f"\r\n--{boundary}--\r\n".encode()
        self.length = sum(len(header) + path.stat().st_size for header, path in self.parts) + len(self.suffix)

    def __iter__(self):
        for header, path in self.parts:
            yield header
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk
        yield self.suffix


class Client:
    def __init__(self, server: str, token: str | None = None, request_timeout: float = 30):
        self.server = server_url(server)
        if token is not None and (not isinstance(token, str) or len(token) > 256 or any(char.isspace() or char == "\0" for char in token)):
            raise ClientError("Invalid control token.", EXIT_AUTH, "unauthorized")
        self.token = token
        self.timeout = request_timeout
        self.opener = build_opener(NoRedirect())

    def open(self, method: str, path: str, data=None, *, multipart: Multipart | None = None, timeout: float | None = None):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if multipart:
            body = multipart
            headers.update({"Content-Type": multipart.content_type, "Content-Length": str(multipart.length)})
        else:
            body = json.dumps(data).encode() if data is not None else None
            if body is not None:
                headers["Content-Type"] = "application/json"
        request = Request(self.server + API + path, data=body, headers=headers, method=method)
        try:
            return self.opener.open(request, timeout=timeout if timeout is not None else self.timeout)
        except HTTPError as error:
            try:
                content = json.loads(error.read().decode())
                detail = content.get("error", {})
                message = detail.get("message") or content.get("detail") or str(error)
                code = detail.get("code", "http_error")
            except (ValueError, UnicodeError, AttributeError, OSError, HTTPTransportError):
                message, code = f"HTTP {error.code}: {error.reason}", "http_error"
            raise ClientError(str(message), EXIT_AUTH if error.code in {401, 403} else 1, code) from None
        except (URLError, OSError, socket.timeout, HTTPTransportError) as error:
            raise ClientError(f"Cannot reach Tetolate: {error}", EXIT_CONNECTION, "connection_error") from None

    def request(self, method: str, path: str, data=None, **kwargs):
        with self.open(method, path, data, **kwargs) as response:
            try:
                return json.load(response)
            except (OSError, HTTPTransportError):
                raise ClientError("Server connection was interrupted.", EXIT_CONNECTION, "connection_error") from None
            except (ValueError, UnicodeError):
                raise ClientError("Server did not return JSON. Check the server address and API version.", 1, "invalid_response") from None

    def download(self, path: str, output: Path, *, deadline: float | None = None) -> dict:
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=output.parent, prefix=".tetolate-download-", suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as target, self.open("GET", path) as response:
                expected = response.headers.get("Content-Length")
                while True:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise ClientError("Timed out downloading the archive.", EXIT_TIMEOUT, "timeout")
                    try:
                        chunk = response.read(1024 * 1024)
                    except (OSError, HTTPTransportError):
                        raise ClientError("Download connection was interrupted.", EXIT_CONNECTION, "connection_error") from None
                    if not chunk:
                        break
                    target.write(chunk)
            size = temporary.stat().st_size
            if expected is not None and size != int(expected):
                raise ClientError("Download ended before the complete archive arrived.", EXIT_CONNECTION, "connection_error")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return {"output": str(output), "bytes": size}


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive finite number.")
    return number


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    for flags, kwargs in [
        (("--server",), {"help": "Server URL; or TETOLATE_SERVER_URL"}),
        (("--config",), {"type": Path, "help": "CLI credential file"}),
        (("--json",), {"action": "store_true", "help": "JSON output for scripts and agents"}),
        (("--request-timeout",), {"type": positive_float, "help": "Seconds per HTTP request (default: 30)"}),
    ]:
        common.add_argument(*flags, default=argparse.SUPPRESS, **kwargs)
    root = argparse.ArgumentParser(description=__doc__, parents=[common])
    groups = root.add_subparsers(dest="group", required=True)

    def command(subparsers, name, **kwargs):
        return subparsers.add_parser(name, parents=[common], **kwargs)

    for name in ("status", "pause", "resume", "logout", "schema"):
        command(groups, name)
    login = command(groups, "login", help="Save a persistent control token using the admin password")
    login.add_argument("--name", default="terminal")
    login.add_argument("--read-only", action="store_true")
    login.add_argument("--password-stdin", action="store_true", help="Read password from stdin instead of prompting")
    password = command(groups, "password", help="Change admin password; sign out browser sessions")
    password.add_argument("--password-stdin", action="store_true", help="Read current, new, and confirmation passwords as three lines")
    vlm = command(groups, "vlm")
    test = command(vlm.add_subparsers(dest="action", required=True), "test", help="Test connection and list models; remember a supplied key on success")
    test.add_argument("--endpoint", help="Default: the job, category, or server endpoint")
    test.add_argument("--category")
    test.add_argument("--job", dest="job_id")
    test.add_argument("--settings-file", type=Path, help="JSON with vlm_base_url and/or vlm_auth_token")
    test.set_defaults(set=[])
    for group in ("jobs", "categories", "series", "tokens"):
        parent = command(groups, group)
        actions = parent.add_subparsers(dest="action", required=True)
        names = {
            "jobs": ("list", "show", "submit", "wait", "logs", "restart", "terminate", "rename", "delete", "rerun", "settings", "export", "download", "series", "publish-notes", "set-translated-title", "translate-title", "retry-title-translation", "pages", "page"),
            "categories": ("list", "create", "rename", "delete", "settings"),
            "series": ("list", "create", "show", "rename", "notes", "instructions", "upload"),
            "tokens": ("list", "revoke", "create"),
        }[group]
        for action in names:
            leaf = command(actions, action)
            if group in {"jobs", "series"}:
                leaf.add_argument("--category", default="default" if action in {"submit", "create"} else None)
            if group == "jobs":
                if action == "submit":
                    leaf.add_argument("archive", type=Path, nargs="?")
                    leaf.add_argument("--images", type=Path, nargs="+", help="Page files in the required page order, instead of an archive")
                    leaf.add_argument("--series", default="")
                    leaf.add_argument("--chapter", default="")
                    leaf.add_argument("--notes-file", type=Path)
                elif action != "list":
                    leaf.add_argument("job_id")
                if action == "list":
                    leaf.add_argument("--state")
                if action in {"submit", "settings"}:
                    leaf.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
                    leaf.add_argument("--settings-file", type=Path)
                if action in {"wait", "logs", "export"}:
                    leaf.add_argument("--timeout", type=positive_float, default=3600, help="Overall time limit in seconds")
                    leaf.add_argument("--interval", type=positive_float, default=2)
                if action == "logs":
                    leaf.add_argument("--follow", action="store_true")
                if action == "rename":
                    leaf.add_argument("name")
                if action == "rerun":
                    leaf.add_argument("--from", dest="stage", choices=["ocr_raw", "ocr_merge", "ocr_structured", "alt_placement", "translations", "placements", "render"], default="ocr_raw")
                    leaf.add_argument("--pages", default="", help="Page numbers or ranges, starting at 0 (0-3,6,8)")
                if action == "set-translated-title":
                    leaf.add_argument("name", help="Translated title; an empty string clears it")
                if action == "retry-title-translation":
                    leaf.add_argument("--manual-title", default="", help="Use a manual title instead of the VLM")
                if action in {"pages", "page"}:
                    leaf.add_argument("--kind", choices=["original", "final"], default="final")
                if action == "page":
                    leaf.add_argument("page", type=int, help="Page index, starting at 0")
                    leaf.add_argument("--output", type=Path, required=True)
                if action in {"export", "download"}:
                    leaf.add_argument("--format", choices=["png", "webp", "jxl"] + (["original"] if action == "download" else []), default="png")
                    leaf.add_argument("--output", type=Path, required=True)
                if action == "export":
                    leaf.add_argument("--title", choices=["original", "translated"], default="original")
                    leaf.add_argument("--alt-title-in-notes", action="store_true")
                    leaf.add_argument("--regenerate", action="store_true")
                if action == "series":
                    leaf.add_argument("series_id", help="Series ID, or an empty string to detach")
                    leaf.add_argument("--chapter", default="")
                if action == "publish-notes":
                    leaf.add_argument("--use-current", action="store_true")
            elif group == "categories":
                if action != "list":
                    leaf.add_argument("name")
                if action == "rename":
                    leaf.add_argument("new_name")
                if action == "delete":
                    leaf.add_argument("--confirmation", required=True)
                if action == "settings":
                    leaf.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
                    leaf.add_argument("--settings-file", type=Path)
            elif group == "series":
                if action == "create":
                    leaf.add_argument("name")
                elif action != "list":
                    leaf.add_argument("series_id")
                if action == "rename":
                    leaf.add_argument("name")
                if action in {"notes", "instructions"}:
                    source = leaf.add_mutually_exclusive_group()
                    source.add_argument("--file", type=Path)
                    source.add_argument("--stdin", action="store_true")
                if action == "upload":
                    leaf.add_argument("archives", type=Path, nargs="+", help="1-200 CBZ/ZIP files; sorted by filename (2 before 10)")
                    leaf.add_argument("--first-chapter", default="1")
                    leaf.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
                    leaf.add_argument("--settings-file", type=Path)
                    leaf.add_argument("--notes-file", type=Path)
            elif group == "tokens" and action == "revoke":
                leaf.add_argument("token_id")
            elif group == "tokens" and action == "create":
                leaf.add_argument("name")
                leaf.add_argument("--read-only", action="store_true")
                leaf.add_argument("--password-stdin", action="store_true")
    return root


def settings_changes(args) -> dict:
    if args.settings_file and not args.settings_file.is_file():
        raise ClientError("Settings file does not exist.", 2, "invalid_setting")
    data = load_config(args.settings_file) if args.settings_file else {}
    boolean_fields = {"pause_after_ocr", "proofread_translations", "write_translation_notes", "alt_placement_enabled",
                      "auto_translate_comicinfo_title", "clear_vlm_auth_token", "clear_paddleocr_vl_auth_token"}
    integer_fields = {"thinking_budget_tokens", "ocr_page_workers", "lama_workers", "imagemagick_workers"}
    for setting in args.set:
        key, separator, value = setting.partition("=")
        if not separator or not key:
            raise ClientError("Use --set KEY=VALUE.", 2, "invalid_setting")
        if key.endswith("auth_token") and not key.startswith("clear_"):
            raise ClientError("Use --settings-file for credentials so keys do not enter shell history.", 2, "invalid_setting")
        if key in boolean_fields:
            if value.lower() not in {"true", "false"}:
                raise ClientError(f"{key} must be true or false.", 2)
            value = value.lower() == "true"
        elif key in integer_fields:
            value = int(value)
        data[key] = value
    return data


def job_path(args, suffix: str = "") -> str:
    path = "/jobs/" + quote(args.job_id, safe="") + suffix
    return path + ("?" + urlencode({"category": args.category}) if args.category else "")


def password_input(args, prompt: str) -> str:
    if args.password_stdin:
        line = sys.stdin.readline()
        if not line:
            raise ClientError("Missing password input.", 2, "invalid_input")
        return line.rstrip("\r\n")
    if not sys.stdin.isatty():
        raise ClientError("Use --password-stdin for non-interactive password input.", 2)
    return getpass.getpass(prompt)


def page_selection(value: str) -> list[int]:
    pages = set()
    for item in value.split(",") if value else []:
        bounds = item.strip().split("-")
        if len(bounds) not in {1, 2} or any(not part.isdecimal() for part in bounds):
            raise ClientError("Use page numbers or ranges: 0-3,6,8.", 2)
        start, end = int(bounds[0]), int(bounds[-1])
        if start > end or end > 100000:
            raise ClientError("Page ranges must increase and cannot exceed 100000.", 2)
        pages.update(range(start, end + 1))
    return sorted(pages)


def wait_job(client: Client, args, *, deadline: float | None = None) -> dict:
    deadline = deadline if deadline is not None else time.monotonic() + args.timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClientError("Timed out waiting for the job.", EXIT_TIMEOUT, "timeout")
        status = client.request("GET", job_path(args), timeout=min(client.timeout, remaining))
        if status["status"] in FINISHED:
            if status["status"] != "complete":
                raise ClientError(f"Job {status['status']}: {status.get('message', '')}", 1, "job_failed")
            return status
        if status["status"] == "paused":
            raise ClientError("Job is paused and requires resume or OCR review.", 1, "job_paused")
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


def read_logs(client: Client, args):
    deadline = time.monotonic() + args.timeout
    cursor = ""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ClientError("Timed out following job logs.", EXIT_TIMEOUT, "timeout")
        path = job_path(args, "/logs")
        path += ("&" if "?" in path else "?") + urlencode({"cursor": cursor})
        result = client.request("GET", path, timeout=min(client.timeout, remaining))
        cursor = result["cursor"]
        if getattr(args, "json", False):
            print(json.dumps(result), flush=True)
        else:
            print(result["text"], end="", flush=True)
        if result["hasMore"]:
            continue
        if not args.follow or result["status"] in FINISHED:
            return None
        if result["status"] == "paused":
            raise ClientError("Job is paused and requires resume or OCR review.", 1, "job_paused")
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


def resolve_series(client: Client, args) -> str:
    if args.category:
        code = args.category
    else:
        matches = [item for item in client.request("GET", "/series")["series"] if item["id"] == args.series_id]
        if len(matches) != 1:
            raise ClientError("Series ID not found or ambiguous. Specify --category.", 1, "not_found")
        code = matches[0]["category"]
    return "/categories/" + quote(code, safe="") + "/series/" + quote(args.series_id, safe="")


def execute(client: Client, args, config_path: Path, config: dict):
    group = args.group
    if group == "login" or (group == "tokens" and args.action == "create"):
        password = password_input(args, "Admin password: ")
        # Authentication must not use a possibly stale token from the client config.
        result = Client(client.server, request_timeout=client.timeout).request("POST", "/auth/login", {
            "password": password, "name": args.name, "read_only": args.read_only,
        })
        if group == "tokens":
            return result  # Show the new token once without replacing the saved login.
        save_config(config_path, {"server": client.server, "token": result["token"]})
        return {"server": client.server, "tokenId": result["id"], "readOnly": result["readOnly"], "config": str(config_path)}
    if not client.token:
        raise ClientError("Run login or set TETOLATE_CONTROL_TOKEN first.", EXIT_AUTH, "unauthorized")
    if group == "password":
        current = password_input(args, "Current password: ")
        new = password_input(args, "New password: ")
        confirmation = password_input(args, "Confirm new password: ")
        return client.request("POST", "/auth/password", {"current_password": current,
                              "new_password": new, "confirm_password": confirmation})
    if group == "vlm":
        data = settings_changes(args)
        if args.endpoint is not None:
            data["vlm_base_url"] = args.endpoint
        data.update({key: value for key, value in {"category": args.category, "job_id": args.job_id}.items() if value is not None})
        return client.request("POST", "/vlm/test", data)
    if group == "logout":
        parts = client.token.split("_", 2)
        if len(parts) != 3:
            raise ClientError("Invalid control token.", EXIT_AUTH)
        result = client.request("DELETE", "/tokens/" + quote(parts[1], safe=""))
        if config.get("token") == client.token:
            save_config(config_path, {key: value for key, value in config.items() if key != "token"})
        return result
    if group in {"status", "pause", "resume", "schema"}:
        return client.request("GET" if group in {"status", "schema"} else "POST", "/openapi.json" if group == "schema" else "/" + group)
    action = args.action
    if group == "tokens":
        return client.request("GET", "/tokens") if action == "list" else client.request("DELETE", "/tokens/" + quote(args.token_id, safe=""))
    if group == "categories":
        if action == "list":
            return client.request("GET", "/categories")
        path = "/categories/" + quote(args.name, safe="")
        if action == "create":
            return client.request("POST", "/categories", {"name": args.name})
        if action == "rename":
            return client.request("PATCH", path, {"name": args.new_name})
        if action == "delete":
            return client.request("DELETE", path, {"confirmation": args.confirmation})
        changes = settings_changes(args)
        return client.request("PATCH" if changes else "GET", path + "/settings", changes if changes else None)
    if group == "series":
        if action == "list":
            return client.request("GET", "/series" + ("?" + urlencode({"category": args.category}) if args.category else ""))
        if action == "create":
            return client.request("POST", "/categories/" + quote(args.category, safe="") + "/series", {"name": args.name})
        path = resolve_series(client, args)
        if action == "show":
            return client.request("GET", path)
        if action == "rename":
            return client.request("PATCH", path, {"name": args.name})
        if action == "upload":
            if len(args.archives) > 200:
                raise ClientError("Select 1-200 CBZ or ZIP chapters.", 2)
            fields = {"first_chapter": args.first_chapter, "settings": json.dumps(settings_changes(args)),
                      "translation_notes": args.notes_file.read_text(encoding="utf-8") if args.notes_file else ""}
            return client.request("POST", path + "/upload", multipart=Multipart([("chapters", file) for file in args.archives], fields))
        value = args.file.read_text(encoding="utf-8") if args.file else sys.stdin.read() if args.stdin else None
        result = client.request("GET" if value is None else "PATCH", path, None if value is None else {action: value})
        return {"id": result["id"], "category": result["category"], action: result[action], "revision": result["revision"]}
    if action == "list":
        filters = {key: value for key, value in {"category": args.category, "state": args.state}.items() if value}
        return client.request("GET", "/jobs" + ("?" + urlencode(filters) if filters else ""))
    if action == "submit":
        if bool(args.archive) == bool(args.images):
            raise ClientError("Supply an archive or --images, not both.", 2)
        fields = {"category": args.category, "settings": json.dumps(settings_changes(args)),
                  "series_id": args.series, "chapter": args.chapter,
                  "translation_notes": args.notes_file.read_text(encoding="utf-8") if args.notes_file else ""}
        files = [("page_images", path) for path in args.images] if args.images else args.archive
        return client.request("POST", "/jobs", multipart=Multipart(files, fields))
    if action == "show":
        return client.request("GET", job_path(args))
    if action == "wait":
        return wait_job(client, args)
    if action == "logs":
        return read_logs(client, args)
    if action == "settings":
        changes = settings_changes(args)
        return client.request("PATCH", job_path(args, "/settings"), changes) if changes else client.request("GET", job_path(args))
    if action in {"restart", "terminate"}:
        return client.request("POST", job_path(args, "/" + action))
    if action == "rename":
        return client.request("PATCH", job_path(args, "/name"), {"name": args.name})
    if action == "set-translated-title":
        return client.request("PATCH", job_path(args, "/translated-title"), {"name": args.name})
    if action == "translate-title":
        return client.request("POST", job_path(args, "/translate-title"))
    if action == "retry-title-translation":
        return client.request("POST", job_path(args, "/retry-title-translation"), {"manual_title": args.manual_title})
    if action in {"pages", "page"}:
        path = job_path(args, "/pages" + ("/" + str(args.page) if action == "page" else ""))
        path += ("&" if "?" in path else "?") + urlencode({"kind": args.kind})
        return client.download(path, args.output) if action == "page" else client.request("GET", path)
    if action == "delete":
        return client.request("DELETE", job_path(args))
    if action == "rerun":
        pages = page_selection(args.pages)
        return client.request("POST", job_path(args, "/rerun"), {"stage": args.stage, "pages": pages})
    if action == "series":
        return client.request("PATCH", job_path(args, "/series"), {"series_id": args.series_id, "chapter": args.chapter})
    if action == "publish-notes":
        return client.request("POST", job_path(args, "/series-notes"), {"use_current": args.use_current})
    deadline = None
    if action == "export":
        deadline = time.monotonic() + args.timeout
        client.request("POST", job_path(args, "/export"), {"format": args.format,
            "title_mode": args.title, "alt_title_in_notes": args.alt_title_in_notes, "regenerate": args.regenerate},
            timeout=min(client.timeout, args.timeout))
        wait_job(client, args, deadline=deadline)
    return client.download(job_path(args, "/download/" + args.format), args.output, deadline=deadline)


def display(result, machine: bool) -> None:
    if result is None:
        return
    if machine:
        print(json.dumps(result, ensure_ascii=False))
        return
    if isinstance(result, dict) and len(result) == 1 and isinstance(next(iter(result.values())), list):
        items = next(iter(result.values()))
        for item in items:
            if isinstance(item, dict):
                keys = ("category", "jobId", "id", "status", "name", "jobName", "page", "filename", "bytes", "readOnly")
                print("\t".join(str(item[key]) for key in keys if key in item) or json.dumps(item, ensure_ascii=False))
            else:
                print(item)
        if not items:
            print("No entries.")
    elif isinstance(result, dict):
        for key, value in result.items():
            print(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}")
    else:
        print(result)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    machine = getattr(args, "json", False)
    try:
        default_config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tetolate" / "cli.json"
        config_path = getattr(args, "config", default_config).expanduser()
        config = load_config(config_path)
        # Credentials from one saved server must never be sent to another server.
        address = server_url(getattr(args, "server", None) or os.environ.get("TETOLATE_SERVER_URL") or config.get("server") or "http://127.0.0.1:8088")
        token = os.environ.get("TETOLATE_CONTROL_TOKEN") or None
        if not token and config.get("server") and server_url(config["server"]) == address:
            token = config.get("token")
        client = Client(address, token, getattr(args, "request_timeout", 30))
        result = execute(client, args, config_path, config)
        display(result, machine)
        return 0
    except ClientError as error:
        print(json.dumps({"error": {"code": error.code, "message": str(error)}}) if machine else f"error: {error}", file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"error": {"code": "invalid_input", "message": str(error)}}) if machine else f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
