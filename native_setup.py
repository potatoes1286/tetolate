"""Interactive native setup. Does not start model services or install a system service."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit

import paddle_ocr_image
import translate_cbz
import web_app
from web_storage import write_json_atomic

ROOT = Path(__file__).resolve().parent


class SetupError(Exception):
    pass


def read_config(path: Path, example: str) -> dict:
    try:
        value = json.loads((path if path.exists() else ROOT / "data/config" / example).read_text())
    except (OSError, ValueError):
        raise SetupError(f"Cannot read {path}. Fix the configuration and run setup again.") from None
    if not isinstance(value, dict):
        raise SetupError(f"{path} must contain a JSON object.")
    return value


def check_magick() -> None:
    magick = shutil.which("magick")
    if not magick:
        raise SetupError("Install ImageMagick 7 with PNG, WebP, and JPEG XL support.")
    try:
        version = subprocess.run([magick, "-version"], check=True, capture_output=True, text=True, timeout=10)
        if not re.search(r"ImageMagick 7\.", version.stdout):
            raise SetupError("ImageMagick 7 is required; another version is on PATH.")
        formats = subprocess.run([magick, "-list", "format"], check=True, capture_output=True, text=True, timeout=10)
        supported = {match[0] for match in re.findall(r"^\s*(PNG|WEBP|JXL)\*?\s+\S+\s+(rw\S*)", formats.stdout, re.M)}
        missing = {"PNG", "WEBP", "JXL"} - supported
        if missing:
            raise SetupError("ImageMagick needs read/write support for: " + ", ".join(sorted(missing)))
        subprocess.run([magick, "-background", "none", "-font", str(ROOT / "data/bundled_fonts/ComicNeue-Bold.ttf"),
                        "label:Tetolate", "null:"], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        raise SetupError("ImageMagick could not render text. Check its fonts and installation.") from None
    print("ImageMagick 7: text rendering, PNG, WebP, and JPEG XL are ready.")


def endpoint_models(endpoint: str, key: str, label: str) -> list[str]:
    try:
        endpoint = translate_cbz.normalize_vlm_base_url(endpoint)
        parsed = urlsplit(endpoint)
        if parsed.query or parsed.fragment or any(ord(char) <= 32 or ord(char) == 127 for char in endpoint):
            raise translate_cbz.PipelineError("Invalid endpoint")
    except translate_cbz.PipelineError:
        raise SetupError(f"{label}: enter an HTTP(S) API address without URL credentials or query parameters.") from None
    if not isinstance(key, str) or not key.isascii() or any(ord(char) <= 32 or ord(char) == 127 for char in key):
        raise SetupError(f"{label}: the API key contains an invalid character.")
    headers = {"Accept": "application/json"}
    if key and key != "not-needed":
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(endpoint + "/models", headers=headers)
    try:
        with urllib.request.build_opener(paddle_ocr_image.NoEndpointRedirect()).open(request, timeout=5) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("Response too large")
        data = json.loads(body)
        models = sorted({item["id"] for item in data["data"]
                         if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()})
        if not models:
            raise ValueError("No models")
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            message = "the API key was rejected"
        elif 300 <= error.code < 400:
            message = "the address redirects; enter its final API address"
        else:
            message = f"HTTP {error.code}; check the API address and port"
        raise SetupError(f"{label}: {message}.") from None
    except (OSError, ValueError, KeyError, TypeError):
        raise SetupError(f"{label}: cannot read the model list. Check the address, port, and running service.") from None
    return models


def prompt(label: str, default: str) -> str:
    return input(f"{label} [{default}]: ").strip() or default


def choose_model(label: str, models: list[str], current: str) -> str:
    for number, model in enumerate(models, 1):
        print(f"  {number}. {model}")
    while True:
        choice = prompt(label + " model (number or name)", current if current in models else models[0])
        if choice in models:
            return choice
        if choice.isdecimal() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Choose one of the listed models.")


def configure(pipeline: dict, *, edit: bool) -> None:
    ocr = pipeline.setdefault("ocr", {})
    if not isinstance(ocr, dict):
        raise SetupError("The ocr configuration must be an object.")
    if edit:
        previous_endpoint = pipeline["base_url"].rstrip("/")
        pipeline["base_url"] = prompt("VLM API address", pipeline["base_url"])
        retained_key = pipeline.get("api_key", "not-needed") if pipeline["base_url"].rstrip("/") == previous_endpoint else "not-needed"
        pipeline["api_key"] = getpass.getpass("VLM API key (blank = keep for the same address, otherwise no key): ") or retained_key
    vlm_models = endpoint_models(pipeline["base_url"], pipeline.get("api_key", "not-needed"), "VLM")
    if edit:
        pipeline["model"] = choose_model("VLM", vlm_models, pipeline.get("model", ""))
        while True:
            engine = prompt("OCR engine (paddleocr_vl or paddle)", ocr.get("engine", paddle_ocr_image.DEFAULT_OCR_ENGINE))
            if engine in paddle_ocr_image.OCR_ENGINES:
                ocr["engine"] = engine
                break
            print("Choose paddleocr_vl for a separate inference service, or paddle for built-in OCR.")
    elif pipeline.get("model") not in vlm_models:
        raise SetupError("The saved VLM model is not listed. Run setup again and choose to update settings.")
    if ocr.get("engine", paddle_ocr_image.DEFAULT_OCR_ENGINE) == "paddleocr_vl":
        endpoint = ocr.get("paddleocr_vl_server_url", paddle_ocr_image.DEFAULT_PADDLEOCR_VL_SERVER_URL)
        key = ocr.get("paddleocr_vl_api_key", paddle_ocr_image.DEFAULT_PADDLEOCR_VL_API_KEY) or ""
        if edit:
            previous_endpoint = endpoint.rstrip("/")
            endpoint = prompt("PaddleOCR-VL API address", endpoint)
            retained_key = key if endpoint.rstrip("/") == previous_endpoint else ""
            key = getpass.getpass("PaddleOCR-VL API key (blank = keep for the same address, otherwise no key): ") or retained_key
        models = endpoint_models(endpoint, key, "PaddleOCR-VL")
        current = ocr.get("paddleocr_vl_model", paddle_ocr_image.DEFAULT_PADDLEOCR_VL_MODEL)
        if edit:
            ocr.update(paddleocr_vl_server_url=endpoint, paddleocr_vl_model=choose_model("PaddleOCR-VL", models, current),
                       paddleocr_vl_api_key=key)
        elif current not in models:
            raise SetupError("The saved OCR model is not listed. Run setup again and choose to update settings.")
    print("Model endpoint checks passed. These checks do not run image inference.")


def main() -> int:
    parser = web_app.server_parser()
    parser.description = "Check and configure a native Tetolate installation."
    parser.add_argument("--no-start", action="store_true", help="Set up and test without starting the server.")
    args = parser.parse_args()
    try:
        check_magick()
        config_path = args.config.expanduser().resolve()
        pipeline_path = config_path.with_name("vlm_config.json")
        pipeline = read_config(pipeline_path, "vlm_config.example.json")
        web = read_config(config_path, "web_config.example.json")
        new_pipeline = not pipeline_path.exists()
        if new_pipeline:
            pipeline["base_url"] = "http://127.0.0.1:8080/v1"
        edit = new_pipeline or prompt("Update existing model settings? (y/N)", "n").lower() == "y"
        configure(pipeline, edit=edit)
        # Validate the complete candidate configuration before changing either
        # real file. Loading configuration does not start jobs or model services.
        with tempfile.TemporaryDirectory(prefix="tetolate-setup-") as directory:
            candidate = Path(directory)
            write_json_atomic(candidate / "vlm_config.json", pipeline, mode=0o600)
            write_json_atomic(candidate / "web_config.json", web, mode=0o600)
            translate_cbz.load_config(candidate / "vlm_config.json", None)
            web_app.load_web_config(candidate / "web_config.json")
        if new_pipeline or edit:
            write_json_atomic(pipeline_path, pipeline, mode=0o600)
        if not config_path.exists():
            write_json_atomic(config_path, web, mode=0o600)
        print("Setup complete. Default login password: changeme (new installations only).")
        if not args.no_start:
            command = [sys.executable, str(ROOT / "web_app.py"), "--config", str(config_path), "--address", args.address]
            if args.port is not None:
                command += ["--port", str(args.port)]
            os.execv(sys.executable, command)
        return 0
    except (SetupError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Setup cancelled.", file=sys.stderr)
        return 1
    except (KeyError, TypeError, ValueError):
        print("error: invalid configuration. Check the configuration files and run setup again.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
