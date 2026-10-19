#!/usr/bin/env python3
"""Compare the saved serial Preview backend with the current batch endpoint."""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Request
from fastapi.responses import FileResponse
from httpx import ASGITransport, AsyncClient
from PIL import Image

import overlay_text
from tests import test_regressions
import web_app


def records_for(count: int, sizing: str) -> list[dict]:
    fonts = ["ComicNeue-Bold.ttf", "ComicNeue-BoldItalic.ttf", "IBMPlexMono-Medium.ttf"]
    return [{"placementRegion": [0, 0, 220 + index % 4 * 20, 180 + index % 3 * 30],
             "englishText": f"Box {index + 1}: We must find another way before the morning comes!",
             "font": fonts[index % len(fonts)], "fill": "black", "stroke": "white",
             "strokeWidth": 3, "gravity": ["center", "northwest", "southeast"][index % 3],
             **({"fontSizeWidthPercent": 1.8 if index % 2 else 2.4} if sizing == "fixed" else {})}
            for index in range(count)]


async def measure(root: Path, records: list[dict], mode: str, serial_method) -> tuple[dict, list[str]]:
    root.mkdir()
    manager, code, job = test_regressions.EditorV2Tests().create_job(root)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        manager.initialize_web_state()
    Image.new("RGB", (1200, 1800), "white").save(manager.original_page_path(code, job, 0))
    manager.initialize_editor_v2(code, job)
    app = web_app.create_app(root / "config/web_config.json")
    app.state.manager = manager

    @app.post("/benchmark/serial")
    async def serial(request: Request):
        record = await request.json()
        path = serial_method(manager, code, job, 0, record)
        return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        client.cookies.set(web_app.ADMIN_COOKIE_NAME, manager.authenticate_admin("changeme"))

        async def request_layers(items):
            if mode == "serial":
                images = []
                for item in items:
                    response = await client.post("/benchmark/serial", json=item)
                    response.raise_for_status()
                    images.append(response.content)
                return images
            response = await client.post(f"/api/job/{code}/{job}/editor/v2/pages/0/text-layers", json={"records": items})
            response.raise_for_status()
            return [base64.b64decode(png) for png in response.json()["layers"]]

        results, signatures = {}, []
        for phase, items in (("cold", records), ("warm", records),
                             ("one_edit", [{**records[0], "englishText": records[0]["englishText"] + " Edited."}])):
            with mock.patch.object(overlay_text, "run_magick", wraps=overlay_text.run_magick) as run:
                started = time.perf_counter()
                images = await request_layers(items)
                elapsed = time.perf_counter() - started
            expected_processes = (len(items) if mode == "serial" else 1) if phase != "warm" else 0
            if run.call_count != expected_processes:
                raise AssertionError(f"Unexpected process count: {mode} {phase}: {run.call_count}")
            results[phase] = {"seconds": elapsed, "magick_processes": run.call_count,
                              "http_requests": len(items) if mode == "serial" else 1}
            if phase == "cold":
                for png in images:
                    with Image.open(io.BytesIO(png)) as image:
                        signatures.append(hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest())
        return results, signatures


async def run(args):
    baseline = args.baseline_root / "web_editor_backend.py"
    spec = importlib.util.spec_from_file_location("serial_editor_reference", baseline)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    serial_method = reference.EditorManagerMixin.render_editor_v2_text_layer

    async def direct(function, *positional, **_kwargs):
        return function(*positional)

    output = {"scope": "Authenticated in-process HTTP: layout, render, cache, file reads and PNG response transfer. Excludes browser, network latency and 200ms debounce.",
              "python": platform.python_version(), "platform": platform.platform(),
              "magick": subprocess.check_output([overlay_text.ensure_magick(), "-version"], text=True).splitlines()[0],
              "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
              "rounds": args.rounds, "sizing": args.sizing, "page_pixels": [1200, 1800], "results": []}
    with tempfile.TemporaryDirectory(prefix="editor-layer-benchmark-") as temporary, \
         mock.patch.object(web_app, "run_in_threadpool", new=direct), \
         mock.patch("anyio.to_thread.run_sync", new=direct):
        root = Path(temporary)
        # Warm library and operating-system font caches equally outside measured trials.
        for mode in ("serial", "batch"):
            await measure(root / f"preheat-{mode}", records_for(3, args.sizing), mode, serial_method)
        for count in args.boxes:
            samples = []
            for trial in range(args.rounds):
                pair, signatures = {}, {}
                modes = ("serial", "batch") if trial % 2 == 0 else ("batch", "serial")
                for mode in modes:
                    # Clear font-object caches for both modes; warm filesystem caches remain shared.
                    overlay_text.loaded_layout_font.cache_clear()
                    pair[mode], signatures[mode] = await measure(root / f"{count}-{trial}-{mode}", records_for(count, args.sizing), mode, serial_method)
                if signatures["serial"] != signatures["batch"]:
                    raise AssertionError("Serial and batch pixel outputs differ")
                samples.append(pair)
                print(f"{count} boxes: trial {trial + 1}/{args.rounds} complete", file=sys.stderr, flush=True)
            medians = {phase: {mode: statistics.median(sample[mode][phase]["seconds"] for sample in samples)
                               for mode in ("serial", "batch")} for phase in ("cold", "warm", "one_edit")}
            for values in medians.values():
                values["speedup"] = values["serial"] / values["batch"]
            output["results"].append({"boxes": count, "median_seconds": medians, "samples": samples, "pixels_equal": True})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True, help="Pre-batch source snapshot containing the original single-layer method")
    parser.add_argument("--boxes", type=int, nargs="+", default=[1, 10, 30])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--sizing", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or any(not 1 <= count <= 256 for count in args.boxes):
        parser.error("Use at least one round and 1 to 256 boxes.")
    result = asyncio.run(run(args))
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
