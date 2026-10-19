from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_source_launchers_always_include_ocr_and_inpaint(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "data/config"
            config.mkdir(parents=True)
            for name in ("web_config.json", "vlm_config.json"):
                (config / name).write_text("{}")
            (root / "input.cbz").touch()
            # Record arguments instead of installing packages or starting a server.
            uv = root / "uv"
            uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            uv.chmod(0o755)
            for launcher in ("run.sh", "run_server.sh"):
                shutil.copyfile(repository / launcher, root / launcher)
                for old_flag in (None, "0"):
                    with self.subTest(launcher=launcher, old_flag=old_flag):
                        env = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}"}
                        env.pop("TETOLATE_LOCAL_PADDLEOCR", None)
                        if old_flag is not None:
                            env["TETOLATE_LOCAL_PADDLEOCR"] = old_flag
                        args = ([str(root / "input.cbz")] if launcher == "run.sh" else [])
                        extra_args = ["--port", "8090"] if launcher == "run_server.sh" else ["--overwrite"]
                        args += extra_args
                        result = subprocess.run(
                            ["bash", str(root / launcher), *args],
                            cwd=root, env=env, text=True, capture_output=True, check=True,
                        )
                        command = result.stdout.splitlines()
                        self.assertEqual(command[:8], ["run", "--locked", "--extra", "inpaint", "--extra", "ocr", "--project", str(root)])
                        self.assertIn("python", command)
                        self.assertEqual(command[-len(extra_args):], extra_args)


if __name__ == "__main__":
    unittest.main()
