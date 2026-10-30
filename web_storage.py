"""Small atomic persistence helpers used by the web application."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def rollback_files(paths, backup_dir: Path):
    """Restore a bounded set of files if a multi-file publication fails.

    The caller must hold its publication lock. This handles exceptions, not
    power loss; each individual replacement remains atomic.
    """
    backups = {}
    for index, path in enumerate(dict.fromkeys(paths)):
        backup = backup_dir / str(index)
        if path.exists():
            shutil.copy2(path, backup)
            backups[path] = backup
        else:
            backups[path] = None
    try:
        yield
    except BaseException:
        for path, backup in reversed(list(backups.items())):
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(backup, path)
        raise


def write_json_atomic(path: Path, data: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = mode if mode is not None else (
        path.stat().st_mode & 0o777 if path.exists() else 0o644
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        os.chmod(temp_path, output_mode)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
