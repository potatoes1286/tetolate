#!/usr/bin/env python3
"""Synchronize release metadata from version.py."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "version.py"
SEMVER_PATTERN = re.compile(r"\d+\.\d+\.\d+")


def read_version() -> str:
    tree = ast.parse(VERSION_FILE.read_text(encoding="utf-8"), str(VERSION_FILE))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "VERSION" for target in targets):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value.strip()
    raise SystemExit(f"{VERSION_FILE} must define VERSION as a string.")


def read_project_metadata() -> tuple[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle).get("project", {})
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not name:
        raise SystemExit("pyproject.toml must define project.name.")
    if not isinstance(version, str) or not SEMVER_PATTERN.fullmatch(version):
        raise SystemExit("pyproject.toml must define project.version as MAJOR.MINOR.PATCH.")
    return name, version


def replace_required(text: str, pattern: str, replacement: str, description: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update {description}.")
    return updated


def update_release_examples(text: str, target_version: str) -> str:
    heading = "## Release container"
    start = text.find(heading)
    if start < 0:
        raise SystemExit(f"Could not find documentation section {heading!r}.")
    updated, count = re.subn(
        r"(v|ghcr\.io/potatoes1286/tetolate:)\d+\.\d+\.\d+\b",
        lambda match: match[1] + target_version,
        text[start:],
    )
    if count == 0:
        raise SystemExit(f"Could not update a version reference in {heading!r}.")
    return text[:start] + updated


def update_file(path: Path, old_text: str, new_text: str) -> None:
    if old_text == new_text:
        return
    path.write_text(new_text, encoding="utf-8")


def synchronize(target_version: str, *, check_only: bool) -> list[Path]:
    project_name, old_version = read_project_metadata()
    if not SEMVER_PATTERN.fullmatch(target_version):
        raise SystemExit("version.py VERSION must be MAJOR.MINOR.PATCH.")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(r"^## " + re.escape(target_version) + r" - \d{4}-\d{2}-\d{2}$", changelog, re.MULTILINE):
        raise SystemExit(
            f"CHANGELOG.md must contain a dated ## {target_version} entry before release."
        )

    updates: list[tuple[Path, str, str]] = []
    pyproject_path = ROOT / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")
    pyproject_updated = replace_required(
        pyproject,
        r'^(version\s*=\s*)"' + re.escape(old_version) + r'"\s*$',
        r'\g<1>"' + target_version + '"',
        "pyproject.toml version",
    ) if old_version != target_version else pyproject
    updates.append((pyproject_path, pyproject, pyproject_updated))

    lock_path = ROOT / "uv.lock"
    lock = lock_path.read_text(encoding="utf-8")
    lock_updated = replace_required(
        lock,
        r'(\[\[package\]\]\nname = "' + re.escape(project_name) + r'"\nversion = )"\d+\.\d+\.\d+"',
        r'\g<1>"' + target_version + '"',
        "uv.lock project version",
    )
    updates.append((lock_path, lock, lock_updated))

    building_path = ROOT / "BUILDING.md"
    building = building_path.read_text(encoding="utf-8")
    building_updated = update_release_examples(building, target_version)
    updates.append((building_path, building, building_updated))

    changed = [path for path, old_text, new_text in updates if old_text != new_text]
    if check_only:
        return changed
    for path, old_text, new_text in updates:
        update_file(path, old_text, new_text)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check synchronization without changing files",
    )
    args = parser.parse_args()
    target_version = read_version()
    changed = synchronize(target_version, check_only=args.check)
    if args.check:
        if changed:
            print("Release metadata is out of date:", file=sys.stderr)
            for path in changed:
                print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(f"Release metadata is synchronized for {target_version}.")
        return 0
    if changed:
        print(f"Synchronized release metadata for {target_version}:")
        for path in changed:
            print(f"  {path.relative_to(ROOT)}")
    else:
        print(f"Release metadata is already synchronized for {target_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
