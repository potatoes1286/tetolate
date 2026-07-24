#!/usr/bin/env python3
"""Merge OCR and translation excerpts into a master text JSON file."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


Key = tuple[int, int]


class InputError(ValueError):
    """Raised when JSON inputs cannot be merged."""


def reject_json_constant(value: str) -> Any:
    raise InputError(f"JSON contains non-finite number: {value}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge OCR or translation excerpts into a master text JSON file.",
    )
    parser.add_argument(
        "--mode",
        choices=("ocr", "translation"),
        required=True,
        help="Type of excerpt being merged.",
    )
    parser.add_argument("master_json", type=Path, help="Existing master JSON file")
    parser.add_argument("excerpt_json", type=Path, help="OCR or translation excerpt JSON")
    parser.add_argument(
        "output_json",
        type=Path,
        help="Merged JSON file to write. May be the same path as master_json.",
    )
    return parser.parse_args()


def load_json_list(path: Path, label: str, allow_missing_or_empty: bool = False) -> list[Any]:
    if allow_missing_or_empty and not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InputError(f"{label} file not found: {path}") from exc

    if allow_missing_or_empty and not content.strip():
        return []

    try:
        data = json.loads(content, parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"Invalid JSON in {label} file {path}: {exc.msg} at line {exc.lineno}"
        ) from exc

    if not isinstance(data, list):
        raise InputError(f"The {label} JSON root must be a list.")

    return data


def validate_object_list(items: list[Any], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise InputError(f"{label} entry {index} must be an object.")
        records.append(item)
    return records


def require_non_negative_int(record: dict[str, Any], field: str, label: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(f"{label} must have {field} as a non-negative integer.")
    return value


def require_string(record: dict[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise InputError(f"{label} must have {field} as a string.")
    return value


def require_bool(record: dict[str, Any], field: str, label: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise InputError(f"{label} must have {field} as a boolean.")
    return value


def require_region(record: dict[str, Any], label: str) -> list[int | float]:
    region = record.get("region")
    if not isinstance(region, list) or len(region) != 4:
        raise InputError(f"{label} must have region as a list of four numbers.")

    for coordinate in region:
        if (
            not isinstance(coordinate, (int, float))
            or isinstance(coordinate, bool)
            or not math.isfinite(coordinate)
        ):
            raise InputError(f"{label} region values must be finite numbers.")

    left, top, right, bottom = region
    if right <= left or bottom <= top:
        raise InputError(f"{label} region must have positive width and height.")

    return region


def entry_key(record: dict[str, Any], label: str) -> Key:
    page = require_non_negative_int(record, "page", label)
    boxno = require_non_negative_int(record, "boxno", label)
    return page, boxno


def validate_master(records: list[dict[str, Any]]) -> dict[Key, dict[str, Any]]:
    indexed: dict[Key, dict[str, Any]] = {}
    for index, record in enumerate(records):
        key = entry_key(record, f"Master entry {index}")
        if key in indexed:
            raise InputError(f"Duplicate master entry for page {key[0]}, boxno {key[1]}.")
        indexed[key] = dict(record)
    return indexed


def validate_excerpt_keys(records: Iterable[dict[str, Any]], label: str) -> None:
    seen: set[Key] = set()
    for index, record in enumerate(records):
        key = entry_key(record, f"{label} entry {index}")
        if key in seen:
            raise InputError(f"Duplicate {label} entry for page {key[0]}, boxno {key[1]}.")
        seen.add(key)


def validate_ocr_entry(record: dict[str, Any], index: int) -> None:
    label = f"OCR excerpt entry {index}"
    entry_key(record, label)
    require_region(record, label)
    require_bool(record, "sfx", label)
    require_bool(record, "openLettering", label)
    require_string(record, "text", label)


def validate_translation_entry(record: dict[str, Any], index: int) -> None:
    label = f"Translation excerpt entry {index}"
    entry_key(record, label)
    require_string(record, "text", label)
    require_string(record, "englishText", label)


def merge_ocr(
    master_by_key: dict[Key, dict[str, Any]],
    excerpt_records: list[dict[str, Any]],
) -> None:
    validate_excerpt_keys(excerpt_records, "OCR excerpt")

    for index, record in enumerate(excerpt_records):
        validate_ocr_entry(record, index)
        key = entry_key(record, f"OCR excerpt entry {index}")
        existing = master_by_key.setdefault(key, {})
        previous_text = existing.get("text")
        if isinstance(previous_text, str) and previous_text != record["text"]:
            existing.pop("englishText", None)
        existing.update(
            {
                "page": record["page"],
                "boxno": record["boxno"],
                "region": record["region"],
                "sfx": record["sfx"],
                "openLettering": record["openLettering"],
                "text": record["text"],
            }
        )


def merge_translation(
    master_by_key: dict[Key, dict[str, Any]],
    excerpt_records: list[dict[str, Any]],
) -> list[str]:
    validate_excerpt_keys(excerpt_records, "translation excerpt")
    warnings: list[str] = []

    for index, record in enumerate(excerpt_records):
        validate_translation_entry(record, index)
        key = entry_key(record, f"Translation excerpt entry {index}")

        if key not in master_by_key:
            raise InputError(
                f"Translation entry {index} references missing master OCR entry "
                f"for page {key[0]}, boxno {key[1]}."
            )

        master_record = master_by_key[key]
        master_text = master_record.get("text")
        if not isinstance(master_text, str):
            raise InputError(
                f"Master entry for page {key[0]}, boxno {key[1]} must have text as a string."
            )

        if master_text != record["text"]:
            raise InputError(
                f"Translation text mismatch for page {key[0]}, boxno {key[1]}: "
                "master text differs from excerpt text."
            )

        master_record["englishText"] = record["englishText"]

    return warnings


def sorted_records(master_by_key: dict[Key, dict[str, Any]]) -> list[dict[str, Any]]:
    return [master_by_key[key] for key in sorted(master_by_key)]


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644

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
            json.dump(records, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")

        os.chmod(temp_path, output_mode)
        os.replace(temp_path, path)
    except OSError as exc:
        raise InputError(f"Failed to write output JSON {path}: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def check_output_path(master_json: Path, excerpt_json: Path, output_json: Path) -> None:
    output_resolved = output_json.resolve()
    if excerpt_json.exists() and output_resolved == excerpt_json.resolve():
        raise InputError("Output JSON must be different from the excerpt JSON.")


def main() -> int:
    args = parse_args()

    try:
        check_output_path(args.master_json, args.excerpt_json, args.output_json)
        allow_empty_master = args.mode == "ocr"
        master_items = load_json_list(args.master_json, "master", allow_empty_master)
        excerpt_items = load_json_list(args.excerpt_json, "excerpt")

        master_records = validate_object_list(master_items, "Master")
        excerpt_records = validate_object_list(excerpt_items, "Excerpt")
        master_by_key = validate_master(master_records)

        if args.mode == "ocr":
            warnings = []
            merge_ocr(master_by_key, excerpt_records)
        else:
            warnings = merge_translation(master_by_key, excerpt_records)

        for warning in warnings:
            print(warning, file=sys.stderr)

        write_json(args.output_json, sorted_records(master_by_key))
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
