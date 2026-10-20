"""Shared ComicInfo metadata for category-series exports."""
from __future__ import annotations

import copy
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from fastapi import HTTPException

import translate_cbz


def parse_common_xml(value: str) -> ET.Element:
    if len(value) > 100_000:
        raise HTTPException(400, "Common ComicInfo XML must be at most 100000 characters.")
    if not value.strip():
        return ET.Element("ComicInfo")
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", value, re.I):
        raise HTTPException(400, "ComicInfo XML cannot contain document types or entities.")
    try:
        root = ET.fromstring(value)
    except ET.ParseError as exc:
        raise HTTPException(400, f"Invalid ComicInfo XML: {exc}") from None
    if root.tag != "ComicInfo":
        raise HTTPException(400, "Common XML must have a ComicInfo root element.")
    return root


def merged_xml(common: str, original: bytes | None, *, series: str, volume: str,
               chapter: str, title: str, page_count: int) -> bytes:
    root = ET.Element("ComicInfo")
    if original and not re.search(br"<!\s*(DOCTYPE|ENTITY)", original, re.I):
        try:
            parsed = ET.fromstring(original)
            if parsed.tag.rsplit("}", 1)[-1] == "ComicInfo":
                root = parsed
        except ET.ParseError:
            pass
    def set_element(element):
        key = element.tag.rsplit("}", 1)[-1]
        for existing in list(root):
            if existing.tag.rsplit("}", 1)[-1] == key:
                root.remove(existing)
        root.append(copy.deepcopy(element))
    common_root = parse_common_xml(common)
    for element in common_root:
        set_element(element)
    common_series = next((e.text for e in common_root if e.tag == "Series"), "")
    for name, value in (("Series", common_series or series), ("Volume", volume),
                        ("Number", chapter), ("Title", title), ("PageCount", str(page_count))):
        element = ET.Element(name)
        element.text = value
        set_element(element)
    for element in list(root):
        if element.tag.rsplit("}", 1)[-1] == "Pages":
            root.remove(element)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def metadata_archive(source: Path | None, destination: Path, common: str, **fields) -> None:
    """Keep non-image source files and emit exactly one merged ComicInfo.xml."""
    original = None
    members = []
    if source is not None:
        with zipfile.ZipFile(source) as archive:
            for info in translate_cbz.validate_cbz_members(archive):
                if not translate_cbz.is_preserved_non_image_member(info):
                    continue
                if info.filename.replace("\\", "/").rsplit("/", 1)[-1].casefold() == "comicinfo.xml":
                    original = archive.read(info)
                else:
                    members.append((info, archive.read(info)))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for info, data in members:
            archive.writestr(info, data)
        archive.writestr("ComicInfo.xml", merged_xml(common, original, **fields))
