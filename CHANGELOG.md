# Changelog

## 0.3.0 - Unreleased

### Added

- Added upload support for ZIP archives that contain comic image pages.

### Changed

- JXL/Webp now parallelizes generation for faster generation
- cbz download now only generates after first request rather than generating them beforehand
- Added model selection to VLM endpoint settings

### Fixed

- Fixed missing function in translate_cbz
- JXL/Webp Quality now rejects fractional values
- Imagemagick checksum added to dockerfile
- Better rejection of oversized images


## 0.2.0 - 2026-08-05

### Added

- Added a five-stage job editor.
- Added API key functionality to API endpoints in webui.
- Added parallel worker controls for PaddleOCR-VL, LaMa, and ImageMagick.


### Changed

- Split the pipeline and web server into smaller modules.
- Process long proofreading jobs in smaller batches.
- Modifications to typesetting and expansion algorithm for better accuracy.
- Modifications to prompts for better accuracy.

### Fixed

- Fixed OCR groups that did not appear in later stages.
- Fixed selected record translation that continued into placement work.

## 0.1.0

- First public experimental release.
