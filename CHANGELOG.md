# Changelog

## 2.0.0 - 2026-10-30

### Added

- First-party native installation with guided setup and documentation.
- CLI and JSON API access. Local CLI needs no login; remote access uses revocable tokens.
- Job editor overhaul; far more functional and easy to use
- Remember session login and manual release checks.
- Added Series, special categories with chapter ordering, shared notes, ComicInfo templates, and chapter/volume collection exports.

### Changed

- Native servers bind to 127.0.0.1 by default; use --listen for network access or --address for a specific address.
- Editor actions can regenerate selected work or finish edits without VLM calls.
- Improved preview speed with cached text layers, batched rendering, and fewer LaMa calls.
- CBZ exports run independently of the translation queue, with PNG, WebP, and JXL output.

### Fixed

- Improved editor saves, resets, OCR box tracking, and text rendering.
- Improved settings persistence, endpoint key handling, and model connection errors.
- Improved job and series queue safety, export invalidation, and failed-edit recovery.
- Fixed release checks, container test inputs, and CLI export timeouts.
- Lots of bug fixes

## 1.0.0 - 2026-09-04

Devnote:

Big update. At this point tetolate feels almost entirely feature complete to me. Save for incorporating new models, fixing edge bugs, I don't think there's much more to add.

If you have any ideas, let me know via the issues.

TL;DR lots of housecleaning + no more image building needed anymore, now pushed via github and you can just run a docker compose.

### Added

- Added GHCR images, public Compose installation, local builds, and bundled fonts.
- Added ZIP uploads, lazy CBZ downloads, and ComicInfo title controls.
- Added category and job naming controls.
- Added translation of titles and editing titles.

### Changed

- Improved VLM model and job setting persistence.
- Installation now uses the published GHCR image.
- Minor UI changes.

### Fixed

- Improved VLM connection errors and missing-model handling.
- Improved validation for image and archive inputs.
- First startup now uses the documented default admin password 'changeme'.


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
