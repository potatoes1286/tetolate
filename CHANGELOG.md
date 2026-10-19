# Changelog

## Unreleased

### Added

- Added a manual GitHub release check beside the web UI version.
- Added an editor stage reset to remove saved edits and protection flags on one page.

### Changed

- Local CLI access no longer needs login; remote access still requires a token.
- Combined the Structure, Erase, and Translation editor tabs, with reruns based on changed fields.
- Typesetting Preview batches uncached layers in one ImageMagick process, keeping per-box caches.
- Simplified editor alignment labels and box-coordinate inputs.
- Grouped password and remote CLI controls under Login settings.

### Fixed

- Resetting editor stages clears discarded pending edits and their out-of-date flags.
- Deleted source boxes stay hidden in later editor tabs; reset restores their saved data.

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
- Added series with ordered chapter uploads and automatic translation-note sharing.
- Added CLI and JSON API access to non-editor web actions, with revocable tokens.

### Changed

- Improved VLM model and job setting persistence.
- Installation now uses the published GHCR image.
- Minor UI changes.

### Fixed

- Remember VLM API keys per endpoint across sessions and new jobs.
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
