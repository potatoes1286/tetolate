# Changelog

## Unreleased

### Added

- Added category-series ComicInfo templates and ZIP collection exports by chapter or volume, in PNG, WebP, or JXL.
- Added Advanced Options editing for queued jobs and a category action to apply saved settings to all queued jobs.
- Added Copy to Category Settings and Copy from Category Settings buttons to each job's Advanced Options.
- Added direct page-number entry in the editor.
- Added a Hide boxes toggle to Typesetting Preview.
- Added line-spacing control to Typesetting.
- Added a 30-day Remember session login option that survives server restarts.
- Added a manual GitHub release check beside the web UI version.
- Added an editor stage reset to remove saved edits and protection flags on one page.
- Added manual rotation and skew controls with sliders and keyboard-friendly inputs.
- Added clean-only boxes in Typesetting, with preview and export support without a VLM model.

### Changed

- Removed job-level series-note buttons; volume and chapter can be edited after generation.
- Series are now categories, with optional volume and decimal chapter numbers. Removed separate series records and selectors.
- Series API and CLI actions now use category names instead of series IDs.
- CBZ exports run independently of the translation queue, with separate progress and errors.
- Manual line breaks follow automatic wrapping until edited, then keep the chosen breaks.
- Editor actions now regenerate one stage, one changed page, all changed pages, or finish edits without VLM calls; regeneration and export generation remove old CBZ exports.
- Preview reuses saved cleaned pages, shows text before LaMa finishes, and refreshes clean-box edits without blocking navigation.
- Renamed Render to Last Render, removed duplicate clean-box controls, and moved status below the editor actions.
- Local CLI access no longer needs login; remote access still requires a token.
- Combined the Structure, Erase, and Translation editor tabs, with reruns based on changed fields.
- Typesetting Preview batches uncached layers in one ImageMagick process, keeping per-box caches.
- Simplified editor alignment labels and box-coordinate inputs.
- Grouped password and remote CLI controls under Login settings.
- Clarified category-setting copy buttons, removed their per-job confirmation, and simplified the login page.

### Fixed

- Outline width 0 disables the outline in editor previews and final renders.
- Category renames wait for starting jobs and CBZ exports, preventing orphaned jobs and stuck exports.
- Chapter and volume numbers cannot expand beyond 20 decimal characters.
- CLI collection exports apply one timeout to generation requests, polling, and downloads; timed-out downloads preserve existing files.
- Job Advanced Options save automatically before opening the editor, including OCR endpoints and API keys.
- OCR checks its inference endpoint before model setup and reports readable connection, authentication, and model errors.
- Source launchers include required OCR dependencies by default.
- Clearing the editor font size restores automatic sizing and stays cleared after saving.
- Typesetting saves preserve pending OCR and translation changes.
- Manual OCR additions reach later editor tabs; canvas multi-selection, grouping, and overlap selection work together.
- Improved automatic font sizing for wide, single-line captions without enlarging multi-line text.
- Category Advanced Options save automatically and survive reloads, including parallel workers.
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
