# Building tetolate

This document covers source builds, native development environments, tests, and
individual container targets. For the normal published-image installation, see
[README.md](README.md).

## Native Environment

Requirements:

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- ImageMagick 7 with PNG, WebP, and JPEG XL support
- An OpenAI-compatible vision-language model endpoint

Install the locked application, LaMa, and PaddleOCR dependencies:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --locked --extra inpaint --extra ocr
```

Copy the configuration examples and protect the resulting files:

```bash
cp data/config/vlm_config.example.json data/config/vlm_config.json
cp data/config/web_config.example.json data/config/web_config.json
chmod 600 data/config/vlm_config.json data/config/web_config.json
```

Review all endpoints before starting the application. Real configuration files
are ignored by Git; never add API keys to examples, logs, or debug archives.
The web server always loads `vlm_config.json` beside `web_config.json`.

## Running From Source

Translate a CBZ from the command line:

```bash
./run.sh book.cbz
```

The default output is `book_translated/`. Available resume, page regeneration,
language, OCR, and output controls are listed by:

```bash
uv run --locked --extra inpaint --extra ocr python translate_cbz.py --help
```

Start the web interface with:

```bash
./run_server.sh
```

Both launchers install the OCR and LaMa dependencies automatically.

Stopped jobs save Advanced Options automatically. Opening the editor waits for
pending saves; a failed save keeps you on the job page. OCR checks the inference
server's model list before local model setup. When Tetolate runs on the host,
use the OCR container's published host port, not its container-network name.

The first render downloads the SHA-256-verified `big-lama.pt` model to the Torch
cache. LaMa uses the bundled runner's fixed settings and needs no configuration.

## Control API and CLI

`python tetolate_cli.py` controls an existing server; `run.sh` runs a separate local
translation. The CLI has no third-party dependencies. It does not read or edit the
server's job files; it reads only its private local credential.

Local commands need no login. At startup, the server creates a mode-0600
`.cli-token.json` in its jobs directory. The CLI reads it from
`$TETOLATE_DATA_DIR/jobs` or `./data/jobs`; use `--jobs-dir` for a custom path.
Automatic discovery applies only to loopback server addresses and bypasses HTTP
proxies. Do not share this file: it grants full control. Explicit environment or
saved control tokens take priority, including read-only tokens.

The JSON API is under `/api/v1/`. It accepts browser admin sessions or
`Authorization: Bearer <control-token>`. Control tokens apply only to this API;
VLM API keys are separate. Remote access still requires a control token. These
tokens survive restarts until revoked; the server stores their hashes, and remote
CLI login saves the token in a mode-0600 configuration file. Browser login and
read-only token restrictions also remain in effect.

On the category upload page, Advanced Options save automatically, including
parallel workers. Wait for “Advanced options saved” before leaving. Defaults
can be saved before selecting a model; new jobs still need one. Auth tokens
keep their separate test/upload save behavior.

Use `schema --json` for endpoint and settings schemas. Settings use snake_case
names, such as `vlm_model`, `pause_after_ocr`, and `ocr_page_workers`. Partial updates
keep other settings. For example:

```bash
python tetolate_cli.py categories settings default --set vlm_model=YOUR_MODEL
python tetolate_cli.py jobs settings JOB_ID --set thinking_budget_tokens=2048
python tetolate_cli.py jobs submit chapter.cbz --series SERIES_ID --chapter 2
python tetolate_cli.py series notes SERIES_ID --file notes.txt
```

`--json` writes results to stdout and errors to stderr. `logs --follow --json`
writes one JSON event per log chunk, with a cursor for continued reads. `wait`
and `export` accept `--timeout` and `--interval`; `--request-timeout` limits each
HTTP request. A paused job returns an error so an agent can request OCR review or
resume. Export reuses an existing archive; use `--regenerate` to replace it and
apply new title options.

Non-editor web actions are available through the CLI. `jobs submit --images`
keeps file order and converts pages to PNG. `series upload` validates the whole
batch, sorts filenames naturally (2 before 10), and assigns chapter orders from
`--first-chapter`, adding 1 per file. It accepts 1-200 CBZ/ZIP files and the same
`--set`, `--settings-file`, and `--notes-file` options as job submission.

`vlm test` returns the endpoint's model list. It uses `--job`, `--category`, or
the server default; `--endpoint` overrides the address. Supply a key through
`--settings-file` with `vlm_auth_token`. Saved keys are reused only for the same
endpoint. A successful test remembers the key. Select a model with `settings`.

`jobs set-translated-title JOB_ID ""` clears the translation. `translate-title`
translates the job name; `retry-title-translation` uses ComicInfo's title when
present and permits the VLM to decline translation. Its `--manual-title` option
does not call the VLM. `jobs pages` lists pages; `jobs page` downloads one.
Both support `--kind original|final`. `rerun --pages` accepts ranges such as
`0-3,6,8`; `--from ocr_merge` matches the web OCR & Merge pass.

`tokens create NAME` shows a new token once without replacing a saved login.
Local filesystem access needs no password; remote access asks for the admin
password. It supports `--read-only` and
`--password-stdin`. `password` changes the admin password and signs out browser
sessions; control tokens remain valid until revoked. For non-interactive use,
`password --password-stdin` reads current, new, and confirmation passwords as
three lines. Passwords and keys are not command-line arguments.

These actions use `/api/v1/auth/password`, `/vlm/test`, job title and page
routes, and `/categories/{code}/series/{series_id}/upload`. Read-only tokens
cannot change passwords, probe endpoints, or submit work. Page-editor actions
are not exposed through the control API or CLI.

Exit codes: 0 success, 1 server/job error or paused job, 2 invalid arguments or
local files, 3 authentication/permission error, 4 wait/follow timeout, 5 connection
error. Non-interactive remote login requires `--password-stdin`.

## Container Image

Build the complete web application, ImageMagick, LaMa, and PaddleOCR runtime:

```bash
docker build --target runtime -t tetolate-core:local .
```

To build and start the local image with standard PaddleOCR, apply the development
override to the public Compose file:

```bash
docker compose -f compose.yaml -f compose.build.yaml up -d --build
```

The `paddleocr-vl` profile adds the CPU llama.cpp OCR service. The main container
continues to handle PaddleOCR-VL layout detection and result parsing:

```bash
docker compose -f compose.yaml -f compose.build.yaml --profile paddleocr-vl up -d --build
```

The LaMa model and PaddleOCR models are downloaded into the shared persistent cache,
not embedded in either image.

## Editor

The editor has OCR & Merge, Text & Translation, and Typesetting tabs. The middle
tab combines source text, erase settings, and English text without merging the
pipeline steps. Regeneration uses the earliest pending change on each page:
source edits start at erase decisions, erase edits start at translation, and
English-text edits start at typesetting. Notes-only edits do not rerun the pipeline.
Reset and Protect stage apply to all three groups in Text & Translation.
In OCR & Merge, new raw boxes form one-box groups. Saved manual boxes that
have no group also appear in the later tabs. Ctrl/Shift-click boxes (or check
raw rows), then use Group checked. Double-click cycles overlapping boxes.
Deleting a source box hides its linked boxes in later tabs. Saved downstream
edits are kept; resetting the source stage restores them. Rendered pages still
need regeneration.
Typesetting Preview starts with the saved cleaned page when available, shows
ImageMagick text layers, then updates the background with LaMa.
Fonts, wrapping, sizing, outlines, and alignment use the same renderer as
Current. Moving boxes reuses cached pixels; other edits refresh the affected
layer after a short delay. Uncached layers share one request and one ImageMagick
process (up to 256 boxes per batch); cached layers are not rendered again.
New layers appear together when their batch finishes.
Preview refreshes cleaning automatically after clean-box edits, undo/redo, and
reset. Navigation does not wait for LaMa; late results from another page are
ignored. Current prepares an exact view when selected. Last Render shows the
last pipeline output. These views do not save edits or update final pages/exports;
use Save and Regenerate page for those. Press Delete to remove a selected clean box.
Use Add clean box in Typesetting to draw an area for LaMa to clean without
adding text. Move, resize, or delete the box; Preview refreshes automatically.
Save and Regenerate page apply it to final pages
and CBZ exports without OCR or VLM calls. Undo/redo includes clean boxes;
Reset stage removes them on that page. Cleaning can alter artwork in the area.
Regenerate stage rebuilds only the selected tab on this page, keeping
protected edits. Regenerate page and Regenerate all pages run only the steps
required by saved edits. Finish Edits renders the current saved values with
LaMa and ImageMagick, without OCR or VLM calls, and marks them as final output.
It does not translate unfinished text. These actions and export generation
delete old CBZ exports; generate a new archive afterward.

Typesetting also has manual rotation (±180°) and horizontal/vertical skew
(±60°). Set them to zero to remove the effect. Positive rotation is clockwise;
horizontal skew, vertical skew, then rotation are applied about the box center.
Preview and final output use the same transform. Save, protection, and reset
include these fields; regeneration needs only Render. The VLM does not set them.

To compare a pre-batch source snapshot with the batch endpoint:

```bash
python scripts/benchmark_editor_layers.py --baseline-root /path/to/pre-batch-source --rounds 5
```

This uses synthetic pages and in-process HTTP, not live jobs or models. It
measures cold renders, warm cache reads, and a single changed box. It excludes
browser work, network latency, and the editor's debounce delay.
Automatic sizing is the default; use `--sizing fixed` to test explicit font sizes.

Measured with five trials on synthetic 30-box pages (4 CPU limit, ImageMagick
7.1.2-13): automatic sizing took 8.36 → 6.36 s; fixed sizing took 5.63 → 3.60 s.
Warm reads took about 57 → 20 ms. One-box automatic edits took 343 → 359 ms.
Rendered text pixels were identical between the serial and batch versions.

## Tests

Run the deterministic regression suite without live OCR or VLM services:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --locked --extra inpaint --extra ocr \
  python -m unittest discover -s tests -v
```

Run the browser editor and update-check tests with Node.js:

```bash
node tests/test_update_check.cjs
node tests/test_editor_reset.cjs
node tests/test_text_editor.cjs
node tests/test_editor_canvas.cjs
```

Basic repository checks used during development are:

```bash
.venv/bin/python -m py_compile *.py
for file in web_editor/*.js; do node --check "$file"; done
bash -n run.sh run_server.sh docker/entrypoint.sh
docker compose config
docker compose --profile paddleocr-vl config
docker compose -f compose.yaml -f compose.build.yaml config
```

The Docker `runtime` build also executes the regression suite before producing the
final image.

## Release container

Container images are published only when a `v*` tag is pushed. The workflow
rejects a tag unless its numeric version is the same as the version in
`pyproject.toml`. Set `VERSION` in `version.py`, add a dated changelog entry,
then synchronize the release metadata:

```bash
python scripts/release.py
```

The script updates `pyproject.toml`, `uv.lock`, `compose.yaml`, and the current
release references in the documentation. Complete the checks above before tagging.

The first release is AMD64-only. The `dev` branch does not publish images. Push
the release tag only after its commit is ready:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow uses its repository `GITHUB_TOKEN`; it needs no personal access
token or added repository secret. After the first successful run, open the
`ghcr.io/potatoes1286/tetolate` package settings and set its visibility to
public if GitHub created it as private.

For the first publication, verify all of the following:

- `docker pull ghcr.io/potatoes1286/tetolate:1.0.0` succeeds when signed out of GHCR.
- The versioned `compose.yaml` starts tetolate from a directory that does not contain the repository source.
- `docker compose ps` reports the published image as healthy.
- Confirm Compose binds the web UI to `127.0.0.1:8088` for local-only access.
