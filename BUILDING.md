# Installing and building tetolate

Native installation and Docker/Podman are first-party supported on x86-64 Linux.
See [README.md](README.md#install) for both quick starts. This guide covers native
configuration, development, tests, and container build targets.

## Native installation

Install [uv](https://docs.astral.sh/uv/) and ImageMagick 7 with PNG, WebP,
and JPEG XL support. Start your VLM endpoint and, if using PaddleOCR-VL,
its separate inference endpoint. Then run:

```bash
./setup.sh
```

Setup installs a managed Python 3.11 environment and the locked application,
OCR, and LaMa dependencies. It checks ImageMagick formats and text rendering,
asks for endpoint addresses and API keys, lists models, and starts the web UI.
Endpoint checks read `/models`; they do not test image inference or start model
services. Choose `paddle` to use built-in OCR without a separate OCR endpoint.

Missing configuration files are created with mode 0600. Existing files stay
unchanged unless you choose to update the model settings. Other saved settings
are kept. Use `./setup.sh --no-start` to configure and test without starting.

New native configurations use local endpoint addresses. Existing endpoint
addresses are not rewritten. Real configuration files are ignored by Git;
never add keys to examples or logs. The web server loads `vlm_config.json`
beside `web_config.json`.

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
The server command binds to `127.0.0.1` by default, even if an old configuration
has a network listen address. `--listen` is shorthand for `--address 0.0.0.0`:

```bash
./run_server.sh --listen
./run_server.sh --address 192.168.1.10 --port 8090
```

The same address options work with `setup.sh` and `web_app.py`. `--port` overrides
the port in the configuration's `listen` field. Change the default password
before allowing network access. The container command passes `--listen`
explicitly; Compose still publishes its port on the host's loopback address.

Login offers Remember session for 30 days, including server restarts. Leave it
unchecked on shared devices. Logout revokes that session; a password change
revokes all browser sessions. Only token hashes are saved in the private web state.

Queued and stopped jobs save Advanced Options automatically; running jobs cannot
change them. Opening the editor waits for
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

Jobs copy defaults when queued. To update an existing queue, use **Apply saved
settings to queued jobs** in the category Advanced Options, or run
`python tetolate_cli.py categories apply-to-queued CATEGORY`. This replaces
queued-job options but keeps translation notes, chapter order, and queue position.
Running jobs are not changed. You can also edit one queued job's Advanced Options.
Use **Copy from Category Settings** in a job's Advanced Options to replace only that
job's settings with its category defaults. Translation notes stay unchanged.
Saved job keys stay with the same endpoint; changing an endpoint removes its old
job key. Set a new OCR key on each job if required.

Use `schema --json` for endpoint and settings schemas. Settings use snake_case
names, such as `vlm_model`, `pause_after_ocr`, and `ocr_page_workers`. Partial updates
keep other settings. For example:

```bash
python tetolate_cli.py categories settings default --set vlm_model=YOUR_MODEL
python tetolate_cli.py jobs settings JOB_ID --set thinking_budget_tokens=2048
python tetolate_cli.py categories create my-series --series
python tetolate_cli.py jobs submit chapter.cbz --category my-series --chapter 2 --volume 1
python tetolate_cli.py series notes my-series --file notes.txt
python tetolate_cli.py series comic-info-xml my-series --file ComicInfo.xml
python tetolate_cli.py series export my-series --format webp --grouping volumes --output volumes.zip
```

`--json` writes results to stdout and errors to stderr. `logs --follow --json`
writes one JSON event per log chunk, with a cursor for continued reads. `wait`
and `export` accept `--timeout` and `--interval`; `--request-timeout` limits each
HTTP request. A paused job returns an error so an agent can request OCR review or
resume. Export reuses an existing archive; use `--regenerate` to replace it and
apply new title options.

CBZ exports start independently of the translation queue and its pause setting.
Different jobs and formats can export at the same time. Each export uses a copy
of the current pages; reruns discard outdated exports. Duplicate requests for
the same job and format are rejected while it is generating. Job details expose
per-format `exports` state (`generating`, `complete`, `failed`, or `cancelled`)
and `exportsRunning`. Export failures do not fail the translation job.
CLI `export` waits for that format, then downloads it; its timeout does not cancel
server work. Interrupted exports must be generated again after a server restart.

Non-editor web actions are available through the CLI. `jobs submit --images`
keeps file order and converts pages to PNG. `series upload` validates the whole
batch, sorts filenames naturally (2 before 10), and assigns chapter orders from
`--first-chapter`, adding 1 per file. It accepts 1-200 CBZ/ZIP files and the same
`--set`, `--settings-file`, and `--notes-file` options as job submission.
Series are categories created with `--series`; series commands take the category
name. `--volume` is optional, and chapters accept decimal numbers including 0.
`series export` returns a ZIP of CBZ files, grouped by chapter or volume. It uses
completed chapters only and runs outside the translation queue without VLM calls.
`series download` downloads the latest completed collection. Settings and exports
are under `/categories/{code}/series` and `/categories/{code}/exports`.

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
routes, and `/categories/{code}/series/upload`. Read-only tokens
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
Extra line spacing adds pixels between lines (-200 to 200). Zero keeps the
font default; negative values tighten the gap. This manual setting affects only
typesetting and uses the same renderer for Preview and final output.
Manual line breaks overrides the box text without changing its saved translation.
The field shows the current automatic breaks until you edit it. Edits fix the
breaks; a smaller box shrinks the font instead of wrapping again. Clear the field
to restore automatic wrapping. Its underlined label has a hover/focus tooltip.
Unprotect lets reruns replace a saved field; Use
generated removes its override.
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
bash -n setup.sh run.sh run_server.sh docker/entrypoint.sh
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

The script updates `pyproject.toml`, `uv.lock`, and the release examples below.
Compose stays on `latest`. Complete the checks above before tagging.

The first release is AMD64-only. The `dev` branch does not publish images. Push
the release tag only after its commit is ready:

```bash
git tag v2.0.0
git push origin v2.0.0
```

The workflow uses its repository `GITHUB_TOKEN`; it needs no personal access
token or added repository secret. After the first successful run, open the
`ghcr.io/potatoes1286/tetolate` package settings and set its visibility to
public if GitHub created it as private.

For the first publication, verify all of the following:

- `docker pull ghcr.io/potatoes1286/tetolate:2.0.0` succeeds when signed out of GHCR.
- `compose.yaml` starts tetolate from a directory that does not contain the repository source.
- `docker compose ps` reports the published image as healthy.
- Confirm Compose binds the web UI to `127.0.0.1:8088` for local-only access.
