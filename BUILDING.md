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
uv run --locked --extra inpaint python translate_cbz.py --help
```

Start the web interface with:

```bash
./run_server.sh
```

Set `TETOLATE_LOCAL_PADDLEOCR=1` when either launcher should synchronize the OCR
extra as well as the smaller default LaMa environment.

The first render downloads the SHA-256-verified `big-lama.pt` model to the Torch
cache. LaMa uses the bundled runner's fixed settings and needs no configuration.

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

## Tests

Run the deterministic regression suite without live OCR or VLM services:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --locked --extra inpaint \
  python -m unittest discover -s tests -v
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
`pyproject.toml`. Before tagging, update `pyproject.toml` and `uv.lock`, add the
release date to `CHANGELOG.md`, and complete the checks above.

The first release is AMD64-only. The `dev` branch does not publish images. Push
the release tag only after its commit is ready:

```bash
git tag v0.3.0
git push origin v0.3.0
```

The workflow uses its repository `GITHUB_TOKEN`; it needs no personal access
token or added repository secret. After the first successful run, open the
`ghcr.io/potatoes1286/tetolate` package settings and set its visibility to
public if GitHub created it as private.

For the first publication, verify all of the following:

- `docker pull ghcr.io/potatoes1286/tetolate:0.3.0` succeeds when signed out of GHCR.
- The versioned `compose.yaml` starts tetolate from a directory that does not contain the repository source.
- `docker compose ps` reports the published image as healthy.
- The container remains published only on `127.0.0.1:8088`.
