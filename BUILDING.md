# Building tetolate

This document covers native development environments, tests, and individual container targets. For the pipeline and normal Docker setup, see [README.md](README.md).

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

Standard PaddleOCR runs in the main container and needs no Compose profile:

```bash
docker compose up -d
```

The `paddleocr-vl` profile adds the CPU llama.cpp OCR service. The main container
continues to handle PaddleOCR-VL layout detection and result parsing:

```bash
docker compose --profile paddleocr-vl up -d
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
```

The Docker `runtime` build also executes the regression suite before producing the
final image.
