# tetolate

[Changelog](CHANGELOG.md)

tetolate is a self-hosted program to run a total end-to-end translation of mangas / comic archives. Which means, manga in (japanese/korean/chinese), manga out english.

It runs entirely on your computer. It is up to you (end user) to run the vision-language model (VLM) for tetolate to use.

Recommended models:
- Gemma 4 12B -- Good, on par with 26B-A4B
- Gemma 4 26B-A4B -- Good, on par with 12B
- Gemma 4 31B -- Best

NOT recommended models:
- Qwen 3.x -- Translation is too stiff and less accurate.

Please note that tetolate is designed for max quality over speed. This is NOT a real time translation software. With 40 tps on Gemma 4 31B, I see ~1 min/page for low text mangas an ~2m30/page for high text mangas.

Example images coming soon.

## Features

Accepts `cbz`, `zip` full of images, or a direct collection of images.

View translated pages in the browser. Generate an optional PNG, WebP, or JXL CBZ download when needed.

Web UI + password to upload, view jobs, and download.

Editor to fix minor errors, primarily OCR misses.

## Requirements

- Docker or Podman
- An OpenAI-compatible vision-language model endpoint -- If running the VLM locally (which is expected), at least 8 GB of free VRAM to run Gemma 4 12B, or for the truly desperate, offload that to RAM. By default tetolate assumes this is on `127.0.0.1:8080/v1`; change `vlm_config.json` if different.
- If using PaddleOCR-VL 1.6 (recommended) 10+ gb of free RAM
- Optional professional comic fonts. Tetolate includes open stand-in fonts when no user fonts are installed.

Tested on x86-64 Linux with Docker and Podman. ARM, Windows, and macOS are not currently tested.

## Install

Start an OpenAI-compatible vision-language model endpoint, then download the
versioned Compose file and start tetolate:

```bash
curl -fsSLO https://raw.githubusercontent.com/potatoes1286/tetolate/v0.3.0/compose.yaml
docker compose up -d
docker compose logs tetolate
```

On first startup, tetolate creates the `data` directories, configuration files,
default prompts, and an admin password automatically. Open
`http://127.0.0.1:8088/admin` and use the password shown in the logs. Configure
and test the translation VLM endpoint in the web UI, then select one of the
models returned by the endpoint. For a VLM that runs on the Docker host, use
`host.docker.internal` instead of `127.0.0.1` in its endpoint URL.

Tetolate includes open stand-in fonts for normal dialogue, thoughts, computer
text, handwriting, and sound effects. To replace them with professional comic
fonts, put your files in `data/fonts` and create `font_use.txt` as described in
[the font configuration guide](https://github.com/potatoes1286/tetolate/blob/v0.3.0/data/fonts/README.md).

The main image includes standard PaddleOCR and the PaddleOCR-VL client. Standard
PaddleOCR needs no extra service. To use the recommended PaddleOCR-VL 1.6
option, download the two GGUF files and chat template listed in the
[model guide](https://github.com/potatoes1286/tetolate/blob/v0.3.0/data/models/README.md)
into `data/models`, then start the optional CPU llama.cpp service:

```bash
docker compose --profile paddleocr-vl up -d
```

Select the matching OCR engine in a job's advanced options. The first OCR or
rendering job downloads its model data and takes longer.

The container only listens to localhost on `127.0.0.1`. To listen to the network, change `127.0.0.1` to `0.0.0.0` in `compose.yaml`. Exposing to WAN is VERY MUCH not advised.

Persistent data is stored in:

- `data/config/`: private runtime configuration
- `data/fonts/`: optional user font files and font-use guidance
- `data/models/`: local PaddleOCR-VL model files
- `data/prompts/`: editable VLM prompt templates
- `data/jobs/`: source files, intermediate stages, logs, and results
- `data/cache/`: PaddleOCR, Hugging Face, Torch, and LaMa model caches

Stopping the containers does not delete jobs or caches.

For source builds and native development, see [BUILDING.md](BUILDING.md).

## Upgrade

Download the Compose file for the new version, pull its image, and recreate the
container. Replace `v0.3.1` with the version that you want to install:

```bash
curl -fsSL https://raw.githubusercontent.com/potatoes1286/tetolate/v0.3.1/compose.yaml -o compose.yaml
docker compose pull
docker compose up -d
```

## Rollback

Download an older versioned Compose file, then pull and recreate the container.
For example, to return to `v0.3.0`:

```bash
curl -fsSL https://raw.githubusercontent.com/potatoes1286/tetolate/v0.3.0/compose.yaml -o compose.yaml
docker compose pull
docker compose up -d
```

The `data` directory is retained. Before a future release that changes stored
data formats, follow that release's backup and rollback notes.

## Usage

- Log in with admin password.
- Change the password from the auto-generated one once inside.
- Create a new category (no relation between jobs in the same category, they are just there to organize).
- Choose a CBZ archive, a ZIP archive that contains image pages, or a list of images.
- Under advanced options, test the translation VLM endpoint, select a returned model, then set translation notes, optional passes, source language, OCR engine, thinking tokens, and page worker counts.
- Wait. (models will be downloaded on your first run, be warned!)
- Download your cbz archive or view in browser.

VLM prompts are plain text files in `data/prompts/`. They are read for each request, so edits apply to the next pass without restarting. See `data/prompts/README.md` for placeholder and custom-directory details.

NOTE: For Gemma 4, the default tokens dedicated per image is relatively low which can cause poor performance. To max tokens per image, if running via llama.cpp add `--image-min-tokens 1120`, or with ExLlamaV3 go to your model's `processor_config.json` and set

```json
"image_processor": {
    ...
    "max_soft_tokens": 1120,
    ...
}
```

## Pipeline

1. ocr_raw -- detect text with PaddleOCR or PaddleOCR-VL
2. ocr_merged -- merge nearby OCR detections with deterministic geometry
3. ocr_structured -- reject false positives, classify text, and set reading order with the VLM
4. alt_placement -- optionally classify text that must not be erased in place
5. translations -- translate each page with the VLM
6. proofreading -- optionally proofread translations in bounded batches
7. translation_notes -- optionally create reusable notes for related translations
8. placements -- determine text regions and styles
9. render -- clean text with LaMa and draw translated text with ImageMagick
10. package -- generate PNG, WebP, and JXL CBZ downloads

## Extra

Third-party software and models keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For legal reasons, be aware that copyright exists

I vibed this with mr. codex for my own personal use. Uploaded to share. This software is experimental. Do not expose it to WAN, etc.
