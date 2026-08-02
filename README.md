# tetolate

This is a program for locally and automatically translating comic archives (`.cbz`) and sets of images from Japanese/Korean/Chinese to English using an AI pipeline consisting of OCR, Vision-Large Language Model (VLM), and cleaning.

It is intended to run on local machines. The software itself is relatively light and should run on any semi-decent computer, the VLM is the bulk of the compute requirements.

For a VLM, I would recommend at minimum Gemma 4 12B, and I personally run with Gemma 4 31B for high-quality translations and Gemma 4 26B-A4B for faster medium-quality translations. I would not recommend running any of the Qwen 3.x models as they produce sub-par translations compared to Gemma 4.

THIS IS NOT REAL TIME TRANSLATION SOFTWARE. With a 3090, Gemma 4 31B, I see about 5-10 min PER PAGE. This is a send, come back later, and pick up.

## Features

Translates a .cbz or a collection of images to an English .cbz.

Web UI to upload, view jobs, and download from.

png, webp, jxl cbz output options, or view in browser.

VERY experimental editing of jobs.

## Pipeline

1. ocr_raw -- initial OCR detection via PaddleOCR or PaddleOCR-VL
2. ocr_structured -- merge detected OCR into text boxes, reject false-positives, order the reading order via VLM
3. alt_placement -- optional, detect open lettering text and classify alternative placements for translation via VLM
4. translations -- translate page by page via VLM
5. proofread -- optional, VLM does one final pass, proofreads
6. translation_notes -- optional, VLM assembles translation notes to carry to future related translations
7. placement -- determine text box sizes
8. render -- clean up text via LaMa, draw over text with ImageMagick
9. package -- generate PNG, WebP, and JXL CBZ downloads.

## Requirements

- Docker or Podman
- An OpenAI-compatible vision-language model endpoint -- If running the VLM locally (which is expected), at least 8 GB of free VRAM to run Gemma 4 12B, or for the truly desperate, offload that to RAM. By default tetolate assumes this is on `127.0.0.1:8080/v1`; change `vlm_config.json` if different.
- If using PaddleOCR-VL 1.6 (recommended) 1+ gb of free RAM
- font files for text -- recommended ones are listed in [the font config's readme](data/fonts/README.md), but it is up to you to obtain them or get alternatives from google fonts or wherever

Tested on x86-64 Linux with Docker and Podman. ARM, Windows, and macOS are not currently tested.

## Install

Installing with self-built docker is recommended.

For detailed building / native install see [BUILDING.md](BUILDING.md).

1. Go start up a llama.cpp instance running the vision-language model.

2. The main image includes standard PaddleOCR and the PaddleOCR-VL client. For OCR, you may choose to either:

- Run standard PaddleOCR directly in the main container (not recommended, faster, frequently misses text in my tests)
- Run PaddleOCR-VL 1.6 in the bundled CPU llama.cpp container (recommended, slower, FAR higher quality)

3. Clone the repository, then prepare the configuration before first startup:

```bash
cp data/config/vlm_config.example.json data/config/vlm_config.json
cp data/config/web_config.example.json data/config/web_config.json
```

4. Set the translation endpoint in `data/config/vlm_config.json` -- please note for docker, 127.0.0.1 does not usually work and to access ports on the host machine you will have to use `host.docker.internal`.

5. Put your font files into `data/fonts` and make a `font_use.txt` file from the example and readme. A suggested list of fonts are provided in `data/fonts/README.md`, though it is up to you to get them.

6. For PaddleOCR-VL 1.6, download the two GGUF files and chat template listed in `data/models/README.md` into `data/models/`, then run. The main container performs layout detection and the profile adds only the CPU llama.cpp endpoint:
```bash
docker compose --profile paddleocr-vl up -d
```

For standard PaddleOCR:
```bash
docker compose up -d
```

7. When you submit a job to translate, select the matching OCR engine in a job's advanced options.

8. On first startup there will be an auto-generated password in the docker logs. If you already detached, use `docker compose logs tetolate | grep password`. You can change it after logging in the webui.

The container only listens to localhost on `127.0.0.1`. To listen to the network, change `127.0.0.1` to `0.0.0.0` in `compose.yaml`. Exposing to WAN is VERY MUCH not advised.

Persistent data is stored in:

- `data/config/`: private runtime configuration
- `data/fonts/`: user-provided font files and font-use guidance
- `data/models/`: local PaddleOCR-VL model files
- `data/jobs/`: source files, intermediate stages, logs, and results
- `data/cache/`: PaddleOCR, Hugging Face, Torch, and LaMa model caches

The first OCR and rendering job downloads model data and therefore takes longer.
Stopping the containers does not delete jobs or caches.

## Usage

- Log in with admin password.
- Change the password from the auto-generated one once inside.
- Create a new category (no relation between jobs in the same category, they are just there to organize).
- Choose a cbz or list of images.
- Under advanced options, put in your translation notes, enable optional runs, source language, ocr engine, paddleocr server & model (should be auto filled), max thinking tokens, then submit job.
- Wait. (models will be downloaded on your first run, be warned!)
- Download your cbz archive or view in browser.

## Extra

Third-party software and models keep their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For legal reasons, be aware that copyright exists

I vibed this with mr. codex for my own personal use. Uploaded to share. This software is experimental. Do not expose it to WAN, etc.
