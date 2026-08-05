# Configuration

Before the first run, create the active configuration:

```bash
cp data/config/vlm_config.example.json data/config/vlm_config.json
cp data/config/web_config.example.json data/config/web_config.json
```

Set the translation VLM endpoint and API key in `vlm_config.json`. The web
server always loads `vlm_config.json` from this directory.

The web advanced options can override the translation VLM token and the
PaddleOCR-VL token for one job. Tetolate stores these overrides in a private
job secrets file and does not put them in job status responses or command logs.

The Docker entrypoint creates the active files from these examples when they
are absent, but preparing them first avoids having to stop after initial startup.
