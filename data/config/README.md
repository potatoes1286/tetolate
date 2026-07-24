# Configuration

Before the first run, create the active configuration:

```bash
cp data/config/vlm_config.example.json data/config/vlm_config.json
cp data/config/web_config.example.json data/config/web_config.json
```

Set the translation VLM endpoint and API key in `vlm_config.json`. The web
server always loads `vlm_config.json` from this directory.

The Docker entrypoint creates the active files from these examples when they
are absent, but preparing them first avoids having to stop after initial startup.
