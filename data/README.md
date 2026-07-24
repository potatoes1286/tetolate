# Runtime Data

This directory is the persistent, user-managed tetolate state:

- `config/`: copy and edit the example configuration files before startup.
- `fonts/`: optional user-provided fonts and `font_use.txt`.
- `models/`: PaddleOCR-VL model files used by the `paddleocr-vl` profile.
- `jobs/`: uploaded inputs, intermediate stages, logs, and output archives.
- `cache/`: downloaded PaddleOCR and LaMa model caches.

Only documentation, examples, and directory placeholders are committed. Runtime
configuration, fonts, models, jobs, and caches remain local.
