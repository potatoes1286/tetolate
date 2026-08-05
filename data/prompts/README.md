# Prompts

Tetolate reads these files for every VLM request. Changes apply to the next request without restarting the server.

Placeholders use `${name}` syntax. Do not rename or remove a placeholder unless the corresponding call in `translate_cbz.py` is also changed. Missing files and missing placeholder values stop the pipeline with an error.

Set `TETOLATE_PROMPTS_DIR` to load a different prompt directory. The custom directory must contain all `.txt` files found here.
