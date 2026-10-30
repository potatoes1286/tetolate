
Use this folder for a local PaddleOCR-VL 1.6 inference service. Native installs
run that service separately and enter its API address in setup. Docker/Podman
can start the bundled CPU llama.cpp service with the `paddleocr-vl` Compose profile.
Built-in PaddleOCR does not need these files.

Download the GGUF model, multimodal projector, and chat template from:

https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF

This folder should contain at least:

- PaddleOCR-VL-1.6-GGUF.gguf
- PaddleOCR-VL-1.6-GGUF-mmproj.gguf
- chat_template.jinja
