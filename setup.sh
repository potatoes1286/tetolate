#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./setup.sh [--no-start] [--listen | --address ADDRESS] [--port PORT] [--config PATH]"
  echo "Checks native requirements, installs Python dependencies, tests model endpoints, and starts Tetolate."
  exit 0
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "error: install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if ! command -v magick >/dev/null 2>&1; then
  echo "error: install ImageMagick 7 with PNG, WebP, and JPEG XL support first." >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
uv python install 3.11
uv sync --locked --managed-python --python 3.11 --extra inpaint --extra ocr --project "$repo_dir"
exec uv run --no-sync --project "$repo_dir" python "$repo_dir/native_setup.py" "$@"
