#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_path="$repo_dir/data/config/web_config.json"
web_script="$repo_dir/web_app.py"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run_server.sh [extra web_app.py args...]

Defaults:
  UV_CACHE_DIR    /tmp/uv-cache
  config          <repository>/data/config/web_config.json

Examples:
  ./run_server.sh
  ./run_server.sh --host 127.0.0.1 --port 8090
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$config_path" ]]; then
  echo "error: missing $config_path" >&2
  echo "copy $repo_dir/data/config/web_config.example.json to $config_path and review the server settings" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv_args=(--locked --extra inpaint --project "$repo_dir")
if [[ "${TETOLATE_LOCAL_PADDLEOCR:-0}" == "1" ]]; then
  uv_args+=(--extra ocr)
fi

uv run "${uv_args[@]}" \
  python "$web_script" --config "$config_path" "$@"
