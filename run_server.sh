#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_path="${TETOLATE_WEB_CONFIG:-$repo_dir/data/config/web_config.json}"
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
  ./run_server.sh --listen
  ./run_server.sh --address 127.0.0.1 --port 8090
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

next_config=false
for argument in "$@"; do
  if $next_config; then
    config_path="$argument"
    next_config=false
  elif [[ "$argument" == "--config" ]]; then
    next_config=true
  elif [[ "$argument" == --config=* ]]; then
    config_path="${argument#--config=}"
  fi
done
if $next_config; then
  echo "error: --config needs a path" >&2
  exit 2
fi

if [[ ! -f "$config_path" ]]; then
  echo "error: missing $config_path" >&2
  echo "run $repo_dir/setup.sh to create and test the native configuration" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv_args=(--locked --extra inpaint --extra ocr --project "$repo_dir")

uv run "${uv_args[@]}" \
  python "$web_script" --config "$config_path" "$@"
