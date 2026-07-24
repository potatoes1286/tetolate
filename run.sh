#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_path="$repo_dir/data/config/vlm_config.json"
translate_script="$repo_dir/translate_cbz.py"

usage() {
  cat >&2 <<'EOF'
Usage:
  ./run.sh INPUT.cbz [OUTPUT_DIR] [extra translate_cbz.py args...]

Defaults:
  OUTPUT_DIR      ./<input-name>_translated
  UV_CACHE_DIR    /tmp/uv-cache
  TETOLATE_LOCAL_PADDLEOCR=1 installs the optional in-process PaddleOCR dependencies
  config          <repository>/data/config/vlm_config.json

Examples:
  ./run.sh book.cbz
  ./run.sh book.cbz out_dir --overwrite
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

input_cbz=$1
shift

if [[ ! -f "$input_cbz" ]]; then
  echo "error: input CBZ not found: $input_cbz" >&2
  exit 1
fi

if [[ ! -f "$config_path" ]]; then
  echo "error: missing $config_path" >&2
  echo "copy $repo_dir/data/config/vlm_config.example.json to $config_path and set your VLM endpoint/model" >&2
  exit 1
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  output_dir=$1
  shift
else
  input_name=$(basename "$input_cbz")
  input_stem=${input_name%.*}
  output_dir="${input_stem}_translated"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv_args=(--locked --extra inpaint --project "$repo_dir")
if [[ "${TETOLATE_LOCAL_PADDLEOCR:-0}" == "1" ]]; then
  uv_args+=(--extra ocr)
fi

uv run "${uv_args[@]}" python "$translate_script" \
  "$input_cbz" "$output_dir" --config "$config_path" "$@"
