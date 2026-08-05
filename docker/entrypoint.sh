#!/bin/sh
set -eu

is_rootless_user_namespace() {
    mapped_root=$(awk 'NR == 1 { print $2 }' /proc/self/uid_map 2>/dev/null || true)
    [ -n "$mapped_root" ] && [ "$mapped_root" != "0" ]
}

mkdir -p /data/config /data/jobs /data/cache /data/fonts /data/models

if [ ! -d /data/prompts ]; then
    cp -R /app/data/prompts /data/prompts
    echo "Created /data/prompts with the default VLM prompt templates." >&2
fi

if [ ! -e /data/config/vlm_config.json ]; then
    install -m 0600 /app/data/config/vlm_config.example.json /data/config/vlm_config.json
    echo "Created /data/config/vlm_config.json; review the endpoint and model settings." >&2
fi

if [ ! -e /data/config/web_config.json ]; then
    install -m 0600 /app/data/config/web_config.example.json /data/config/web_config.json
    echo "Created /data/config/web_config.json." >&2
fi

if [ "$(id -u)" = "0" ]; then
    if is_rootless_user_namespace; then
        chown root:root \
            /data/config \
            /data/config/vlm_config.json \
            /data/config/web_config.json \
            /data/jobs \
            /data/cache \
            /data/prompts
        chmod 0750 /data/config /data/jobs /data/cache /data/prompts
        chmod 0600 /data/config/vlm_config.json /data/config/web_config.json
        exec "$@"
    fi
    chown -R tetolate:tetolate /data/prompts
    chown tetolate:tetolate /data /data/config /data/jobs /data/cache
    chown tetolate:tetolate /data/config/vlm_config.json /data/config/web_config.json
    exec gosu tetolate:tetolate "$@"
fi

exec "$@"
