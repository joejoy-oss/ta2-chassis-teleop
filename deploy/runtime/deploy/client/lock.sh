#!/usr/bin/env bash
# Regenerate client/requirements.lock from client/requirements.in. Resolves inside the client's base image with its own
# /usr/bin/python3 (3.10), so markers and wheels match the deploy interpreter; uv is the binary from the pinned uv image.
#   usage: client/lock.sh            (proxy: HTTPS_PROXY / https_proxy, if set, is passed through; network: host)
set -euo pipefail
cd "$(dirname "$0")"
BASE_IMAGE="$(sed -n 's/^ARG BASE_IMAGE=//p' Dockerfile)"
UV_IMAGE="ghcr.io/astral-sh/uv@$(sed -n 's/^ARG UV_IMAGE_DIGEST=//p' Dockerfile)"
proxy="${HTTPS_PROXY:-${https_proxy:-}}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cid="$(docker create "$UV_IMAGE")"
docker cp "$cid:/uv" "$tmp/uv" >/dev/null
docker rm "$cid" >/dev/null
docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -m 4g --memory-swap 4g \
    -e "https_proxy=$proxy" -e "http_proxy=$proxy" -e "HTTPS_PROXY=$proxy" -e "HTTP_PROXY=$proxy" \
    -v "$tmp/uv:/usr/local/bin/uv:ro" -v "$PWD:/work" -w /work --entrypoint /usr/local/bin/uv \
    "$BASE_IMAGE" pip compile --python /usr/bin/python3 --generate-hashes --no-cache \
    --custom-compile-command "client/lock.sh" requirements.in -o requirements.lock
