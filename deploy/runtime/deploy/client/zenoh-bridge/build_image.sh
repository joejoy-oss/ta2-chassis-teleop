#!/usr/bin/env bash
# Build openpi-xpu-client:v0.2.1 (Dockerfile.v0.2.1: v0.2.0 plus the patched zenoh-bridge-ros2dds) with buildx on this
# machine, from a temporary context holding the Dockerfile and the binary.
#   BRIDGE=/path/to/zenoh-bridge-ros2dds-1.9.0-p1 client/zenoh-bridge/build_image.sh
#   env: BRIDGE (required; its sha256 must be the Dockerfile's BRIDGE_SHA256), IMAGE (default openpi-xpu-client),
#        TAG (default v0.2.1). v0.2.0 (sha256:bdf07374...) must be loaded here: FROM names it by that digest.
# Ship it as B2 did v0.2.0 (README.md of client/): docker save | gzip -1, sha256, rsync, docker load, and check that
# the image ID there is the one printed here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${BRIDGE:?set BRIDGE to the patched binary}"
IMAGE="${IMAGE:-openpi-xpu-client}"
TAG="${TAG:-v0.2.1}"
want="$(sed -n 's/^ARG BRIDGE_SHA256=//p' "$HERE/Dockerfile.v0.2.1")"
echo "$want  $BRIDGE" | sha256sum -c
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp "$HERE/Dockerfile.v0.2.1" "$ctx/Dockerfile"
cp "$BRIDGE" "$ctx/zenoh-bridge-ros2dds-1.9.0-p1"
docker buildx build --load --progress plain -t "$IMAGE:$TAG" "$ctx"
docker image inspect "$IMAGE:$TAG" --format 'image {{.Id}} layers {{len .RootFS.Layers}} version {{index .Config.Labels "org.opencontainers.image.version"}} bridge {{index .Config.Labels "zenoh.bridge-ros2dds"}} {{index .Config.Labels "zenoh.bridge-ros2dds.sha256"}}'
