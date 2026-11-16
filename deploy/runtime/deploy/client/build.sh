#!/usr/bin/env bash
# Build openpi-xpu-client (client/Dockerfile) with buildx on this machine.
#   usage: client/build.sh [extra docker buildx build args, e.g. --no-cache]
#   env:   IMAGE (default openpi-xpu-client), TAG (default client/VERSION; also tagged :latest),
#          APT_MIRROR (default http://mirrors.ustc.edu.cn/ubuntu), ROS_APT_MIRROR (default empty: packages.ros.org
#          through the proxy; e.g. http://mirrors.ustc.edu.cn/ros2/ubuntu): build-time only, the image keeps the
#          original sources;
#          OPENPI_DIR (default ../openpi next to this repo): its HEAD commit goes into the image label;
#          proxy: HTTPS_PROXY → https_proxy → HTTP_PROXY → http_proxy, passed as predefined build args (not stored in
#          the image) with network host, so a proxy on 127.0.0.1 is reachable.
#   The FROM images are pinned by digest and must be local or pullable (osrf/ros:humble-desktop, ghcr.io/astral-sh/uv).
#   The va-builder stage downloads the libva / gmmlib / media-driver / libva-utils tarballs from GitHub (proxy needed
#   here) and checks them against the SHA256s in the Dockerfile (also listed in SHA256SUMS).
set -euo pipefail
cd "$(dirname "$0")"
IMAGE="${IMAGE:-openpi-xpu-client}"
TAG="${TAG:-$(cat VERSION)}"
APT_MIRROR="${APT_MIRROR-http://mirrors.ustc.edu.cn/ubuntu}"
ROS_APT_MIRROR="${ROS_APT_MIRROR-}"
OPENPI_DIR="${OPENPI_DIR:-$(cd .. && cd .. && pwd)/openpi}"
OPENPI_COMMIT="$(git --no-optional-locks -C "$OPENPI_DIR" rev-parse HEAD)"
LOCK_SHA256="$(sha256sum requirements.lock | cut -d' ' -f1)"
VAPP_SHA256="$(sha256sum va_vpp/vapp.c | cut -d' ' -f1)"
https_p="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
http_p="${HTTP_PROXY:-${http_proxy:-${https_p}}}"
proxy_args=()
if [ -n "$https_p" ]; then
    for k in HTTP_PROXY http_proxy; do proxy_args+=(--build-arg "$k=$http_p"); done
    for k in HTTPS_PROXY https_proxy; do proxy_args+=(--build-arg "$k=$https_p"); done
    proxy_args+=(--build-arg "NO_PROXY=localhost,127.0.0.1" --build-arg "no_proxy=localhost,127.0.0.1")
    echo "[build] proxy https=$https_p http=$http_p (network host)" >&2
fi
echo "[build] $IMAGE:$TAG openpi=$OPENPI_COMMIT lock=$LOCK_SHA256 vapp=$VAPP_SHA256 apt=${APT_MIRROR:-<orig>} ros=${ROS_APT_MIRROR:-<orig>}" >&2
docker buildx build --load --network host --progress plain \
    "${proxy_args[@]}" \
    --build-arg "APT_MIRROR=$APT_MIRROR" --build-arg "ROS_APT_MIRROR=$ROS_APT_MIRROR" \
    --build-arg "OPENPI_COMMIT=$OPENPI_COMMIT" --build-arg "IMAGE_VERSION=$TAG" --build-arg "LOCK_SHA256=$LOCK_SHA256" \
    --build-arg "VAPP_SHA256=$VAPP_SHA256" \
    -t "$IMAGE:$TAG" "$@" .
docker tag "$IMAGE:$TAG" "$IMAGE:latest"
echo "[build] tagged $IMAGE:$TAG and $IMAGE:latest" >&2
