#!/usr/bin/env bash
# 构建 openpi-xpu-env 镜像（用 docker compose build 驱动 BuildKit；B70 上没有 buildx 插件）。
#   用法: scripts/build.sh [额外 docker compose build 参数，如 --no-cache]
#   环境变量:
#     IMAGE       镜像名        (默认 openpi-xpu-env)
#     TAG         镜像 tag      (默认 VERSION 文件内容；同时打 :latest)
#     APT_MIRROR  构建期替换 Ubuntu 源（如 http://mirrors.ustc.edu.cn/ubuntu），构建结束前恢复，不留在镜像里
#     代理        依次识别 HTTPS_PROXY → https_proxy → HTTP_PROXY → http_proxy，任一非空即以 build-arg 传入；
#                 compose 文件固定 network: host，127.0.0.1 上的代理（如 SSH 反向隧道）才可达。
#                 NO_PROXY/no_proxy 取环境值，默认 localhost,127.0.0.1。代理 build-arg 不写进镜像 Env/history。
#   构建前要求本地已有 FROM 里按 digest 钉住的两个镜像（ubuntu:24.04、ghcr.io/astral-sh/uv:0.13.5），
#   以及 deps/umd-host/ 下与 SHA256SUMS 一致的 .deb（见 README）。
set -euo pipefail
cd "$(dirname "$0")/.."

export IMAGE="${IMAGE:-openpi-xpu-env}"
export TAG="${TAG:-$(cat VERSION)}"
export OPENPI_COMMIT="$(cat deps/OPENPI_COMMIT)"
export APT_MIRROR="${APT_MIRROR:-}"

https_p="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
http_p="${HTTP_PROXY:-${http_proxy:-${https_p}}}"
export NO_PROXY="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}" no_proxy="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}"
if [ -n "${https_p}" ]; then
    export HTTP_PROXY="${http_p}" HTTPS_PROXY="${https_p}" http_proxy="${http_p}" https_proxy="${https_p}"
    echo "[build] using proxy https=${https_p} http=${http_p} no_proxy=${NO_PROXY} (network: host)" >&2
else
    echo "[build] no proxy configured" >&2
fi

(cd deps/umd-host && sha256sum -c SHA256SUMS) >&2
echo "[build] ${IMAGE}:${TAG} (openpi commit ${OPENPI_COMMIT}, apt mirror ${APT_MIRROR:-<none>})" >&2
docker compose -f docker-compose.yml build --progress=plain "$@"
docker tag "${IMAGE}:${TAG}" "${IMAGE}:latest"
echo "[build] tagged ${IMAGE}:${TAG} and ${IMAGE}:latest" >&2
