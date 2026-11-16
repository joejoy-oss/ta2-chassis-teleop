#!/usr/bin/env bash
# 阶段 1 冒烟：安装 uv 0.13.5 + Python 3.11.17 + torch 2.12.1+xpu / torchvision 0.27.1+xpu 到 /opt/venv。
#   用法（容器内，root）：setup_torch.sh [torch_version] [torchvision_version]
#   uv 缓存放 /work/uv-cache（两组容器共享，只下载一次）。
set -euxo pipefail
TORCH_VER="${1:-2.12.1+xpu}"
TV_VER="${2:-0.27.1+xpu}"
UV_VER=0.13.5
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10

if ! command -v uv >/dev/null || [ "$(uv --version | awk '{print $2}')" != "${UV_VER}" ]; then
  mkdir -p /work/dl
  f=/work/dl/uv-${UV_VER}-x86_64-unknown-linux-gnu.tar.gz
  [ -s "$f" ] || curl -fsSL --retry 5 -o "$f" "https://github.com/astral-sh/uv/releases/download/${UV_VER}/uv-x86_64-unknown-linux-gnu.tar.gz"
  curl -fsSL --retry 5 "https://github.com/astral-sh/uv/releases/download/${UV_VER}/uv-x86_64-unknown-linux-gnu.tar.gz.sha256" -o "$f.sha256.remote"
  echo "$(awk '{print $1}' "$f.sha256.remote")  $f" | sha256sum -c -
  tar -xzf "$f" -C /usr/local/bin --strip-components=1 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx
fi
uv --version

uv python install 3.11.17
uv venv --python 3.11.17 /opt/venv
# extra-index（pytorch-xpu）优先于默认 PyPI；torch/torchvision 的 +xpu 本地版本只在 xpu 索引上
uv pip install --python /opt/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/xpu \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" numpy
echo "=== pip list (torch / triton / intel runtime)"
uv pip list --python /opt/venv/bin/python | grep -iE "^(torch|torchvision|triton|pytorch-triton|intel|dpcpp|onemkl|mkl|tbb|umf|tcm|impi|numpy)" || true
du -sh /opt/venv /work/uv-cache
