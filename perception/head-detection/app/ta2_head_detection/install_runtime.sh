#!/usr/bin/env bash
set -euo pipefail

root=/data/ta2/head-detection
mkdir -p "$root/models"
python3 -m venv "$root/venv-cpu"
python="$root/venv-cpu/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"$python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0' 'torchvision==0.25.0'
"$python" -m pip install --index-url https://pypi.org/simple 'ultralytics==8.4.92' 'opencv-python==4.13.0.92'
"$python" -m pip check
"$python" -m pip freeze > "$root/requirements-cpu.lock"

weights="$root/models/yolo26n.pt"
checksum=9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
check_weights() { printf '%s  %s\n' "$checksum" "$1" | sha256sum --check --status; }
if ! check_weights "$weights"; then
  for url in \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
    https://hf.co/Ultralytics/YOLO26/resolve/main/yolo26n.pt; do
    if curl --fail --location --silent --show-error --connect-timeout 8 --max-time 90 \
        --output "$weights.download" "$url" && check_weights "$weights.download"; then
      mv "$weights.download" "$weights"
      break
    fi
  done
fi
check_weights "$weights"
