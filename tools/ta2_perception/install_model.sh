#!/bin/bash
set -euo pipefail
PY=/data/ta2/perception/venv/bin/python
# CPU PyTorch avoids downloading CUDA libraries on the Intel server.
"$PY" -m pip install --disable-pip-version-check --timeout 30 --retries 1 \
  --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
"$PY" -m pip install --disable-pip-version-check --timeout 30 --retries 1 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'ultralytics>=8.3,<9' 'openvino>=2024.4' 'numpy==1.26.4' 'opencv-python==4.10.0.84'
# Ultralytics requires opencv-python; restore the matching contrib build for ChArUco.
"$PY" -m pip install --disable-pip-version-check --force-reinstall --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python-headless==4.10.0.84
"$PY" -m pip freeze > /data/ta2/perception/installed-model-requirements.txt
"$PY" - <<'PY'
import cv2, openvino as ov, torch
from ultralytics import YOLO
print('OpenCV',cv2.__version__,'ChArUco',hasattr(cv2.aruco,'CharucoDetector'))
print('Torch',torch.__version__)
core=ov.Core()
for device in core.available_devices:
    print(device,core.get_property(device,'FULL_DEVICE_NAME'))
PY
