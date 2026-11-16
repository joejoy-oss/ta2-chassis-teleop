"""阶段 1 (c)：验证按 deps/uv.lock 同步出的环境（xpu 组）。用法：/opt/openpi-venv/bin/python verify_deps.py"""

import importlib.metadata as md
import json
import os
import sys

out = {"python": sys.version.split()[0]}
for dist in ("torch", "torchvision", "triton-xpu", "torchcodec", "av", "transformers", "jax", "jaxlib", "flax", "numpy", "lerobot"):
    try:
        out[dist] = md.version(dist)
    except md.PackageNotFoundError:
        out[dist] = None

import torch  # noqa: E402

out["xpu_available"] = torch.xpu.is_available()
out["xpu_device_count"] = torch.xpu.device_count()
out["xpu_name"] = torch.xpu.get_device_name(0) if torch.xpu.is_available() else None

import transformers  # noqa: E402

out["transformers_ok"] = transformers.__version__ == "4.53.2"

import torchcodec  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402,F401

out["torchcodec_import"] = torchcodec.__version__

import av  # noqa: E402

out["av_import"] = av.__version__
out["av_ffmpeg"] = {k: v for k, v in av.library_versions.items() if k in ("libavcodec", "libavformat")}

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: E402

out["lerobot_dataset_import"] = lerobot_dataset.__name__

import jax  # noqa: E402

out["jax_devices"] = [str(d) for d in jax.devices()]
out["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS")

print("VERIFY_DEPS " + json.dumps(out, ensure_ascii=False))
