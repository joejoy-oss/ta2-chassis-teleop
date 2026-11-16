"""Benchmark a served policy end to end on one machine.

Runs the same `WebsocketPolicyServer` that `serve_policy.py` runs, in a thread of this process, and drives it with
`WebsocketClientPolicy` over loopback. Each request is timed three ways: the client round trip, the server's
`infer_ms`, and the policy's own stages (`policy_timing`). The client round trip includes (de)serialization and the
websocket, but no network, so it is a lower bound on what a robot client measures.

One server configuration (checkpoint x compile mode x RTC mode) per process, so compilation and memory start from
scratch. Steps: build the policy, run `serve_policy.warmup` (timed: with empty compile caches this is the first
compilation), then for each observation format send `--warmup-requests` untimed and `--requests` timed requests. With
RTC enabled, every request after the very first continues the previous chunk with fixed prefix_start, delay and
horizon, so all timed requests take the constrained path.

Observation formats (TeleAvatar v2): "raw" is side-by-side stereo frames at the resolution of the training videos,
head 1920x3840 and wrists 800x2560, like `make_teleavatar_example`; `TeleavatarInputs` crops one eye out of each.
"dataset" (a misnomer kept for the existing results) is single eyes, head 960x960 and wrists 400x640: what the v2
robot client (`examples/teleavatar_v2`) sends, its RTP interface having already split the composite camera frame into
single-eye views. "model" is 224x224, the model's input size. Each is generated once with a fixed seed and resent
unchanged.

Example (Intel GPU, guided RTC, compiled):

    python scripts/benchmark_serving.py --out /tmp/guided_default.json \\
        --policy.config pi05_teleavatar_v2 --policy.dir /ckpt/pi05_teleavatar_v2_pytorch \\
        --policy.pytorch-device xpu --policy.pytorch-compile-mode default --rtc.enabled --rtc.mode GUIDED
"""

import dataclasses
import json
import logging
import os
import pathlib
import platform
import threading
import time
import traceback

import numpy as np
from openpi_client import websocket_client_policy
import tyro

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server

try:  # `python -m scripts.benchmark_serving` or imported by tests
    from . import serve_policy
except ImportError:  # `python scripts/benchmark_serving.py`
    import serve_policy  # type: ignore[no-redef]

# Image sizes (height, width) of the observation formats, see the module docstring.
FORMATS: dict[str, dict[str, tuple[int, int]]] = {
    "raw": {"head": (1920, 3840), "wrist": (800, 2560)},
    "dataset": {"head": (960, 960), "wrist": (400, 640)},
    "model": {"head": (224, 224), "wrist": (224, 224)},
}


@dataclasses.dataclass
class Args:
    # Output JSON file.
    out: pathlib.Path
    policy: serve_policy.Checkpoint
    rtc: serve_policy.RTCArgs = dataclasses.field(default_factory=serve_policy.RTCArgs)
    # Observation formats to time, in order (see the module docstring).
    formats: tuple[str, ...] = ("raw", "dataset", "model")
    # Untimed requests before the timed ones, per format.
    warmup_requests: int = 10
    # Timed requests per format.
    requests: int = 100
    # If > 0, send this many timed requests on the first format only (a stability run), instead of the per-format runs.
    stability: int = 0
    # Skip serve_policy.warmup, so that the first client request triggers compilation (keepalive check).
    skip_warmup: bool = False
    # RTC fields of every request after the first (steady state of the docs example: delay 9, horizon 18).
    prefix_start: int = 18
    inference_delay: int = 9
    execution_horizon: int = 18
    prompt: str = "Insert all the colored shape blocks into their matching slots on the puzzle board."
    seed: int = 0
    port: int = 8765


def make_observation(fmt: str, rng: np.random.Generator, prompt: str) -> dict:
    sizes = FORMATS[fmt]
    return {
        "observation/state": rng.random(62),
        "observation/images/left_color": rng.integers(256, size=(*sizes["wrist"], 3), dtype=np.uint8),
        "observation/images/right_color": rng.integers(256, size=(*sizes["wrist"], 3), dtype=np.uint8),
        "observation/images/head_camera": rng.integers(256, size=(*sizes["head"], 3), dtype=np.uint8),
        "prompt": prompt,
    }


def _memory() -> dict:
    """Host MemAvailable and this process's VmRSS / VmHWM, in GiB."""
    out = {}
    for path, keys in (("/proc/meminfo", ("MemAvailable",)), ("/proc/self/status", ("VmRSS", "VmHWM"))):
        for line in pathlib.Path(path).read_text().splitlines():
            name, _, value = line.partition(":")
            if name in keys:
                out[name] = int(value.split()[0]) / 2**20
    return out


def _device_memory(device: str | None) -> dict:
    import torch

    if device == "xpu" and torch.xpu.is_available():
        return {
            "max_memory_allocated_GiB": torch.xpu.max_memory_allocated() / 2**30,
            "max_memory_reserved_GiB": torch.xpu.max_memory_reserved() / 2**30,
        }
    if device == "cuda" and torch.cuda.is_available():
        return {
            "max_memory_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "max_memory_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        }
    return {}


def _compiled_graphs() -> int:
    import torch._dynamo.utils

    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])  # noqa: SLF001


def _summary(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
        "max": float(v.max()),
        "min": float(v.min()),
    }


class _Client:
    """Sends observations, chaining the RTC fields when the server runs RTC."""

    def __init__(self, args: Args):
        self._args = args
        self._client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=args.port)
        self._prev_chunk_id = 0

    def infer(self, obs: dict) -> tuple[float, dict]:
        request = dict(obs)
        if self._args.rtc.enabled and self._prev_chunk_id:
            request[_policy.RTC_PREV_CHUNK_ID] = self._prev_chunk_id
            request[_policy.RTC_PREFIX_START] = self._args.prefix_start
            request[_policy.RTC_INFERENCE_DELAY] = self._args.inference_delay
            request[_policy.RTC_EXECUTION_HORIZON] = self._args.execution_horizon
        start = time.perf_counter()
        result = self._client.infer(request)
        elapsed = time.perf_counter() - start
        if self._args.rtc.enabled:
            self._prev_chunk_id = int(result[_policy.RTC_CHUNK_ID])
        return elapsed, result


def _timed(client: _Client, obs: dict, count: int, record: dict) -> None:
    for _ in range(count):
        elapsed, result = client.infer(obs)
        record.setdefault("client_ms", []).append(elapsed * 1000)
        for key, value in result.get("server_timing", {}).items():
            record.setdefault(f"server_{key}", []).append(value)
        for key, value in result.get("policy_timing", {}).items():
            record.setdefault(f"policy_{key}", []).append(value)


def main(args: Args) -> None:
    import torch

    report: dict = {
        "args": json.loads(json.dumps(dataclasses.asdict(args), default=str)),
        "torch": torch.__version__,
        "host": platform.node(),
        "env": {k: os.environ.get(k) for k in ("ZE_AFFINITY_MASK", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
        "memory_start": _memory(),
    }
    rng = np.random.default_rng(args.seed)
    observations = {fmt: make_observation(fmt, rng, args.prompt) for fmt in args.formats}

    serve_args = serve_policy.Args(env=serve_policy.EnvMode.TELEAVATAR, policy=args.policy, rtc=args.rtc)
    start = time.monotonic()
    policy = serve_policy.create_policy(serve_args)
    report["build_policy_s"] = time.monotonic() - start
    device = args.policy.pytorch_device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"
    report["memory_after_build"] = _memory()

    if not args.skip_warmup:
        report["warmup_s"] = serve_policy.warmup(policy, serve_policy.WARMUP_EXAMPLE[serve_policy.EnvMode.TELEAVATAR]())
    report["graphs_after_warmup"] = _compiled_graphs()

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="127.0.0.1", port=args.port, metadata=policy.metadata
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = _Client(args)

    report["runs"] = {}
    report["errors"] = []
    try:
        if args.skip_warmup:
            # The first request compiles inside the server's event loop; see whether the connection survives it.
            elapsed, _ = client.infer(observations[args.formats[0]])
            report["first_request_s"] = elapsed
        if args.stability > 0:
            fmt = args.formats[0]
            record: dict = {}
            _timed(client, observations[fmt], args.warmup_requests, {})
            graphs = _compiled_graphs()
            _timed(client, observations[fmt], args.stability, record)
            record["new_graphs"] = _compiled_graphs() - graphs
            report["runs"][f"stability_{fmt}"] = record
        else:
            for fmt in args.formats:
                record = {}
                _timed(client, observations[fmt], args.warmup_requests, {})
                graphs = _compiled_graphs()
                _timed(client, observations[fmt], args.requests, record)
                record["new_graphs"] = _compiled_graphs() - graphs
                report["runs"][fmt] = record
    except Exception:  # Record and still write what was measured.
        report["errors"].append(traceback.format_exc())
        logging.exception("Benchmark failed")

    report["graphs_at_end"] = _compiled_graphs()
    report["device_memory"] = _device_memory(device)
    report["memory_end"] = _memory()
    report["summary"] = {
        name: {key: _summary(values) for key, values in record.items() if isinstance(values, list)}
        for name, record in report["runs"].items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for name, summary in report["summary"].items():
        client_ms = summary.get("client_ms", {})
        logging.info(
            "%s: client p50 %.1f p95 %.1f max %.1f ms; model p50 %.1f ms",
            name,
            client_ms.get("p50", float("nan")),
            client_ms.get("p95", float("nan")),
            client_ms.get("max", float("nan")),
            summary.get("policy_infer_ms", {}).get("p50", float("nan")),
        )
    logging.info("wrote %s", args.out)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
