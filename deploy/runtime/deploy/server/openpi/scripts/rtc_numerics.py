"""Numerics of Real-Time Chunking on a real pi05_teleavatar_v2 checkpoint: JAX vs PyTorch (CPU / XPU).

Model level: every backend runs in its own process (a full-size model takes 13-20 GB of host memory),
reads the same transformed observations, noise and previous chunk, and writes the normalized action
chunk of every (sample, combo) pair to one ``[N, C, 30, 32]`` .npy plus a ``.json`` sidecar:

    rtc_numerics.py extract-real --dataset DIR --episode 0 --frame 1430 --out real0.npz
    rtc_numerics.py make-inputs  --config pi05_teleavatar_v2 --ckpt JAX_CKPT --asset-id ID \\
                                 --seeds 100-115 --real real0.npz real1.npz --out inputs.npz
    rtc_numerics.py jax-prev     --config ... --ckpt JAX_CKPT --inputs inputs.npz --out prev.npz
    rtc_numerics.py run-jax      --config ... --ckpt JAX_CKPT --inputs ... --prev ... --dtype float32 --out jax_fp32.npy
    rtc_numerics.py run-torch    --config ... --ckpt PT_CKPT  --inputs ... --prev ... --dtype bfloat16 --device xpu --out xpu.npy
    rtc_numerics.py calibrate    --ref jax_fp32.npy --alt jax_fp32_alt.npy --draws d1.npy d2.npy ... --out criteria.json
    rtc_numerics.py evaluate     --criteria criteria.json --ref jax_fp32.npy --prev prev.npz --cand xpu.npy --kind bf16

Policy level: one sequence of ``rtc/*`` requests through a ``Policy`` (JAX or PyTorch), recording the raw
chunks, the served actions and what reached ``sample_actions``; ``policy-compare`` checks two runs.

Weights are the checkpoint rounded to bf16, as served. ``--dtype float32`` computes in float32 on those
bf16 values (JAX: params cast up and config dtype float32; PyTorch: ``model.to(float32)``).
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import resource
import sys
import time

import numpy as np

ACTION_HORIZON = 30
ACTION_DIM = 32
NUM_STEPS = 10
# Teleavatar uses the first 16 action dims: [left arm (7), left gripper, right arm (7), right gripper].
VALID_DIMS = 16
GROUPS = {"J": (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14), "G": (7, 15)}
# The client has executed this many steps of the previous chunk when it asks for the next one.
PREFIX_START = 5
DELAY_HORIZON = ((0, 30), (4, 8), (9, 18), (12, 24))
OBS_KEYS = (
    "image",
    "image_mask",
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "token_ar_mask",
    "token_loss_mask",
)
GRIPPER_ZERO_CROSSING = 0.10  # trigger value where the gripper effort changes sign


def _log(msg: str) -> None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"[{time.strftime('%H:%M:%S')} maxrss={rss_gb:.1f}G] {msg}", flush=True)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, payload) -> None:
    pathlib.Path(path).write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")


# --- combos ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Combo:
    name: str
    mode: str | None = None  # None (no RTC), "guided" or "trained"
    schedule: str = "exp"
    trained_prefix_noise: bool = False
    delay: int = 0
    horizon: int = ACTION_HORIZON

    def rtc_config(self):
        from openpi.models import rtc as _rtc

        return _rtc.RTCConfig(
            mode=_rtc.RTCMode(self.mode),
            prefix_attention_schedule=_rtc.PrefixAttentionSchedule(self.schedule),
            trained_prefix_noise=self.trained_prefix_noise,
        )


def all_combos() -> list[Combo]:
    """The fixed table: no RTC, then each variant over every (delay, horizon). Grouped by variant, so a
    compiled backend sees each new graph once and then only (delay, horizon) changes."""
    variants = (
        ("guided_exp", "guided", "exp", False),
        ("guided_linear", "guided", "linear", False),
        ("guided_ones", "guided", "ones", False),
        ("trained", "trained", "exp", False),
        ("trained_noise", "trained", "exp", True),
    )
    combos = [Combo("none")]
    for name, mode, schedule, noise in variants:
        for delay, horizon in DELAY_HORIZON:
            combos.append(Combo(f"{name}_d{delay}_h{horizon}", mode, schedule, noise, delay, horizon))
    return combos


def select_combos(spec: str) -> list[Combo]:
    """`all`, or comma-separated name prefixes (`none,trained` also picks `trained_noise_*`)."""
    combos = all_combos()
    if spec == "all":
        return combos
    prefixes = tuple(p.strip() for p in spec.split(",") if p.strip())
    chosen = [c for c in combos if c.name.startswith(prefixes)]
    if not chosen:
        raise ValueError(f"no combo matches {spec!r}")
    return chosen


def prefix_weights_np(start: int, end: int, total: int, schedule: str) -> np.ndarray:
    """NumPy copy of `openpi.models.rtc.get_prefix_weights`."""
    start, end = int(np.clip(start, 0, total)), int(np.clip(end, 0, total))
    start = min(start, end)
    idx = np.arange(total)
    if schedule == "ones":
        return (idx < end).astype(np.float64)
    if schedule == "zeros":
        return (idx < start).astype(np.float64)
    ramp = 1.0 - (idx - start + 1) / (end - start + 1)
    weights = np.clip(np.where(idx < start, 1.0, ramp), 0.0, 1.0)
    weights = np.where(idx >= end, 0.0, weights)
    if schedule == "exp":
        weights = weights * np.expm1(weights) / (math.e - 1.0)
    return weights


# --- inputs ------------------------------------------------------------------------------------


def _train_config(name: str, **model_overrides):
    from openpi.training import config as _config

    train_config = _config.get_config(name)
    return dataclasses.replace(train_config, model=dataclasses.replace(train_config.model, **model_overrides))


def _halve(image: np.ndarray) -> np.ndarray:
    """2x2 box average with integer rounding, uint8 in and out."""
    h, w, c = image.shape
    blocks = image.astype(np.uint16).reshape(h // 2, 2, w // 2, 2, c).sum(axis=(1, 3))
    return ((blocks + 2) // 4).astype(np.uint8)


def extract_real(args) -> None:
    """One LeRobot frame, cut the way the deployed client sends it (`examples/teleavatar_v2/env.py`): one eye
    per camera -- the head's left eye, the inner eye of each wrist camera, as `TeleavatarInputs` crops them --
    at half resolution (head 960x960, wrists 400x640). The client's own resize is not reproduced exactly."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = pathlib.Path(args.dataset)
    dataset = LeRobotDataset(root.name, root=root, episodes=[args.episode], video_backend="pyav")
    index = next(i for i in range(len(dataset.hf_dataset)) if int(dataset.hf_dataset[i]["frame_index"]) == args.frame)
    item = dataset[index]
    assert int(item["episode_index"]) == args.episode
    assert int(item["frame_index"]) == args.frame

    def to_uint8(chw) -> np.ndarray:  # as `teleavatar_v2_policy._parse_image` treats LeRobot's float frames
        return (255 * chw.numpy()).astype(np.uint8).transpose(1, 2, 0)

    head = to_uint8(item["observation.images.head_camera"])
    left = to_uint8(item["observation.images.left_color"])
    right = to_uint8(item["observation.images.right_color"])
    raw = {
        "observation/state": item["observation.state"].numpy().astype(np.float64),
        "observation/images/head_camera": _halve(head[:, : head.shape[1] // 2]),
        "observation/images/left_color": _halve(left[:, left.shape[1] // 2 :]),
        "observation/images/right_color": _halve(right[:, : right.shape[1] // 2]),
        "prompt": np.asarray(item["task"]),
    }
    np.savez(args.out, **raw)
    for key, value in raw.items():
        print(f"{key}: {value.dtype} {value.shape}" + (f" {value}" if value.ndim == 0 else ""))
    print(
        f"source: {root} episode {args.episode} frame {args.frame} (index {index}), timestamp {float(item['timestamp']):.4f}"
    )
    _log(f"wrote {args.out}")


def _load_raw(path: str) -> dict:
    raw = dict(np.load(path))
    raw["prompt"] = str(raw["prompt"])
    return raw


def _parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        seeds.extend(range(int(lo), int(hi or lo) + 1))
    return seeds


def make_inputs(args) -> None:
    from openpi import transforms
    from openpi.policies import teleavatar_v2_policy
    from openpi.training import checkpoints as _checkpoints

    train_config = _train_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats_path = pathlib.Path(args.ckpt) / "assets" / args.asset_id / "norm_stats.json"
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.ckpt) / "assets", args.asset_id)
    # The input side of `policy_config.create_trained_policy`, without the model.
    transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    raws, sources = [], []
    for seed in _parse_seeds(args.seeds):
        np.random.seed(seed)  # make_teleavatar_example draws from the global RNG
        raws.append(teleavatar_v2_policy.make_teleavatar_example())
        sources.append({"kind": "synthetic", "seed": seed})
    for path in args.real or []:
        raws.append(_load_raw(path))
        sources.append({"kind": "real", "file": path, "sha256": _sha256(path)})
    flat: dict[str, list] = {}
    for raw in raws:
        data = transform(raw)
        for key in OBS_KEYS:
            if key not in data:
                continue
            if isinstance(data[key], dict):
                for sub, value in data[key].items():
                    flat.setdefault(f"{key}/{sub}", []).append(np.asarray(value))
            else:
                flat.setdefault(key, []).append(np.asarray(data[key]))
    batch = {k: np.stack(v) for k, v in flat.items()}
    rng = np.random.default_rng(args.noise_seed)
    shape = (len(raws), train_config.model.action_horizon, train_config.model.action_dim)
    batch["noise"] = rng.standard_normal(shape, dtype=np.float32)
    batch["prev_noise"] = rng.standard_normal(shape, dtype=np.float32)
    np.savez(args.out, **batch)
    for key, value in batch.items():
        print(f"{key}: {value.dtype} {value.shape}")
    _write_json(
        args.out + ".json",
        {
            "config": args.config,
            "asset_id": args.asset_id,
            "norm_stats": str(norm_stats_path),
            "norm_stats_sha256": _sha256(norm_stats_path),
            "noise_seed": args.noise_seed,
            "samples": sources,
            "prompts": [str(r.get("prompt", "")) for r in raws],
        },
    )
    _log(f"wrote {args.out} ({len(raws)} samples)")


def _load_inputs(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    raw = dict(np.load(path))
    noise, prev_noise = raw.pop("noise"), raw.pop("prev_noise")
    obs: dict = {}
    for key, value in raw.items():
        if "/" in key:
            outer, inner = key.split("/", 1)
            obs.setdefault(outer, {})[inner] = value
        else:
            obs[key] = value
    return obs, noise, prev_noise


def _load_prev(path: str) -> np.ndarray:
    prev = np.load(path)
    assert int(prev["prefix_start"]) == PREFIX_START
    return prev["aligned"]


# --- JAX ---------------------------------------------------------------------------------------


def _jax_model(config: str, ckpt: str, dtype: str):
    """The served weights (checkpoint rounded to bf16), optionally cast up to float32 leaf by leaf."""
    import jax.numpy as jnp

    from openpi.models import model as _model

    params = _model.restore_params(pathlib.Path(ckpt) / "params", dtype=jnp.bfloat16)
    if dtype == "float32":

        def upcast(tree):
            for key, value in tree.items():
                if isinstance(value, dict):
                    upcast(value)
                else:
                    tree[key] = value.astype(jnp.float32)
                    del value

        upcast(params)
    model = _train_config(config, dtype=dtype).model.load(params)
    del params
    return model


def _jax_sampler(model):
    from openpi.shared import nnx_utils

    # `rtc_config` selects the traced graph; delay and horizon are arrays, as in `Policy`.
    return nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_config",))


def jax_prev(args) -> None:
    """The previous chunk every backend is constrained against: JAX float32, unconstrained, from `prev_noise`,
    then left-aligned for `PREFIX_START` executed steps and right-padded with zeros, as `Policy` does."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    model = _jax_model(args.config, args.ckpt, "float32")
    obs_np, _, prev_noise = _load_inputs(args.inputs)
    obs = _model.Observation.from_dict(jax.tree.map(jnp.asarray, obs_np))
    sample = _jax_sampler(model)
    raw = []
    for i in range(prev_noise.shape[0]):
        one = jax.tree.map(lambda x, i=i: x[i : i + 1], obs)
        raw.append(
            np.asarray(sample(jax.random.key(0), one, num_steps=NUM_STEPS, noise=jnp.asarray(prev_noise[i : i + 1])))
        )
        _log(f"prev chunk {i}")
    raw = np.concatenate(raw).astype(np.float32)
    aligned = np.zeros_like(raw)
    aligned[:, : ACTION_HORIZON - PREFIX_START] = raw[:, PREFIX_START:]
    np.savez(args.out, raw=raw, aligned=aligned, prefix_start=PREFIX_START)
    _log(f"wrote {args.out} {raw.shape}")


class _Runner:
    """Fills `[N, C, 30, 32]` combo by combo, saving a partial file after each combo so that an interrupted
    run resumes where it stopped."""

    def __init__(self, out: str, n: int, combos: list[Combo], meta: dict):
        self.out, self.combos, self.meta = out, combos, meta
        self.partial = out + ".partial.npy"
        self.actions = np.full((n, len(combos), ACTION_HORIZON, ACTION_DIM), np.nan, dtype=np.float32)
        if pathlib.Path(self.partial).exists():
            previous = np.load(self.partial)
            assert previous.shape == self.actions.shape, (previous.shape, self.actions.shape)
            self.actions = previous
            _log(
                f"resuming from {self.partial}: {int(np.isfinite(previous[..., 0, 0]).sum())} of {n * len(combos)} done"
            )
        self.meta.update({"combos": [dataclasses.asdict(c) for c in combos], "seconds": {}, "argv": sys.argv})

    def run(self, call) -> None:
        for ci, combo in enumerate(self.combos):
            todo = [i for i in range(self.actions.shape[0]) if not np.isfinite(self.actions[i, ci, 0, 0])]
            if not todo:
                continue
            times = []
            for i in todo:
                t0 = time.perf_counter()
                self.actions[i, ci] = call(i, combo)
                times.append(time.perf_counter() - t0)
            self.meta["seconds"][combo.name] = times
            self.after_combo(combo)
            np.save(self.partial, self.actions)
            _log(f"{combo.name}: {len(todo)} samples, first {times[0]:.2f} s, median {float(np.median(times)):.2f} s")
        assert np.isfinite(self.actions).all()
        np.save(self.out, self.actions)
        _write_json(self.out + ".json", self.meta)
        pathlib.Path(self.partial).unlink()
        _log(f"wrote {self.out} {self.actions.shape}")

    def after_combo(self, combo: Combo) -> None:
        pass


def run_jax(args) -> None:
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    _log(f"jax {jax.__version__} devices {jax.devices()}")
    model = _jax_model(args.config, args.ckpt, args.dtype)
    obs_np, noise, _ = _load_inputs(args.inputs)
    prev = _load_prev(args.prev)
    obs = _model.Observation.from_dict(jax.tree.map(jnp.asarray, obs_np))
    sample = _jax_sampler(model)
    configs: dict[Combo, object] = {}

    def call(i: int, combo: Combo) -> np.ndarray:
        one = jax.tree.map(lambda x: x[i : i + 1], obs)
        kwargs = {}
        if combo.mode is not None:
            kwargs = {
                "prev_chunk": jnp.asarray(prev[i : i + 1]),
                "inference_delay": jnp.asarray(combo.delay, dtype=jnp.int32),
                "execution_horizon": jnp.asarray(combo.horizon, dtype=jnp.int32),
                "rtc_config": configs.setdefault(combo, combo.rtc_config()),
            }
        actions = sample(jax.random.key(0), one, num_steps=NUM_STEPS, noise=jnp.asarray(noise[i : i + 1]), **kwargs)
        return np.asarray(actions, dtype=np.float32)[0]

    meta = {
        "backend": "jax",
        "dtype": args.dtype,
        "jax": jax.__version__,
        "inputs_sha256": _sha256(args.inputs),
        "prev_sha256": _sha256(args.prev),
    }
    _Runner(args.out, noise.shape[0], select_combos(args.combos), meta).run(call)


# --- PyTorch -----------------------------------------------------------------------------------


def _restore_rope_inv_freq(model) -> int:
    """For code older than the RoPE fix: recompute each rotary embedding's `inv_freq` in float32."""
    import torch

    n = 0
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
            exact, _ = module.rope_init_fn(module.config, module.inv_freq.device)
            module.inv_freq = exact.to(torch.float32)
            module.original_inv_freq = module.inv_freq
            n += 1
    return n


def _torch_model(args):
    import torch

    compile_mode = None if args.compile == "none" else args.compile
    train_config = _train_config(args.config, pytorch_compile_mode=compile_mode)
    # As `policy_config.create_trained_policy` loads it.
    model = train_config.model.load_pytorch(train_config, str(pathlib.Path(args.ckpt) / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if args.dtype == "float32":
        model = model.to(torch.float32)
    if args.fix_rope:
        _log(f"rope inv_freq recomputed in float32 for {_restore_rope_inv_freq(model)} modules (old code only)")
    return model.to(args.device).eval().requires_grad_(requires_grad=False), compile_mode


def _sync(device: str) -> None:
    import torch

    if device.startswith("xpu"):
        torch.xpu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def run_torch(args) -> None:
    import torch

    from openpi.models import model as _model

    if args.threads:
        torch.set_num_threads(args.threads)
    _log(f"torch {torch.__version__}, threads {torch.get_num_threads()}, device {args.device}")
    model, compile_mode = _torch_model(args)
    _log(f"model loaded ({args.dtype}, compile={compile_mode})")
    obs_np, noise, _ = _load_inputs(args.inputs)
    prev = _load_prev(args.prev)
    device = args.device

    def call(i: int, combo: Combo) -> np.ndarray:
        one = {
            k: (
                {kk: torch.from_numpy(vv[i : i + 1]).to(device) for kk, vv in v.items()}
                if isinstance(v, dict)
                else torch.from_numpy(v[i : i + 1]).to(device)
            )
            for k, v in obs_np.items()
        }
        kwargs = {}
        if combo.mode is not None:
            # 0-d tensors, as `Policy` passes them: a changing delay is data, never a new graph.
            kwargs = {
                "prev_chunk": torch.from_numpy(prev[i : i + 1]).to(device),
                "inference_delay": torch.tensor(combo.delay, device=device),
                "execution_horizon": torch.tensor(combo.horizon, device=device),
                "rtc_config": combo.rtc_config(),
            }
        actions = model.sample_actions(
            device,
            _model.Observation.from_dict(one),
            noise=torch.from_numpy(noise[i : i + 1]).to(device),
            num_steps=NUM_STEPS,
            **kwargs,
        )
        actions = actions.float().cpu().numpy()[0]
        _sync(device)
        return actions

    meta = {
        "backend": "torch",
        "device": device,
        "dtype": args.dtype,
        "compile": compile_mode,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "inputs_sha256": _sha256(args.inputs),
        "prev_sha256": _sha256(args.prev),
        "compiled_graphs": {},
    }
    if device.startswith("xpu"):
        meta["xpu_device"] = torch.xpu.get_device_name(0)
        meta["xpu_driver"] = getattr(torch.xpu.get_device_properties(0), "driver_version", None)
    runner = _Runner(args.out, noise.shape[0], select_combos(args.combos), meta)

    def after_combo(combo: Combo) -> None:
        # Graphs compiled so far (prefix pass + one denoising graph per variant): recorded per combo so that
        # "a new (delay, horizon) never compiles" can be checked.
        import torch._dynamo.utils

        graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])  # noqa: SLF001
        meta["compiled_graphs"][combo.name] = graphs
        if device.startswith("xpu"):
            meta["xpu_max_memory_allocated_GiB"] = torch.xpu.max_memory_allocated() / 2**30

    runner.after_combo = after_combo
    runner.run(call)


# --- statistics --------------------------------------------------------------------------------


def _load_run(path: str) -> tuple[np.ndarray, list[str]]:
    actions = np.load(path)
    meta = json.loads(pathlib.Path(path + ".json").read_text())
    return actions.astype(np.float64), [c["name"] for c in meta["combos"]]


def _pools(n: int, n_synthetic: int) -> dict[str, list[int]]:
    return {"S": list(range(n_synthetic)), "R": list(range(n_synthetic, n))}


def _stats(values: np.ndarray) -> dict[str, float]:
    v = values.ravel()
    return {
        "p50": float(np.quantile(v, 0.5)),
        "p99": float(np.quantile(v, 0.99)),
        "mean": float(v.mean()),
        "max": float(v.max()),
        "n": int(v.size),
    }


def _group(diff: np.ndarray, samples: list[int], group: str) -> np.ndarray:
    """`diff` is [N, C?, 30, 32]-shaped |a - b|; select samples and one dim group."""
    return diff[samples][..., list(GROUPS[group])]


def _sensitive(ref: np.ndarray, alt: np.ndarray, names: list[str], ratio: float = 5.0) -> list[dict]:
    """Trajectories (sample, dim) whose float32 accumulation-order noise stands out: max over steps divided by
    the largest such max among all other trajectories of the same combo exceeds `ratio`."""
    noise = np.abs(alt - ref)[..., :VALID_DIMS]  # [N, C, 30, 16]
    found = []
    for ci, name in enumerate(names):
        peak = noise[:, ci].max(axis=1)  # [N, 16]
        flat = peak.ravel()
        order = np.argsort(flat)[::-1]
        if flat[order[1]] <= 0:
            continue
        for k in order:
            others = flat[order[0]] if k != order[0] else flat[order[1]]
            r = flat[k] / others if others > 0 else np.inf
            if r <= ratio:
                break
            s, d = np.unravel_index(k, peak.shape)
            found.append({"combo": name, "sample": int(s), "dim": int(d), "r": float(r), "max_abs": float(flat[k])})
    return found


def calibrate(args) -> None:
    """JAX-only statistics, written before any PyTorch result is compared (see the phase 4 pre-registration)."""
    ref, names = _load_run(args.ref)
    alt, alt_names = _load_run(args.alt)
    assert alt_names == names
    draws = []
    for path in args.draws:
        d, d_names = _load_run(path)
        assert d_names == names, path
        draws.append(d)
    n = ref.shape[0]
    pools = _pools(n, args.synthetic)

    # Bitwise-identical draws are one draw.
    distinct: list[int] = []
    for k, d in enumerate(draws):
        if not any(np.array_equal(d, draws[j]) for j in distinct):
            distinct.append(k)
    used = [draws[k] for k in distinct]
    if len(used) < 3:
        raise SystemExit(f"only {len(used)} distinct bf16 draws; the pre-registration asks for at least 3")

    out = {
        "created": time.strftime("%F %T %Z"),
        "files": {},
        "pools": pools,
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "draws": [args.draws[k] for k in distinct],
        "duplicate_draws": [args.draws[k] for k in range(len(draws)) if k not in distinct],
        "combos": names,
        "eps": {},
        "spread": {},
        "thresholds": {},
    }
    for path in [args.ref, args.alt, *args.draws]:
        out["files"][path] = _sha256(path)

    dev = [np.abs(d - ref) for d in used]  # D_k = |bf16_k - fp32|
    pairs = [(i, j) for i in range(len(used)) for j in range(i + 1, len(used))]
    spread = {(i, j): np.abs(used[i] - used[j]) for i, j in pairs}
    for pool_name, samples in pools.items():
        if not samples:
            continue
        eps, spr = {}, {}
        for g in GROUPS:
            eps.setdefault("_pooled", {})[g] = _stats(np.concatenate([_group(d, samples, g).ravel() for d in dev]))
            spr.setdefault("_pooled", {})[g] = {
                "p99_max_pair": max(float(np.quantile(_group(s, samples, g), 0.99)) for s in spread.values()),
                "mean_max_pair": max(float(_group(s, samples, g).mean()) for s in spread.values()),
            }
            for ci, name in enumerate(names):
                eps.setdefault(name, {})[g] = _stats(
                    np.concatenate([_group(d[:, ci], samples, g).ravel() for d in dev])
                )
                spr.setdefault(name, {})[g] = {
                    "p99_max_pair": max(
                        float(np.quantile(_group(s[:, ci], samples, g), 0.99)) for s in spread.values()
                    ),
                    "mean_max_pair": max(float(_group(s[:, ci], samples, g).mean()) for s in spread.values()),
                }
        out["eps"][pool_name], out["spread"][pool_name] = eps, spr

    eps, spr = out["eps"]["S"], out["spread"]["S"]
    out["thresholds"] = {
        "fp32": {"max_abs": 1e-2, "min_cos": 0.999},
        "bf16": {
            "pooled": {g: {"p99": 2 * eps["_pooled"][g]["p99"], "mean": 2 * eps["_pooled"][g]["mean"]} for g in GROUPS},
            "per_combo_p99_cap": {name: {g: 4 * eps[name][g]["p99"] for g in GROUPS} for name in names},
        },
        "compile_vs_eager": {
            "pooled": {g: {"p99": 2 * spr["_pooled"][g]["p99_max_pair"]} for g in GROUPS},
            "per_combo_p99_cap": {name: {g: 4 * spr[name][g]["p99_max_pair"] for g in GROUPS} for name in names},
        },
        "pull_ratio_max": 0.5,
        "pull_ratio_fallback_abs": 0.05,
    }
    out["sensitive"] = _sensitive(ref, alt, names)
    noise = np.abs(alt - ref)[..., :VALID_DIMS].max(axis=2)  # [N, C, 16]
    out["order_noise_top3"] = {}
    for ci, name in enumerate(names):
        flat = noise[:, ci].ravel()
        top = np.argsort(flat)[::-1][:4]
        out["order_noise_top3"][name] = [
            {
                "sample": int(k // VALID_DIMS),
                "dim": int(k % VALID_DIMS),
                "max_abs": float(flat[k]),
                "ratio_to_4th": float(flat[k] / flat[top[3]]) if flat[top[3]] > 0 else None,
            }
            for k in top[:3]
        ]
    out["fp32_order_noise"] = {
        name: float(np.abs(alt - ref)[:, ci, :, :VALID_DIMS].max()) for ci, name in enumerate(names)
    }
    _write_json(args.out, out)
    _log(f"wrote {args.out}")
    _print_calibration(out)


def _print_calibration(out: dict) -> None:
    print(f"distinct draws {len(out['draws'])}: {out['draws']}; duplicates {out['duplicate_draws']}")
    for g in GROUPS:
        e, s = out["eps"]["S"]["_pooled"][g], out["spread"]["S"]["_pooled"][g]
        print(
            f"S pooled {g}: eps p50 {e['p50']:.3g} p99 {e['p99']:.3g} mean {e['mean']:.3g} max {e['max']:.3g}; "
            f"draw spread p99 {s['p99_max_pair']:.3g}"
        )
    print(f"{'combo':24s} " + " ".join(f"{g}:eps_p99 {g}:eps_mean {g}:spread_p99" for g in GROUPS))
    for name in out["combos"]:
        cells = []
        for g in GROUPS:
            e, s = out["eps"]["S"][name][g], out["spread"]["S"][name][g]
            cells.append(f"{e['p99']:.3g} {e['mean']:.3g} {s['p99_max_pair']:.3g}")
        print(f"{name:24s} " + " | ".join(cells))
    print(f"sensitive trajectories (r > 5): {len(out['sensitive'])}")
    for item in out["sensitive"]:
        print(f"  {item}")


def _cos_min(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest per-sample cosine over the valid dims of one combo, [N, 30, 32] each."""
    cos = []
    for x, y in zip(a[..., :VALID_DIMS], b[..., :VALID_DIMS], strict=True):
        cos.append(float(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y))))
    return min(cos)


def _pull_ratio(
    actions: np.ndarray, unconstrained: np.ndarray, prev: np.ndarray, combo: Combo, samples: list[int]
) -> float:
    """sum_w |guided - prev| / sum_w |unconstrained - prev| over the valid part of the previous chunk, weighted by
    the combo's prefix weights."""
    w = prefix_weights_np(combo.delay, combo.horizon, ACTION_HORIZON, combo.schedule)
    w[ACTION_HORIZON - PREFIX_START :] = 0.0  # the zero padding is not part of the previous chunk
    w = w[None, :, None]
    guided = (w * np.abs(actions[samples] - prev[samples])[..., :VALID_DIMS]).sum()
    free = (w * np.abs(unconstrained[samples] - prev[samples])[..., :VALID_DIMS]).sum()
    return float(guided / free)


def _quantile_unnormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _effort(trigger: np.ndarray) -> np.ndarray:
    return np.where(trigger < 0.10, 2.0 * (1.0 - trigger / 0.10), -1.6 * (trigger - 0.10) / 0.90)


def _physical(cand: np.ndarray, ref: np.ndarray, norm_stats_path: str, samples: list[int]) -> dict:
    """Errors as the robot sees them: joints in rad, grippers in Nm split by the reference's segment."""
    stats = json.loads(pathlib.Path(norm_stats_path).read_text())["norm_stats"]["actions"]
    q01, q99 = np.asarray(stats["q01"])[:VALID_DIMS], np.asarray(stats["q99"])[:VALID_DIMS]
    a = _quantile_unnormalize(cand[samples][..., :VALID_DIMS], q01, q99)
    b = _quantile_unnormalize(ref[samples][..., :VALID_DIMS], q01, q99)
    joints = list(GROUPS["J"])
    out = {"joint_rad": _stats(np.abs(a - b)[..., joints])}
    grip = list(GROUPS["G"])
    ea, eb = _effort(a[..., grip]), _effort(b[..., grip])
    opening = b[..., grip] < GRIPPER_ZERO_CROSSING
    diff = np.abs(ea - eb)
    for seg, mask in (("open", opening), ("close", ~opening)):
        out[f"gripper_Nm_{seg}"] = _stats(diff[mask]) if mask.any() else None
    return out


def evaluate(args) -> None:
    criteria = json.loads(pathlib.Path(args.criteria).read_text())
    ref, names = _load_run(args.ref)
    prev = _load_prev(args.prev).astype(np.float64)
    combos = {c.name: c for c in all_combos()}
    th = criteria["thresholds"]
    pools = criteria["pools"]
    report = {
        "criteria": args.criteria,
        "criteria_sha256": _sha256(args.criteria),
        "ref": args.ref,
        "kind": args.kind,
        "candidates": {},
    }
    ok_all = True
    for path in args.cand:
        cand, cand_names = _load_run(path)
        assert cand_names == names, (path, cand_names)
        res = {"sha256": _sha256(path), "per_combo": {}, "checks": []}
        ok = True

        def check(name: str, passed: bool, detail: str, *, res=res) -> bool:  # noqa: FBT001
            res["checks"].append({"check": name, "pass": bool(passed), "detail": detail})
            return passed

        diff = np.abs(cand - ref)
        everyone = pools["S"] + pools["R"]
        for ci, name in enumerate(names):
            row = {
                "max_abs_all": float(diff[everyone, ci, :, :VALID_DIMS].max()),
                "min_cos_all": _cos_min(cand[everyone, ci], ref[everyone, ci]),
            }
            for pool, samples in pools.items():
                if samples:
                    row[pool] = {g: _stats(_group(diff[:, ci], samples, g)) for g in GROUPS}
            res["per_combo"][name] = row
            combo = combos[name]
            # Structural: a trained prefix is the previous chunk, bit for bit.
            if combo.mode == "trained" and combo.delay > 0:
                same = np.array_equal(cand[:, ci, : combo.delay], prev[:, : combo.delay])
                ok &= check(f"{name}: trained prefix == prev (bitwise)", same, "")
            if args.kind == "fp32":
                ok &= check(
                    f"{name}: fp32 max|d| <= {th['fp32']['max_abs']}",
                    row["max_abs_all"] <= th["fp32"]["max_abs"],
                    f"{row['max_abs_all']:.3g}",
                )
                ok &= check(
                    f"{name}: fp32 min cos >= {th['fp32']['min_cos']}",
                    row["min_cos_all"] >= th["fp32"]["min_cos"],
                    f"{row['min_cos_all']:.7f}",
                )
            if args.kind == "bf16":
                for g in GROUPS:
                    p99 = row["S"][g]["p99"]
                    cap = th["bf16"]["per_combo_p99_cap"][name][g]
                    ok &= check(f"{name}: bf16 {g} p99 <= 4 eps_c", p99 <= cap, f"{p99:.3g} <= {cap:.3g}")
            if combo.mode == "guided":
                rho = _pull_ratio(cand[:, ci], cand[:, names.index("none")], prev, combo, everyone)
                rho_ref = _pull_ratio(ref[:, ci], ref[:, names.index("none")], prev, combo, everyone)
                row["pull_ratio"], row["pull_ratio_ref"] = rho, rho_ref
                if args.kind == "jax":
                    pass  # JAX bf16 draws: the pull ratio is recorded, not gated
                elif rho_ref < th["pull_ratio_max"]:
                    ok &= check(
                        f"{name}: guided pull ratio < {th['pull_ratio_max']}",
                        rho < th["pull_ratio_max"],
                        f"{rho:.3f} (JAX fp32 {rho_ref:.3f})",
                    )
                else:  # the algorithm itself does not reach 0.5 here: the port must match JAX
                    ok &= check(
                        f"{name}: pull ratio within {th['pull_ratio_fallback_abs']} of JAX fp32 (JAX >= {th['pull_ratio_max']})",
                        abs(rho - rho_ref) <= th["pull_ratio_fallback_abs"],
                        f"{rho:.3f} vs {rho_ref:.3f}",
                    )
        if args.kind == "bf16":
            pooled = {}
            for g in GROUPS:
                values = _group(diff, pools["S"], g)[..., :]
                stats = _stats(values)
                pooled[g] = stats
                lim = th["bf16"]["pooled"][g]
                ok &= check(
                    f"pooled bf16 {g} p99 <= 2 eps",
                    stats["p99"] <= lim["p99"],
                    f"{stats['p99']:.3g} <= {lim['p99']:.3g}",
                )
                ok &= check(
                    f"pooled bf16 {g} mean <= 2 eps",
                    stats["mean"] <= lim["mean"],
                    f"{stats['mean']:.3g} <= {lim['mean']:.3g}",
                )
            res["pooled_S"] = pooled
            if pools["R"]:
                res["pooled_R_report_only"] = {g: _stats(_group(diff, pools["R"], g)) for g in GROUPS}
        if args.norm_stats:
            res["physical_S"] = _physical(cand, ref, args.norm_stats, pools["S"])
            if pools["R"]:
                res["physical_R"] = _physical(cand, ref, args.norm_stats, pools["R"])
        sens = [(s["sample"], s["dim"], s["combo"]) for s in criteria.get("sensitive", [])]
        res["sensitive_excluded_max"] = {}
        for ci, name in enumerate(names):
            mask = np.ones(diff.shape[0:1] + diff.shape[2:], dtype=bool)  # [N, 30, 32]
            for s, d, c in sens:
                if c == name:
                    mask[s, :, d] = False
            mask[..., VALID_DIMS:] = False
            res["sensitive_excluded_max"][name] = float(diff[:, ci][mask].max())
        graphs = json.loads(pathlib.Path(path + ".json").read_text()).get("compiled_graphs") or {}
        if json.loads(pathlib.Path(path + ".json").read_text()).get("compile"):
            # The counter must have seen the first compile, or it is not counting on this torch version.
            ok &= check(
                "compiled-graph counter is live", bool(graphs) and graphs[names[0]] > 0, str(graphs.get(names[0]))
            )
        if any(graphs.values()):
            # Within one variant (mode, schedule, prefix noise), a new (delay, horizon) must not compile anything.
            for variant in sorted({n.rsplit("_d", 1)[0] for n in names if n != "none"}):
                counts = [graphs[n] for n in names if n.rsplit("_d", 1)[0] == variant]
                ok &= check(
                    f"{variant}: no compile after its first (delay, horizon)", len(set(counts)) == 1, str(counts)
                )
            res["compiled_graphs"] = graphs
        res["pass"] = bool(ok)
        ok_all &= ok
        report["candidates"][path] = res
        _print_candidate(path, res, names, args.kind)
    report["pass"] = bool(ok_all)
    if args.out:
        _write_json(args.out, report)
        _log(f"wrote {args.out}")
    print(f"OVERALL {'PASS' if ok_all else 'FAIL'}")


def _print_candidate(path: str, res: dict, names: list[str], kind: str) -> None:
    print(f"== {path} ({kind})")
    print(f"{'combo':24s} {'max|d|':>9s} {'min cos':>10s} " + " ".join(f"{g}:p99 {g}:mean" for g in GROUPS) + "  pull")
    for name in names:
        row = res["per_combo"][name]
        cells = " ".join(f"{row['S'][g]['p99']:.3g} {row['S'][g]['mean']:.3g}" for g in GROUPS)
        pull = f"{row['pull_ratio']:.3f}/{row['pull_ratio_ref']:.3f}" if "pull_ratio" in row else ""
        print(f"{name:24s} {row['max_abs_all']:9.3g} {row['min_cos_all']:10.7f} {cells}  {pull}")
    failed = [c for c in res["checks"] if not c["pass"]]
    print(f"checks: {len(res['checks'])}, failed: {len(failed)}")
    for c in failed:
        print(f"  FAIL {c['check']}: {c['detail']}")
    for key in ("pooled_S", "pooled_R_report_only", "physical_S", "physical_R"):
        if key in res:
            print(f"{key}: {json.dumps(res[key])}")


def compare_pair(args) -> None:
    """Compile vs eager (or any two runs of one backend): |a - b| against the draw-spread thresholds."""
    criteria = json.loads(pathlib.Path(args.criteria).read_text())
    a, names = _load_run(args.a)
    b, b_names = _load_run(args.b)
    assert names == b_names
    th = criteria["thresholds"]["compile_vs_eager"]
    pools = criteria["pools"]
    diff = np.abs(a - b)
    ok = True
    lines = []
    for g in GROUPS:
        stats = _stats(_group(diff, pools["S"], g))
        passed = stats["p99"] <= th["pooled"][g]["p99"]
        ok &= passed
        lines.append(
            f"pooled {g}: p99 {stats['p99']:.3g} <= {th['pooled'][g]['p99']:.3g} {'ok' if passed else 'FAIL'}; "
            f"mean {stats['mean']:.3g} max {stats['max']:.3g}"
        )
    for ci, name in enumerate(names):
        cells = []
        for g in GROUPS:
            p99 = float(np.quantile(_group(diff[:, ci], pools["S"], g), 0.99))
            cap = th["per_combo_p99_cap"][name][g]
            ok &= p99 <= cap
            cells.append(f"{g} p99 {p99:.3g} (cap {cap:.3g}{'' if p99 <= cap else ' FAIL'})")
        lines.append(f"{name:24s} max {diff[:, ci, :, :VALID_DIMS].max():.3g}  " + "  ".join(cells))
    print("\n".join(lines))
    if pools["R"]:
        print("R (report only): " + json.dumps({g: _stats(_group(diff, pools["R"], g)) for g in GROUPS}))
    print(f"OVERALL {'PASS' if ok else 'FAIL'}")


# --- policy level ------------------------------------------------------------------------------

# Request k > 0 reports (prefix_start, inference_delay, execution_horizon) for the chunk returned at k - 1.
POLICY_REQUESTS = (None, (5, 4, 8), (9, 9, 18), (12, 12, 24), (30, 4, 8))
POLICY_SEEDS = (300, 301, 302, 303, 304)


def _policy_transforms(train_config, norm_stats):
    """The transform lists of `policy_config.create_trained_policy` (no repack transforms, no default prompt)."""
    from openpi import transforms

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    inputs = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    outputs = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    return inputs, outputs


def _policy_config(args):
    train_config = _train_config(args.config)
    if args.delta is not None:
        train_config = dataclasses.replace(
            train_config, data=dataclasses.replace(train_config.data, use_delta_joint_actions=args.delta)
        )
    return train_config


def _build_policy(args, rtc_config):
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config
    from openpi.training import checkpoints as _checkpoints

    train_config = _policy_config(args)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(args.ckpt) / "assets", args.asset_id)
    if args.backend == "jax" and args.dtype == "float32":
        model = _jax_model(args.config, args.ckpt, "float32")
        inputs, outputs = _policy_transforms(train_config, norm_stats)
        return _policy.Policy(
            model,
            transforms=inputs,
            output_transforms=outputs,
            metadata=train_config.policy_metadata,
            rtc_config=rtc_config,
        )
    if args.backend == "torch":
        mode = None if args.compile == "none" else args.compile
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=mode)
        )
    policy = policy_config.create_trained_policy(
        train_config,
        args.ckpt,
        norm_stats=norm_stats,
        rtc_config=rtc_config,
        pytorch_device=args.device if args.backend == "torch" else None,
    )
    inputs, _ = _policy_transforms(train_config, norm_stats)
    # The float32 JAX policy above is built from `_policy_transforms`; check it mirrors create_trained_policy.
    served_types = [type(t).__name__ for t in policy._input_transforms]  # noqa: SLF001
    assert served_types == [type(t).__name__ for t in inputs], served_types
    if args.backend == "torch" and args.dtype == "float32":
        import torch

        policy._model.to(torch.float32)  # noqa: SLF001  (in place; `_sample_actions` is its bound method)
    return policy


def policy_run(args) -> None:
    from openpi.policies import policy as _policy
    from openpi.policies import teleavatar_v2_policy

    rtc_config = Combo("policy", args.mode.split("_")[0], args.schedule, args.mode == "trained_noise").rtc_config()
    policy = _build_policy(args, rtc_config)
    delta = getattr(_policy_config(args).data, "use_delta_joint_actions", None)
    _log(f"policy built: backend {args.backend} {args.dtype} delta {delta} rtc {rtc_config}")

    calls = []
    sample_actions = policy._sample_actions  # noqa: SLF001

    def spy(*a, **kw):
        calls.append(
            {
                "prev_chunk": "prev_chunk" in kw,
                "inference_delay": int(kw["inference_delay"]) if "inference_delay" in kw else None,
                "execution_horizon": int(kw["execution_horizon"]) if "execution_horizon" in kw else None,
            }
        )
        return sample_actions(*a, **kw)

    policy._sample_actions = spy  # noqa: SLF001
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logging.getLogger(_policy.__name__).addHandler(_Capture(level=logging.WARNING))
    rng = np.random.default_rng(args.noise_seed)
    raw_chunks, served, states, ids, horizons, warnings = [], [], [], [], [], []
    chunk_id = 0
    for k, (seed, request) in enumerate(zip(POLICY_SEEDS, POLICY_REQUESTS, strict=True)):
        np.random.seed(seed)
        obs = teleavatar_v2_policy.make_teleavatar_example()
        if request is not None:
            prefix_start, delay, horizon = request
            obs = {
                **obs,
                _policy.RTC_PREV_CHUNK_ID: chunk_id,
                _policy.RTC_PREFIX_START: prefix_start,
                _policy.RTC_INFERENCE_DELAY: delay,
                _policy.RTC_EXECUTION_HORIZON: horizon,
            }
        n_records = len(records)
        result = policy.infer(obs, noise=rng.standard_normal((ACTION_HORIZON, ACTION_DIM), dtype=np.float32))
        chunk_id = int(result[_policy.RTC_CHUNK_ID])
        ids.append(chunk_id)
        horizons.append(int(result[_policy.RTC_ACTION_HORIZON]))
        raw_chunks.append(np.asarray(policy._rtc_prev_chunk, dtype=np.float32))  # noqa: SLF001
        served.append(np.asarray(result["actions"], dtype=np.float64))
        states.append(np.asarray(obs["observation/state"], dtype=np.float64))
        warnings.append([r.getMessage() for r in records[n_records:]])
        _log(f"request {k}: {request} -> chunk {chunk_id}, sample_actions {calls[-1]}, warnings {warnings[-1]}")
    np.savez(args.out, raw=np.stack(raw_chunks), served=np.stack(served), states=np.stack(states))
    _write_json(
        args.out + ".json",
        {
            "argv": sys.argv,
            "backend": args.backend,
            "dtype": args.dtype,
            "mode": args.mode,
            "schedule": args.schedule,
            "delta": delta,
            "requests": list(POLICY_REQUESTS),
            "seeds": list(POLICY_SEEDS),
            "noise_seed": args.noise_seed,
            "chunk_ids": ids,
            "action_horizons": horizons,
            "sample_actions_calls": calls,
            "warnings": warnings,
        },
    )
    _log(f"wrote {args.out}")


def _policy_structure(run: dict, served: np.ndarray, mode: str) -> list[tuple[str, bool, str]]:
    checks = []
    n = len(POLICY_REQUESTS)
    checks.append(("chunk ids 1..n", run["chunk_ids"] == list(range(1, n + 1)), str(run["chunk_ids"])))
    checks.append(
        (
            "rtc/action_horizon == 30",
            all(h == ACTION_HORIZON for h in run["action_horizons"]),
            str(run["action_horizons"]),
        )
    )
    for k, request in enumerate(POLICY_REQUESTS):
        call, warns = run["sample_actions_calls"][k], run["warnings"][k]
        if request is None:
            checks.append((f"request {k}: unconstrained", not call["prev_chunk"], str(call)))
        elif request[0] >= ACTION_HORIZON:
            checks.append(
                (f"request {k}: prefix_start {request[0]} >= 30 -> unconstrained", not call["prev_chunk"], str(call))
            )
            checks.append((f"request {k}: warning logged", any("fully consumed" in w for w in warns), str(warns)))
        else:
            ok = (
                call["prev_chunk"] and call["inference_delay"] == request[1] and call["execution_horizon"] == request[2]
            )
            checks.append((f"request {k}: constrained with delay {request[1]}, horizon {request[2]}", ok, str(call)))
            if mode.startswith("trained"):
                # The pinned steps re-enter physical space as the previous chunk's, after any re-anchoring.
                prefix_start, delay, _ = request
                new, old = served[k][:delay], served[k - 1][prefix_start : prefix_start + delay]
                err = np.abs(new - old) - (1e-3 * np.abs(old) + 1e-4)
                checks.append(
                    (
                        f"request {k}: served prefix == previous chunk (rtol 1e-3, atol 1e-4)",
                        bool((err <= 0).all()),
                        f"max |d| {np.abs(new - old).max():.3g}",
                    )
                )
    return checks


def policy_compare(args) -> None:
    runs = []
    for path in (args.a, args.b):
        data = np.load(path)
        runs.append((dict(data), json.loads(pathlib.Path(path + ".json").read_text())))
    (a, meta_a), (b, meta_b) = runs
    ok = True
    for label, (data, meta) in zip((args.a, args.b), runs, strict=True):
        for name, passed, detail in _policy_structure(meta, data["served"], meta["mode"]):
            ok &= passed
            print(f"{'ok  ' if passed else 'FAIL'} {label}: {name} {detail}")
    assert meta_a["requests"] == meta_b["requests"]
    assert meta_a["seeds"] == meta_b["seeds"]
    for k in range(len(POLICY_REQUESTS)):
        x, y = a["raw"][k][:, :VALID_DIMS].astype(np.float64), b["raw"][k][:, :VALID_DIMS].astype(np.float64)
        cos = float(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x) * np.linalg.norm(y)))
        passed = cos >= args.min_cos
        ok &= passed
        served = np.abs(a["served"][k] - b["served"][k])
        joints, grip = list(GROUPS["J"]), list(GROUPS["G"])
        print(
            f"{'ok  ' if passed else 'FAIL'} chunk {k + 1}: raw cos {cos:.6f} (>= {args.min_cos}), raw max|d| {np.abs(x - y).max():.3g}; "
            f"served joints max {served[:, joints].max():.3g} rad, grippers max {served[:, grip].max():.3g} Nm"
        )
    print(f"OVERALL {'PASS' if ok else 'FAIL'}")


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract-real")
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=extract_real)

    p = sub.add_parser("make-inputs")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="checkpoint dir holding assets/<asset-id>/norm_stats.json")
    p.add_argument("--asset-id", required=True)
    p.add_argument("--seeds", default="100-115", help="make_teleavatar_example seeds, e.g. 100-115")
    p.add_argument("--real", nargs="*", help="extract-real outputs, appended after the synthetic samples")
    p.add_argument("--noise-seed", type=int, default=4)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=make_inputs)

    p = sub.add_parser("jax-prev")
    for name in ("--config", "--ckpt", "--inputs", "--out"):
        p.add_argument(name, required=True)
    p.set_defaults(fn=jax_prev)

    for name, fn in (("run-jax", run_jax), ("run-torch", run_torch)):
        p = sub.add_parser(name)
        for flag in ("--config", "--ckpt", "--inputs", "--prev", "--out"):
            p.add_argument(flag, required=True)
        p.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
        p.add_argument("--combos", default="all", help="`all` or comma-separated name prefixes, e.g. none,trained")
        if name == "run-torch":
            p.add_argument("--device", default="cpu")
            p.add_argument("--compile", default="none", help="none, or a torch.compile mode such as default")
            p.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0: leave the default)")
            p.add_argument(
                "--fix-rope", action="store_true", help="recompute RoPE inv_freq in float32 (code before 7ec5f8f)"
            )
        p.set_defaults(fn=fn)

    p = sub.add_parser("calibrate")
    p.add_argument("--ref", required=True, help="JAX float32")
    p.add_argument("--alt", required=True, help="JAX float32 with another accumulation order")
    p.add_argument("--draws", nargs="+", required=True, help="JAX bfloat16 draws, accumulation order varied")
    p.add_argument("--synthetic", type=int, default=16, help="the first N samples form the calibration pool S")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=calibrate)

    p = sub.add_parser("evaluate")
    p.add_argument("--criteria", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--prev", required=True)
    p.add_argument("--cand", nargs="+", required=True)
    p.add_argument("--kind", choices=("fp32", "bf16", "jax"), required=True)
    p.add_argument("--norm-stats", help="norm_stats.json, to report errors in rad / Nm")
    p.add_argument("--out")
    p.set_defaults(fn=evaluate)

    p = sub.add_parser("compare-pair")
    p.add_argument("--criteria", required=True)
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=compare_pair)

    p = sub.add_parser("policy-run")
    p.add_argument("--backend", choices=("jax", "torch"), required=True)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--asset-id", required=True)
    p.add_argument("--mode", choices=("guided", "trained", "trained_noise"), required=True)
    p.add_argument("--schedule", default="exp")
    p.add_argument(
        "--delta",
        action=argparse.BooleanOptionalAction,
        help="override use_delta_joint_actions of the config (default: the config's own; both v2 configs use delta)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--compile", default="none")
    p.add_argument("--noise-seed", type=int, default=7)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=policy_run)

    p = sub.add_parser("policy-compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--min-cos", type=float, default=0.99)
    p.set_defaults(fn=policy_compare)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _log(f"{args.cmd}: {' '.join(sys.argv[1:])}")
    args.fn(args)


if __name__ == "__main__":
    main()
