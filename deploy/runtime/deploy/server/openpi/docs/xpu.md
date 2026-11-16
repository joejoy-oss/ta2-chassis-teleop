# Inference on Intel GPUs (XPU)

The PyTorch model serves on Intel Arc Pro B70 GPUs (Xe2 / Battlemage), with or without
[Real-Time Chunking](real_time_chunking.md). Only inference runs on XPU; training stays on the CUDA / JAX
setup. The checkpoints below are `pi05_teleavatar_v2` at step 19999 and `pi05_teleavatar_v2_rtc` at step 7499
("19999" and "7499"). Unless noted otherwise, everything was measured on a server with two B70 cards and a Xeon
Gold 5418N; what is noted ran on the CPU of a development machine.

## Environment

Use the `openpi-xpu-env` image (repository `openpi-xpu-env`, tag `v0.1.0`). It holds the environment only:
Python 3.11.17, torch 2.12.1+xpu, triton-xpu 3.7.1, the Intel GPU user-mode driver (compute-runtime
26.09.37435.12, IGC 2.30.3, Level Zero 1.28.0) and the compiler toolchain `torch.compile` needs. This
repository is mounted at `/app`. That repository's README covers loading the image, `scripts/run.sh`, the
compile cache, picking a card and CPU pinning. The commands below are run from its root, and all of them except
the `ln -s` start a container with `scripts/run.sh`. With `XPU=0 GPUS=none` the container gets no GPU and
prints a `NOTE` that `/dev/dri` is missing, which is expected there.

The XPU dependency set is pinned in `openpi-xpu-env/deps`, not in this repository's `pyproject.toml`.

## Converting a checkpoint

XPU serving needs a PyTorch checkpoint. Convert the JAX checkpoint once, in bf16, on CPU. Mount the step
directory, the one holding `params/` and `assets/`; the output goes to `OUTPUT_DIR`:

```bash
OPENPI_DIR=/path/to/openpi XPU=0 GPUS=none OUTPUT_DIR=/path/to \
  DOCKER_RUN_EXTRA="-e JAX_PLATFORMS=cpu -v /path/to/jax_checkpoint/19999:/jax:ro" \
  scripts/run.sh python examples/convert_jax_model_to_pytorch.py --checkpoint-dir /jax \
  --config-name pi05_teleavatar_v2 --output-path /outputs/pi05_teleavatar_v2_pytorch --precision bfloat16
```

It needs about 32 GB of memory. In this image on the B70 server, converting 7499 took 63 s, with up to 29 GiB of
anonymous memory (sampled once a second); on the development machine, in the CUDA training image
(`IMAGE=openpi-env TAG=v0.1.0`), converting 19999 took 105 s with RAM plus swap at about 32 GB. `--inspect-only`
instead of `--output-path` only lists the checkpoint's parameters.

The script copies the checkpoint's `assets/` (the norm stats) next to the weights. The teleavatar v2 configs
use the placeholder `repo_id="path-to-dataset"`, so the policy looks for `assets/path-to-dataset/`; link it to
the asset directory the checkpoint came with (the container runs as root, so the output belongs to root):

```bash
ln -s align5_vertical_tilted_slow_45fps /path/to/pi05_teleavatar_v2_pytorch/assets/path-to-dataset
```

Spot checks on the development machine found the converted weights bit-identical to the JAX parameters cast to
bf16, and two conversions of 7499, on the development machine and on the B70, are identical tensor by tensor.
The exception is `gemma_expert.lm_head.weight`: it is unused and not loaded from the JAX checkpoint, so it keeps
its random initialization and the file hash differs between two conversions; compare conversions tensor by tensor
without it.

## Serving

```bash
OPENPI_DIR=/path/to/openpi DOCKER_RUN_EXTRA="--cpuset-cpus=3,27 -p 8000:8000 -v /path/to/pi05_teleavatar_v2_pytorch:/ckpt:ro" \
  scripts/run.sh python scripts/serve_policy.py --env TELEAVATAR --warmup \
  policy:checkpoint --policy.config pi05_teleavatar_v2 --policy.dir /ckpt \
  --policy.pytorch-device xpu --policy.pytorch-compile-mode default
```

- `--policy.pytorch-device xpu`: without it the first available of cuda, xpu and cpu is used.
- `--policy.pytorch-compile-mode default`: the config's `max-autotune` compiles for 4 to 7 minutes on the B70
  with no consistent latency gain. `default` compiles in 65 to 101 s from an empty cache and warms up in about
  22 s from a full one; `none` runs eagerly.
- `--warmup` compiles before the server starts listening, so the first client request does not wait. It runs
  the example observation of `--env`, which has to match the config: `TELEAVATAR` here.
- The PaliGemma tokenizer is read from `OPENPI_DATA_HOME` (default `~/.cache/openpi`), and downloaded there
  from Google Cloud Storage the first time. Without internet access, copy `big_vision/paligemma_tokenizer.model`
  into it first.
- `--cpuset-cpus=3,27` pins the server to the two hyperthreads of one physical core (`lscpu -e=CPU,CORE`).
  `guided` needs it on a host with a power-saving CPU governor: unpinned, its slowest request was 492 ms.
- `pi05_teleavatar_v2` and `pi05_teleavatar_v2_rtc` predict the arm joints relative to the current state
  (`use_delta_joint_actions=True`, grippers absolute). A checkpoint has to be served with the setting it was
  trained with.

Real-Time Chunking (the flags go before `policy:checkpoint`):

- `guided`: add `--rtc.enabled --rtc.mode GUIDED`. Works with any checkpoint, about 1.7x the latency.
- `trained`: add `--rtc.enabled --rtc.mode TRAINED` and serve a checkpoint trained with
  `rtc_training_max_delay > 0`, such as `pi05_teleavatar_v2_rtc` (`--policy.config pi05_teleavatar_v2_rtc`).
  Do not add `--rtc.trained-prefix-noise` for such a checkpoint, see [Limitations](real_time_chunking.md#limitations).

The robot client is unchanged: `examples/teleavatar_v2/main.py`, with `--rtc` for an RTC server.

## Checking end to end without the robot

`examples/teleavatar_v2/fake_env_rtc.py` runs `main.py`'s client loop against a stand-in robot whose joints
follow the commands exactly. It needs no GPU. Against the server above:

```bash
OPENPI_DIR=/path/to/openpi XPU=0 GPUS=none DOCKER_RUN_EXTRA="--network host" \
  scripts/run.sh python examples/teleavatar_v2/fake_env_rtc.py --rtc --num-episodes 1 --record /outputs/check.json
```

Drop `--rtc` for a server without `--rtc.enabled`. The client logs `RTC calibrated: inference_delay=...` once,
and a warning for every chunk that arrives later than the calibrated delay; `--record` writes every executed
action with the request it came from. Without `--observation`, every camera sends random pixels.

The [end-to-end table](#end-to-end) was run with a recorded frame, the default two episodes of 600 steps, and the
client pinned to other cores than the server:

```bash
OPENPI_DIR=/path/to/openpi XPU=0 GPUS=none \
  DOCKER_RUN_EXTRA="--network host --cpuset-cpus=4,5,28,29 -v /path/to/frame.npz:/frame.npz:ro" \
  scripts/run.sh python examples/teleavatar_v2/fake_env_rtc.py --rtc --observation /frame.npz --record /outputs/e2e.json
```

The frame is frame 831, the middle one, of episode 0 of the LeRobot dataset `align5_rec_tilted_45fps`, which
comes from the same source as the training data. This cuts it out of the dataset the way the robot client sends
it, into `OUTPUT_DIR/frame.npz`. It runs in the CUDA training image, without a GPU: `extract-real` decodes the
videos with LeRobot's `pyav` backend, which needs `torchvision.io.VideoReader`, and the torchvision of the XPU
image (0.27.1) does not have it.

```bash
OPENPI_DIR=/path/to/openpi IMAGE=openpi-env TAG=v0.1.0 XPU=0 GPUS=none OUTPUT_DIR=/path/to \
  DOCKER_RUN_EXTRA="-v /path/to/align5_rec_tilted_45fps:/data/align5_rec_tilted_45fps:ro" \
  scripts/run.sh python scripts/rtc_numerics.py extract-real --dataset /data/align5_rec_tilted_45fps \
  --episode 0 --frame 831 --out /outputs/frame.npz
```

Two frames were tried, the middle frames of episode 0 of `align5_rec_vertical_45fps` and
`align5_rec_tilted_45fps`; this is the one whose unconstrained first chunk moves the arms more (median step 0.0181
against 0.0015 rad, JAX, 19999). With random pixels instead, the control loop row came out the same in a check,
the arm and gripper rows did not.

## Tests

All tests run on CPU. The suite was run in this image (torch 2.12.1) and in the CUDA training image (torch
2.7.1), with one pytest process per test file:

```bash
OPENPI_DIR=/path/to/openpi XPU=0 GPUS=none DOCKER_RUN_EXTRA="-e JAX_PLATFORMS=cpu" scripts/run.sh bash -c '
for f in $(python -m pytest -p no:cacheprovider -q -m "not manual" --collect-only 2>/dev/null | grep "::" | cut -d: -f1 | uniq); do
    python -m pytest -p no:cacheprovider -q -m "not manual" "$f"
done'
```

`pytest` collects `src`, `scripts`, `packages` and `examples/teleavatar_v2`.

- The `manual` tests download checkpoints. Five others need network access: `test_fast_tokenizer` (Hugging Face
  hub), the three `gs://` tests in `download_test.py`, and `test_with_real_dataset` (a LeRobot dataset, unless it is
  cached). With `scripts/run.sh`, `PROXY=1` passes the host's proxy into the container.
- `scripts/train_test.py` needs more than 24 GiB of memory: capped at 24 GiB it was killed, capped at 64 GiB it
  passed, with a cgroup peak of 38 GiB that includes the page cache. On a 30 GB development machine the suite ran
  out of memory in a single pytest process; one process per test file got through all of it except
  `train_test.py`.

## Measurements

### Numerics against JAX

Same bf16 weights on both sides, same noise, actions compared in the normalized space the model predicts in.
JAX ran on the CPU of the development machine, PyTorch on that CPU and on the B70.

| | 19999 | 7499 |
| --- | --- | --- |
| PyTorch CPU fp32 vs JAX fp32, max abs difference over all modes run (cosine 1.0) | 3.2e-5 | 9.8e-5 |
| `trained`: pinned prefix equals the previous chunk | bit-exact | bit-exact |
| `guided`: weighted distance to the previous chunk, guided / unguided (has to be < 0.5) | 0.136 to 0.355 | no guided runs |
| Without an RTC config, output against the code before the RTC port | identical | identical |
| Changing `inference_delay` / `execution_horizon` recompiles | no | no |

The guided ratios are JAX's; PyTorch matches them to 3 decimals. XPU bf16 against JAX fp32 passes a criterion
calibrated on how far JAX's own bf16 results are from its fp32 ones. XPU compiled against XPU eager passes a
criterion calibrated on the spread between JAX's own bf16 runs, pooled over all combinations of mode, delay and
horizon, but not on every combination alone.

### Latency

Robot-resolution requests (single eyes, head 960x960, wrists 400x640, as `examples/teleavatar_v2/main.py`
sends them), client and server in one process over loopback, `default` compile mode, warm cache, server pinned
to one physical core; client round trip in ms:

| RTC | p50 | p95 | max |
| --- | --- | --- | --- |
| none | 136.1 | 140.0 | 140.6 |
| `guided` | 233.4 | 237.3 | 240.1 |
| `trained` | 140.6 | 144.7 | 150.0 |
| `trained` + prefix noise | 138.5 | 142.9 | 151.8 |

A real network adds at least 34.4 ms per request on 1 GbE for the 4.3 MB of images (3.4 ms on 10 GbE).
Raw stereo frames at the resolution of the training videos are 34 MB per request: with them, `guided` calibrated
an `inference_delay` of 18 to 20 steps and its action queue ran empty about 70 times in two episodes.

### End to end

`fake_env_rtc.py` at 45 Hz on the recorded frame above, at robot resolution, 2 episodes x 600 steps per column,
server pinned to CPUs 3 and 27, client to CPUs 4, 5, 28 and 29, delta actions. A stall is a step longer than two
control periods (44.4 ms):

| | no RTC (19999) | `guided` (19999) | `trained` (7499) | `trained` + prefix noise (7499) |
| --- | --- | --- | --- | --- |
| control loop | 32.7 Hz, 37 stalls per episode of about 180 ms | 44.87 Hz, no stall, longest step 34 ms | 44.87 Hz, no stall, longest step 38 ms | 44.87 Hz, no stall, longest step 40 ms |
| client round trip in the episodes, p50 / p95 / max (ms) | 156.4 / 159.2 / 161.1 | 241.1 / 246.1 / 257.7 | 147.8 / 153.2 / 175.4 | 146.6 / 151.4 / 158.6 |
| calibrated `inference_delay` / `execution_horizon` | | 13 / 26 | 9 / 18 | 8 / 16 |
| chunks arriving late | | 0 | 0 | 0 |
| arm joints, median step at a splice / inside a chunk (rad) | 0.0163 / 0.0048 | 0.0191 / 0.0065 | 0.0047 / 0.0047 | 0.0051 / 0.0050 |
| arm joints, median step into row `inference_delay` (rad) | | 0.0041 | 0.0051 | 0.0353 |
| left gripper torque, median step at a splice / inside a chunk (N·m) | 0.611 / 0.0075 (82x) | 0.0082 / 0.0021 (3.9x) | 0.0010 / 0.0013 (0.76x) | 0.0015 / 0.0018 (0.85x) |

Three more `guided` runs gave arm splice / inside-chunk ratios of 3.04, 3.08 (a single episode) and 2.86, against
2.92 in the table. The last of them had one stall, a step of 46 ms: the only one in about 4,200 `guided` steps on
this frame at robot resolution. The table's runs and the first two repeats reached the server through its
container's network namespace (`--network container:<server>`); the last one ran the
[table's command](#checking-end-to-end-without-the-robot), with `--network host`.

The stand-in's joints follow the commands exactly, so a new delta chunk starts from where the old one left the
arm even without RTC; a real arm lags its commands. The arm-joint rows are therefore not a prediction for the
robot. The control loop and the grippers, which are absolute, show what RTC changes here.

## Open points for maintainers

- The bf16 rounding fix for the rotary frequencies only covers `to_bfloat16_for_selected_params`. A plain
  `model.to(torch.bfloat16)` still rounds them; overriding `_apply` on `GemmaRotaryEmbedding` in
  `transformers_replace` would close that. Upstream openpi has the same code.
- `gemma_expert.lm_head.weight` could be fixed or left out of the converted file, so that conversions can be
  compared by file hash.
- The PyTorch time embedding is computed in float64, JAX's in float32 (8.2e-5 apart at dimension 64), one of the
  known sources of the remaining difference.
