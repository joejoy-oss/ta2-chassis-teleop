"""Real-Time Chunking on `PI0Pytorch`: the cases of `openpi.models.rtc_test` that concern
inference, plus parity with the JAX `Pi0` on identical weights.

All on CPU in float32, with dummy-sized Gemmas and the compiler off (`pytorch_compile_mode=None`)
except where recompilation itself is under test.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch._dynamo.testing

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import rtc
from openpi.models_pytorch import jax_conversion
from openpi.models_pytorch import rtc_pytorch
from openpi.shared import nnx_utils

ACTION_HORIZON = 12
ACTION_DIM = 8
BATCH = 2
NUM_STEPS = 4


def _make_models(*, pi05: bool, perturbation: float = 0.15, seed: int = 0):
    """A JAX `Pi0` with every parameter perturbed as in `openpi.models.rtc_test`, and a `PI0Pytorch`
    with the same weights."""
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        pi05=pi05,
        pytorch_compile_mode=None,
    )
    model = config.create(jax.random.key(seed))
    # The adaRMS modulation is zero-initialised, gating every block to the identity.
    graphdef, state = nnx.split(model)
    keys = iter(jax.random.split(jax.random.key(seed + 100), len(jax.tree.leaves(state))))
    state = jax.tree.map(lambda x: x + perturbation * jax.random.normal(next(keys), x.shape, x.dtype), state)
    jax_model = nnx.merge(graphdef, state)
    torch_model = jax_conversion.pytorch_model_from_jax(jax_model, config).requires_grad_(requires_grad=False)
    return config, jax_model, torch_model


@pytest.fixture(scope="module")
def models():
    # Each pair holds a full-size SigLIP twice; build them once for the module.
    return {pi05: _make_models(pi05=pi05) for pi05 in (False, True)}


@pytest.fixture(scope="module")
def parity_models():
    """Like `models`, but perturbed by 0.02 instead of 0.15.

    0.15 suits the behavioural tests (it makes the zero-initialised adaRMS gates non-trivial), but
    it leaves the network so badly conditioned that the backends disagree beyond the tolerances
    even without RTC: SigLIP's residual stream reaches ~1e3 and float32 rounding grows ~1.3x per
    block (each backend is ~3% off a float64 run after 27 blocks), and the sinusoidal time
    embedding, float64 in PyTorch and float32 in JAX (~8e-5 apart), is amplified as well. At 0.02
    the backends agree to rounding noise, while mistakes in the RTC logic still move the output by
    ~1e-3 or more. Pairs are built lazily, one at a time, to bound memory.
    """
    cache = {}

    def get(*, pi05: bool):
        if pi05 not in cache:
            cache.clear()
            cache[pi05] = _make_models(pi05=pi05, perturbation=0.02)
        return cache[pi05]

    yield get
    cache.clear()


def _inputs(config: pi0_config.Pi0Config, *, batch: int = BATCH, seed: int = 0) -> dict[str, np.ndarray]:
    """Random observation, noise and previous chunk, with some image and token padding."""
    rng = np.random.default_rng(seed)
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    image_mask = {k: np.ones(batch, bool) for k in keys}
    image_mask["right_wrist_0_rgb"][-1] = False
    token_len = np.linspace(config.max_token_len, config.max_token_len // 2, batch).astype(int)
    return {
        "image": {k: rng.uniform(-1, 1, (batch, *_model.IMAGE_RESOLUTION, 3)).astype(np.float32) for k in keys},
        "image_mask": image_mask,
        "state": rng.normal(size=(batch, config.action_dim)).astype(np.float32),
        "tokenized_prompt": rng.integers(0, 257_152, (batch, config.max_token_len)).astype(np.int32),
        "tokenized_prompt_mask": np.arange(config.max_token_len)[None, :] < token_len[:, None],
        "noise": rng.normal(size=(batch, config.action_horizon, config.action_dim)).astype(np.float32),
        "prev_chunk": rng.normal(size=(batch, config.action_horizon, config.action_dim)).astype(np.float32),
    }


def _torch_args(inputs: dict) -> tuple[_model.Observation, torch.Tensor, torch.Tensor]:
    obs = {k: v for k, v in inputs.items() if k not in ("noise", "prev_chunk")}
    obs = jax.tree.map(torch.from_numpy, obs)
    # The PyTorch model takes channels-first images.
    obs["image"] = {k: v.permute(0, 3, 1, 2) for k, v in obs["image"].items()}
    return (
        _model.Observation.from_dict(obs),
        torch.from_numpy(inputs["noise"]),
        torch.from_numpy(inputs["prev_chunk"]),
    )


def _jax_args(inputs: dict) -> tuple[_model.Observation, jax.Array, jax.Array]:
    obs = {k: v for k, v in inputs.items() if k not in ("noise", "prev_chunk")}
    return (
        _model.Observation.from_dict(jax.tree.map(jnp.asarray, obs)),
        jnp.asarray(inputs["noise"]),
        jnp.asarray(inputs["prev_chunk"]),
    )


# --- Prefix weights and guidance weight -------------------------------------------------


def test_prefix_weights_linear_matches_reference_ramp():
    # linspace(1, 0, n + 2)[1:-1]: the transition excludes both endpoints.
    weights = rtc_pytorch.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.LINEAR)
    torch.testing.assert_close(weights[:3], torch.ones(3))
    torch.testing.assert_close(weights[3:7], torch.tensor([0.8, 0.6, 0.4, 0.2]), atol=1e-6, rtol=0)
    torch.testing.assert_close(weights[7:], torch.zeros(5))
    torch.testing.assert_close(weights[3:7], torch.linspace(1, 0, 4 + 2)[1:-1], atol=1e-6, rtol=0)


def test_prefix_weights_hard_schedules():
    ones = rtc_pytorch.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.ONES)
    torch.testing.assert_close(ones, torch.tensor([1.0] * 7 + [0.0] * 5))
    zeros = rtc_pytorch.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.ZEROS)
    torch.testing.assert_close(zeros, torch.tensor([1.0] * 3 + [0.0] * 9))


def test_prefix_weights_exp_decays_faster_but_pins_the_endpoints():
    linear = rtc_pytorch.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.LINEAR)
    exp = rtc_pytorch.get_prefix_weights(3, 7, 12, rtc.PrefixAttentionSchedule.EXP)
    torch.testing.assert_close(exp[:3], torch.ones(3))
    torch.testing.assert_close(exp[7:], torch.zeros(5))
    assert torch.all(exp[3:7] < linear[3:7])


@pytest.mark.parametrize(("start", "end"), [(0, 0), (5, 5), (12, 12), (9, 4), (20, 30)])
def test_prefix_weights_degenerate_ranges_stay_in_bounds(start, end):
    weights = rtc_pytorch.get_prefix_weights(start, end, 12, rtc.PrefixAttentionSchedule.LINEAR)
    assert weights.shape == (12,)
    assert torch.all((weights >= 0.0) & (weights <= 1.0))
    assert torch.all(torch.diff(weights) <= 1e-6)  # non-increasing


def test_guidance_weight_is_symmetric_and_clipped():
    # Invariant under t <-> 1-t, bottoms out at 2, diverges at both ends.
    assert float(rtc_pytorch.guidance_weight(0.5, 10.0)) == pytest.approx(2.0)
    assert float(rtc_pytorch.guidance_weight(0.3, 100.0)) == pytest.approx(
        float(rtc_pytorch.guidance_weight(0.7, 100.0))
    )
    assert float(rtc_pytorch.guidance_weight(1.0, 10.0)) == pytest.approx(10.0)
    assert float(rtc_pytorch.guidance_weight(0.001, 10.0)) == pytest.approx(10.0)


@pytest.mark.parametrize("schedule", list(rtc.PrefixAttentionSchedule))
@pytest.mark.parametrize(
    ("start", "end"), [(0, 0), (0, 12), (3, 7), (4, 8), (5, 5), (9, 4), (9, 18), (12, 12), (20, 30), (-3, 4)]
)
def test_prefix_weights_match_jax(schedule, start, end):
    expected = np.asarray(rtc.get_prefix_weights(start, end, 12, schedule))
    # Ints and 0-d tensors (what `PI0Pytorch.sample_actions` receives from the policy) alike.
    for args in ((start, end), (torch.tensor(start), torch.tensor(end))):
        weights = rtc_pytorch.get_prefix_weights(*args, 12, schedule)
        assert weights.dtype == torch.float32
        np.testing.assert_allclose(weights.numpy(), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("max_guidance_weight", [0.0, 2.5, 10.0, 1e3])
def test_guidance_weight_matches_jax(max_guidance_weight):
    for t in np.linspace(0.0, 1.0, 41, dtype=np.float32):
        expected = float(rtc.guidance_weight(float(t), max_guidance_weight))
        actual = float(rtc_pytorch.guidance_weight(torch.tensor(t), max_guidance_weight))
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6)


# --- Sampling ------------------------------------------------------------------------------


@pytest.mark.parametrize("pi05", [False, True])
def test_sampling_without_rtc_is_unchanged(models, pi05):
    config, _, model = models[pi05]
    obs, noise, prev = _torch_args(_inputs(config))

    baseline = model.sample_actions("cpu", obs, noise=noise, num_steps=NUM_STEPS)
    # A previous chunk without a config must not engage RTC, nor a config without a previous
    # chunk (the first chunk of an episode).
    assert torch.equal(baseline, model.sample_actions("cpu", obs, noise=noise, num_steps=NUM_STEPS, prev_chunk=prev))
    assert torch.equal(
        baseline, model.sample_actions("cpu", obs, noise=noise, num_steps=NUM_STEPS, rtc_config=rtc.RTCConfig())
    )


@pytest.mark.parametrize("pi05", [False, True])
def test_guided_pulls_the_chunk_towards_the_previous_one(models, pi05):
    config, _, model = models[pi05]
    obs, noise, prev = _torch_args(_inputs(config))

    def distance(max_guidance_weight):
        # ONES over the full horizon isolates guidance from the weight schedule.
        config_rtc = rtc.RTCConfig(
            mode=rtc.RTCMode.GUIDED,
            prefix_attention_schedule=rtc.PrefixAttentionSchedule.ONES,
            max_guidance_weight=max_guidance_weight,
        )
        actions = model.sample_actions(
            "cpu",
            obs,
            noise=noise,
            num_steps=NUM_STEPS,
            prev_chunk=prev,
            inference_delay=0,
            execution_horizon=ACTION_HORIZON,
            rtc_config=config_rtc,
        )
        return float((actions - prev).abs().mean())

    unguided, guided = distance(0.0), distance(10.0)
    assert guided < 0.5 * unguided, f"guidance did not pull the chunk closer: {unguided=} {guided=}"


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("trained_prefix_noise", [False, True])
def test_trained_mode_pins_the_prefix_exactly(models, pi05, trained_prefix_noise):
    config, _, model = models[pi05]
    obs, noise, prev = _torch_args(_inputs(config))
    delay = 4

    actions = model.sample_actions(
        "cpu",
        obs,
        noise=noise,
        num_steps=NUM_STEPS,
        prev_chunk=prev,
        inference_delay=torch.tensor(delay),
        execution_horizon=torch.tensor(8),
        rtc_config=rtc.RTCConfig(mode=rtc.RTCMode.TRAINED, trained_prefix_noise=trained_prefix_noise),
    )
    assert torch.equal(actions[:, :delay], prev[:, :delay])
    assert not torch.equal(actions[:, delay:], prev[:, delay:])


def _compiled_frame_counts(model, obs, noise, requests) -> list[int]:
    """Run `requests` (sample_actions kwargs) with both forward passes compiled, returning the number
    of frames dynamo has compiled after each one. The counting backend runs the graphs eagerly."""
    # The model compiles its two forward passes (prefix KV cache, denoising step), never the
    # sampling loop, so the per-request delay and horizon cannot reach a compiled graph.
    torch._dynamo.reset()  # noqa: SLF001
    counter = torch._dynamo.testing.CompileCounter()  # noqa: SLF001
    model._prefix_kv_fn = torch.compile(model._prefix_kv, backend=counter)  # noqa: SLF001
    model._denoise_fn = torch.compile(model.denoise_step, backend=counter)  # noqa: SLF001
    frame_counts = []
    try:
        for kwargs in requests:
            model.sample_actions("cpu", obs, noise=noise, num_steps=2, **kwargs)
            frame_counts.append(counter.frame_count)
    finally:
        model._prefix_kv_fn, model._denoise_fn = model._prefix_kv, model.denoise_step  # noqa: SLF001
        torch._dynamo.reset()  # noqa: SLF001
    print(f"compiled frames after each request: {frame_counts}")
    return frame_counts


def _rtc_request(prev, delay, horizon, **config) -> dict:
    return {
        "prev_chunk": prev,
        "inference_delay": torch.tensor(delay),
        "execution_horizon": torch.tensor(horizon),
        "rtc_config": rtc.RTCConfig(**config),
    }


def test_changing_delay_and_horizon_does_not_recompile(models):
    config, _, model = models[True]
    obs, noise, prev = _torch_args(_inputs(config, batch=1))

    frame_counts = _compiled_frame_counts(
        model, obs, noise, [_rtc_request(prev, d, h) for d, h in [(1, 5), (3, 9), (6, 11), (2, 7)]]
    )
    # One frame per compiled forward pass (no graph breaks), all of them during the first request:
    # a delay that changes every step never retriggers compilation.
    assert frame_counts == [2] * len(frame_counts), frame_counts


@pytest.mark.parametrize("pi05", [False, True])
def test_serving_compiles_one_denoising_graph_per_variant(models, pi05):
    config, _, model = models[pi05]
    obs, noise, prev = _torch_args(_inputs(config, batch=1))

    frame_counts = _compiled_frame_counts(
        model,
        obs,
        noise,
        [
            # An episode's first chunk has nothing to agree with: unconstrained.
            {"rtc_config": rtc.RTCConfig()},
            _rtc_request(prev, 1, 5),
            _rtc_request(prev, 3, 9),
            _rtc_request(prev, 1, 5, mode=rtc.RTCMode.TRAINED),
            _rtc_request(prev, 3, 9, mode=rtc.RTCMode.TRAINED),
            _rtc_request(prev, 3, 9, mode=rtc.RTCMode.TRAINED, trained_prefix_noise=True),
        ],
    )
    # The prefix pass compiles once. The denoising step compiles once per variant: without
    # gradients, with gradients (guided), and with per-token timesteps (trained). Re-noised
    # trained prefixes keep one timestep per sample and reuse the first graph. A server warms up
    # by sending one request of each variant it will see.
    assert frame_counts == [2, 3, 3, 4, 4, 4], frame_counts


# --- Parity with the JAX model -------------------------------------------------------------

_PARITY_CASES = {
    # name: (rtc_config, inference_delay, execution_horizon, max |torch - jax| tolerance)
    "no_rtc": (None, 0, None, 1e-4),
    "guided_exp": (rtc.RTCConfig(mode=rtc.RTCMode.GUIDED), 3, 9, 5e-4),
    "trained": (rtc.RTCConfig(mode=rtc.RTCMode.TRAINED), 4, 8, 1e-4),
    "trained_noise": (rtc.RTCConfig(mode=rtc.RTCMode.TRAINED, trained_prefix_noise=True), 4, 8, 1e-4),
    # Out of range: both backends clip the delay and the horizon to the action horizon. A delay that reaches the
    # end guides (or pins) the whole chunk, which makes the horizon irrelevant, so the horizon is checked separately
    # with a delay inside the chunk.
    "guided_overflow": (rtc.RTCConfig(mode=rtc.RTCMode.GUIDED), ACTION_HORIZON, ACTION_HORIZON + 5, 5e-4),
    "guided_horizon_overflow": (rtc.RTCConfig(mode=rtc.RTCMode.GUIDED), 3, ACTION_HORIZON + 5, 5e-4),
    "trained_overflow": (rtc.RTCConfig(mode=rtc.RTCMode.TRAINED), ACTION_HORIZON + 2, ACTION_HORIZON + 5, 1e-4),
}


@pytest.mark.parametrize("case", list(_PARITY_CASES))
@pytest.mark.parametrize("pi05", [False, True])
def test_matches_jax_on_the_same_weights(parity_models, pi05, case):
    config, jax_model, torch_model = parity_models(pi05=pi05)
    rtc_config, delay, horizon, atol = _PARITY_CASES[case]
    inputs = _inputs(config, seed=1)

    jax_obs, jax_noise, jax_prev = _jax_args(inputs)
    jax_sample = nnx_utils.module_jit(jax_model.sample_actions, static_argnames=("rtc_config",))
    expected = jax_sample(
        jax.random.key(0),
        jax_obs,
        num_steps=NUM_STEPS,
        noise=jax_noise,
        prev_chunk=jax_prev if rtc_config else None,
        inference_delay=jnp.asarray(delay, dtype=jnp.int32),
        execution_horizon=None if horizon is None else jnp.asarray(horizon, dtype=jnp.int32),
        rtc_config=rtc_config,
    )
    expected = np.asarray(expected)

    torch_obs, torch_noise, torch_prev = _torch_args(inputs)
    actual = torch_model.sample_actions(
        "cpu",
        torch_obs,
        noise=torch_noise,
        num_steps=NUM_STEPS,
        prev_chunk=torch_prev if rtc_config else None,
        inference_delay=torch.tensor(delay),
        execution_horizon=None if horizon is None else torch.tensor(horizon),
        rtc_config=rtc_config,
    ).numpy()

    if rtc_config is not None and rtc_config.mode is rtc.RTCMode.TRAINED:
        # Both pin the executed prefix bit for bit; compare the rest numerically.
        pinned = min(delay, ACTION_HORIZON)
        np.testing.assert_array_equal(actual[:, :pinned], inputs["prev_chunk"][:, :pinned])
        np.testing.assert_array_equal(expected[:, :pinned], inputs["prev_chunk"][:, :pinned])
        if pinned == ACTION_HORIZON:
            print(f"parity {case} pi05={pi05}: whole chunk pinned on both sides")
            return
        actual, expected = actual[:, pinned:], expected[:, pinned:]
    max_abs = float(np.abs(actual - expected).max())
    print(
        f"parity {case} pi05={pi05}: max |torch - jax| = {max_abs:.3g} (tolerance {atol}, scale {np.abs(expected).max():.3g})"
    )
    assert max_abs <= atol, f"{case}: max |torch - jax| = {max_abs:.3g} > {atol} (scale {np.abs(expected).max():.3g})"
