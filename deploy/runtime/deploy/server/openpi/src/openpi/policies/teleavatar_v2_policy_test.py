"""TeleAvatar v2 with the chassis (include_chassis_actions): the 19-dim outputs, and the policy refusing to start with
action norm stats that do not cover all 19 dims (chassis review S2). No model is built: the tests stub the model loading,
the tokenizer and the Policy, and pass the norm stats in."""

import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import teleavatar_v2_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


def _stats(dims: int) -> _normalize.NormStats:
    rng = np.random.default_rng(dims)
    return _normalize.NormStats(
        mean=rng.normal(size=dims),
        std=rng.uniform(0.5, 2.0, size=dims),
        q01=-rng.uniform(0.5, 1.0, size=dims),
        q99=rng.uniform(0.5, 1.0, size=dims),
    )


def _norm_stats(action_dims: int) -> dict[str, _normalize.NormStats]:
    return {"state": _stats(14), "actions": _stats(action_dims)}


@pytest.mark.parametrize(
    ("include_chassis_actions", "action_dims", "refused"),
    [(False, 16, False), (False, 19, False), (True, 19, False), (True, 32, False), (True, 16, True), (True, 18, True)],
)
def test_outputs_check_the_action_norm_stats(include_chassis_actions, action_dims, refused):
    outputs = teleavatar_v2_policy.TeleavatarOutputs(include_chassis_actions=include_chassis_actions)
    assert isinstance(outputs, _transforms.NormStatsRequirement)
    if refused:
        with pytest.raises(ValueError, match="all 19 action dims") as info:
            outputs.check_norm_stats(_norm_stats(action_dims))
        assert f"'mean': {action_dims}" in str(info.value)
    else:
        outputs.check_norm_stats(_norm_stats(action_dims))


def test_the_quantiles_count_too():
    stats = _stats(19)
    short = _normalize.NormStats(mean=stats.mean, std=stats.std, q01=stats.q01[:16], q99=stats.q99[:16])
    with pytest.raises(ValueError, match=r"\{'q01': 16, 'q99': 16\}"):
        teleavatar_v2_policy.TeleavatarOutputs(include_chassis_actions=True).check_norm_stats(
            {"state": _stats(14), "actions": short}
        )


def test_no_action_norm_stats_only_matter_with_the_chassis():
    with pytest.raises(ValueError, match="no action norm stats"):
        teleavatar_v2_policy.TeleavatarOutputs(include_chassis_actions=True).check_norm_stats({"state": _stats(14)})
    teleavatar_v2_policy.TeleavatarOutputs().check_norm_stats(None)


@pytest.mark.parametrize("include_chassis_actions", [False, True])
def test_outputs_return_19_dims_with_the_chassis_and_16_without(include_chassis_actions):
    rng = np.random.default_rng(0)
    model_actions = rng.uniform(-1.0, 1.0, size=(30, 32)).astype(np.float32)
    state = rng.uniform(-1.0, 1.0, size=14)
    outputs = teleavatar_v2_policy.TeleavatarOutputs(
        use_delta_joint_actions=True, include_chassis_actions=include_chassis_actions
    )
    actions = outputs({"actions": model_actions, "state": state})["actions"]
    assert actions.shape == (30, 19 if include_chassis_actions else 16)
    np.testing.assert_allclose(actions[:, 0:7], model_actions[:, 0:7] + state[0:7], rtol=1e-6)
    np.testing.assert_allclose(actions[:, 8:15], model_actions[:, 8:15] + state[7:14], rtol=1e-6)
    for gripper in (7, 15):
        effort = teleavatar_v2_policy._gripper_trigger_to_effort(model_actions[:, gripper])  # noqa: SLF001
        np.testing.assert_allclose(actions[:, gripper], effort)
    if include_chassis_actions:
        # The chassis velocity as the model gave it: no delta, no gripper curve.
        np.testing.assert_array_equal(actions[:, 16:19], model_actions[:, 16:19])


@pytest.fixture
def stubbed_loading(monkeypatch):
    """create_trained_policy without a checkpoint: the params, the model, the tokenizer and the Policy are stand-ins.
    Returns the keyword arguments each Policy was built with."""
    built = []
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(pi0_config.Pi0Config, "load", lambda self, params, **kwargs: "model")
    monkeypatch.setattr(_tokenizer, "PaligemmaTokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(_policy, "Policy", lambda model, **kwargs: built.append(kwargs) or "policy")
    return built


@pytest.mark.parametrize(
    ("config_name", "action_dims", "refused"),
    [
        ("pi05_teleavatar_v2_chassis", 16, True),
        ("pi05_teleavatar_v2_chassis", 19, False),
        ("pi05_teleavatar_v2_rtc", 16, False),
    ],
)
def test_create_trained_policy_refuses_short_action_norm_stats_for_the_chassis(
    stubbed_loading, tmp_path, config_name, action_dims, refused
):
    """The norm stats checked are the ones the policy would use (passed in here, normally the checkpoint's), and a
    refusal comes before any Policy exists. Without the chassis nothing changes."""
    config = _config.get_config(config_name)
    norm_stats = _norm_stats(action_dims)
    if refused:
        with pytest.raises(ValueError, match="all 19 action dims"):
            _policy_config.create_trained_policy(config, tmp_path, norm_stats=norm_stats)
        assert stubbed_loading == []
        return
    assert _policy_config.create_trained_policy(config, tmp_path, norm_stats=norm_stats) == "policy"
    (kwargs,) = stubbed_loading
    (unnormalize,) = [t for t in kwargs["output_transforms"] if isinstance(t, _transforms.Unnormalize)]
    assert unnormalize.norm_stats is norm_stats
    (outputs,) = [t for t in kwargs["output_transforms"] if isinstance(t, teleavatar_v2_policy.TeleavatarOutputs)]
    assert outputs.include_chassis_actions == (config_name == "pi05_teleavatar_v2_chassis")
