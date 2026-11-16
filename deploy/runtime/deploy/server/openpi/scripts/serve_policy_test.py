import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import rtc as _rtc
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import policy as _policy
from openpi.policies import teleavatar_v2_policy
from openpi.training import config as _config

from . import serve_policy

_ACTION_DIM = 8


def _example() -> dict:
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    return {
        "image": {k: np.zeros((*_model.IMAGE_RESOLUTION, 3), np.uint8) for k in keys},
        "image_mask": dict.fromkeys(keys, np.True_),
        "state": np.zeros(_ACTION_DIM, np.float32),
        "tokenized_prompt": np.zeros(200, np.int32),
        "tokenized_prompt_mask": np.ones(200, bool),
    }


@pytest.mark.parametrize("mode", [None, _rtc.RTCMode.GUIDED, _rtc.RTCMode.TRAINED])
def test_warmup_runs_the_constrained_request_and_resets(mode):
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=_ACTION_DIM,
        action_horizon=12,
        pi05=True,
        pytorch_compile_mode=None,
    )
    model = pi0_pytorch.PI0Pytorch(config)
    constrained = []
    sample_actions = model.sample_actions

    def spy(*args, **kwargs):
        constrained.append(kwargs.get("prev_chunk") is not None)
        return sample_actions(*args, **kwargs)

    model.sample_actions = spy
    rtc_config = None if mode is None else _rtc.RTCConfig(mode=mode)
    policy = _policy.Policy(model, is_pytorch=True, rtc_config=rtc_config)

    seconds = serve_policy.warmup(policy, _example())

    assert constrained == ([False] if mode is None else [False, True])
    assert len(seconds) == len(constrained)
    assert all(s > 0 for s in seconds)
    if mode is not None:
        # The RTC history is gone, so the first client request starts a new chunk sequence.
        assert policy.infer(_example())[_policy.RTC_CHUNK_ID] == 1


@pytest.mark.parametrize(("override", "expected"), [(None, "max-autotune"), ("none", None), ("default", "default")])
def test_checkpoint_overrides_the_compile_mode(monkeypatch, override, expected):
    created = []
    monkeypatch.setattr(
        serve_policy._policy_config,  # noqa: SLF001
        "create_trained_policy",
        lambda train_config, *args, **kwargs: created.append(train_config),
    )
    checkpoint = serve_policy.Checkpoint(config="pi05_teleavatar_v2", dir="unused", pytorch_compile_mode=override)

    serve_policy.create_policy(serve_policy.Args(policy=checkpoint))

    (train_config,) = created
    assert train_config.model.pytorch_compile_mode == expected
    assert train_config.name == "pi05_teleavatar_v2"


def test_a_warmup_failure_points_at_the_env_flag(monkeypatch):
    class _RejectingPolicy:
        def __init__(self) -> None:
            self.metadata: dict = {}

        def infer(self, obs: dict) -> dict:
            raise KeyError("observation/state")

        def reset(self) -> None:
            pass

    monkeypatch.setattr(serve_policy, "create_policy", lambda args: _RejectingPolicy())

    with pytest.raises(RuntimeError, match="--env TELEAVATAR") as excinfo:
        serve_policy.main(serve_policy.Args(warmup=True))
    assert "ALOHA_SIM" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, KeyError)


@pytest.mark.parametrize("name", ["pi05_teleavatar_v2", "pi05_teleavatar_v2_rtc"])
def test_the_pi05_teleavatar_v2_configs_serve_delta_arm_actions(name):
    # The checkpoints of these configs were trained on delta arm actions; serving them as absolute positions
    # commands the arms to the deltas.
    train_config = _config.get_config(name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    (inputs,) = data_config.data_transforms.inputs
    (outputs,) = data_config.data_transforms.outputs
    assert isinstance(inputs, teleavatar_v2_policy.TeleavatarInputs)
    assert isinstance(outputs, teleavatar_v2_policy.TeleavatarOutputs)
    assert inputs.use_delta_joint_actions
    assert outputs.use_delta_joint_actions
