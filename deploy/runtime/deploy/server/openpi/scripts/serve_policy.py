from collections.abc import Callable
import dataclasses
import enum
import logging
import socket
import time

from openpi_client import base_policy as _base_policy
import tyro

from openpi.models import rtc as _rtc
from openpi.policies import aloha_policy
from openpi.policies import droid_policy
from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import teleavatar_v2_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    TELEAVATAR = "teleavatar"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Device for PyTorch checkpoints (e.g., "cuda", "xpu", "cpu"). If not provided, the first available of
    # cuda, xpu and cpu is used.
    pytorch_device: str | None = None
    # torch.compile mode for PyTorch checkpoints, overriding the config's `pytorch_compile_mode`: one of "default",
    # "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", or "none" to run eagerly. If not provided,
    # the config's value is used.
    pytorch_compile_mode: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class RTCArgs:
    """Real-Time Chunking: lets the robot keep executing while the policy is queried.

    Needs an RTC-aware client (`openpi_client.rtc.RTCActionBroker`); an ordinary client just
    gets unconstrained chunks. See docs/real_time_chunking.md.
    """

    enabled: bool = False
    # "guided": training-free, works with any checkpoint, ~1.5x inference cost.
    # "trained": hard prefix pinning, free, needs Pi0Config(rtc_training_max_delay > 0).
    mode: _rtc.RTCMode = _rtc.RTCMode.GUIDED
    # Shape of the weight decay between inference_delay and execution_horizon.
    prefix_attention_schedule: _rtc.PrefixAttentionSchedule = _rtc.PrefixAttentionSchedule.EXP
    # Upper clip on the guidance weight ("guided" only).
    max_guidance_weight: float = 10.0
    # "trained" only: re-noise the prefix so an unretrained checkpoint stays in distribution.
    trained_prefix_noise: bool = False

    def create(self) -> _rtc.RTCConfig | None:
        if not self.enabled:
            return None
        return _rtc.RTCConfig(
            mode=self.mode,
            prefix_attention_schedule=self.prefix_attention_schedule,
            max_guidance_weight=self.max_guidance_weight,
            trained_prefix_noise=self.trained_prefix_noise,
        )


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies, and to pick the example
    # observation for --warmup.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Run the policy on an example observation for --env before serving, so that the first client request does not
    # wait for compilation. With RTC enabled, the constrained request of the configured mode is run as well.
    warmup: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Real-Time Chunking settings.
    rtc: RTCArgs = dataclasses.field(default_factory=RTCArgs)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None, rtc_config: _rtc.RTCConfig | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            rtc_config=rtc_config,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    rtc_config = args.rtc.create()
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.policy.pytorch_compile_mode is not None:
                mode = None if args.policy.pytorch_compile_mode == "none" else args.policy.pytorch_compile_mode
                train_config = dataclasses.replace(
                    train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=mode)
                )
            return _policy_config.create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                pytorch_device=args.policy.pytorch_device,
                rtc_config=rtc_config,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, rtc_config=rtc_config)


# Example observations for --warmup, in the format of each environment's client.
WARMUP_EXAMPLE: dict[EnvMode, Callable[[], dict]] = {
    EnvMode.ALOHA: aloha_policy.make_aloha_example,
    EnvMode.ALOHA_SIM: aloha_policy.make_aloha_example,
    EnvMode.DROID: droid_policy.make_droid_example,
    EnvMode.LIBERO: libero_policy.make_libero_example,
    EnvMode.TELEAVATAR: teleavatar_v2_policy.make_teleavatar_example,
}


def warmup(policy: _base_policy.BasePolicy, example: dict) -> list[float]:
    """Send the requests that compile the policy before a client does, and return their wall times in seconds.

    The first request is unconstrained. With RTC enabled, a second request continues the first chunk,
    which takes the constrained path of the configured mode (a different graph for "guided", and for
    "trained" without prefix noise). The RTC history is dropped afterwards, so the first client starts
    from an unconstrained chunk.
    """
    seconds = []
    start = time.monotonic()
    result = policy.infer(example)
    seconds.append(time.monotonic() - start)
    logging.info("Warmup: unconstrained request took %.1f s", seconds[-1])
    if _policy.RTC_CHUNK_ID in result:
        start = time.monotonic()
        policy.infer(
            {
                **example,
                _policy.RTC_PREV_CHUNK_ID: result[_policy.RTC_CHUNK_ID],
                _policy.RTC_PREFIX_START: 1,
                _policy.RTC_INFERENCE_DELAY: 1,
            }
        )
        seconds.append(time.monotonic() - start)
        logging.info("Warmup: RTC request took %.1f s", seconds[-1])
    policy.reset()
    return seconds


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.warmup:
        logging.info("Warming up the policy on the %s example observation", args.env.value)
        try:
            warmup(policy, WARMUP_EXAMPLE[args.env]())
        except Exception as e:
            # --env defaults to ALOHA_SIM, and the input transforms of other policies reject its example with an
            # error that does not mention the flag.
            raise RuntimeError(
                f"--warmup failed on the {args.env.name} example observation. If the policy is not for that "
                "environment, pass the matching --env, e.g. --env TELEAVATAR for the teleavatar configs."
            ) from e

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
