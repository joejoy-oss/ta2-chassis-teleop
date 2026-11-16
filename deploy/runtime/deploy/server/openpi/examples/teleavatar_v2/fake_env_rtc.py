#!/usr/bin/env python3
"""
Runs the client loop of main.py against a policy server, with a stand-in for the robot.

The stand-in shows the policy one fixed camera frame per camera (a recorded one from
--observation, or random pixels) and makes the joint positions follow the commanded ones
exactly, so the state in each observation is where the previous commands took it (this is
what moves the anchor of delta actions under RTC). The client side is main.py's: the same
RTCActionBroker / ActionChunkBroker set-up, stepped by Runtime at --control-frequency.
Meant for checking Real-Time Chunking end to end (docs/real_time_chunking.md). --record
writes every executed action with the request and chunk index it came from, and every
request with the whole chunk it returned, for offline analysis of the splices.

Usage:
    # Start a policy server first, e.g. with guided RTC:
    uv run scripts/serve_policy.py --env TELEAVATAR --warmup --rtc.enabled policy:checkpoint \
        --policy.config=pi05_teleavatar_v2 \
        --policy.dir=checkpoints/pi05_teleavatar_v2/my_experiment/20000

    # Then:
    python examples/teleavatar_v2/fake_env_rtc.py --rtc --num-episodes 2 --max-episode-steps 600 \
        --observation frame.npz --record /tmp/guided.json
"""

import dataclasses
import json
import logging
import pathlib
import platform
import time
import traceback
from typing import Literal

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import base_policy as _base_policy
from openpi_client import rtc as _rtc
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime import subscriber as _subscriber
from openpi_client.runtime.agents import policy_agent as _policy_agent
from typing_extensions import override
import tyro

# (height, width) of the views the v2 robot client sends: single eyes, split from the RTP
# composite frame by rtp_video_interface.
ROBOT_IMAGE_SIZES = {"head_camera": (960, 960), "left_color": (400, 640), "right_color": (400, 640)}
# Added to every chunk before the broker slices it, so that each executed action carries the
# request it came from and its index in that request's chunk.
REQUEST_KEY = "fake_env/request"
INDEX_KEY = "fake_env/index"


@dataclasses.dataclass
class Args:
    """Command-line arguments. The client settings are main.py's."""

    remote_host: str = "127.0.0.1"
    remote_port: int = 8000

    control_frequency: float = 45.0
    """Control loop frequency in Hz."""

    open_loop_horizon: int = 16
    """Number of actions to execute before querying the policy again. Ignored with --rtc."""

    rtc: bool = False
    """Use RTCActionBroker (needs a server started with --rtc.enabled) instead of ActionChunkBroker."""

    rtc_calibration_steps: int = 5
    rtc_warmup_steps: int = 2
    rtc_inference_delay: int | None = None
    rtc_execution_horizon: int | None = None

    num_episodes: int = 2

    max_episode_steps: int = 600
    """Episode length. Runtime takes one step more than this; the stand-in never ends an episode itself."""

    image_format: Literal["robot", "raw"] = "robot"
    """robot: what the v2 robot client sends, single eyes, head 960x960 and wrists 400x640.
    raw: side-by-side stereo at the resolution of the training videos, head 1920x3840 and
    wrists 800x2560 (the shape of make_teleavatar_example); the robot frame upsampled 2x,
    in both eyes."""

    observation: pathlib.Path | None = None
    """.npz with `observation/state`, `observation/images/{head_camera,left_color,right_color}`
    at robot resolution and `prompt`, e.g. a frame written by `scripts/rtc_numerics.py
    extract-real`. Default: random pixels and state (--seed)."""

    prompt: str | None = None
    """Language instruction. Default: the one in --observation, else main.py's default."""

    seed: int = 0

    record: pathlib.Path | None = None
    """Write the episodes and requests here as JSON, also when the run fails."""


def to_raw(frame: np.ndarray) -> np.ndarray:
    """A side-by-side stereo frame showing `frame` at twice its resolution in both eyes, so it
    survives whichever eye TeleavatarInputs crops."""
    eye = np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)
    return np.concatenate([eye, eye], axis=1)


def load_scene(args: Args) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """The camera frames (robot resolution), the initial 62-dim state and the prompt."""
    prompt = "Stack the three blocks"  # main.py's default
    if args.observation is None:
        rng = np.random.default_rng(args.seed)
        images = {name: rng.integers(256, size=(*hw, 3), dtype=np.uint8) for name, hw in ROBOT_IMAGE_SIZES.items()}
        recorded_state = rng.random(62)
    else:
        with np.load(args.observation) as data:
            images = {name: np.asarray(data[f"observation/images/{name}"]) for name in ROBOT_IMAGE_SIZES}
            recorded_state = np.asarray(data["observation/state"], dtype=np.float64)
            if "prompt" in data:
                prompt = str(data["prompt"])
        for name, hw in ROBOT_IMAGE_SIZES.items():
            image = images[name]
            if image.shape != (*hw, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{args.observation}: {name} is {image.dtype} {image.shape}, expected uint8 {(*hw, 3)}"
                )
    # The first 62 values of the v2 training layout: positions, velocities, efforts (16 each, same
    # joint order), then end-effector poses (14); recorded datasets carry more. The robot client
    # (ros2_interface.py) sends only the first 48, as float32 with the gripper slots 7, 15, 39 and
    # 47 left at 0. The policy reads the 14 arm positions alone, so the model sees no difference.
    state = np.zeros(62)
    state[: min(62, len(recorded_state))] = recorded_state[:62]
    return images, state, args.prompt or prompt


class FakeTeleavatarEnvironment(_environment.Environment):
    """Stands in for TeleavatarEnvironment: fixed camera frames, joints that reach every command."""

    def __init__(self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str):
        self._images = images
        self._initial_state = np.asarray(state, dtype=np.float64).copy()
        self._state = self._initial_state.copy()
        self._prompt = prompt

    @override
    def reset(self) -> None:
        self._state = self._initial_state.copy()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        return {
            # A copy: the RTC broker sends this from its worker thread while the loop moves on.
            "observation/state": self._state.copy(),
            "observation/images/left_color": self._images["left_color"],
            "observation/images/right_color": self._images["right_color"],
            "observation/images/head_camera": self._images["head_camera"],
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action["actions"])
        if actions.shape != (16,):
            raise ValueError(f"Expected 16-dim action, got shape {actions.shape}")
        # Actions: left arm (7), left gripper effort, right arm (7), right gripper effort.
        self._state[0:7] = actions[0:7]
        self._state[8:15] = actions[8:15]
        self._state[32 + 7] = actions[7]
        self._state[32 + 15] = actions[15]


class RequestLog(_base_policy.BasePolicy):
    """Wraps the policy the broker queries: records every request and tags its chunk."""

    def __init__(self, policy: _base_policy.BasePolicy, clock):
        self._policy = policy
        self._clock = clock
        self.episode = 0
        self.requests: list[dict] = []

    @override
    def infer(self, obs: dict) -> dict:  # Brokers keep at most one request in flight.
        sent = self._clock()
        result = self._policy.infer(obs)
        received = self._clock()
        actions = np.asarray(result["actions"])
        number = len(self.requests)
        self.requests.append(
            {
                "episode": self.episode,
                "sent_s": sent,
                "received_s": received,
                "rtc": {key: int(value) for key, value in obs.items() if key.startswith("rtc/")},
                "positions": np.asarray(obs["observation/state"])[:16].tolist(),
                "chunk_id": result.get(_rtc.RTC_CHUNK_ID),
                "actions": actions.tolist(),
                "server_timing": result.get("server_timing", {}),
                "policy_timing": result.get("policy_timing", {}),
            }
        )
        return {**result, REQUEST_KEY: np.full(len(actions), number), INDEX_KEY: np.arange(len(actions))}

    @override
    def reset(self) -> None:
        self._policy.reset()


class StepLog(_subscriber.Subscriber):
    """Records every executed action, and the RTC broker's current delay and horizon."""

    def __init__(self, requests: RequestLog, broker: _base_policy.BasePolicy, clock):
        self._requests = requests
        self._broker = broker
        self._clock = clock
        self.episodes: list[list[dict]] = []

    @override
    def on_episode_start(self) -> None:
        # Runtime resets the agent before this, so a request still in flight was logged under
        # the previous episode.
        self._requests.episode = len(self.episodes)
        self.episodes.append([])

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        step = {
            "t_s": self._clock(),
            "request": int(action[REQUEST_KEY]),
            "index": int(action[INDEX_KEY]),
            "actions": np.asarray(action["actions"]).tolist(),
        }
        if isinstance(self._broker, _rtc.RTCActionBroker):
            step["inference_delay"] = self._broker.inference_delay
            step["execution_horizon"] = self._broker.execution_horizon
        self.episodes[-1].append(step)

    @override
    def on_episode_end(self) -> None:
        pass


def main(args: Args) -> None:
    images, state, prompt = load_scene(args)
    if args.image_format == "raw":
        images = {name: to_raw(frame) for name, frame in images.items()}
    environment = FakeTeleavatarEnvironment(images, state, prompt)
    logging.info(
        "Fake environment: %s images %s, prompt '%s'",
        args.image_format,
        {name: image.shape for name, image in images.items()},
        prompt,
    )

    start = time.perf_counter()

    def clock() -> float:
        return time.perf_counter() - start

    record: dict = {
        "args": json.loads(json.dumps(dataclasses.asdict(args), default=str)),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "images": {name: list(image.shape) for name, image in images.items()},
        "prompt": prompt,
        "initial_state": state.tolist(),
        "error": None,
    }
    requests = steps = None
    try:
        ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.remote_host, port=args.remote_port)
        logging.info("Connected to policy server. Metadata: %s", ws_client_policy.get_server_metadata())
        requests = RequestLog(ws_client_policy, clock)
        if args.rtc:
            broker = _rtc.RTCActionBroker(
                policy=requests,
                config=_rtc.RTCBrokerConfig(
                    control_frequency=args.control_frequency,
                    warmup_steps=args.rtc_warmup_steps,
                    calibration_steps=args.rtc_calibration_steps,
                    inference_delay=args.rtc_inference_delay,
                    execution_horizon=args.rtc_execution_horizon,
                ),
            )
        else:
            broker = action_chunk_broker.ActionChunkBroker(policy=requests, action_horizon=args.open_loop_horizon)
        steps = StepLog(requests, broker, clock)
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(policy=broker),
            subscribers=[steps],
            max_hz=args.control_frequency,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
        runtime.run()
        broker.reset()  # Waits for a request still in flight, so that the record has it.
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        if args.record is not None:
            # Copies: the RTC broker's worker thread may still be finishing a request.
            record["episodes"] = list(steps.episodes) if steps is not None else []
            record["requests"] = list(requests.requests) if requests is not None else []
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(record))
            logging.info("Recorded %s", args.record)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    main(tyro.cli(Args))
