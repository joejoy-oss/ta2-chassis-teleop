"""Tests for fake_env_rtc.py against a policy server that runs a scripted policy."""

import json
import logging
import re
import socket
import threading
import time

import fake_env_rtc  # pytest puts this directory on sys.path
import numpy as np
from openpi_client import base_policy as _base_policy
import pytest

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import teleavatar_v2_policy
from openpi.serving import websocket_policy_server

CHUNK = 30


class _ScriptedPolicy(_base_policy.BasePolicy):
    """Answers request n with a chunk whose row k is n + k / 100 in every dimension, plus a chunk id when `rtc`,
    like an RTC-enabled server, `latency_s` after taking the request. Records the positions and the RTC fields of
    every request as it arrives."""

    def __init__(self, *, rtc: bool, latency_s: float = 0.0):
        self._rtc = rtc
        self._latency_s = latency_s
        self.requests: list[dict] = []

    def infer(self, obs: dict) -> dict:
        self.requests.append(
            {
                "positions": np.asarray(obs["observation/state"])[:16].tolist(),
                "rtc": {key: int(value) for key, value in obs.items() if key.startswith("rtc/")},
            }
        )
        time.sleep(self._latency_s)
        number = len(self.requests)
        result = {"actions": np.zeros((CHUNK, 16)) + number + np.arange(CHUNK)[:, None] / 100}
        if self._rtc:
            result[_policy.RTC_CHUNK_ID] = number
        return result


def _serve(policy: _base_policy.BasePolicy) -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # The client retries a refused connection only every 5 s.
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.05)
    return port


def _run(tmp_path, policy: _ScriptedPolicy, **kwargs) -> dict:
    args = fake_env_rtc.Args(
        remote_port=_serve(policy), control_frequency=200.0, record=tmp_path / "run.json", **kwargs
    )
    fake_env_rtc.main(args)
    record = json.loads(args.record.read_text())
    assert record["error"] is None
    # Every executed action is the row of the chunk it is tagged with.
    for steps in record["episodes"]:
        for step in steps:
            assert step["actions"] == record["requests"][step["request"]]["actions"][step["index"]]
    # What the server received is what was recorded.
    assert [r["positions"] for r in record["requests"]] == [r["positions"] for r in policy.requests]
    assert [r["rtc"] for r in record["requests"]] == [r["rtc"] for r in policy.requests]
    return record


def test_raw_frames_carry_the_robot_frame_in_the_eye_the_policy_keeps():
    images, state, prompt = fake_env_rtc.load_scene(fake_env_rtc.Args())
    raw = {name: fake_env_rtc.to_raw(frame) for name, frame in images.items()}
    example = teleavatar_v2_policy.make_teleavatar_example()
    assert {name: frame.shape for name, frame in raw.items()} == {
        name: example[f"observation/images/{name}"].shape for name in raw
    }
    transform = teleavatar_v2_policy.TeleavatarInputs(model_type=_model.ModelType.PI05)

    def model_images(frames: dict) -> dict:
        obs = {"observation/state": state, "prompt": prompt}
        obs.update({f"observation/images/{name}": frame for name, frame in frames.items()})
        return transform(obs)["image"]

    from_robot, from_raw = model_images(images), model_images(raw)
    assert set(from_robot) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    for key, image in from_robot.items():
        np.testing.assert_array_equal(from_raw[key], np.repeat(np.repeat(image, 2, axis=0), 2, axis=1))


def test_scene_from_a_recorded_frame(tmp_path):
    rng = np.random.default_rng(1)
    frames = {
        f"observation/images/{name}": rng.integers(256, size=(*hw, 3), dtype=np.uint8)
        for name, hw in fake_env_rtc.ROBOT_IMAGE_SIZES.items()
    }
    np.savez(tmp_path / "frame.npz", **frames, **{"observation/state": np.arange(72.0), "prompt": "insert the blocks"})

    images, state, prompt = fake_env_rtc.load_scene(fake_env_rtc.Args(observation=tmp_path / "frame.npz"))

    assert prompt == "insert the blocks"
    np.testing.assert_array_equal(state, np.arange(62.0))  # the first 62 recorded values
    for name, image in images.items():
        np.testing.assert_array_equal(image, frames[f"observation/images/{name}"])
    assert fake_env_rtc.load_scene(fake_env_rtc.Args(observation=tmp_path / "frame.npz", prompt="other"))[2] == "other"

    frames["observation/images/head_camera"] = fake_env_rtc.to_raw(frames["observation/images/head_camera"])
    np.savez(tmp_path / "raw.npz", **frames, **{"observation/state": np.arange(72.0)})
    with pytest.raises(ValueError, match="expected uint8"):
        fake_env_rtc.load_scene(fake_env_rtc.Args(observation=tmp_path / "raw.npz"))


def test_rtc_run_continues_each_chunk_and_the_state_follows_the_commands(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    record = _run(tmp_path, _ScriptedPolicy(rtc=True), rtc=True, max_episode_steps=40)
    episodes, requests = record["episodes"], record["requests"]
    assert [len(steps) for steps in episodes] == [41, 41]  # Runtime takes max_episode_steps + 1
    # The delay and horizon recorded with the first step are the ones the broker logs once calibrated.
    (calibrated,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("RTC calibrated")]
    delay, horizon = re.search(r"inference_delay=(\d+) .*execution_horizon=(\d+)", calibrated).groups()
    assert [episodes[0][0]["inference_delay"], episodes[0][0]["execution_horizon"]] == [int(delay), int(horizon)]

    def blocking(episode: int) -> int:
        first_step = episodes[episode][0]["t_s"]
        return sum(1 for r in requests if r["episode"] == episode and r["received_s"] <= first_step)

    # Warm start: 2 + 5 requests before the first action. Episode 2 reuses the calibration.
    assert blocking(0) == 7
    assert blocking(1) == 1
    initial = record["initial_state"][:16]
    for steps in episodes:
        launched = {}
        for s, step in enumerate(steps):
            launched.setdefault(step["request"], s - step["index"])  # the step its chunk's row 0 belongs to
        assert len(launched) > 3
        for number, p in launched.items():
            request = requests[number]
            if p == 0:
                assert request["positions"] == initial
                continue
            # Sent at step p, after the command of step p - 1 moved the arms there.
            previous = steps[p - 1]["actions"]
            assert request["positions"] == [*previous[0:7], initial[7], *previous[8:15], initial[15]]
            # And it continues the chunk step p executes, from the row step p executes.
            assert request["rtc"][_policy.RTC_PREV_CHUNK_ID] == requests[steps[p]["request"]]["chunk_id"]
            assert request["rtc"][_policy.RTC_PREFIX_START] == steps[p]["index"]
            assert request["rtc"][_policy.RTC_INFERENCE_DELAY] >= 1
            # Step p records the delay and horizon the broker sent the request with.
            assert steps[p]["inference_delay"] == request["rtc"][_policy.RTC_INFERENCE_DELAY]
            assert steps[p]["execution_horizon"] == request["rtc"][_policy.RTC_EXECUTION_HORIZON]


def test_a_request_in_flight_when_the_run_ends_is_recorded(tmp_path):
    # Answers take about 6 control steps, and an execution horizon of 8 is raised to inference_delay + 1, so the
    # broker sends the next request as soon as one is answered: one is in flight when the run ends.
    policy = _ScriptedPolicy(rtc=True, latency_s=0.03)
    args = fake_env_rtc.Args(
        remote_port=_serve(policy),
        control_frequency=200.0,
        rtc=True,
        rtc_execution_horizon=8,
        num_episodes=1,
        max_episode_steps=40,
        record=tmp_path / "run.json",
    )
    fake_env_rtc.main(args)
    time.sleep(0.2)  # Any request the client sent has reached the server by now.
    record = json.loads(args.record.read_text())
    assert record["requests"][-1]["received_s"] > record["episodes"][-1][-1]["t_s"]
    assert [r["positions"] for r in record["requests"]] == [r["positions"] for r in policy.requests]


def test_the_stand_in_moves_the_arms_and_the_gripper_efforts_to_the_command():
    images = {name: np.zeros((*hw, 3), dtype=np.uint8) for name, hw in fake_env_rtc.ROBOT_IMAGE_SIZES.items()}
    initial = np.arange(62.0)
    environment = fake_env_rtc.FakeTeleavatarEnvironment(images, initial, "a prompt")
    command = 100.0 + np.arange(16)
    environment.apply_action({"actions": command})
    expected = initial.copy()
    expected[0:7], expected[8:15] = command[0:7], command[8:15]  # arm positions
    expected[39], expected[47] = command[7], command[15]  # gripper efforts; the gripper positions 7, 15 stay
    np.testing.assert_array_equal(environment.get_observation()["observation/state"], expected)
    environment.reset()
    np.testing.assert_array_equal(environment.get_observation()["observation/state"], initial)


def test_open_loop_run_takes_a_new_chunk_every_horizon(tmp_path):
    record = _run(tmp_path, _ScriptedPolicy(rtc=False), open_loop_horizon=8, max_episode_steps=20)
    for steps in record["episodes"]:
        assert [step["index"] for step in steps] == [s % 8 for s in range(21)]
        assert [step["request"] for step in steps] == [steps[0]["request"] + s // 8 for s in range(21)]
    assert all(r["rtc"] == {} for r in record["requests"])


def test_rtc_run_against_a_server_without_rtc_stops_and_still_records(tmp_path):
    args = fake_env_rtc.Args(remote_port=_serve(_ScriptedPolicy(rtc=False)), rtc=True, record=tmp_path / "run.json")
    with pytest.raises(RuntimeError, match="--rtc.enabled"):
        fake_env_rtc.main(args)
    record = json.loads(args.record.read_text())
    assert "--rtc.enabled" in record["error"]
    assert len(record["requests"]) == 1
