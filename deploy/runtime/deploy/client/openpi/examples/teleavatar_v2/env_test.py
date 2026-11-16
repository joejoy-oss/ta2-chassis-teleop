"""Tests for the ROS2 thread's teardown in env.py and for main.py's exit status, with a small ROS2 node in place of
the robot interface (no robot, video or policy server). They need rclpy and gi, i.e. the client image; skipped
elsewhere.

Without the teardown the ROS2 thread is still waiting inside rclpy when the interpreter exits, and the process ends in
"terminate called without an active exception" (abort, core dump; B3 review 应-4)."""

import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def interface_class(valid_observations: int):
    """A stand-in for ros2_interface.TeleavatarROS2Interface: the interface's 200 Hz timer (it keeps the executor
    waking up), initial data at once, `valid_observations` observations and then none (stale sensors)."""
    import numpy as np
    from rclpy.node import Node

    class FakeInterface(Node):
        def __init__(self, **_settings):
            super().__init__("teleavatar_env_test")
            self.create_timer(0.005, lambda: None)
            self.observations = 0
            self.actions = 0
            self.destroyed = 0

        def wait_for_initial_data(self, timeout: float) -> bool:
            return True

        def get_observation(self):
            self.observations += 1
            if self.observations > valid_observations:
                return None
            image = np.zeros((4, 4, 3), np.uint8)
            return {"state": np.zeros(48), "images": {"left_color": image, "right_color": image, "head_camera": image}}

        def publish_action(self, actions) -> None:
            self.actions += 1

        def destroy_node(self) -> None:
            self.destroyed += 1
            print("fake interface: destroy_node", file=sys.stderr, flush=True)
            super().destroy_node()

    return FakeInterface


def test_close_stops_the_ros2_thread_and_tears_down_once(monkeypatch):
    pytest.importorskip("rclpy")
    pytest.importorskip("gi")
    import rclpy

    monkeypatch.syspath_prepend(str(REPO))
    from examples.teleavatar_v2 import env as _env

    monkeypatch.setattr(_env.ros2_interface, "TeleavatarROS2Interface", interface_class(valid_observations=2))
    environment = _env.TeleavatarEnvironment(video_decode="cpu")
    interface = environment._ros_interface  # noqa: SLF001
    thread = environment._ros_thread  # noqa: SLF001
    try:
        assert thread.is_alive()
        assert rclpy.ok()
        for _ in range(2):
            assert set(environment.get_observation()) >= {"observation/state", "prompt"}
        with pytest.raises(_env.ObservationUnavailableError):
            environment.get_observation()
    finally:
        environment.close()
    assert not thread.is_alive()
    assert interface.destroyed == 1
    assert not rclpy.ok()
    environment.close()
    assert interface.destroyed == 1


DRIVER = """
import logging
import sys

import numpy as np

import env_test
from examples.teleavatar_v2 import env as _env
from examples.teleavatar_v2 import main as _main


class Policy:
    def __init__(self, host, port):
        pass

    def get_server_metadata(self):
        return {}

    def infer(self, obs):
        return {"actions": np.zeros((30, 16))}

    def reset(self):
        pass


_env.ros2_interface.TeleavatarROS2Interface = env_test.interface_class(int(sys.argv[1]))
_main._websocket_client_policy.WebsocketClientPolicy = Policy
logging.basicConfig(level=logging.INFO, force=True)
args = _main.Args(control_frequency=200.0, num_episodes=1, max_episode_steps=int(sys.argv[2]), video_decode="cpu")
sys.exit(_main.main(args))
"""


@pytest.mark.parametrize(
    ("valid_observations", "max_episode_steps", "status"),
    [(5, 50, 3), (1000, 5, 0)],  # the sensors go stale at step 6 (exit 3) / the episode ends (exit 0)
)
def test_main_exits_with_its_status_after_tearing_down(valid_observations, max_episode_steps, status):
    """main.py's whole path in a fresh interpreter: the exit status is main()'s, the node is destroyed before
    "Shutdown complete.", and the process does not abort on its way out."""
    pytest.importorskip("rclpy")
    pytest.importorskip("gi")
    pytest.importorskip("tyro")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(HERE), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, "-c", DRIVER, str(valid_observations), str(max_episode_steps)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == status, result.stderr[-3000:]
    assert "terminate called" not in result.stderr
    destroyed = result.stderr.index("fake interface: destroy_node")
    assert destroyed < result.stderr.index("Shutdown complete.")
    if status == 3:
        assert "Stopping: Observation unavailable" in result.stderr
