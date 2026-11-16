"""Tests for the chassis velocity path (--enable-chassis; chassis review M1, M2, S1, S6).

Each test runs the real TeleavatarEnvironment and TeleavatarROS2Interface, or main.py's whole path, in a fresh
interpreter (this file run as a script, `driver ...`), next to a DDS subscriber in another process (`subscriber`)
standing in for the zenoh bridge, as the chassis review's sub.py did. There is no robot, video stream or policy server:
the interface takes off as soon as the subscriber is matched (QoS volatile: it must not miss the first commands),
observations are dummies and the policy is a stand-in. They need rclpy and gi, i.e. the client image; skipped
elsewhere.

Covered: 16- and 19-dim actions and both mismatches (ValueError before anything is published); NaN / inf become zero;
the speed limits clamp; the watchdog commands zero once commands stop and lets go when they resume; on the exception
path and after Ctrl+C (SIGINT), SIGTERM and SIGHUP, sent by main.py to itself or by another process, the subscriber's
last chassis message is zero, acknowledged before the video decoder stops, with no thread exception and the documented
exit status; a teardown step that fails (nobody subscribed, so the zero velocity cannot be confirmed; the video decoder
failing to stop) ends with EXIT_TEARDOWN_FAILED and no "Shutdown complete."; executor callbacks still on their way
when the ROS2 thread tears down do not run into the destroyed node; without --enable-chassis there is no chassis topic
and Ctrl+C ends as before. The interface's stop() publishes the zero velocity at most 3 times, waiting 0.5 s each time
for the acknowledgements (chassis re-review B): on its own with a stand-in publisher, and in main.py with a subscriber
that acknowledges nothing (frozen), where it gives up after 3 x 0.5 s with EXIT_TEARDOWN_FAILED.
"""

import json
import logging
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

ZERO = (0.0, 0.0, 0.0)
VEL = (0.1, -0.05, 0.2)  # within the default limits (0.15 m/s, 0.4 m/s, 0.4 rad/s)
VEL2 = (-0.12, 0.3, -0.35)
PERIOD = 1.0 / 45.0
# Not to be seen in any clean teardown: the symptoms of rclpy's own signal handler shutting the context down under
# the running node (chassis review M1, and the ros2-executor traceback carried over to stage 4).
THREAD_FAILURES = ("Exception in thread", "never retrieved", "context is not valid", "context is invalid")


# --------------------------------------------------------------------------------------------- the two processes


def _subscriber() -> int:
    """One JSON line per /api/chassis/velocity message (monotonic receive time, data); on SIGTERM one with the number of
    /api/left_arm/joint_cmd messages, then exit."""
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("chassis_test_subscriber")
    joint_cmds = [0]

    def on_chassis(msg: Float32MultiArray) -> None:
        print(json.dumps({"t": time.monotonic(), "chassis": list(msg.data)}), flush=True)

    def on_joint_cmd(_msg: JointState) -> None:
        joint_cmds[0] += 1
        if joint_cmds[0] % 20 == 0:  # progress, for a test that waits for the arm to be driven
            print(json.dumps({"arm": joint_cmds[0]}), flush=True)

    node.create_subscription(Float32MultiArray, "/api/chassis/velocity", on_chassis, 100)
    node.create_subscription(JointState, "/api/left_arm/joint_cmd", on_joint_cmd, 100)
    print(json.dumps({"ready": True}), flush=True)
    while not stop.is_set():
        rclpy.spin_once(node, timeout_sec=0.05)
    print(json.dumps({"joint_cmd": joint_cmds[0]}), flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _use_test_interface(*, wait_for_subscriber: bool, video_stop_fails: bool = False) -> None:
    """Make the environment build the real interface minus the robot: the video pipeline on a free port nothing sends
    to, initial data as soon as a subscriber is matched, dummy observations."""
    from examples.teleavatar_v2 import env as _env
    from examples.teleavatar_v2 import ros2_interface as _ros2_interface

    class Interface(_ros2_interface.TeleavatarROS2Interface):
        def __init__(self, **settings):
            super().__init__(rtp_port=_free_udp_port(), **settings)
            if video_stop_fails:
                video_stop = self._video.stop

                def failing_stop() -> None:
                    video_stop()
                    raise RuntimeError("video decoder failed to stop (test)")

                self._video.stop = failing_stop

        def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
            if wait_for_subscriber:
                key = "chassis" if "chassis" in self.action_publishers else "left_arm"
                deadline = time.monotonic() + 10.0
                while self.action_publishers[key].get_subscription_count() < 1:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"no subscriber matched on {key} within 10 s")
                    time.sleep(0.02)
            return True

        def get_observation(self):
            image = np.zeros((4, 4, 3), np.uint8)
            images = dict.fromkeys(("left_color", "right_color", "head_camera"), image)
            return {"state": np.zeros(48, np.float32), "images": images}

    _env.ros2_interface.TeleavatarROS2Interface = Interface


def _action(velocity=VEL, dim: int = 19) -> dict:
    action = np.zeros(dim, np.float32)
    action[0:7] = [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
    action[8:15] = [0.0, -0.5, 0.0, -1.0, 0.0, 0.0, 0.0]
    if dim == 19:
        action[16:19] = velocity
    return {"actions": action}


def _report(**fields) -> None:
    print(json.dumps(fields), flush=True)


def _drive(environment, velocity, seconds: float) -> float:
    """Actions at 45 Hz for `seconds`; returns the monotonic time of the last one."""
    end = time.monotonic() + seconds
    last = time.monotonic()
    while time.monotonic() < end:
        last = time.monotonic()
        environment.apply_action(_action(velocity))
        time.sleep(PERIOD)
    return last


def _env_scenario(name: str) -> int:
    """TeleavatarEnvironment directly: `values`, `mismatch-on`, `mismatch-off`, `watchdog[-nointerp]`."""
    from examples.teleavatar_v2 import env as _env

    _use_test_interface(wait_for_subscriber=True)
    environment = _env.TeleavatarEnvironment(
        enable_chassis=name != "mismatch-off",
        control_frequency=45.0,
        interpolate=not name.endswith("nointerp"),
        video_decode="cpu",
    )
    try:
        if name == "values":
            for velocity in VALUES:
                environment.apply_action(_action(velocity))
                time.sleep(0.05)
        elif name == "mismatch-on":
            with pytest.raises(ValueError, match="Expected 19-dim action"):
                environment.apply_action(_action(dim=16))
        elif name == "mismatch-off":
            with pytest.raises(ValueError, match="Expected 16-dim action"):
                environment.apply_action(_action(dim=19))
            _report(publishers=sorted(environment._ros_interface.action_publishers))  # noqa: SLF001
            for _ in range(10):
                environment.apply_action(_action(dim=16))
                time.sleep(PERIOD)
        elif name.startswith("watchdog"):
            time.sleep(0.5)  # nothing commanded yet: the watchdog must keep quiet
            first = time.monotonic()
            pause = _drive(environment, VEL, 0.5)
            time.sleep(1.0)
            resume = time.monotonic()
            _drive(environment, VEL2, 0.3)
            _report(first=first, pause=pause, resume=resume)
        else:
            raise ValueError(f"unknown scenario {name}")
    finally:
        _report(close=environment.close())
    return 0


def _delay_executor_callbacks(delay: float) -> None:
    """Every callback the MultiThreadedExecutor hands to its thread pool starts `delay` s late, so that at the end some
    are still on their way while the ROS2 thread tears down."""
    from rclpy import executors

    init = executors.MultiThreadedExecutor.__init__

    def late_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        submit = self._executor.submit

        def late_submit(fn, /, *fn_args, **fn_kwargs):
            def late():
                time.sleep(delay)
                return fn(*fn_args, **fn_kwargs)

            return submit(late)

        self._executor.submit = late_submit

    executors.MultiThreadedExecutor.__init__ = late_init


def _main_scenario(how: str) -> int:
    """main.py's whole path with a stand-in policy; `how` it ends: sigint, sigterm, sighup (sent to itself from inside
    the 6th inference), async-sigint (from another thread, at no particular point of the loop), exception, done (the
    episode's 60 steps), nosub (sigint, nobody subscribed), video-stop-fails (sigint), late-callbacks (sigint, the
    executor's callbacks 50 ms late), off (sigint, no chassis), run and off-run (only by a signal from outside, with and
    without the chassis)."""
    from examples.teleavatar_v2 import main as _main

    chassis = how not in ("off", "off-run")
    _use_test_interface(wait_for_subscriber=how != "nosub", video_stop_fails=how == "video-stop-fails")
    if how == "late-callbacks":
        _delay_executor_callbacks(0.05)
    signals = {"sigterm": signal.SIGTERM, "sighup": signal.SIGHUP}

    class Policy:
        def __init__(self, host, port):
            self.calls = 0

        def get_server_metadata(self):
            return {}

        def infer(self, obs):
            self.calls += 1
            if self.calls == 1 and how == "async-sigint":
                threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT)).start()
            if self.calls == 6 and how not in ("async-sigint", "done", "run", "off-run"):
                if how == "exception":
                    raise RuntimeError("the policy server failed (test)")
                os.kill(os.getpid(), signals.get(how, signal.SIGINT))
            actions = np.zeros((30, 19 if chassis else 16), np.float32)
            if chassis:
                actions[:, 16:19] = VEL
            return {"actions": actions}

        def reset(self):
            pass

    _main._websocket_client_policy.WebsocketClientPolicy = Policy  # noqa: SLF001
    logging.basicConfig(level=logging.INFO, force=True, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = _main.Args(
        control_frequency=45.0,
        open_loop_horizon=8,
        num_episodes=1,
        max_episode_steps=60 if how == "done" else 0,
        video_decode="cpu",
        enable_chassis=chassis,
    )
    return _main.main(args)


VALUES = [
    VEL,
    (float("nan"), 0.0, 0.0),
    VEL,
    (0.0, float("inf"), 0.0),
    VEL,
    (1.0, -1.0, 2.0),
    (-0.2, 0.39, -0.41),
    VEL,
]
"""The `values` scenario's chassis commands, 50 ms apart; EXPECTED_VALUES is what must arrive (then the zero of the
teardown)."""
EXPECTED_VALUES = [VEL, ZERO, VEL, ZERO, VEL, (0.15, -0.4, 0.4), (-0.15, 0.39, -0.4), VEL, ZERO]


# --------------------------------------------------------------------------------------------- the tests


def _require_ros() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("gi")


def _run(*driver_args: str, subscriber: bool = True, send: signal.Signals | None = None, freeze: bool = False):
    """Runs the driver next to a subscriber. With `send`, sends the driver that signal from here once the subscriber
    has seen 20 chassis messages (200 arm commands without the chassis); with `freeze` as well, stops the subscriber
    (SIGSTOP) just before, so that it acknowledges nothing from then on, and continues it once the driver has exited.
    Returns (the driver's result, the chassis messages [(t, data)], the number of arm commands, the driver's JSON
    reports); with `send`, the reports also get "signal_to_exit": the seconds from the signal to the driver's exit."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")])}
    command = [sys.executable, str(pathlib.Path(__file__).resolve())]
    lines: list[dict] = []
    sub = None
    sent = None
    if subscriber:
        sub = subprocess.Popen(
            [*command, "subscriber"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=REPO
        )
        ready = sub.stdout.readline()
        assert ready.strip() == '{"ready": true}', ready + sub.stderr.read()

        def read() -> None:
            # Appended one by one as they arrive: the signal below waits for them.
            lines.extend(json.loads(line) for line in sub.stdout if line.startswith("{"))

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        driver = subprocess.Popen([*command, "driver", *driver_args], stdout=out, stderr=err, env=env, cwd=REPO)
        try:
            if send is not None:
                deadline = time.monotonic() + 60.0
                while driver.poll() is None and time.monotonic() < deadline:
                    if sum("chassis" in line for line in lines) >= 20 or sum("arm" in line for line in lines) >= 10:
                        if freeze:
                            sub.send_signal(signal.SIGSTOP)
                        sent = time.monotonic()
                        driver.send_signal(send)
                        break
                    time.sleep(0.01)
            returncode = driver.wait(timeout=120)
            ended = time.monotonic()
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait()
            if sub is not None:
                if freeze:
                    sub.send_signal(signal.SIGCONT)  # before SIGTERM: a stopped process would not end
                time.sleep(0.5)  # whatever is still on its way
                sub.send_signal(signal.SIGTERM)
                sub.wait(timeout=30)
                reader.join(timeout=30)
        out.seek(0)
        err.seek(0)
        result = subprocess.CompletedProcess(driver.args, returncode, out.read(), err.read())
    chassis = [(line["t"], line["chassis"]) for line in lines if "chassis" in line]
    joint_cmd = next((line["joint_cmd"] for line in lines if "joint_cmd" in line), None)
    reports = {}
    for line in result.stdout.splitlines():
        if line.startswith("{"):
            reports.update(json.loads(line))
    if sent is not None:
        reports["signal_to_exit"] = ended - sent
    return result, chassis, joint_cmd, reports


def _explain(result, chassis) -> str:
    return f"rc {result.returncode}\nstderr:\n{result.stderr[-6000:]}\nchassis: {chassis[-12:]}"


def _assert_no_thread_failure(stderr: str) -> None:
    for symptom in THREAD_FAILURES:
        assert symptom not in stderr, symptom


def _close(a, b) -> bool:
    return np.allclose(a, b, rtol=0.0, atol=1e-6)


def test_values_are_sanitized_and_clamped_and_the_last_one_is_zero():
    _require_ros()
    result, chassis, _, reports = _run("env", "values")
    assert result.returncode == 0, _explain(result, chassis)
    assert reports["close"] is True
    received = [data for _, data in chassis]
    assert len(received) == len(EXPECTED_VALUES), _explain(result, chassis)
    for got, expected in zip(received, EXPECTED_VALUES, strict=True):
        assert _close(got, expected), (got, expected)
    # The limits hold for the float32 that is sent, too.
    limits = np.float32([0.15, 0.4, 0.4])
    assert all(np.all(np.abs(np.float32(data)) <= limits) for data in received)
    assert "Invalid chassis velocity" in result.stderr
    assert "Chassis command clamped to the speed limits: vx 1.000->0.150, vy -1.000->-0.400, wz 2.000->0.400" in (
        result.stderr
    )
    _assert_no_thread_failure(result.stderr)


@pytest.mark.parametrize("scenario", ["mismatch-on", "mismatch-off"])
def test_a_mismatched_action_fails_before_anything_is_published(scenario):
    _require_ros()
    result, chassis, joint_cmd, reports = _run("env", scenario)
    assert result.returncode == 0, _explain(result, chassis)
    assert reports["close"] is True
    if scenario == "mismatch-on":
        # Nothing but the zero velocity of the teardown.
        assert [data for _, data in chassis] == [list(ZERO)], _explain(result, chassis)
    else:
        assert reports["publishers"] == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
        assert chassis == []
        assert joint_cmd > 0  # the 16-dim actions after it drove the arm
    _assert_no_thread_failure(result.stderr)


@pytest.mark.parametrize("scenario", ["watchdog", "watchdog-nointerp"])
def test_the_watchdog_stops_the_chassis_when_commands_stop(scenario):
    _require_ros()
    result, chassis, _, reports = _run("env", scenario)
    assert result.returncode == 0, _explain(result, chassis)
    assert reports["close"] is True
    first, pause, resume = reports["first"], reports["pause"], reports["resume"]
    assert all(t > first for t, _ in chassis), "a message before the first command"
    # While commands come at 45 Hz: only them.
    assert all(_close(data, VEL) for t, data in chassis if t < pause + 0.25), _explain(result, chassis)
    # Then zero from 0.3 s after the last command on, at 50 Hz, until the next one.
    zeros = [t for t, data in chassis if pause < t < resume and _close(data, ZERO)]
    assert len(zeros) >= 20, _explain(result, chassis)
    assert 0.3 <= zeros[0] - pause <= 0.45, zeros[0] - pause
    first_vel2 = next(i for i, (_, data) in enumerate(chassis) if _close(data, VEL2))
    last_vel = max(i for i, (_, data) in enumerate(chassis[:first_vel2]) if _close(data, VEL))
    assert all(_close(data, ZERO) for _, data in chassis[last_vel + 1 : first_vel2]), _explain(result, chassis)
    # Commands again: the watchdog lets go; the teardown's zero comes last.
    assert all(_close(data, VEL2) for _, data in chassis[first_vel2:-1]), _explain(result, chassis)
    assert _close(chassis[-1][1], ZERO)
    assert "No chassis command for" in result.stderr
    assert "Chassis commands resumed after" in result.stderr
    _assert_no_thread_failure(result.stderr)


def _assert_stopped_with_a_zero_velocity(run, status: int, stop_line: str, tracebacks: int = 0) -> None:
    """The subscriber's last chassis message is zero, acknowledged before the video decoder stops, after a while of
    commands; the exit status is main.py's documented one, with "Shutdown complete." and no thread failure or ERROR."""
    result, chassis, joint_cmd, _ = run
    stderr = result.stderr
    assert result.returncode == status, _explain(result, chassis)
    assert stop_line in stderr
    _assert_no_thread_failure(stderr)
    assert stderr.count("Traceback") == tracebacks, _explain(result, chassis)
    assert "ERROR" not in stderr, _explain(result, chassis)
    acknowledged = stderr.index("Zero chassis velocity sent and acknowledged by 1 subscriber(s)")
    assert acknowledged < stderr.index("Stopped decoder after") < stderr.index("Shutdown complete.")
    assert sum(not _close(data, ZERO) for _, data in chassis) >= 20, _explain(result, chassis)
    assert _close(chassis[-1][1], ZERO), _explain(result, chassis)
    assert joint_cmd > 0


@pytest.mark.parametrize(
    ("how", "status", "stop_line"),
    [
        ("sigint", 0, "Stopping robot (Ctrl+C pressed)"),
        ("async-sigint", 0, "Stopping robot (Ctrl+C pressed)"),
        ("sigterm", 143, "Stopping robot (SIGTERM received)"),
        ("sighup", 129, "Stopping robot (SIGHUP received)"),
        ("exception", 1, "RuntimeError: the policy server failed (test)"),
        ("done", 0, "Episode completed."),
    ],
)
def test_main_ends_with_an_acknowledged_zero_velocity(how, status, stop_line):
    _require_ros()
    tracebacks = 1 if how == "exception" else 0
    _assert_stopped_with_a_zero_velocity(_run("main", how), status, stop_line, tracebacks)


@pytest.mark.parametrize(
    ("sig", "status", "stop_line"),
    [
        (signal.SIGINT, 0, "Stopping robot (Ctrl+C pressed)"),
        (signal.SIGTERM, 143, "Stopping robot (SIGTERM received)"),
        (signal.SIGHUP, 129, "Stopping robot (SIGHUP received)"),
    ],
)
def test_a_signal_from_outside_stops_it_the_same_way(sig, status, stop_line):
    """As a terminal's Ctrl+C, a kill or a hang-up do: the signal comes from another process, at no particular point
    of the control loop."""
    _require_ros()
    _assert_stopped_with_a_zero_velocity(_run("main", "run", send=sig), status, stop_line)


def test_the_teardown_waits_for_callbacks_still_on_their_way():
    """Callbacks the executor has handed to its thread pool but not run yet must not run into the node's destruction
    ("cannot use Destroyable because destruction was requested", printed as "never retrieved" at exit): rclpy Humble's
    MultiThreadedExecutor.shutdown() does not wait for them, so the ROS2 thread does."""
    _require_ros()
    _assert_stopped_with_a_zero_velocity(_run("main", "late-callbacks"), 0, "Stopping robot (Ctrl+C pressed)")


@pytest.mark.parametrize(
    ("how", "error"),
    [
        ("nosub", "Zero chassis velocity reached no subscriber"),
        ("video-stop-fails", "ROS2 thread: node teardown (video decoder, node) failed"),
    ],
)
def test_a_failed_teardown_step_ends_with_its_own_exit_status(how, error):
    _require_ros()
    result, chassis, _, _ = _run("main", how, subscriber=how != "nosub")
    stderr = result.stderr
    assert result.returncode == 4, _explain(result, chassis)  # main.EXIT_TEARDOWN_FAILED
    assert error in stderr
    assert "Shutdown complete." not in stderr
    assert "Shutdown failed" in stderr
    _assert_no_thread_failure(stderr)
    if how == "video-stop-fails":
        # The zero velocity went out first all the same.
        assert stderr.index("Zero chassis velocity sent and acknowledged") < stderr.index(error)
        assert _close(chassis[-1][1], ZERO)


class _Publisher:
    """The chassis publisher, for stop() on its own: wait_for_all_acked answers from `acks`, one per call, and notes
    the timeout it was given."""

    def __init__(self, acks: list[bool], subscribers: int):
        self.acks = acks
        self.subscribers = subscribers
        self.timeouts_ns: list[int] = []

    def wait_for_all_acked(self, timeout) -> bool:
        self.timeouts_ns.append(timeout.nanoseconds)
        return self.acks[len(self.timeouts_ns) - 1]

    def get_subscription_count(self) -> int:
        return self.subscribers


class _Logger:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, message: str, **_kwargs) -> None:
        self.lines.append(("INFO", message))

    def error(self, message: str, **_kwargs) -> None:
        self.lines.append(("ERROR", message))


@pytest.mark.parametrize(
    ("acks", "subscribers", "confirmed", "log"),
    [
        (
            [False, False, False],
            1,
            False,
            (
                "ERROR",
                "Zero chassis velocity NOT acknowledged by the 1 matched subscriber(s) after 3 attempts, 0.5 s each",
            ),
        ),
        (
            [False, True],
            1,
            True,
            ("INFO", "Zero chassis velocity sent and acknowledged by 1 subscriber(s) (attempt 2)"),
        ),
        ([True], 0, False, ("ERROR", "Zero chassis velocity reached no subscriber")),
    ],
)
def test_stop_tries_3_times_waiting_0_5_s_each(monkeypatch, acks, subscribers, confirmed, log):
    """stop() on its own, with a stand-in publisher (chassis re-review B): the acknowledgements took 179-200 ms right
    after the 45 Hz commands, so each attempt waits 0.5 s for them, not 0.2 s; still at most 3 attempts. Nobody
    subscribed: no second attempt (wait_for_all_acked is True at once then). Once confirmed, stop() publishes nothing
    more."""
    _require_ros()
    monkeypatch.syspath_prepend(str(REPO))
    from examples.teleavatar_v2 import ros2_interface as _ros2_interface

    assert _ros2_interface._CHASSIS_STOP_ACK_TIMEOUT == 0.5  # noqa: SLF001
    assert _ros2_interface._CHASSIS_STOP_ATTEMPTS == 3  # noqa: SLF001
    publisher = _Publisher(acks, subscribers)
    published = []
    interface = types.SimpleNamespace(
        _enable_chassis=True,
        _chassis_stop_confirmed=False,
        _chassis_stopping=False,
        _chassis_lock=threading.Lock(),
        action_publishers={"chassis": publisher},
        _publish_chassis_cmd=lambda velocity: published.append(list(velocity)),
        logger=_Logger(),
    )
    stop = _ros2_interface.TeleavatarROS2Interface.stop
    assert stop(interface) is confirmed
    assert published == [list(ZERO)] * len(acks)
    assert publisher.timeouts_ns == [500_000_000] * len(acks)
    assert interface._chassis_stopping is True  # noqa: SLF001
    assert len(interface.logger.lines) == 1
    level, message = interface.logger.lines[0]
    assert level == log[0]
    assert message.startswith(log[1]), message
    if confirmed:
        assert stop(interface) is True
        assert len(published) == len(acks)


def test_an_unacknowledged_zero_velocity_is_reported_after_3_times_0_5_s():
    """main.py with a subscriber that acknowledges nothing from Ctrl+C on (frozen with SIGSTOP; the zenoh bridge stuck):
    the zero velocity is published 3 times, 0.5 s each, then an ERROR says so, and main.py exits with
    EXIT_TEARDOWN_FAILED."""
    _require_ros()
    result, chassis, _, reports = _run("main", "run", send=signal.SIGINT, freeze=True)
    stderr = result.stderr
    assert result.returncode == 4, _explain(result, chassis)  # main.EXIT_TEARDOWN_FAILED
    assert "Stopping robot (Ctrl+C pressed)" in stderr
    assert (
        "Zero chassis velocity NOT acknowledged by the 1 matched subscriber(s) after 3 attempts, 0.5 s each" in stderr
    ), _explain(result, chassis)
    assert "Shutdown complete." not in stderr
    assert "Shutdown failed" in stderr
    assert reports["signal_to_exit"] >= 3 * 0.5, reports
    _assert_no_thread_failure(stderr)


@pytest.mark.parametrize("outside", [False, True])
def test_without_the_chassis_ctrl_c_ends_as_before(outside):
    _require_ros()
    result, chassis, joint_cmd, _ = _run("main", "off-run", send=signal.SIGINT) if outside else _run("main", "off")
    stderr = result.stderr
    assert result.returncode == 0, _explain(result, chassis)
    assert "Chassis control: OFF" in stderr
    assert chassis == []
    assert joint_cmd > 0
    assert "Zero chassis velocity" not in stderr
    assert "Traceback" not in stderr
    _assert_no_thread_failure(stderr)
    assert stderr.index("Stopping robot (Ctrl+C pressed)") < stderr.index("Stopped decoder after")
    assert stderr.index("Stopped decoder after") < stderr.index("Shutdown complete.")


def _cli(argv: list[str]) -> int:
    if argv == ["subscriber"]:
        return _subscriber()
    if argv[:2] == ["driver", "env"]:
        return _env_scenario(argv[2])
    if argv[:2] == ["driver", "main"]:
        return _main_scenario(argv[2])
    raise SystemExit(f"usage: {sys.argv[0]} subscriber | driver env SCENARIO | driver main HOW")


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
