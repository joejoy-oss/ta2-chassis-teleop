#!/usr/bin/env python3
"""
Main entry point for running Teleavatar robot with OpenPI policy.

This script uses the standard openpi_client.runtime framework for clean,
modular robot control with remote policy inference.

Usage:
    # Start policy server first (in another terminal):
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_teleavatar_v2 \
        --policy.dir=checkpoints/pi05_teleavatar_v2/my_experiment/20000

    # Then run this script:
    python examples/teleavatar_v2/main.py --remote-host 127.0.0.1

Chassis (--enable-chassis): the actions are 19-dim, the 16 arm and gripper dims followed by the chassis velocity
[vx, vy, wz] (m/s, m/s, rad/s), which is published once per action on /api/chassis/velocity. The server must serve
19-dim actions: --policy.config=pi05_teleavatar_v2_chassis (with a 16-dim server, or without --enable-chassis against
a 19-dim one, the first action fails with ValueError before anything is published). Every chassis command is clamped
to --chassis-max-vx / --chassis-max-vy / --chassis-max-wz (0.15 m/s, 0.4 m/s, 0.4 rad/s, confirmed by the owner on
2026-11-05: the robot's API does not limit the speed, TA2 developer documentation §4.5); when no chassis command has
come for --chassis-cmd-timeout (0.3 s; the control loop stalled), a watchdog commands zero velocity until the next
one, sooner than the robot, which keeps the last target for 1 s (§4.5). Stopping, main.py first commands zero
velocity and waits until the subscribers (the zenoh bridge) acknowledge it, then shuts the rest down.

Exit status: 0 when the episodes are done or after Ctrl+C in the control loop; 143 or 129 (128 + the signal number)
after SIGTERM or SIGHUP in the control loop, which stop it the way Ctrl+C does; 3 (EXIT_SENSORS_LOST) when the robot's
sensor data goes missing or stale, e.g. the link to the robot broke (the loop stops rather than act on frozen
sensors); 4 (EXIT_TEARDOWN_FAILED) when, after any of these, a step of the teardown failed (an ERROR says which, e.g.
the zero chassis velocity was not acknowledged) and "Shutdown complete." is not logged; 1 on any other error, with its
traceback. In every case the zero chassis velocity (with --enable-chassis) goes out first, then the ROS2 thread and the
video decoder are shut down. Ctrl+C while still waiting for the first sensor data ends the program with
KeyboardInterrupt, SIGTERM or SIGHUP with TerminationRequested (status 1), after the same teardown.

Stopping it: SIGTERM and SIGHUP stop it as Ctrl+C does only when they reach it. Run with docker exec (on the B70,
openpi-xpu-env's client/run_client.sh exec), main.py gets no signal when its terminal or SSH session goes away: it keeps
driving the robot. docker stop sends it none either: it is SIGKILLed with the rest of the container, and no zero
velocity goes out (the robot clears its target 1 s after the last command). Stop it with Ctrl+C in its terminal or,
from another one, with `client/run_client.sh exec pkill -INT -f examples/teleavatar_v2/main.py`; `run_client.sh stop`
sends it SIGINT first and waits for it to end.
"""

import dataclasses
import logging
import signal
from typing import Literal

from openpi_client import action_chunk_broker
from openpi_client import rtc as _rtc
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from examples.teleavatar_v2 import env as _env
from examples.teleavatar_v2 import ros2_interface as _ros2_interface

EXIT_SENSORS_LOST = 3
"""Exit status when the robot's sensor data goes missing or stale (env.ObservationUnavailableError)."""

EXIT_TEARDOWN_FAILED = 4
"""Exit status when a step of the teardown failed (an ERROR says which), e.g. the zero chassis velocity was not
acknowledged: the chassis may still be moving."""


class TerminationRequested(BaseException):
    """SIGTERM or SIGHUP, raised in the main thread so that main.py stops the way it does on Ctrl+C. A BaseException,
    like KeyboardInterrupt, so that no `except Exception` on the way catches it."""

    def __init__(self, signum: int):
        self.signum = signum
        self.signal_name = signal.Signals(signum).name
        super().__init__(self.signal_name)


def _raise_termination(signum: int, _frame) -> None:
    raise TerminationRequested(signum)


@dataclasses.dataclass
class Args:
    """Command-line arguments for Teleavatar deployment."""

    # Remote policy server settings
    remote_host: str = "127.0.0.1"
    """IP address of the policy server (e.g., '192.168.1.100')"""

    remote_port: int = 8000
    """Port of the policy server"""

    # Control settings
    control_frequency: float = 45.0
    """Control loop frequency in Hz (rate at which the policy command is stepped)"""

    interp_frequency: float = 200.0
    """Rate (Hz) to republish des_q, interpolated from the command stream (~200 Hz)."""

    interpolate: bool = True
    """Interpolate des_q up to interp_frequency; --no-interpolate publishes raw (ZOH)."""

    enable_chassis: bool = False
    """Enable the 3-dim chassis velocity action [vx, vy, wz]: 19-dim actions, from a server started with
    --policy.config=pi05_teleavatar_v2_chassis."""

    chassis_max_vx: float = _ros2_interface.DEFAULT_CHASSIS_MAX_VELOCITY[0]
    """--enable-chassis: |vx| of each chassis command is clamped to this (m/s). The default was confirmed by the owner
    on 2026-11-05; the robot's API does not limit the speed (TA2 developer documentation §4.5)."""

    chassis_max_vy: float = _ros2_interface.DEFAULT_CHASSIS_MAX_VELOCITY[1]
    """--enable-chassis: |vy| of each chassis command is clamped to this (m/s); default confirmed by the owner."""

    chassis_max_wz: float = _ros2_interface.DEFAULT_CHASSIS_MAX_VELOCITY[2]
    """--enable-chassis: |wz| of each chassis command is clamped to this (rad/s); default confirmed by the owner."""

    chassis_cmd_timeout: float = _ros2_interface.DEFAULT_CHASSIS_CMD_TIMEOUT
    """--enable-chassis: when no chassis command has come for this long (s), i.e. the control loop stalled, a watchdog
    commands zero velocity until the next one (the robot itself would keep the last target for 1 s, §4.5)."""

    open_loop_horizon: int = 16
    """Number of actions to execute before querying the policy again. Must not
    exceed the action chunk length of the trained model (30 for the teleavatar
    configs). Ignored when --rtc is set."""

    # Video decoding (the RTP/H265 camera stream)
    video_decode: Literal["nvidia", "cpu", "va"] = "nvidia"
    """H.265 decoding: "nvidia" (nvh265dec + CUDA colour conversion), "cpu" (avdec_h265 + videoconvert, for
    machines without an NVIDIA GPU) or "va" (vah265dec on an Intel GPU such as the B70 server's, with the colour
    conversion on the GPU straight into memory; needs the openpi-xpu-client image and a mapped render node)."""

    video_cpu_threads: int = 1
    """--video-decode cpu: avdec_h265 max-threads. 0 lets gst-libav choose: the host's CPU count, at most 16, whatever
    the container's CPU limit. Frame threads add a frame of latency each."""

    video_cpu_thread_type: Literal["auto", "frame", "slice"] = "auto"
    """--video-decode cpu: avdec_h265 thread-type; "slice" only helps streams encoded with WPP."""

    video_convert_threads: int = 4
    """--video-decode cpu: videoconvert n-threads."""

    video_va_decoder: str = "vah265dec"
    """--video-decode va: the VA decoder element (vah265dec when one render node is mapped into the container; with
    several, the others are named after their node, e.g. varenderD129h265dec)."""

    # Real-Time Chunking
    rtc: bool = False
    """Query the policy asynchronously and splice chunks instead of stalling the control
    loop. Requires a server started with --rtc.enabled. See docs/real_time_chunking.md."""

    rtc_calibration_steps: int = 5
    """Warm start: inferences to time before executing anything, to size the delay and
    execution horizon. 0 uses --rtc-inference-delay instead."""

    rtc_warmup_steps: int = 2
    """Inferences discarded before timing starts (the first JAX call compiles)."""

    rtc_inference_delay: int | None = None
    """Override the measured delay, in control steps."""

    rtc_execution_horizon: int | None = None
    """Override the execution horizon, in control steps. Default: 2x the delay."""

    # Task settings
    prompt: str = "Stack the three blocks"
    """Language instruction for the robot"""

    # Episode settings
    num_episodes: int = 100
    """Number of episodes to run"""

    max_episode_steps: int = 0
    """Maximum steps per episode (0 = unlimited)"""


def main(args: Args) -> int:
    """Main function to run Teleavatar with policy inference. Returns the exit status (see the module docstring)."""

    # rclpy installs no signal handlers (env.py): Ctrl+C is Python's KeyboardInterrupt, and SIGTERM (kill) and SIGHUP
    # (the terminal went away) become TerminationRequested, so that all three end in the same teardown. Set before the
    # environment exists: a signal while it waits for the first sensor data tears it down as well. Under docker exec
    # (run_client.sh exec on the B70) neither a closed terminal or SSH session nor docker stop sends main.py any of them;
    # docker stop SIGKILLs it (module docstring).
    signal.signal(signal.SIGTERM, _raise_termination)
    signal.signal(signal.SIGHUP, _raise_termination)

    logging.info("=" * 60)
    logging.info("Teleavatar OpenPI Deployment")
    logging.info("=" * 60)
    logging.info(f"Policy server: ws://{args.remote_host}:{args.remote_port}")
    logging.info(f"Control frequency: {args.control_frequency} Hz")
    if args.interpolate:
        logging.info(f"des_q interp publish: {args.interp_frequency} Hz (ZOH-staircase fix)")
    else:
        logging.info("des_q interpolation: OFF (raw commands, original ZOH behavior)")
    if args.enable_chassis:
        logging.info(
            f"Chassis control: ON (19-dim actions; limits vx {args.chassis_max_vx} m/s, vy {args.chassis_max_vy} m/s, "
            f"wz {args.chassis_max_wz} rad/s; zero velocity after {args.chassis_cmd_timeout} s without a command)"
        )
    else:
        logging.info("Chassis control: OFF")
    if args.rtc:
        logging.info("Real-Time Chunking: ON (async inference)")
    else:
        logging.info(f"Open-loop horizon: {args.open_loop_horizon} steps")
    logging.info(f"Prompt: '{args.prompt}'")
    logging.info("=" * 60)

    # Create WebSocket client policy
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.remote_host,
        port=args.remote_port,
    )

    # Get and log server metadata
    metadata = ws_client_policy.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {metadata}")

    # Create policy agent with action chunking
    if args.rtc:
        broker = _rtc.RTCActionBroker(
            policy=ws_client_policy,
            config=_rtc.RTCBrokerConfig(
                control_frequency=args.control_frequency,
                warmup_steps=args.rtc_warmup_steps,
                calibration_steps=args.rtc_calibration_steps,
                inference_delay=args.rtc_inference_delay,
                execution_horizon=args.rtc_execution_horizon,
            ),
        )
    else:
        broker = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.open_loop_horizon,  # Execute this many actions before querying
        )
    agent = _policy_agent.PolicyAgent(policy=broker)

    # Create Teleavatar environment (from here on, its ROS2 thread runs until environment.close())
    # Note: Images are kept at original resolution to match training data
    environment = _env.TeleavatarEnvironment(
        prompt=args.prompt,
        control_frequency=args.control_frequency,
        interp_frequency=args.interp_frequency,
        interpolate=args.interpolate,
        enable_chassis=args.enable_chassis,
        chassis_max_velocity=(args.chassis_max_vx, args.chassis_max_vy, args.chassis_max_wz),
        chassis_cmd_timeout=args.chassis_cmd_timeout,
        video_decode=args.video_decode,
        video_cpu_threads=args.video_cpu_threads,
        video_cpu_thread_type=args.video_cpu_thread_type,
        video_convert_threads=args.video_convert_threads,
        video_va_decoder=args.video_va_decoder,
    )

    status = 0
    try:
        # Create runtime
        runtime = _runtime.Runtime(
            environment=environment,
            agent=agent,
            subscribers=[],  # Could add video recording, logging, etc.
            max_hz=args.control_frequency,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

        # Run!
        logging.info("\nStarting robot control loop...")
        logging.info("Press Ctrl+C to stop\n")
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n\nStopping robot (Ctrl+C pressed)...")
    except TerminationRequested as exc:
        logging.info(f"\n\nStopping robot ({exc.signal_name} received)...")
        status = 128 + exc.signum
    except _env.ObservationUnavailableError as exc:
        logging.error(f"Stopping: {exc}")
        status = EXIT_SENSORS_LOST
    finally:
        # No return in here: it would swallow an exception on its way out (status 1, with its traceback).
        teardown_ok = environment.close()
        if teardown_ok:
            logging.info("Shutdown complete.")
        else:
            logging.error("Shutdown failed: a teardown step went wrong (see the ERROR above)")
    return status if teardown_ok else EXIT_TEARDOWN_FAILED


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',  # note: %(asctime)s below carries ms via the format string
        force=True
    )

    # Parse arguments and run
    args: Args = tyro.cli(Args)
    sys.exit(main(args))
