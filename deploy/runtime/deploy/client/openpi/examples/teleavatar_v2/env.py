#!/usr/bin/env python3
"""
Environment wrapper for Teleavatar robot using openpi_client.runtime framework.
"""

import logging
import signal
import threading
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.teleavatar_v2 import ros2_interface
from examples.teleavatar_v2 import rtp_pipeline

# Ignored while close() commands zero chassis velocity: a second Ctrl+C (or a SIGTERM / SIGHUP) must not cut it short.
_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class ObservationUnavailableError(RuntimeError):
    """The robot's sensor data is missing or stale (e.g. the link to the robot broke): the control loop stops rather
    than act on frozen sensors."""


class TeleavatarEnvironment(_environment.Environment):
    """Environment for Teleavatar dual-arm robot."""

    def __init__(
        self,
        prompt: str = "pick a toy and put it in the basket using left gripper",
        control_frequency: float = 30.0,
        interp_frequency: float = 200.0,
        interpolate: bool = True,
        enable_chassis: bool = False,
        chassis_max_velocity: tuple[float, float, float] = ros2_interface.DEFAULT_CHASSIS_MAX_VELOCITY,
        chassis_cmd_timeout: float = ros2_interface.DEFAULT_CHASSIS_CMD_TIMEOUT,
        video_decode: str = "nvidia",
        video_cpu_threads: int = 1,
        video_cpu_thread_type: str = "auto",
        video_convert_threads: int = 4,
        video_va_decoder: str = "vah265dec",
    ):
        """Initialize Teleavatar environment.

        Args:
            prompt: Default language instruction for the policy
            control_frequency: Rate (Hz) apply_action is called; sets the interp ramp duration.
            interp_frequency: Rate (Hz) the interface republishes interpolated des_q (~200 Hz).
            interpolate: Enable des_q interpolation (False = raw ZOH publish).
            enable_chassis: Accept and publish the 3-dim chassis velocity action.
            chassis_max_velocity: With enable_chassis, the limits [vx (m/s), vy (m/s), wz (rad/s)] each chassis command
                is clamped to.
            chassis_cmd_timeout: With enable_chassis, the watchdog commands zero velocity when no chassis command has
                come for this long (s).
            video_decode: H.265 decoding, "nvidia" (nvh265dec), "cpu" (avdec_h265) or "va" (Intel GPU, vah265dec).
            video_cpu_threads, video_cpu_thread_type, video_convert_threads: decoder threads, libav
                thread type and videoconvert threads for video_decode="cpu" (see rtp_pipeline.py).
            video_va_decoder: the VA decoder element for video_decode="va" (vah265dec with one render node).

        Note: Images are NOT resized here. All three views arrive from the
        RTP/H265 composite stream (rtp_video_interface) already split to single
        eyes (head 960×960, wrists 400×640; see
        ros2_interface._POLICY_TO_RTP_VIEW), matching the crops applied to
        the training videos.
        """
        self._prompt = prompt
        self._control_frequency = control_frequency
        self._interp_frequency = interp_frequency
        self._interpolate = interpolate
        self._enable_chassis = enable_chassis
        # Checked here, in the caller's thread: the interface is built in the ROS2 thread, where an error would only
        # show as a timeout.
        ros2_interface.chassis_limits(chassis_max_velocity, chassis_cmd_timeout)
        self._chassis_max_velocity = tuple(chassis_max_velocity)
        self._chassis_cmd_timeout = chassis_cmd_timeout
        self._video_decode = video_decode
        self._video_cpu_threads = video_cpu_threads
        self._video_cpu_thread_type = video_cpu_thread_type
        self._video_convert_threads = video_convert_threads
        self._video_va_decoder = video_va_decoder

        # Initialize ROS2 interface in a separate thread
        self._ros_interface: Optional[ros2_interface.TeleavatarROS2Interface] = None
        self._ros_thread: Optional[threading.Thread] = None
        self._ros_stop = threading.Event()
        self._ros_errors: list[str] = []  # what failed in the ROS2 thread (its executor, its teardown); see close()
        self._chassis_zero_ok: bool | None = None  # close()'s zero chassis velocity: confirmed or not (None: not yet)
        self._init_ros2()

        logging.info(f"TeleavatarEnvironment initialized with prompt: '{prompt}'")

    def _init_ros2(self):
        """Initialize ROS2 in a background thread and wait for initial sensor data.

        The executor runs in a daemon thread until close(), which makes that
        thread tear down in order (executor, node with the video decoder, rclpy).
        Only one environment per process is supported (rclpy.init() cannot be
        called twice). If anything here fails, close() runs before the error
        propagates.

        rclpy installs no signal handlers (SignalHandlerOptions.NO): its own
        would shut the context down on Ctrl+C or SIGTERM while the main thread
        still has to publish the zero chassis velocity, and under the running
        executor. Ctrl+C reaches the main thread as Python's KeyboardInterrupt;
        main.py turns SIGTERM and SIGHUP into an exception too.
        """
        import rclpy
        from rclpy.signals import SignalHandlerOptions

        # Event to signal when executor starts spinning
        spin_started = threading.Event()

        def ros_step(what: str, step) -> None:
            # One step of the thread's work: a failure is logged with its traceback and noted for close(), and the
            # steps after it still run.
            try:
                step()
            except Exception:
                logging.exception(f"ROS2 thread: {what} failed")
                self._ros_errors.append(what)

        def ros_spin():
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            self._ros_interface = ros2_interface.TeleavatarROS2Interface(
                control_frequency=self._control_frequency,
                interp_frequency=self._interp_frequency,
                interpolate=self._interpolate,
                enable_chassis=self._enable_chassis,
                chassis_max_velocity=self._chassis_max_velocity,
                chassis_cmd_timeout=self._chassis_cmd_timeout,
                video_decode=self._video_decode,
                video_cpu_threads=self._video_cpu_threads,
                video_cpu_thread_type=self._video_cpu_thread_type,
                video_convert_threads=self._video_convert_threads,
                video_va_decoder=self._video_va_decoder,
            )

            # Spin in background
            executor = rclpy.executors.MultiThreadedExecutor()
            executor.add_node(self._ros_interface)

            # Signal that spinning is about to start
            spin_started.set()

            def spin() -> None:
                # spin_once with a timeout instead of spin(): close() can end the loop within 0.1 s
                while rclpy.ok() and not self._ros_stop.is_set():
                    executor.spin_once(timeout_sec=0.1)

            def shutdown_executor() -> None:
                executor.shutdown()
                # rclpy Humble's MultiThreadedExecutor.shutdown() does not wait for the callbacks it has already handed
                # to its thread pool (newer rclpy does): one could still run into destroy_node() below and fail with
                # "cannot use Destroyable because destruction was requested", printed at exit as an exception "never
                # retrieved". Wait for them here.
                pool = getattr(executor, "_executor", None)
                if pool is not None:
                    pool.shutdown(wait=True)

            try:
                ros_step("the executor", spin)
            finally:
                ros_step("executor shutdown", shutdown_executor)
                ros_step("node teardown (video decoder, node)", self._ros_interface.destroy_node)
                if rclpy.ok():  # no rclpy signal handler: still up, unless something else shut it down
                    ros_step("rclpy shutdown", rclpy.shutdown)

        self._ros_thread = threading.Thread(target=ros_spin, name="ros2-executor", daemon=True)
        self._ros_thread.start()
        try:
            self._wait_for_ros2(spin_started)
        except BaseException:
            self.close()
            raise

    def _wait_for_ros2(self, spin_started: threading.Event) -> None:
        import time

        # Wait for ROS2 interface object to be created
        timeout = 10.0
        start_time = time.time()
        while self._ros_interface is None and time.time() - start_time < timeout:
            time.sleep(0.1)

        if self._ros_interface is None:
            raise RuntimeError("Failed to initialize ROS2 interface object within timeout")

        logging.info("ROS2 interface object created, waiting for executor to start spinning...")

        # Wait for executor to start spinning
        if not spin_started.wait(timeout=5.0):
            raise RuntimeError("ROS2 executor failed to start spinning")

        logging.info("ROS2 executor started, waiting for initial sensor data...")

        # Now wait for initial sensor data (callbacks can now be triggered)
        if not self._ros_interface.wait_for_initial_data(timeout=30.0):
            if self._video_decode == rtp_pipeline.VA:
                decode_check = f"gst-inspect-1.0 {self._video_va_decoder}; vainfo; the render node is mapped (--device)"
            else:
                decode_check = "; ".join(
                    f"gst-inspect-1.0 {e}" for e in rtp_pipeline.BACKEND_ELEMENTS[self._video_decode]
                )
            raise RuntimeError(
                "Failed to receive initial sensor data. "
                "Please check that the RTP video stream and ROS2 joint topics are up:\n"
                "  - S100 is pushing the RTP/H265 stream to this machine (default port 8890, payload 96)\n"
                f"  - GStreamer H265 decode works ({decode_check})\n"
                "  - ROS_DOMAIN_ID matches the robot (export ROS_DOMAIN_ID=29) in this shell\n"
                "  ros2 topic list\n"
                "  ros2 topic echo /left_arm/joint_states --once"
            )

        logging.info("ROS2 interface initialized successfully with sensor data")

    def close(self, timeout: float = 10.0) -> bool:
        """Tear down. First, with the chassis enabled, command zero chassis velocity and confirm it (see
        TeleavatarROS2Interface.stop), while the video, the executor and the node still run; SIGINT, SIGTERM and SIGHUP
        are ignored meanwhile, so a second Ctrl+C cannot cut it short. Then stop the ROS2 thread and wait for its
        teardown: executor (and the callbacks it has started), node (stops the video decoder first), rclpy.

        Returns True when all of it went well; False when the zero chassis velocity was not confirmed, a step in the
        ROS2 thread failed (each logged an ERROR) or the thread did not stop within `timeout`.

        Call it before the interpreter exits (main.py does). Left running, the thread is still waiting inside rclpy
        when the interpreter finalizes; CPython then ends the daemon thread with pthread_exit, whose unwinding through
        rclpy's C++ code calls std::terminate ("terminate called without an active exception", abort, core dump).
        Safe to call more than once.
        """
        if self._enable_chassis and self._chassis_zero_ok is None and self._ros_interface is not None:
            self._chassis_zero_ok = self._stop_chassis_uninterrupted()
        ok = self._chassis_zero_ok is not False
        self._ros_stop.set()
        thread = self._ros_thread
        if thread is None or thread is threading.current_thread():
            return ok
        thread.join(timeout)
        if thread.is_alive():
            logging.error(f"ROS2 thread did not stop within {timeout:.0f} s")
            return False
        return ok and not self._ros_errors

    def _stop_chassis_uninterrupted(self) -> bool:
        """The interface's stop() with SIGINT, SIGTERM and SIGHUP ignored, their handlers restored afterwards. Only
        in the main thread, the only one that can set handlers (and Python runs them only there)."""
        if threading.current_thread() is not threading.main_thread():
            return self._ros_interface.stop()
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in _STOP_SIGNALS}
        try:
            return self._ros_interface.stop()
        finally:
            for sig, handler in previous.items():
                if handler is not None:  # None: a handler not set from Python, which cannot be put back
                    signal.signal(sig, handler)

    @override
    def reset(self) -> None:
        """Reset the environment.

        For Teleavatar, this is a no-op as we don't have a reset mechanism.
        In a real deployment, you might want to move to a home position here.
        """
        logging.info("Environment reset called (no-op for Teleavatar)")

    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete.

        For Teleavatar, episodes never complete automatically - they must be
        terminated by the user (e.g., Ctrl+C).
        """
        return False

    @override
    def get_observation(self) -> dict:
        """Get current observation from robot sensors.

        Returns:
            Dictionary with keys:
                - 'state': 48-dim proprioceptive state
                - 'images': Dict of camera images in (H, W, C) format at ORIGINAL resolution
                - 'prompt': Language instruction

        Note: Image formats match the (cropped) training data:
            - left_color: RTP split `left_wrist_right_eye`, 400×640×3 (H,W,C)
            - right_color: RTP split `right_wrist_left_eye`, 400×640×3 (H,W,C)
            - head_camera: RTP split `head_left_eye`, 960×960×3 (H,W,C)
        All views are split from the RTP composite frame by the
        RTP video interface; the policy's _parse_image will handle any
        remaining format conversion if needed.
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        # Get raw observation from ROS2
        raw_obs = self._ros_interface.get_observation()
        if raw_obs is None:
            raise ObservationUnavailableError(
                "Observation unavailable (sensor data missing or stale — see the "
                "interface log above). Stopping instead of acting on frozen sensors."
            )

        # Process images: keep original resolution AND keep (H, W, C) format
        # Policy's _parse_image will handle format conversion if needed
        # Return with the exact keys expected by teleavatar_v2_policy.py
        return {
            'observation/state': raw_obs['state'],
            'observation/images/left_color': image_tools.convert_to_uint8(raw_obs['images']['left_color']),
            'observation/images/right_color': image_tools.convert_to_uint8(raw_obs['images']['right_color']),
            'observation/images/head_camera': image_tools.convert_to_uint8(raw_obs['images']['head_camera']),
            'prompt': self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        """Apply action to the robot.

        Args:
            action: Dictionary containing 'actions' key with a 16-dim arm action,
                or a 19-dim arm + chassis action when enabled.
        """
        if self._ros_interface is None:
            raise RuntimeError("ROS2 interface not initialized")

        if 'actions' not in action:
            raise ValueError(f"Action dict must contain 'actions' key, got: {action.keys()}")

        actions = action['actions']
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions, dtype=np.float32)

        # Ensure correct shape
        expected_dim = 19 if self._enable_chassis else 16
        if actions.shape != (expected_dim,):
            raise ValueError(f"Expected {expected_dim}-dim action, got shape {actions.shape}")

        # Publish to ROS2
        self._ros_interface.publish_action(actions)

    def set_prompt(self, prompt: str):
        """Update the language instruction prompt.

        Args:
            prompt: New language instruction
        """
        self._prompt = prompt
        logging.info(f"Updated prompt to: '{prompt}'")
