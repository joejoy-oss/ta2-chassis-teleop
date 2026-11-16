# TeleAvatar V2 — Deployment & Data Format

TeleAvatar V2 at a glance: all three cameras are stereo and are tiled into **a single RTP/H.265 video stream** (decoded with GStreamer — no ROS2 camera topics); the platform has built-in low-level control, so position commands are published directly to `/api/<arm>/joint_cmd` (**no** PD relay needed); grippers use a left/right symmetric piecewise force curve.

- **Training configs**: `pi0_teleavatar_v2` / `pi05_teleavatar_v2` / `pi0_teleavatar_v2_low_mem_finetune`
- **Data config class**: `LeRobotTeleavatarV2DataConfig`
- **Actions**: `pi05_teleavatar_v2` and `pi05_teleavatar_v2_rtc` predict the 14 arm joints relative to the current state (`use_delta_joint_actions=True`; the grippers stay absolute), in training and serving alike. Serve a checkpoint with the setting it was trained with, and compute its norm stats with it too.
- **Policy module**: [`teleavatar_v2_policy.py`](../../src/openpi/policies/teleavatar_v2_policy.py)
- **Dataset conversion**: [rosbag_to_dataset_TA2](https://github.com/dexteleop/rosbag_to_dataset_TA2)

For installation, the fine-tuning workflow, and the shared data format, see the [main README](../../README.md).

## 📑 Table of Contents

- [Client Dependencies](#-client-dependencies)
- [Robot-Side Setup](#-robot-side-setup)
  - [1. Switch the Robot to API Mode](#1-switch-the-robot-to-api-mode)
  - [2. Point the Video Stream at Your Host](#2-point-the-video-stream-at-your-host)
  - [3. Bridge ROS2 Topics with zenoh](#3-bridge-ros2-topics-with-zenoh)
- [Deployment Data Flow](#-deployment-data-flow)
- [Deployment Steps](#-deployment-steps)
  - [1. Zero the Robot (Client)](#1-zero-the-robot-client)
  - [2. Start the Policy Server (Server)](#2-start-the-policy-server-server)
  - [3. Run the Robot Client (Client)](#3-run-the-robot-client-client)
- [Topics & Video Stream](#-topics--video-stream)
- [Camera Format](#-camera-format)
- [Gripper Mapping](#-gripper-mapping)
- [Additional Tools](#-additional-tools)

## 📦 Client Dependencies

Camera images arrive as an RTP/H.265 video stream (receiver implemented in [`rtp_video_interface.py`](rtp_video_interface.py)), which requires GStreamer and PyGObject. The current `environment.yml` already includes these dependencies; if your client environment was created from an older `environment.yml`, install them into the conda environment:

```bash
conda install -c conda-forge pygobject gst-python gstreamer \
    gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav
```

Environment self-check for the default NVIDIA decoding (`nvidia-smi` must work and all GStreamer elements must be present):

```bash
nvidia-smi
for e in rtph265depay h265parse nvh265dec cudaconvert cudadownload appsink; do
  gst-inspect-1.0 "$e" >/dev/null && echo "$e OK" || echo "$e MISSING"
done
```

A host without an NVIDIA GPU decodes on an Intel GPU with VA-API (`--video-decode va`) or on the CPU (`--video-decode cpu`), see [Run the Robot Client](#3-run-the-robot-client-client). The VA backend needs `vah265dec` (the `va` plugin of gst-plugins-bad), a render node (`/dev/dri/renderD12x`, in a container `--device` plus `--group-add` of the render group) and `libvapp.so`, a small C library that converts the decoded frames with VA-API video processing (built from `openpi-xpu-env/client/va_vpp/vapp.c`); the CPU backend needs `avdec_h265` (gst-libav) and `videoconvert`. Neither needs `nvh265dec cudaconvert cudadownload` or `nvidia-smi`. The `openpi-xpu-client` image (`openpi-xpu-env/client/`) has both; it runs the client on the B70 inference server with `--video-decode va`.

## 🔌 Robot-Side Setup

Before deploying, the robot must be switched out of VR/teleoperation mode and connected to your client host. This setup is one-time (the configuration is persisted across reboots). The robot ships wired into the LAN; connect your client host to the same LAN. The robot's IP (referred to as `<ROBOT_IP>` below) can be checked in remoteApp — see the TA2 user manual for the configuration UI walkthrough.

### 1. Switch the Robot to API Mode

In the **System Config** panel (via remoteApp or a display attached to the robot), set:

- **Run mode**: API (instead of VR)
- **Left / right arm**: enabled, with control mode set to **joint** — the policy publishes joint position commands to `/api/<arm>/joint_cmd`

Click **Save config** (persists the change), then **Apply** (takes effect immediately).

### 2. Point the Video Stream at Your Host

In the same config panel, fill the **remote IP** field with your client host's IP. The robot then pushes the combined RTP/H.265 camera stream to your host over UDP (default port 8890 — see [Topics & Video Stream](#-topics--video-stream)), so the host only has to be reachable from the robot: on the robot's subnet, or behind a router the robot can send through.

### 3. Bridge ROS2 Topics with zenoh

The robot exposes its ROS2 topics across machines through zenoh: the robot runs a zenoh router, and your host connects to it with `zenoh-bridge-ros2dds`, after which the robot's topics are discoverable on your host as if they were local.

Prerequisites: ROS2 Humble and `zenoh-bridge-ros2dds` installed on the client host, and `<ROBOT_IP>:9000` (TCP) reachable from it.

```bash
# Terminal 1: connect to the robot's zenoh router (keep this running)
export ROS_DOMAIN_ID=29
export ROS_DISTRO=humble
zenoh-bridge-ros2dds -e tcp/<ROBOT_IP>:9000
```

Verify from another terminal:

```bash
export ROS_DOMAIN_ID=29
ros2 topic list
ros2 topic echo /right_arm/joint_states
```

> The bridge terminal must stay open — closing it disconnects the robot. **Every terminal that talks to the robot (including `zero.py` / `main.py` below) must export the same `ROS_DOMAIN_ID=29`**, otherwise the bridged topics are not discoverable. If the client runs inside a container, make sure the container network can reach `<ROBOT_IP>:9000`. A host that cannot reach it (outside the robot's LAN) can forward the port with SSH from a machine that can, and run the bridge in client mode against the forwarded port; `openpi-xpu-env/client/README.md` describes this setup.

## 🔄 Deployment Data Flow

```
cameras (RTP/H.265 stream) ┐
                           ├→  ros2_interface  →  main.py (env)  ──WebSocket──▶  serve_policy (policy)
joints (ROS2)              ┘                          ▲                                 │
                                                      └─ 16-dim actions (19: chassis) ◀─┘
                                                      │
                                                      ▼
                                  /api/left_arm/joint_cmd,  /api/right_arm/joint_cmd  (position commands)
                                  /api/left_gripper/cmd,    /api/right_gripper/cmd    (trigger commands)
                                  /api/chassis/velocity     (with --enable-chassis: velocity commands)
```

> `serve_policy` runs on the **server** (uv environment); `zero.py` and `main.py` run on the **client** (conda + ROS2 environment). ROS2 traffic between the robot and the client goes through the zenoh bridge (see [Robot-Side Setup](#-robot-side-setup)). Do **not** run V1's `arm_pd_controller.py` on V2 — it publishes to the same topics as V2's `ros2_interface` and would conflict.

## 🚀 Deployment Steps

The steps below assume [Robot-Side Setup](#-robot-side-setup) is complete: the robot is in API mode with joint control, the video stream is pushing to your host, and the zenoh bridge is running.

### 1. Zero the Robot (Client)

```bash
export ROS_DOMAIN_ID=29
python examples/teleavatar_v2/zero.py
```

Interpolates from the current joint positions to a preset home pose over 5 seconds (the target pose is hardcoded in the script; joint limits are read from `arm_config.yml` at the repository root), publishing at 100 Hz to `/api/{left,right}_arm/joint_cmd` and setting `/api/fsm/enable=1` to enable the robot. It exits automatically once both arms converge.

> Zeroing must finish **before** starting `main.py` — both publish commands to `/api/{arm}/joint_cmd` and would conflict if run simultaneously. Alternatively, use [`zero_from_episode.py`](zero_from_episode.py) to reset to the starting pose of a dataset episode (see [Additional Tools](#-additional-tools)).

### 2. Start the Policy Server (Server)

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_teleavatar_v2 \
    --policy.dir=checkpoints/pi0_teleavatar_v2/my_experiment/20000
```

`--policy.config` must match the training config of the checkpoint (the two generations differ in image cropping and gripper conversion; a mismatch fails silently). `--policy.dir` must contain `assets/<dataset_name>/norm_stats.json` (packaged automatically at training time). Default port: `8000`.

### 3. Run the Robot Client (Client)

```bash
export ROS_DOMAIN_ID=29
python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --prompt "stack the three blocks"
```

Common flags (see `Args` in `main.py`): `--control-frequency` (control loop rate), `--open-loop-horizon` (steps executed before re-querying the policy), `--prompt` (language instruction). The client performs action chunking via `ActionChunkBroker`.

The control loop (`Runtime` in `openpi_client`) runs its steps at fixed times, one period of `--control-frequency` apart: when a sleep returns late, the next one is shorter, so the delays do not add up, and a step that runs over its period restarts the schedule from there instead of being followed by catch-up steps. Before, each step slept a full period after the previous sleep had returned, so the loop ran a little slower than asked (about 44.4 Hz for 45 Hz on the B70 server). This applies to every client built on `Runtime`.

`main.py` stops when the robot's sensor data goes missing or stale (for example when the link to the robot breaks) rather than act on frozen sensors, and exits with status 3. It exits with 0 when the episodes are done or after Ctrl+C in the control loop, with 143 or 129 (128 + the signal number) after SIGTERM or SIGHUP there, which stop it the way Ctrl+C does when they reach it (run with `docker exec`, neither a closed terminal nor `docker stop` sends them: see **Stopping it** below), and with 1 and a traceback on any other error. In each case it first commands zero chassis velocity (with `--enable-chassis`, below), then shuts down the ROS 2 thread and the camera decoder; when a step of that teardown fails, it logs an ERROR instead of "Shutdown complete." and exits with 4 (an error on its way out keeps 1). Ctrl+C while it still waits for the first sensor data ends it with KeyboardInterrupt, SIGTERM or SIGHUP with TerminationRequested (status 1).

**Chassis** (`--enable-chassis`, off by default), for a checkpoint that also drives the chassis, served with `--policy.config=pi05_teleavatar_v2_chassis`: the actions are 19-dim, the 16 above followed by the chassis velocity `[vx, vy, wz]` (m/s, m/s, rad/s), which `main.py` publishes once per action on `/api/chassis/velocity`. A server and a client that disagree (16 against 19 dims) fail at the first action, before anything is published. The robot's API does not limit the speed, and when commands stop it keeps the last target for 1 s (TA2 developer documentation §4.5), so the client:

- clamps every command to `--chassis-max-vx`, `--chassis-max-vy`, `--chassis-max-wz` (0.15 m/s, 0.4 m/s, 0.4 rad/s; confirmed by the owner on 2026-11-05), with a throttled warning, and replaces a velocity that is not 3 finite numbers by zero, with a throttled error;
- commands zero velocity when no chassis command has come for `--chassis-cmd-timeout` (0.3 s) because the control loop stalled (the RTC queue ran dry, a slow server), until the next one;
- stopping (Ctrl+C, SIGTERM or SIGHUP when they reach it, an error, the end of the episodes), commands zero velocity first, before the video, the ROS 2 thread and the node go, and waits until the zenoh bridge's DDS reader acknowledges it (0.5 s at a time, publishing it again up to 3 times in all; in tests the acknowledgement took up to about 0.2 s); a second Ctrl+C meanwhile is ignored (one after that, while the rest shuts down, still cuts the teardown short, as before). When it cannot be confirmed (nothing subscribed, no acknowledgement), an ERROR says so and the exit status is 4.

The robot clears its target as soon as the zero velocity arrives and slows down at up to 0.6 m/s² (§4.5); if the client dies without sending it (SIGKILL, out of memory, a broken link), the robot clears the target by itself after 1 s.

**Stopping it** when it runs with `docker exec`, as in the B70's client container (`openpi-xpu-env/client/run_client.sh exec`): `main.py` gets no signal when its terminal or SSH session goes away, and keeps driving the robot; `docker stop` (also `docker kill`, `docker rm -f`) sends it none either and SIGKILLs it with the rest of the container, so no zero velocity goes out and only the robot's 1 s timeout stops the chassis. So:

- stop `main.py` only with Ctrl+C in its terminal or, from another terminal, with `client/run_client.sh exec pkill -INT -f examples/teleavatar_v2/main.py`. `client/run_client.sh stop` does the same first: SIGINT to `main.py` and `zero.py` in the container, then it waits up to `STOP_WAIT` s (default 10) for them to end before it stops the container. Do not `docker stop` the container yourself;
- when the SSH connection drops while `main.py` runs, it goes on: use the robot's emergency stop on site, then reconnect and stop `main.py` as above;
- on an ERROR "the chassis may still be moving" or exit status 4, watch the chassis and use the emergency stop if needed.

Before enabling the chassis, with the bridge connected and `main.py` not started yet, run `ros2 topic info -v /api/chassis/velocity` on the client (in the B70's container: `client/run_client.sh exec ros2 topic info -v /api/chassis/velocity`). It must list a SUBSCRIPTION by the node `zenoh_bridge_ros2dds` with `Reliability: RELIABLE`. With none, the robot's chassis node is not matched and the commands would not reach it: do not run. With BEST_EFFORT, the "acknowledged" in the log proves nothing. The first drive uses the default limits above, 0.15 m/s, 0.4 m/s and 0.4 rad/s (decided by the owner on 2026-11-05). They were set from the chassis checkpoint's norm stats, not chosen as a low speed for a first drive: 0.4 m/s sideways is not slow for one. Confirm the direction of each axis first (TA2 developer documentation §4.5), with someone at the emergency stop throughout; check once that the chassis starts slowing down at Ctrl+C rather than going on for about 1 s.

Camera decoding: by default the stream is decoded on an NVIDIA GPU (`nvh265dec`). On a host with an Intel GPU, add `--video-decode va`: `vah265dec` decodes into GPU memory and [`va_frame.py`](va_frame.py) converts each frame on the GPU with VA-API video processing (`libvapp.so`), written straight into the client's memory, then hands over the RGB frame (BT.709 limited range in, full range out, as the NVIDIA and CPU paths convert this stream). `--video-va-decoder` names the decoder element (default `vah265dec`; with more than one render node visible, GStreamer names the others e.g. `varenderD129h265dec`, so map only the one you want). The conversion is set up before the pipeline starts playing, for frames of 1280×2720 (the robot's composite); a stream of another size is handled after a one-time rebuild. The parts that do not need the decoder's VA display (cv2, the buffers) are done before the UDP port is bound: nothing reads the socket until the pipeline plays. On the B70 the media engine's clock decides most of the decode time: at its default minimum (400 MHz) the robot's stream (1280×2720, all intra, about 39 Mbit/s) took about 10.5 ms per frame at p95. With the minimum raised to 1500 MHz, this code took 6.25–6.27 ms per frame at p95 on card 0 (`renderD128`, three runs) and 6.26 / 6.28 ms on card 1 (`renderD129`, two runs), each run replaying a recording of the robot's stream for 10 minutes (stage B3 of the B70 client project). `openpi-xpu-env/client/run_client.sh` raises the minimum while the client container runs (`gpu_minfreq.sh`). The decoder's own first frame is slow (about 36 ms), so the frame after it can be dropped once at the start of the stream. Without an Intel or NVIDIA GPU, add `--video-decode cpu` to decode with `avdec_h265` and convert with `videoconvert`; `--video-cpu-threads` (default 1), `--video-cpu-thread-type` (default `auto`) and `--video-convert-threads` (default 4) set the decoder threads, the libav thread type and the conversion threads. Each libav frame thread delays every frame by one frame interval (about 22 ms at 45 fps), and slice threads only help a stream encoded with WPP (wavefront parallel processing): otherwise libavcodec decodes the slices or tiles of a picture one after another. The CPU pipeline receives, parses and decodes on separate threads, so a frame that is slow to decode does not keep the socket from being read. Whether a CPU keeps up depends on the stream and the host: on the B70 server (Xeon Gold 5418N), decoding and converting a 20 Mbit/s 1280×2720 stream without WPP on one thread takes about 20.6 ms per frame on average and about 58 ms for a keyframe, too close to the 22 ms frame interval, and more threads do not help such a stream. A stream encoded with WPP kept up there at 20 and 30 Mbit/s, twice each in 3-minute runs, with `--video-cpu-thread-type slice --video-cpu-threads 8` and the default 4 conversion threads (slice threads add no delay). Set the thread type and count explicitly: `--video-cpu-threads 0` lets gst-libav take the host's CPU count, at most 16, whatever the container's CPU limit (16 threads on that server), and with `auto` the thread type depends on whether the source is live. The server's decoding cores sometimes stay at the base clock (1.8 GHz) after decoding starts, for anything from a few seconds to a whole 3-minute run, and the latency is higher meanwhile. Measure with `test.py --stats-json` (below) before relying on it.

## 📡 Topics & Video Stream

Topics used by `ros2_interface.py`:

**Subscribed (observations)**

| Topic                             | Type           | Purpose                    |
| --------------------------------- | -------------- | -------------------------- |
| `/left_arm/joint_states`          | `JointState`   | Left arm joint states      |
| `/right_arm/joint_states`         | `JointState`   | Right arm joint states     |

**Cameras (RTP/H.265 video stream, not ROS2)**

The S100 downsamples the head (2 eyes) + both wrist (2 eyes each) cameras, aligns them by exposure time, and tiles them into a single 2720×1280 frame, which is H.265-encoded and streamed over RTP (default port 8890, payload 96, ~45 fps). On the client, `RTPH265VideoInterface` in [`rtp_video_interface.py`](rtp_video_interface.py) decodes it with GStreamer (`nvh265dec`, `vah265dec` with `--video-decode va`, or `avdec_h265` with `--video-decode cpu`) and crops six fixed sub-views; `ros2_interface` maps three of them to policy inputs (this mapping has been verified on the real robot and matches the inner-eye crops used at training time):

| Policy observation key | RTP sub-view            | Size (H×W) | Description                    |
| ---------------------- | ----------------------- | ---------- | ------------------------------ |
| `head_camera`          | `head_left_eye`         | 960×960    | Head left eye                  |
| `left_color`           | `left_wrist_right_eye`  | 400×640    | Left wrist right eye (inner)   |
| `right_color`          | `right_wrist_left_eye`  | 400×640    | Right wrist left eye (inner)   |

> If the sender's tiling layout changes, update the crop coordinates in `TELEAVATAR_SPLIT_REGIONS` inside the interface accordingly; use [`test.py`](test.py) to test stream reception.

**Published (actions)**

| Topic                         | Type         | Publisher        | Purpose                            |
| ----------------------------- | ------------ | ---------------- | ---------------------------------- |
| `/api/left_arm/joint_cmd`     | `JointState` | ros2_interface   | Left arm target joint positions    |
| `/api/right_arm/joint_cmd`    | `JointState` | ros2_interface   | Right arm target joint positions   |
| `/api/left_gripper/cmd`       | `Float32`    | ros2_interface   | Left gripper trigger command       |
| `/api/right_gripper/cmd`      | `Float32`    | ros2_interface   | Right gripper trigger command      |
| `/api/chassis/velocity`       | `Float32MultiArray` | ros2_interface | Chassis velocity `[vx, vy, wz]` (`--enable-chassis` only) |
| `/api/fsm/enable`             | `Float32`    | ros2_interface   | Enable FSM                         |

## 📷 Camera Format

All three cameras are **side-by-side stereo**; a single eye is cropped from each before being fed to the model:

| Dataset key     | Raw resolution (dataset videos) | Model input key     | Processing at training time (`TeleavatarInputs`)      |
| --------------- | ------------------------------- | ------------------- | ------------------------------------------------------ |
| `head_camera`   | 3840×1920 (2:1 stereo)          | `base_0_rgb`        | Crop the left eye → 1920×1920 square main view          |
| `left_color`    | 2560×800 (stereo)               | `left_wrist_0_rgb`  | Crop the **right eye** → 1280×800 (inner, facing the table center) |
| `right_color`   | 2560×800 (stereo)               | `right_wrist_0_rgb` | Crop the **left eye** → 1280×800 (inner, facing the table center)  |

> The V2 head camera is mounted **right-side-up**: `rotate_head_camera=False` (the default of `LeRobotTeleavatarV2DataConfig`); **training and inference must be consistent**. The crop logic uses a `width >= 2 * height` aspect-ratio check: raw stereo frames (head 2:1, wrists 3.2:1) are cropped, while already-cropped single-eye frames (1:1, 1.6:1) are skipped automatically. At inference, the client takes the pre-split single-eye views directly from the RTP stream (head 960×960, wrists 400×640 — half the resolution of the dataset videos), which hit the skip branch; the policy side then resizes everything to 224×224.

## 🤏 Gripper Mapping

The platform accepts a trigger value in `[0, 1]` and maps it to an effort (Nm) through a piecewise linear curve (left/right symmetric; positive = opening, negative = closing, zero crossing at trigger = 0.10):

```
trigger <  0.10:  effort = +2.0 × (1 − trigger / 0.10)     # opening segment: +2.0 → 0
trigger >= 0.10:  effort = −1.6 × (trigger − 0.10) / 0.90  # closing segment:  0 → −1.6
```

In practice: fully open → 0.0, grasping → 0.6–1.0 (1.0 is the rated maximum closing force); avoid the 0.10–0.30 range, which sits near the zero crossing where the force is minimal. At training time, `TeleavatarInputs` applies the inverse of this curve to convert the dataset's gripper efforts to trigger values, and the model output is converted back after inference. The conversion happens before dataset norm-stats normalization — if you change it, you must re-run `scripts/compute_norm_stats.py`.

## 🧰 Additional Tools

- [`replay_episode.py`](replay_episode.py): Replay a dataset episode on the real robot (first eases from the current pose to frame 0, then replays at the recorded frame rate; `--dry-run` needs no ROS2 and only prints a trajectory summary):

  ```bash
  python examples/teleavatar_v2/replay_episode.py --dataset <dataset_path> --episode 0 [--speed 0.5]
  ```

- [`zero_from_episode.py`](zero_from_episode.py): Ease both arms to the pose of a given frame of an episode (frame 0 by default, `--frame -1` for the last frame), exiting once converged:

  ```bash
  python examples/teleavatar_v2/zero_from_episode.py --dataset <dataset_path> --episode 0 [--frame -1]
  ```

- [`test.py`](test.py): RTP reception / decoding / six-way split-and-save test, for checking the video pipeline before deployment. It takes the same decoding flags as `main.py`. `--stats-json` measures the decoder instead of (or as well as) recording: every frame's `decode_latency` (the value the interface logs, from the parsed access unit to the decoded RGB frame, including any wait for the decoder) with its percentiles, fps per 10 s window, RTP sequence gaps, frames dropped before the appsink (with when, in seconds after the first frame, and how many came later than the first second), and `receive_wait_ms_max`, the longest a packet waited before it was parsed (not part of `decode_latency`). With `--video-decode va` it adds the conversion's set-up times, its per-frame steps and the decoder's own latency of the first 45 frames:

  ```bash
  python examples/teleavatar_v2/test.py --video-decode va --duration-s 600 --stats-json decode_stats.json
  ```
