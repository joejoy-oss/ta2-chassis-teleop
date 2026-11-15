# TA2 chassis teleop

Scripts for the TeleAvatar 2 (TA2) robot from the B70 inference host: chassis test and stepper scripts,
standalone arm poses, a one-command VLA grasp run, and a web remote (Chinese UI) with a software
emergency stop, VLA grasp button, hold-to-move chassis control, trajectory recording and path replay.

```
browser (any device on the robot router) --HTTP 8088--> B70: web_move.py
                                                           | stdin/stdout JSON
                                       openpi-xpu-client container: teleop.py (ROS 2 node)
                                                           | zenoh bridge (inside the container)
                                                         robot 192.168.31.218
```

Measured on the real robot (see [docs/measurements.md](docs/measurements.md)): **+vx = right, +vy = forward,
+wz = CCW**. The developer doc's "x forward, y left" is wrong for this robot.

## Files

| file | runs where | what |
|---|---|---|
| `chassis_test.py` | container, `/outputs/tools/` | one constant-velocity move with self-checks and a stop report |
| `move.sh` | B70 `/root/` | `./move.sh forward|back|left|right [cm]`, `all`, `stop` |
| `teleop.py` | container, `/outputs/tools/` | long-running ROS node behind the web remote: steps, hold, odometry, path following |
| `web_move.py` | B70 `/root/` | web server on port 8088 (page, E-STOP, VLA grasp, arms, chassis) |
| `vla/vla_run.sh` | B70 `/root/` | reset arms to the training start pose, then run the VLA policy (default 120 s) |
| `install_b70.sh` | B70 | copies the files into place and restarts the web remote (never moves the robot) |
| `paths/001-004.csv` | B70 `/root/ta2_paths/` | recorded L-shaped paths; run in order they form a closed loop |
| `tests/` | anywhere | fake rclpy + robot simulator and `sim_paths.py` |

## Setup on B70

1. The official client image must be loaded and running (see the organisers' `deploy/client/README.md`):
   ```bash
   cd /data/ta2/software/deploy
   export OPENPI_DIR=$(pwd)/client/openpi
   ZENOH_ENDPOINT=tcp/192.168.31.218:9000 client/run_client.sh start
   ```
   Do not start a second zenoh bridge on the host. Stop the container only with `client/run_client.sh stop`.
2. Robot: dev config -> run mode API, chassis enabled -> Save config -> Apply -> reboot.
3. From a checkout of this repo on B70: `./install_b70.sh`. The web remote does not start by itself after
   B70 reboots; run it again.

## Web remote (http://192.168.31.228:8088)

Single column, Chinese labels, works on a phone. From top to bottom:

- **Top bar** (always visible): connection, API mode, current activity, and the only red button,
  **急停 (E-STOP)**; the **Esc** key does the same. See Safety below.
- **VLA 抓取**: the big button 开始抓取（60 秒） runs `vla/vla_run.sh` for 60 s at max 0.5 rad/s with the
  prompt 抓取沙包 (Start pose first, then the policy). While it runs the page shows a progress bar and a
  停止 button; the line below shows the last result (steps and end reason). It needs the policy server
  container (`sandbag-step4999-policy`) running and refuses to start while anything else moves.
  Below it, small arm buttons: 起始姿态 (training start pose), 抬起 (raise), 放下 (lower).
- **底盘**: direction pad 前进 / 后退 / 左 / 右 and 逆时针 / 顺时针. Click (< 0.3 s): 10 cm or 15 deg.
  Press and hold: 0.10 m/s or 0.2 rad/s until release. The centre 停 is the soft STOP: chassis, paths,
  arm script and VLA run stop, arms hold where they are.
- **路径与轨迹** (folded): wheel-odometry pose and robot footprint; recording (start / stop / save as CSV,
  copy in `/root/ta2_trails/`, reset origin); paths from `/root/ta2_paths/` plus any loaded from the
  browser. Run follows one path from the current pose (its first point is anchored there); Run all follows
  them in name order, each from where the previous ended. Corners are driven through without stopping;
  in Run all the robot stands still for 10 s after each path before starting the next one (`PATH_PAUSE`).
  Paths do not depend on the page: once started they run to the end even if the phone loses Wi-Fi or
  the page is closed; only 停 / 急停 end them early. Click a row to preview it on the map.
- **日志** (folded): recent events, including VLA output and E-STOP results.

CSV format (written by Save, read by Paths): `t_s,x_m,y_m,heading_deg`, x = right, y = forward, heading CCW,
relative to the origin.

## Safety

- **急停 (E-STOP)**, top bar or Esc: stops the chassis and any path, sends SIGINT to every
  `guarded_grasp.py` (VLA) and `arm_pose.py` in the container whoever started them (force-killed after
  3 s), then publishes `/api/fsm/enable` 0.0 for 2 s so the robot PAUSEs: motors stay powered and the arms
  hold their pose (they do not drop); grippers keep their last state. It is a software stop and needs B70,
  the network and the page to work: keep someone at the robot's hardware emergency stop.
- While a VLA run, an arm move or a path is active, the other motion buttons are refused, so two programs
  never command the arms at once.
- Every move requires API mode + chassis enabled and fresh wheel feedback (< 0.3 s).
- Hold is a dead-man switch: the browser sends a heartbeat every 0.1 s; no heartbeat for 0.4 s -> stop.
  One hold is capped at 10 s.
- Paths ignore the page connection; they stop at the end, on 停 / 急停, or on the robot-side checks:
  stale wheel feedback, 5 s without progress, timeout, or not in API mode. Run all is cancelled the
  same way. Paths longer than 6 m are refused.
- If `web_move.py` exits, `teleop.py` sees stdin close, sends zero and exits.
- `chassis_test.py` limits: planar speed 0.15 m/s (`--vmax` up to 0.3), 0.3 rad/s, 12 s; Ctrl+C sends zero.
- Commands from `docker exec` do not get SIGHUP when an SSH session drops: keep someone at the e-stop.
- Anyone on the robot router can open the web remote; there is no password.

## Command line

```bash
ssh -t root@192.168.31.228 ./move.sh forward       # 10 cm; -t so Ctrl+C reaches the robot
ssh -t root@192.168.31.228 ./move.sh left 5
ssh -t root@192.168.31.228 ./move.sh all           # forward, back, left, right
cd /data/ta2/software/deploy && client/run_client.sh exec python3 /outputs/tools/chassis_test.py 0 0.05 0 --dur 2
```

## Offline tests

```bash
python3 tests/sim_paths.py      # teleop.py + fake robot, follows paths/001-004 (0.15 m/s, 3 s pauses)
```

`TELEOP_SPEED` overrides the drive speed for tests only. Odometry uses `J_INV` from real runs but has not
been checked with a tape measure.

## VLA grasp (`vla/vla_run.sh`)

```bash
ssh -t root@192.168.31.228 /root/vla_run.sh run_01            # 120 s, max 0.5 rad/s
ssh -t root@192.168.31.228 /root/vla_run.sh run_02 60 0.25    # 60 s, max 0.25 rad/s
```

The first argument is a new record name; output goes to `/root/openpi-xpu/client/outputs/<name>/` (report,
commands, inferences, observations). Steps: `arm.sh start` (training start pose, grippers open), one
live observation (`live_observation_probe.py`), then `guarded_grasp.py --execute` with prompt 抓取沙包
against the policy server on `127.0.0.1:8000` (container `sandbag-step4999-policy`, pi05 chassis config,
sandbag step 4999). `guarded_grasp.py` and `live_observation_probe.py` live in the container's
`/outputs/tools/` and are not part of this repo; they rate-limit every joint, clamp targets to the joint
limits, and abort on stale sensors or mode, joint tracking error > 0.2 rad, or another arm publisher.
Ctrl+C in the SSH session, 停, or 急停 stops a run; the arms then hold where they are. The web button
runs the same script without a terminal (it picks `docker exec -i` instead of `-it`).

## Arms (standalone, `arm/`)

Separate from the chassis code. The web remote runs `arm_pose.py` as its own process (Arms panel: Start pose
10 s, Raise 8 s, Lower 15 s); `move.sh` never touches the arms.

| file | runs where | what |
|---|---|---|
| `arm/arm_pose.py` | container, `/outputs/arm/` | both arms to the official `zero.py` raised pose (`raise`), back to the pose saved before raising (`lower`), or to the VLA training start pose (`start`) |
| `arm/train_start_pose.json` | container, `/outputs/arm/` | training start pose: frame 0 of sandbag episode 2, the start closest to the median of all 36 episodes |
| `arm/arm.sh` | B70 `/root/` | `./arm.sh check`, `./arm.sh raise [--time 8]`, `./arm.sh lower`, `./arm.sh start` |

Needs run mode API, both arms enabled and in **joint** control mode. The script holds the current pose
with the FSM heartbeat until READY, then moves with a min-jerk profile (peak joint speed <= 0.35 rad/s).
If a joint lags its command by > 0.3 rad for 0.5 s (possible contact), or on Ctrl+C, it holds where it is
and pauses the FSM (`/api/fsm/enable` 0.0). Raising moves the arms forward and up: clear the space in front.

Before VLA inference run `start`: the policy only saw arms low and forward over the table at the start of
an episode; starting from the raised pose gives off-distribution first actions. `start` also commands both grippers
open (`/api/{arm}_gripper/cmd` 0.0 = full opening force, sent with every joint command during the move): in
136 of 137 training episodes (sandbag_trial_20261114_v1, grasp_red_sandbag) the gripper starts fully open
(~1.05 rad, the dataset maximum) with trigger 0.0. The base position is not set by it: put the base where it
stood during data collection.
