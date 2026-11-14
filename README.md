# TA2 chassis teleop

Scripts for driving the TeleAvatar 2 (TA2) mobile base from the B70 inference host: a test script, a
command-line stepper, and a small web remote with hold-to-move, trajectory display/recording and
recorded-path replay.

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
| `web_move.py` | B70 `/root/` | web server on port 8088 |
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

- **Direction buttons** Forward / Back / Left / Right / CCW / CW. Click (< 0.3 s): 10 cm or 15 deg.
  Press and hold: 0.05 m/s or 0.2 rad/s until release. STOP stops everything.
- **Trajectory**: wheel-odometry pose and robot footprint. Start / Stop recording, Save (CSV download and a
  copy in `/root/ta2_trails/`), Reset (origin = current pose).
- **Paths**: CSVs from `/root/ta2_paths/` plus any loaded with "Load files". Run follows one path from the
  current pose (the path's first point is anchored there); Run all follows them in name order, each from
  where the previous ended. The robot stops for 10 s at every corner (turn > 45 deg, `CORNER_PAUSE`).
  Click a row to preview it on the map.

CSV format (written by Save, read by Paths): `t_s,x_m,y_m,heading_deg`, x = right, y = forward, heading CCW,
relative to the origin.

## Safety

- Every move requires API mode + chassis enabled and fresh wheel feedback (< 0.3 s).
- Hold is a dead-man switch: the browser sends a heartbeat every 0.1 s; no heartbeat for 0.4 s -> stop.
  One hold is capped at 10 s.
- Paths stop when the page stops polling for 1 s, on 5 s without progress, on timeout, or on STOP
  (Run all is cancelled). Paths longer than 6 m are refused.
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

## Arms (standalone, `arm/`)

Separate from the chassis code; nothing in the web remote or `move.sh` touches the arms.

| file | runs where | what |
|---|---|---|
| `arm/arm_pose.py` | container, `/outputs/arm/` | both arms to the official `zero.py` raised pose, or back to the saved start pose |
| `arm/arm.sh` | B70 `/root/` | `./arm.sh check`, `./arm.sh raise [--time 8]`, `./arm.sh lower` |

Needs run mode API, both arms enabled and in **joint** control mode. The script holds the current pose
with the FSM heartbeat until READY, then moves with a min-jerk profile (peak joint speed <= 0.35 rad/s).
If a joint lags its command by > 0.3 rad for 0.5 s (possible contact), or on Ctrl+C, it holds where it is
and pauses the FSM (`/api/fsm/enable` 0.0). Raising moves the arms forward and up: clear the space in front.
