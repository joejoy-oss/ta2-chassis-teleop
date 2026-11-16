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
- **VLA 抓取**: the big button 开始抓取（同全流程第 3 步） runs exactly full-flow step 3 (`mission.run`
  `{"flow":"grasp","attempts":5|3|1}`) with the currently selected model in `vla_models.json`: each attempt resets
  (Start pose, grippers open) and runs up to 45 s at max 1.0 rad/s; a grasp stops it holding in place, an empty close
  (1 s), a hold without lift (10 s) or the time limit moves on to the next attempt; all failed → Start pose. **选择 / 切换模型** opens `/models`,
  listing only RTC models: v3 step 20000 (checkpoint 19999, port 8005, default `grasp_v3_step20000_rtc`),
  step 13000 (8004), step 10000 (8003), and v2 step 4999 (8000), with their container states. Selection is saved
  in `/root/ta2_vla_selected.json` without starting motion and is also used as the default model for
  **开始全流程** unless overridden via API parameters or environment variables. Switching is refused during
  motion. Below it, small arm buttons: 起始姿态 (training start pose), 起始姿态·保持夹爪, 右臂下降 5 cm,
  松开右爪, 闭合右爪, 抬起 (raise), 放下 (lower).
- **路线**: 绿 → 红 runs `001` → pause 10 s → `002`; 红 → 绿 runs `003` → pause 10 s → `004`.
  The card shows the current path, pause countdown and last result.
  Place the robot at the start with the recorded heading first. Routes use Run all's continuous corner
  handling, pause only between paths (`PATH_PAUSE`), and keep running if the page disconnects.
  Once started, a route accepts only 急停 (including during a pause); every other web control command,
  including 停 and release, is refused with HTTP 409. Page polling only refreshes the display:
  the server supplies the robot heartbeat, so no phone connection or requests are required to finish.
  Missing paths and busy starts are refused. Robot-side fault checks still apply.
- **底盘**: direction pad 前进 / 后退 / 左 / 右 and 逆时针 / 顺时针. Click (< 0.3 s): 10 cm or 15 deg.
  Press and hold: 0.10 m/s or 0.2 rad/s until release. The centre 停 is the soft STOP: chassis, paths,
  arm script and VLA run stop, arms hold where they are. During a route, use 急停 instead.
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
ssh -t root@192.168.31.228 /root/vla_run.sh run_01            # 120 s, max 1.0 rad/s
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


## 全流程：绿→红→抓取→红→绿→放下

网页最上方「开始全流程」和 API `mission.run` 共用 `mission.py`、序列锁及急停。
页面显示当前步骤、抓取次数、抓取手、到达停留倒计时和结果。运行及中止清理期间仅接受急停；
软停、cancel 和所有其他 POST 均拒绝（`MISSION_ESTOP_ONLY`）。接口和客户端示例见 [docs/API.md](docs/API.md)。
`/api/v1` 还支持单手任意下降 `arm.down`、单次模型/时长及抓住即停的 `vla.grasp`，以及只读夹爪、预检、模型列表；`ta2_client.py` 提供对应方法。

预检确认底盘连接/API 模式、策略容器、四段路线、双臂反馈新鲜、当前双爪都未夹持后，执行：

1. 路线 001→002（PATH_PAUSE=10 秒），到红区再停 ARRIVE_PAUSE=10 秒。
2. 最多 ATTEMPTS=5 次：启动只读监视器，调用 VLA 包装脚本（每次策略 ATTEMPT_S=45 秒，限速 1.0，RESET=1，RTC 跟随所选模型）。
3. 任一手持续满足抓取判定则立即中断 VLA，记下 hand；start_hold 收臂后确认仍夹持。
   某只手空夹 1 秒（且另一只手没夹着东西）、或夹着却 10 秒没抬起，本次尝试立即判失败，停止整个 VLA 链
   （复位脚本、开跑前探测、guarded_grasp），进入下一次，不再等满 45 秒。
4. 路线 003→004，持续监测夹持；若掉落，当前路径完成后终止，不启动下一路径或放置。到达停 10 秒。
5. `arm_down.sh <hand> 10`，再 `open_<hand>`，成功结束。五次全失败则 start 复位，停在红区。

监视器 `arm/grasp_monitor.py` 在客户端容器中由 `python3 -u` 运行，每 0.1 秒输出 JSON。
关节、夹爪、末端各自必须在 0.3 秒内有有效反馈；一条流新鲜不能掩盖另一条停滞。
「夹着东西」（实心、薄片都算）= 夹爪在用力（grip_eff ≤ −1.0 N·m）且 grip_pos ≥ 该手空夹位置 + 0.01 rad、≤ 0.75 rad。
空夹位置按手标定（HOLD_EMPTY_RIGHT / HOLD_EMPTY_LEFT）：右爪 −0.0485 rad（2026-11-15 五次实测，波动 < 0.0004），
左爪暂用 −0.059（只测过一次，需用网页「夹爪标定记录」补测）。右爪实测：实体 0.37–0.56、薄片 −0.034–0.13、空夹 −0.0485。
力矩不区分空夹与夹住（空夹也有 −1.71），只表示夹爪在用力。
抓取成功 = 夹着东西且相对该次闭合高度抬升 ≥ 0.04 m，连续 1.5 秒。每次尝试只有在看到夹爪张开（> 0.75）之后才开始判定，
夹爪每次重新张开都会清空闭合高度和计时（上一次尝试留下的合爪不会被误判）。每次尝试的夹爪读数以 ~10 Hz 写入全流程日志（grip 事件）。
反馈丢失不会作为普通失败继续尝试。

配置集中在 `mission.py` 顶部，并支持同名环境变量：ATTEMPTS、ATTEMPT_S、PROMPT、ARRIVE_PAUSE、
PLACE_CM、HOLD_EMPTY_RIGHT/LEFT、HOLD_MARGIN、HOLD_POS_MAX、HOLD_EFF_MAX、LIFT_MIN、HOLD_PERSIST、EMPTY_PERSIST、NO_LIFT_S、MISSION_LOG_DIR。
策略端口使用 MISSION_PORT（或 POLICY_PORT），默认 8000；`web_move.py` 的 PORT 仍为 HTTP 端口 8088。
每次 API 请求可覆盖 attempts/prompt/port。MONITOR_TIMEOUT 默认 0.6 秒为监视器断流看门狗；
SCRIPT_TIMEOUT 默认 90 秒为手臂/下降脚本时限，同时作为每次 VLA 复位和取帧的额外余量。
所有等待均检查急停和反馈。路线的原有超时、拐角不停和服务器心跳保持不变。
外部命令替换：TELEOP_CMD、VLA_CMD、ARM_CMD、ARM_DOWN_CMD（不带 side/cm）、MONITOR_CMD，
以及原有 ESTOP_DEXEC/ARM_STOP；按带引号的 shell 命令行解析，但不通过 shell 执行。
任务 JSONL 默认写到 `/root/ta2_mission/<id>.jsonl`，summary 包含终态、每次尝试和各步耗时。

左臂推广：`arm.sh open_left|close_left` 与右臂对称，另一只爪不改指令；
`arm_down.sh <left|right> <cm> [--dry-run]` 根据对应 `${side}_fk_fit.json` 计算 IK，再调用现有 goto。
两侧高度下限均为各自肩坐标系 EE z=−0.40 m，网页原右臂下降按钮改用 `right 5`。

B70 审查和标定时需要：

- 左臂与右臂 FK/IK 拟合文件已置于 `arm/left_fk_fit.json`（2401 样本，中位残差 0.23 mm，p95 0.88 mm）与
  `arm/right_fk_fit.json`（891 样本，中位残差 0.25 mm，p95 1.01 mm）；如机械结构调整，可在仿真容器重跑
  `ik_offset.py --arm left|right --fit-samples samples.npz --save-fit <arm>_fk_fit.json`。
- 验证左/右臂的末端高度方向、拟合残差、下降 dry-run 与 −0.40 m 下限，再进行实物下降验证。
- 标定空夹、夹住沙包、收臂及搬运时的位置/力矩阈值和持物反馈抖动，确认 4 cm/1.5 s 判定合适。
- 确认现场路线起点、两区桌面高度、10 cm 放置距离、策略端口和容器；复核 VLA 中断后夹爪保持，
  并验证软件急停的 FSM PAUSE、监视器消息类型和频率。离线测试不能确认这些真机条件。

本机离线测试（只启动 localhost HTTP 和假 ROS，不连接机器人）：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

端到端用真实 teleop.py + tests/fakeros 驱动缩短的测试 CSV，假 VLA/手臂/下降及场景 JSON 监视器；
真实 001–004 CSV 未修改。覆盖左/右成功、五次失败、途中掉落、急停、POST 门控、预检拒绝、
反馈断流、脚本失败及原 API/手臂按钮回归。无需 ROS、Docker 或 MuJoCo；MuJoCo 数值与实物验证留给 B70。


## v3 checkpoint 10000 deployment (2026-11-15)

The previously deployed model is v3 step 10000, trained on the same 122 demonstrations.
`vla/vla_grasp_v3_step10000.sh` pins `grasp-v3-step10000-policy` on localhost port 8003,
prompt 抓取红色沙包, default 60 s and max 1.0 rad/s. The matching `_rtc.sh` runner uses
`RTC=1` with the existing shared runner. Only the `_rtc` entry appears in `/models`; older RTC
models remain available. Checkpoint 8000 is downloaded only and is not registered or served.

B70 JAX weights: `/data/ta2/model_exports/grasp_v3_step10000`.
Converted weights: `/data/ta2/model_exports/grasp_v3_step10000_pytorch`.
Deployment and offline smoke evidence: `/data/ta2/model_exports/grasp_v3_eval/step10000_deployment`.


## v3 checkpoint 13000 deployment (2026-11-15)

The previously registered default checkpoint is v3 step 13000, with the same 122 demonstrations and prompt
抓取红色沙包. The independent `grasp-v3-step13000-policy` service uses GPU0 / localhost port 8004.
`vla/vla_grasp_v3_step13000.sh` and its `_rtc.sh` variant retain the current 60-second and 1.0 rad/s
settings and use the existing shared runner and training start pose. Older models remain available.
JAX checkpoint: `/data/ta2/model_exports/grasp_v3_step13000`.
PyTorch checkpoint: `/data/ta2/model_exports/grasp_v3_step13000_pytorch`.
Verification and offline smoke evidence: `/data/ta2/model_exports/grasp_v3_eval/step13000_deployment`.

Only the four `_rtc` model IDs are registered. A saved, removed synchronous ID falls back
at startup to the configured default (now `grasp_v3_step20000_rtc`). Synchronous shell scripts remain installed because
the RTC wrappers call them with `RTC=1`.

## 接口文档 / 一键测试

主页面入口在新标签打开 `/api-docs`：中文移动端文档覆盖所有 `/api/v1` 接口和九种动作，
参数选项从 capabilities / models 读取，可复制 curl 与 Python 客户端示例。
「测试」直接发送真实请求，不弹确认框；运动卡带「会动机器人」标记，动作自动轮询终态并支持取消。
顶部常驻急停、停止及连接状态；路线和全流程仍只接受急停。
安装脚本会把 `api_docs.html` 复制到 `/root/`，与 `web_move.py` 同目录。

## v3 final 20000-step deployment (2026-11-16)

The final checkpoint directory is `19999` (zero-based training index for 20000 steps).
The RTC entry `grasp_v3_step20000_rtc` replaces the v3 step5000 entry and becomes the default.
Its service `grasp-v3-step20000-policy` uses GPU1 / localhost port8005, prompt 抓取红色沙包,
60 seconds and max1.0rad/s. The shared runner and training start pose are unchanged.
Step10000, step13000 and morning v2 remain available. Old v3 step5000 weights and its stopped
container are retained for rollback. Deployment evidence is in
`/data/ta2/model_exports/grasp_v3_eval/step20000_deployment`; deployment does not start robot motion.
