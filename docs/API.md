# TA2 编排 API 与 Python 客户端

`web_move.py` 在现有 HTTP 服务中提供 `/api/v1`，默认地址为
`http://192.168.31.228:8088`。所有请求和响应均使用 JSON；动作异步启动，客户端可选择轮询等待。
API 与网页共用机器人、序列锁、底层动作函数和急停；全流程任务由 `mission.py` 编排现有积木。

## 使用边界

- **软件急停不等于硬件急停。** 软件急停依赖主机、网络、进程和机器人通信；现场必须有人能操作硬件急停。
- 无鉴权、无 TLS，仅限可信的场地内网。不要把端口暴露到公网。
- 路径从当前位置和朝向出发，先确认起点、朝向和周围空间。关闭客户端、等待超时或网络断开不会取消路径。
- **路线运行期间（含路径间暂停），所有 POST 只放行 `/estop` 和 `/api/v1/estop`。**
  软停、cancel、新动作、网页其他按钮均拒绝；API 返回 `409 ROUTE_ESTOP_ONLY`。
- 单独的 `vla.grasp` 固定执行 60 秒，速度上限 0.5 rad/s，提示词「抓取沙包」。动作成功表示程序正常执行结束，**不判断抓取是否成功**。
- 全流程运行及中止清理期间，所有其他 POST 返回 `409 MISSION_ESTOP_ONLY`，只放行两个急停端点。
- 同时只执行一个动作。网页的全流程创建 API 动作记录；其他网页动作只占用忙状态。

## 端点总览

| 方法 | 路径 | 成功响应 |
|---|---|---|
| GET | `/api/v1/health` | 200，服务和忙状态 |
| GET | `/api/v1/state` | 200，结构化快照 |
| GET | `/api/v1/capabilities` | 200，动作参数、已加载路径、路线、姿态、VLA 常量 |
| GET | `/api/v1/actions?limit=20` | 200，`{"actions": [...]}`，新的在前 |
| GET | `/api/v1/actions/<id>` | 200，动作记录 |
| POST | `/api/v1/actions` | 202，`{"action_id": "...", "action": {...}}` |
| POST | `/api/v1/stop` | 200，`{"ok": true}` |
| POST | `/api/v1/estop` | 200，`{"ok": true}`，急停已请求 |
| POST | `/api/v1/actions/<id>/cancel` | 200，`{"ok": true}` |

POST 控制端点不需要请求体；启动动作使用 `Content-Type: application/json`。
动作记录时间为含时区的 ISO 8601 UTC 时间；mission 的暂停截止和事件时间使用 Unix 秒。`heading_deg` 为度，底层弧度已转换；x 向右、y 向前。

## 只读请求与响应

`GET /api/v1/health`：

```json
{"ok":true,"connected":true,"api_mode":true,"estop_active":false,"busy":false,"current_action":null}
```

`ok` 表示 HTTP 服务可用，不能作为机器人就绪证明。`api_mode` 在模式未知时为 null。
`current_action` 为 API 动作 ID 或 null；网页动作运行时也会 `busy: true`。

`GET /api/v1/state`：

```json
{
  "robot":{"connected":true,"api_mode":true},
  "chassis":{"pose":{"x_m":0.0,"y_m":0.0,"heading_deg":0.0},"moving":null},
  "arms":{"busy":false,"last":"arms start: done"},
  "vla":{"running":false,"last":"VLA: not run yet"},
  "sequence":{"names":[],"index":-1,"route":null,"pause_left_s":0},
  "estop":{"active":false,"last":null},
  "mission":{"id":null,"status":"idle","step":null,"hand":null,"attempt":0,"attempts":5,"pause_until":null,"result":null,"attempt_results":[],"durations":{},"active":false,"pause_left_s":0},
  "current_action":null
}
```

不包含 trail 或 log。`chassis.moving` 为底层活动描述字符串或 null；`sequence.index` 从 0 开始，
未开始时为 -1。此处 `current_action` 为完整记录或 null；中止后可能短暂保留记录，直到底层工作线程退出，
期间仍拒绝新动作。中止后底层退出时会补全 result，并追加最终 JSON 日志。

`GET /api/v1/capabilities`（路径列表取当前已加载内容）：

```json
{
  "actions":{
    "chassis.step":{"direction":["forward","back","left","right","ccw","cw"]},
    "path.run":{"name":"loaded path name"},
    "path.run_all":{},"route.run":{"route":["g2r","r2g"]},
    "arm.pose":{"pose":["raise","lower","start","start_hold","down5","open_right","close_right","open_left","close_left"]},"vla.grasp":{},"odom.reset":{},
    "mission.run":{"attempts":"optional integer","prompt":"optional string","port":"optional integer"}
  },
  "paths":["001","002","003","004"],
  "routes":[
    {"id":"g2r","name":"绿 → 红","paths":["001","002"]},
    {"id":"r2g","name":"红 → 绿","paths":["003","004"]}
  ],
  "arm_poses":["raise","lower","start","start_hold","down5","open_right","close_right","open_left","close_left"],
  "vla":{"seconds":60,"max_rad_s":0.5,"prompt":"抓取沙包"}
}
```

`GET /api/v1/actions?limit=1`，limit 为 1–200 的整数，默认 20：

```json
{"actions":[{"id":"a12b34c56d78","type":"chassis.step","params":{"direction":"forward"},"status":"succeeded","created_at":"2026-11-15T02:00:00+00:00","ended_at":"2026-11-15T02:00:01+00:00","result":{"why":"done","teleop":{"event":"idle","kind":"step","dir":"forward","why":"done"}},"error":null}]}
```

`GET /api/v1/actions/a12b34c56d78` 返回上面数组中的单条完整记录。
不存在或已被淘汰的 ID 返回：

```json
{"error":{"code":"NOT_FOUND","message":"action not found: a12b34c56d78"}}
```

历史只在内存保存最近 200 条，服务重启清空；每条动作的底层工作结束时通过后台写入器追加一行 JSON 到
`ACTION_LOG`（默认 `/root/ta2_actions.jsonl`）。写失败记录到服务日志，不影响控制。日志不会在启动时回放。

## 启动动作

`POST /api/v1/actions` 请求示例：

```json
{"type":"route.run","params":{"route":"g2r"}}
```

响应 HTTP 202（异步接受；立即动作或快速失败也可能已进入终态）：

```json
{"action_id":"a12b34c56d78","action":{"id":"a12b34c56d78","type":"route.run","params":{"route":"g2r"},"status":"running","created_at":"2026-11-15T02:00:00+00:00","ended_at":null,"result":null,"error":null}}
```

| type | params 示例 | 执行与结束判定 |
|---|---|---|
| `chassis.step` | `{"direction":"forward"}` | forward/back/left/right/ccw/cw；现有定时点动，teleop idle 原因决定结果 |
| `path.run` | `{"name":"001"}` | 名称不含 `.csv`；现有单路径，`path done` 成功 |
| `path.run_all` | `{}` | 当前所有路径按名称顺序，现有序列 runner 的 `finished` 成功 |
| `route.run` | `{"route":"g2r"}` | g2r=001→002，r2g=003→004；复用序列 runner，保留 `route_last` |
| `arm.pose` | `{"pose":"start"}` | 所有 `ARM_TIME` 动作（见 capabilities）；`down5` 调用 `arm_down.sh right 5`；脚本退出码和 `arm_last` |
| `vla.grasp` | `{}` | 检查推理容器后运行现有脚本；正常跑满时限 `time_limit` 成功 |
| `mission.run` | `{"attempts":5,"prompt":"抓取沙包","port":8000}`（均可省略） | 预检通过后异步执行全流程；监测抓取成功并用同一只手放置 |
| `odom.reset` | `{}` | 发送现有 `reset_pose`，发送成功立即 succeeded |

参数必须匹配动作类型；无参数动作可以省略 params。`mission.run` 的可选参数仅限 attempts/prompt/port；其余动作参数不变。
序列保留现有路径间 `PATH_PAUSE`（默认 10 秒）。

VLA 的终态 result 保留脚本 JSON 的原始字段，并带 `status`、`actuations`、`error`、`report`、
`tag`、`returncode`。脚本未提供 report 时使用现有输出目录约定路径；该路径不表示报告文件已验证存在。
例如：

```json
{"status":"time_limit","actuations":42,"error":null,"report":"/root/openpi-xpu/client/outputs/vla_web_20261115_100000/report.json","tag":"vla_web_20261115_100000","returncode":0}
```

## 停止与取消

`POST /api/v1/stop` 请求现有软停：停止底盘、路径、手臂和 VLA，当前 API 动作标记 aborted / STOPPED。

`POST /api/v1/actions/a12b34c56d78/cancel` 对仍运行的该动作执行同样软停。
动作已结束返回 `409 NOT_RUNNING`；未知 ID 返回 `404 NOT_FOUND`。

`POST /api/v1/estop` 绕过序列锁立即请求现有急停，当前 API 动作标记 aborted / ESTOP。
它停止底盘和路径，中断容器内手臂/VLA 脚本（3 秒后必要时强杀），随后发布 FSM PAUSE。

三个端点的成功响应均为 `{"ok":true}`。急停响应表示请求已接受，
不表示物理停止或 FSM PAUSE 已确认；轮询 state 的 estop.active / estop.last 查看执行情况。
`estop_active` 仅表示急停流程正在执行，不是需要解锁的锁存状态。
**路线或全流程运行时 stop 和 cancel 均拒绝，只允许 estop。**

## 状态与错误码

| status | 含义 |
|---|---|
| `running` | 已接受、执行中 |
| `succeeded` | 正常结束：done / path done / finished / VLA time_limit 等 |
| `failed` | 超时、受阻、反馈丢失、模式不符、脚本错误等 |
| `aborted` | STOP、cancel 或急停中止 |

所有 API HTTP 错误使用统一格式：

```json
{"error":{"code":"BUSY","message":"wait until the robot is still"}}
```

| code | HTTP / 动作错误 | 含义 |
|---|---|---|
| `BAD_REQUEST` | 400 | 未知动作、参数或 JSON 不合法、limit 越界 |
| `UNKNOWN_PATH` | 400 | 路径不存在、路线所需路径缺失或无路径 |
| `UNKNOWN_ROUTE` | 400 | 非 g2r / r2g |
| `NOT_FOUND` | 404 | 端点或动作 ID 不存在 |
| `BUSY` | 409 | 网页或 API 动作仍占用机器人 |
| `ESTOP_ACTIVE` | 409 | 急停流程执行中 |
| `ROUTE_ESTOP_ONLY` | 409 | 路线期间只允许急停；优先于其他 POST 检查 |
| `MISSION_ESTOP_ONLY` | 409 | 全流程运行及清理期间仅接受急停，优先于路线门控和参数检查 |
| `MISSION_PREFLIGHT_FAILED` | 409 | 模式、路径、反馈或夹持预检失败，未开始移动 |
| `NOT_RUNNING` | 409 | 取消目标已经结束 |
| `NOT_CONNECTED` | 503 | 动作启动前 teleop 未连接 |
| `POLICY_DOWN` | 503 | VLA 推理容器未运行或检查失败 |
| `TELEOP_DISCONNECTED` | 动作错误 / 503 控制错误 | teleop 断开或写入失败 |
| `TIMEOUT` | 动作错误 | 运动超时 |
| `STUCK` | 动作错误 | 路径无进展 |
| `FEEDBACK_LOST` | 动作错误 | 底盘或手臂反馈不新鲜或丢失 |
| `NOT_API_MODE` | 动作错误 | 非 API 模式、底盘未启用或模式未知 |
| `SCRIPT_ERROR` | 动作错误 | 手臂/VLA 脚本非零退出或启动异常 |
| `ACTION_FAILED` | 动作错误 | 其他底层异常结束原因 |
| `STOPPED` / `ESTOP` | 动作错误 | 软停（含 cancel）/ 急停 |
| `INTERNAL_ERROR` | 500 | 非预期的服务端异常 |

动作执行中的失败通过记录的 status/error 返回，查询本身仍为 HTTP 200；error.message 保留原始原因。

## Python 客户端

复制仓库根目录的 `ta2_client.py` 即可使用，无第三方依赖，Python 3.8+。

```python
from ta2_client import TA2Client, TA2Error, TA2Timeout, TA2ActionFailed

c = TA2Client("http://192.168.31.228:8088", timeout=5)  # 单次 HTTP 超时
print(c.health())
print(c.state())
print(c.capabilities())
aid = c.start("chassis.step", direction="forward")
record = c.wait(aid, timeout=30, poll=0.5, raise_on_fail=True)
print(c.action(aid))
print(c.actions(limit=20))
```

便捷方法 `run_route(route)`、`run_path(name)`、`run_all()`、`arm_pose(pose)`、`vla_grasp()`、
`step(direction)`、`reset_odom()` 默认 `wait=False`，返回 action_id；`wait=True` 返回终态记录。
都支持 timeout、poll、raise_on_fail。`wait` 默认 `raise_on_fail=False`，直接返回失败/中止记录。

HTTP 错误抛 `TA2Error`，含 code、message、http_status；网络错误使用 CONNECTION_ERROR，
http_status 为 None。等待超时抛 `TA2Timeout`，**不会自动停机器人**。
`raise_on_fail=True` 时终态非 succeeded 抛 `TA2ActionFailed`，其 record 保留完整记录。

「绿→红 → VLA 抓取 → 红→绿」示例（执行前确认路线与作业空间）：

```python
from ta2_client import TA2Client, TA2Error

c = TA2Client("http://192.168.31.228:8088", timeout=5)
try:
    c.run_route("g2r", wait=True, timeout=300, raise_on_fail=True)
    grasp = c.vla_grasp(wait=True, timeout=180, raise_on_fail=True)
    print(grasp["result"])  # 程序执行结果；不代表已抓到沙包
    c.run_route("r2g", wait=True, timeout=300, raise_on_fail=True)
except (TA2Error, KeyboardInterrupt):
    c.estop()  # 此编排主动选择出错时请求急停，客户端 wait 自身不会停机器人
    raise
```

控制示例：`c.stop()`、`c.cancel(aid)`、`c.estop()`；路线中只能使用最后一个。
命令行检查：

```bash
python3 ta2_client.py --url http://192.168.31.228:8088 state
python3 ta2_client.py --url http://192.168.31.228:8088 estop
```


## 全流程任务

```json
{"type":"mission.run","params":{"attempts":5,"prompt":"抓取沙包","port":8000}}
```

params 可省略。attempts 为 1 到配置 ATTEMPTS 的整数（默认最多 5），port 为 1–65535 的整数；
prompt 为非空字符串（最多 500 字符；当前 VLA 包装脚本限制，不接受单引号、换行、NUL）。
`c.run_mission(attempts=5, wait=True, timeout=1800, raise_on_fail=True)` 可等待结果。
客户端等待超时不会停止任务。

预检要求 teleop 已连接且 API 模式、四段路径已加载、策略容器运行、两臂的关节/夹爪/末端反馈都新鲜，
两只夹爪均不符合 holding 判定；不通过时拒绝请求，不执行复位或移动。
随后步骤为 `preflight → g2r → grasp → stow → r2g → place`：
001→002，到达停 10 秒；最多 5 次 VLA（每次策略 45 秒，限速 0.5，RESET=1）；
成功立即中断 VLA，记录 `hand`；start_hold 收臂；003→004，到达停 10 秒；
`arm_down.sh <hand> 10`，`open_<hand>`。五次失败执行 start 复位，停在红区。
单次 VLA 非零或跑满均计失败并重试；其他脚本非零、反馈断开、路线未完成均终止任务并停底盘。
回程掉落记录后完成当前路径即停止，不再启动下一路径或放置；反馈断开和急停立即请求停底盘。

`/status` 和 `/api/v1/state` 均有 `mission`：id、status、active（包含清理）、step、hand、
attempt/attempts、pause_left_s、result、attempt_results、各步骤 durations（秒）。
成功为 succeeded；失败为 failed；急停立即显示 aborted，后台清理结束后 active=false。
每次任务的 `<MISSION_LOG_DIR>/<id>.jsonl` 记录步骤、尝试、命令、命令输出（包装成 JSON 事件），
最后一行为 summary，含结果、各次尝试与抓取手、步骤耗时。任务日志无法创建时不开始移动。
配置和真机标定事项见 README 的全流程章节。
