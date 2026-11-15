# TA2 编排 API 与 Python 客户端

`web_move.py` 在现有 HTTP 服务中提供 `/api/v1`，默认地址为
`http://192.168.31.228:8088`。所有请求和响应均使用 JSON；动作异步启动，客户端可选择轮询等待。
API 与网页共用机器人、序列锁、底层动作函数和急停；全流程任务由 `mission.py` 编排现有积木。

在线文档与测试页为 `/api-docs`（主页面入口在新标签打开）。每张卡可编辑参数、直接测试、
复制 curl / ta2_client.py 示例并查看 HTTP 状态码、耗时与 JSON；动作自动轮询并支持取消。
测试会发送真实请求，不弹确认框，运动卡带明显标记；停止与急停仍遵循下述门控规则。

## 使用边界

- **软件急停不等于硬件急停。** 软件急停依赖主机、网络、进程和机器人通信；现场必须有人能操作硬件急停。
- 无鉴权、无 TLS，仅限可信的场地内网。不要把端口暴露到公网。
- 路径从当前位置和朝向出发，先确认起点、朝向和周围空间。关闭客户端、等待超时或网络断开不会取消路径。
- **路线运行期间（含路径间暂停），所有 POST 只放行 `/estop` 和 `/api/v1/estop`。**
  软停、cancel、新动作、网页其他按钮均拒绝；API 返回 `409 ROUTE_ESTOP_ONLY`。
- `vla.grasp` 默认使用网页选中模型的时长、限速和提示词；可指定单次 model / seconds。默认不判断抓取成功，只有 `stop_on_grasp: true` 才会按全流程判据检测并提前停止。
- 全流程运行及中止清理期间，所有其他 POST 返回 `409 MISSION_ESTOP_ONLY`，只放行两个急停端点。
- 同时只执行一个动作。网页的全流程创建 API 动作记录；其他网页动作只占用忙状态。

## 端点总览

| 方法 | 路径 | 成功响应 |
|---|---|---|
| GET | `/api/v1/health` | 200，服务和忙状态 |
| GET | `/api/v1/state` | 200，结构化快照 |
| GET | `/api/v1/grippers` | 200，左右夹爪和手腕实时读数 |
| GET | `/api/v1/preflight?model=<id>` | 200，`{"ok": bool, "checks": [...]}`；model 可省略 |
| GET | `/api/v1/models` | 200，模型、RTC、容器运行和选中状态 |
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
    "arm.pose":{"pose":["raise","lower","start","start_hold","down5","open_right","close_right","open_left","close_left"]},
    "arm.down":{"hand":["left","right"],"cm":"number from 0.5 to 15"},
    "vla.grasp":{
      "model":"optional registered model id (default: selected model)",
      "seconds":"optional number from 1 to 120 (default: model seconds)",
      "stop_on_grasp":"optional boolean (default: false)"
    },"odom.reset":{},
    "mission.run":{"attempts":"optional integer","prompt":"optional string","port":"optional integer","rtc":"optional boolean (default: the selected model's mode)"}
  },
  "paths":["001","002","003","004"],
  "routes":[
    {"id":"g2r","name":"绿 → 红","paths":["001","002"]},
    {"id":"r2g","name":"红 → 绿","paths":["003","004"]}
  ],
  "arm_poses":["raise","lower","start","start_hold","down5","open_right","close_right","open_left","close_left"],
  "vla":{"seconds":60,"max_rad_s":1.0,"prompt":"抓取红色沙包","model":"grasp_v3_step13000_rtc"}
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
| `arm.down` | `{"hand":"left","cm":7.5}` | left/right，0.5–15 cm（含端点）；复用手臂执行/忙/停止逻辑，底层 `arm_down.sh left 7.5` 自动限位 |
| `vla.grasp` | `{"model":"grasp_v3_step10000_rtc","seconds":60,"stop_on_grasp":true}`（均可省略） | 单次选择模型和时长；可在任一手抓取成功后立即停 VLA |
| `mission.run` | `{"attempts":5,"prompt":"抓取沙包","port":8000}`（均可省略） | 预检通过后异步执行全流程；监测抓取成功并用同一只手放置 |
| `odom.reset` | `{}` | 发送现有 `reset_pose`，发送成功立即 succeeded |

参数必须匹配动作类型；无参数动作可以省略 params。`mission.run` 保留现有 attempts/prompt/port/rtc 参数与行为。

`mission.run` 的 `flow`（可选）：`v1`（默认，第 1–6 步）、`v2`（全流程 v2：1–6 步后接第 7 步语音交互模块）、
`voice`（只运行第 7 步，用于调试）。第 7 步的子步骤在 `state.mission.voice.steps` 里实时给出：
7.1 识别口令 → 7.2 真机抓取（抓住即停，同第 3 步）→ 7.3 收臂（start_hold）→ 7.4 下降 10 cm（arm_down）
→ 7.5 松开该手 → 7.6 确认交接（夹爪张开且未夹持）→ 7.7 完成播报（Mac /confirm）。每步含 `id/name/uses/status/detail`。
网页入口：`POST /mission/run/v2`、`POST /mission/run/voice`。
序列保留现有路径间 `PATH_PAUSE`（默认 10 秒）。

VLA 的终态 result 保留脚本 JSON 的原始字段，并带 `status`、`actuations`、`error`、`report`、
`tag`、`returncode`、`model`、`rtc`、`seconds`、`grasped`、`hand`。脚本未提供 report 时使用现有输出目录约定路径；该路径不表示报告文件已验证存在。
例如：

```json
{"status":"time_limit","actuations":42,"error":null,"report":"/root/openpi-xpu/client/outputs/vla_web_20261115_100000/report.json","tag":"vla_web_20261115_100000","returncode":0,"model":"grasp_v3_step13000_rtc","rtc":true,"seconds":60,"grasped":null,"hand":null}
```

## 单独下降与抓取

```json
{"type":"arm.down","params":{"hand":"right","cm":7.5}}
```

仍返回 HTTP 202 和 action_id。成功时动作 status 为 succeeded，result 示例：

```json
{"why":"arms down: done","returncode":0,"output":"...最后最多 8 行有效脚本输出..."}
```

失败保留退出码与输出，动作 error 使用 SCRIPT_ERROR。cm 必须是有限数字，不能是字符串或布尔值。
原 `arm.pose` 的 `down5` 仍按原逻辑调用 `arm_down.sh right 5`。

```json
{"type":"vla.grasp","params":{"model":"grasp_v3_step10000_rtc","seconds":60,"stop_on_grasp":true}}
```

model 必须是 `/api/v1/models` 的 RTC id；同步版 id 已删除，使用时返回 UNKNOWN_MODEL。
默认注册模型为 `grasp_v3_step13000_rtc`；省略时绑定请求开始时的网页选择。
保存的已删除 id 在服务启动时回退到该默认值。
指定 model 不保存选择，也不改变网页。seconds 为 1–120 的有限数字，默认该模型的 seconds；
stop_on_grasp 必须是布尔值，默认 false。网页「开始抓取」仍走旧逻辑，不启用检测。

启用检测前必须拿到新鲜监控数据，否则 HTTP 503 MONITOR_DOWN，VLA 不启动。
每次调用独立初始化左右检测器，每帧使用 `mission.grasp_update`，沿用闭爪高度、负载、抬升和持续时间判据。
任一手成功后调用与网页 STOP 相同的 `end_vla_runner()` 和容器内 `pkill -INT`，不复位、不收臂；
动作 status 为 succeeded，即使主动中断导致脚本非零退出。result 示例：

```json
{"grasped":true,"hand":"left","model":"grasp_v3_step10000_rtc","rtc":true,"seconds":60,"status":"succeeded","actuations":42,"error":null,"report":"/root/openpi-xpu/client/outputs/vla_trial/report.json","tag":"vla_trial","returncode":130}
```

seconds 是配置的策略运行时长，不是实际耗时；actuations 在脚本没提供报告时为 null。
未启用检测时 grasped / hand 为 null；启用后未检测到成功时 grasped 为 false、hand 为 null，
脚本正常跑满仍可 succeeded / time_limit，所以应看 grasped 判断是否抓到。
运行中反馈断流会停止 VLA 并记 failed / FEEDBACK_LOST。
用户 STOP／急停仍记 aborted / STOPPED 或 ESTOP，不会被检测成功覆盖。

## 夹爪、预检与模型（只读）

`GET /api/v1/grippers`：

```json
{"left":{"pos":1.05,"eff":0.2,"ee_z":-0.30,"fresh":true,"holding":false},"right":{"pos":0.39,"eff":-2.5,"ee_z":-0.23,"fresh":true,"holding":true},"age_s":0.02,"source":"monitor"}
```

pos 单位 rad，eff 单位 Nm，ee_z 单位 m。holding 直接使用 `mission.holding` 的位置／力矩判据；
它不等于经过抬升与持续时间确认的抓取成功。
首次读取按 `mission.Mission._monitor_cmd` 启动只读 `mission.Monitor`，支持相同的 MONITOR_CMD 覆盖；
不启动容器，不发送运动或 FSM 指令。监控进程退出后下一次请求自动重启；最多等待 3 秒取得数值读数，
启动失败、退出或仍无读数返回 503 MONITOR_DOWN。已有读数变陈旧时保留读数并返回 fresh=false、实际 age_s；
预检将其标记为失败，启用抓取检测的动作则必须等到新鲜反馈，否则返回 MONITOR_DOWN。

夹爪接口与单次抓取共用一个后台监控，避免重复订阅与重复进程；每次抓取注册独立检测器，结束即移除。
全流程保留自己的监控和原有行为。STOP／急停可能杀掉后台监控，后续读取会自动恢复。

`GET /api/v1/preflight` 或 `GET /api/v1/preflight?model=grasp_v3_step10000_rtc`：

```json
{"ok":true,"checks":[
  {"name":"policy","ok":true,"detail":"grasp_v3_step10000_rtc"},
  {"name":"left_gripper","ok":true,"detail":"fresh=True, holding=False"},
  {"name":"right_gripper","ok":true,"detail":"fresh=True, holding=False"},
  {"name":"connected","ok":true,"detail":"chassis connection"},
  {"name":"api_mode","ok":true,"detail":"robot API mode"},
  {"name":"estop_clear","ok":true,"detail":"emergency stop inactive"},
  {"name":"idle","ok":true,"detail":"no web or API action running"},
  {"name":"paths","ok":true,"detail":"001–004 loaded"}
]}
```

model 默认当前选中模型，不改变选择。检查失败仍返回 HTTP 200、ok=false，并在对应项 detail 中说明原因；
例如监控失败对应两只 gripper 的 ok=false 和 MONITOR_DOWN 原因，策略失败对应 policy 的 POLICY_DOWN 原因。
未知 model 返回 400 UNKNOWN_MODEL。该接口不锁定后续动作，也不保证检查后机器人状态不变。

`GET /api/v1/models` 示例（models 数组实际包含全部注册模型）：

```json
{"selected":"grasp_v3_step13000_rtc","models":[{"id":"grasp_v3_step13000_rtc","name":"最新模型 v3 · 13000 步 · RTC 流畅模式","training":"122 段演示 · 11 月 15 日","prompt":"抓取红色沙包","seconds":60,"speed":1.0,"rtc":true,"running":true,"selected":true}]}
```

每项含 public_model 全部字段、推理容器的 running 和 selected 标记；只读 Docker 状态，不启动或停止服务。

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
| `UNKNOWN_MODEL` | 400 | 未注册的 model id |
| `NOT_FOUND` | 404 | 端点或动作 ID 不存在 |
| `BUSY` | 409 | 网页或 API 动作仍占用机器人 |
| `ESTOP_ACTIVE` | 409 | 急停流程执行中 |
| `ROUTE_ESTOP_ONLY` | 409 | 路线期间只允许急停；优先于其他 POST 检查 |
| `MISSION_ESTOP_ONLY` | 409 | 全流程运行及清理期间仅接受急停，优先于路线门控和参数检查 |
| `MISSION_PREFLIGHT_FAILED` | 409 | 模式、路径、反馈或夹持预检失败，未开始移动 |
| `NOT_RUNNING` | 409 | 取消目标已经结束 |
| `NOT_CONNECTED` | 503 | 动作启动前 teleop 未连接 |
| `POLICY_DOWN` | 503 | VLA 推理容器未运行或检查失败 |
| `MONITOR_DOWN` | 503 | 监控无法启动、已退出、3 秒内无数值读数；启用检测的抓取还要求新鲜反馈 |
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
`arm_down(hand, cm)`、`step(direction)`、`reset_odom()` 默认 `wait=False`，返回 action_id；`wait=True` 返回终态记录。
都支持 timeout、poll、raise_on_fail。`wait` 默认 `raise_on_fail=False`，直接返回失败/中止记录。

```python
print(c.models())
print(c.preflight(model="grasp_v3_step10000_rtc"))
print(c.grippers())
c.arm_down("left", 7.5, wait=True, raise_on_fail=True)
record = c.vla_grasp(model="grasp_v3_step10000_rtc", seconds=60,
                     stop_on_grasp=True, wait=True, raise_on_fail=True)
print(record["result"]["grasped"], record["result"]["hand"])
```

`vla_grasp()` 和 `vla_grasp(wait=True)` 保持兼容；完整签名为
`vla_grasp(model=None, seconds=None, stop_on_grasp=False, wait=False, timeout=300, poll=0.5, raise_on_fail=False)`。
`grippers()`、`preflight(model=None)`、`models()` 直接返回上述 JSON 对象。

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
