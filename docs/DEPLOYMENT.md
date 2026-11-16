# TA2 推理、部署与工具交接

本次整理于 2026-11-16，从 B70 当前文件读取并归档。保留已有控制逻辑和本机未提交的
`HOLD_PERSIST=0.5` 修改；没有向机器人发动作、重启服务或把本提交部署到现场。

## 代码与运行路径

| 仓库目录 | B70 对应位置 | 内容 |
|---|---|---|
| 根目录、`arm/`、`vla/`、`voice/` | `/root/`、客户端 `/outputs/tools` 和 `/outputs/arm` | 网页/API、编排、底盘、手臂、RTC 守护客户端、语音 |
| `tools/` | `/data/ta2/tools/` | 数采下载、转换、筛选、推理监控、视觉、感知和抓取判定 |
| `home/` | `/data/ta2/home/` | 静态工具总台 |
| `perception/head-depth/` | `/data/ta2/head-depth/` | OpenVINO 头部双目深度 |
| `perception/head-detection/` | `/data/ta2/head-detection/` | 头部检测网页、推理及安装脚本 |
| `deploy/config/` | `/data/ta2/perception/`、`/data/ta2/grasp/` | 现场配置、头部标定 |
| `deploy/systemd/` | `/etc/systemd/system/` | 现场服务单元快照 |
| `deploy/runtime/` | 原主办方软件包与历史部署脚本，见其 README | XPU 服务端、RTC 客户端、JAX 转换、ROSbag 转换、镜像构建资料 |

各脚本仍使用现场绝对路径；目录整理不表示可以在任意电脑直接运行真机程序。
`deploy/SOURCE_MANIFEST.json` 记录本次采集文件的来源和 SHA-256。
第三方源码保留其 LICENSE、NOTICE 和原目录；本仓库不替上游重新授予许可。

## 当前模型与参数

| 项目 | 2026-11-16 只读核查结果 |
|---|---|
| 现场选择 | `grasp_v3_step20000_rtc`，v3 **20000 步**，检查点目录 `19999` |
| v3 训练 | 122 段演示、28,406 帧，`pi05_teleavatar_v2_chassis` |
| v3 策略服务 | `grasp-v3-step20000-policy`，GPU1，`127.0.0.1:8005` |
| JAX 导出 | `/data/ta2/model_exports/grasp_v3_step20000`，20 个正式文件、12,440,023,546 字节 |
| PyTorch 导出 | `/data/ta2/model_exports/grasp_v3_step20000_pytorch`，权重 7,233,650,408 字节，另有 config/norm_stats |
| 归一化 | `assets/grasp_20261115_valid/norm_stats.json`；部署目录内兼容 alias 指向相同统计 |
| 提示词 | `抓取红色沙包` |
| 策略设置 | TRAINED RTC，单次动作时限 60 秒，最大关节速度 1.0 rad/s |
| registry 默认 | `grasp_v4_clean81_step10000_rtc`，GPU0、8006，81 段清洗数据、10000 步 |
| 其他保留服务 | v3 13000:8004、v3 10000:8003、v2 4999:8000 |

**registry 默认与现场已选择模型是两个状态。** 现场选择保存在
`/root/ta2_vla_selected.json`，应查 `GET /status` 的 `vla_selected`。
本提交不覆盖这个状态文件，不会切换模型。v3 5000 的脚本保留用于追溯，其服务此前已停止。

代码交付中的成功判定为持物抬高至少 4 cm、连续 0.5 秒；采集时 B70 部署文件仍为
1.5 秒。空夹 2 秒会先停止旧 VLA、原地张开该爪，再在本次剩余时间内重启推理。
路径之间默认暂停 12 秒，到达后暂停 3 秒。模型启动、复位、取帧、RTC 校准时间应与
实际动作时限区分；网页墙钟时间不能直接视为单次推理延迟。

权重单独备份到 [Google Drive 的 ta2/grasp_v3_step20000](https://drive.google.com/drive/folders/1QYLED2FFipbOZ_jQChz1mgfZnwEtew_V)。
以该目录的文件清单与校验结果判断备份完整性；Git 不包含权重、演示、优化器状态或镜像。

## 页面入口

B70 与机器人处于同一路由器；下表主机为 `192.168.31.228`。

| 端口 | 页面 | 源码 |
|---|---|---|
| 8088 | 机器人遥控、VLA、全流程、API 文档 | `web_move.py`、`mission.py` |
| 8090 | 数采下载控制台 | `tools/ta2_download_web.py`、`ta2_download.py` |
| 8093 | 数据筛选台 | `tools/ta2_review_server.py` |
| 8095 | 数据转化台 | `tools/ta2_convert_server.py` |
| 8096 | 推理监控台 | `tools/ta2_infer_monitor.py`、`infer_monitor_ui.html` |
| 8097 | 机器人视频查看器 | `tools/ta2_vision_server.py` |
| 8099 | B70 语音前端 | `voice/b70_frontend.py` |
| 8100 | 感知与标定 | `tools/ta2_perception/` |
| 8101 | 腕部视频抓取判定 | `tools/ta2_grasp/` |
| 8102 | 头部双目深度 | `perception/head-depth/` |
| 8103 | 工具总台 | `home/index.html` |
| 8104 | 头部检测工作台 | `perception/head-detection/` |

监控台支持 p50/p95、推理预算、循环与执行耗时、RTC 窗口、动作块、关节跟踪和历史 run。
部分旧 RTC run 没记录推理耗时，此时页面会明确显示不可用；不补造数据。
视频查看器与推理客户端共用机器人 UDP 8890，不能同时接收；`vla_run.sh` 会按已有流程停查看器。

## 恢复顺序

1. 准备主办方镜像与模型资产。实际运行镜像及脱敏启动参数在
   `deploy/runtime/container-recipes.json`，对应源码与锁文件见该目录 README。
   XPU 依赖以镜像和 `openpi-xpu-env` 的锁文件为准，不在上游 OpenPI 源码下随意 `uv sync`。
2. 将 `deploy/runtime/deploy/` 的主办方源码恢复到 `/data/ta2/software/deploy/`。
   客户端 bind mount 必须指向恢复的 `client/openpi`；服务端指向 `server/openpi`。
   将独立备份的模型放回上表路径，核对清单及 SHA-256 后使用已归档的部署 recipe。
   历史 `cutover` 脚本会改变模型目录/registry，仅用于追溯，不自动重放。
3. 官方 ROS2 Humble 与 zenoh 桥在 `openpi-xpu-client` 容器。按现有官方 `run_client.sh`
   启动；不要另起宿主机桥。桥配置及客户端入口包含在源码快照内。
4. 在 B70 仓库运行 `./install_b70.sh` 安装机器人控制代码并重启 **8088**。
   该脚本不启动抓取，但会替换运行代码并重启网页，需在现场空闲时执行。
5. 工具网页源码可用 `python3 deploy/install_tools.py --target /data/ta2` 复制。
   此命令保留已有 JSON/环境配置，不安装 systemd、不启停服务、不下载模型。
   临时恢复演练可将 `--target` 指向空目录。
6. 感知/深度/检测的 Python 环境按各子目录 requirements/安装脚本准备；模型权重另备。
   审阅 `deploy/systemd/` 的路径后再按需安装具体服务。`tools/ta2_services.sh status`
   与 `tools/ta2_monitor_ops.sh doctor` 可查看现场工具状态。

本次只校验源码与离线流程，没有重建大型镜像或进行真机抓取验收。上游代码有些直接来自
主办方离线软件包，无法独立确认完整 Git revision；保留镜像标签与逐文件哈希来标识来源。

## 离线验证与已知限制

```bash
python3 -m unittest discover -s tests -v
python3 tools/ta2_infer_monitor.py --selftest --no-frames
```

第一组使用假 ROS、假动作脚本与 localhost HTTP；第二组使用合成 run，不连接策略服务或机器人。
仅作归档的上游源码不意味着本次已验证全部可选模型、安装器或训练工具。

本次结果：84 项网页/API/编排离线测试通过；监控合成自检通过；B70 原客户端容器中的
15 项 guard/stub 执行测试通过（没有连接机器人或运行真实策略）。229 个 Python 文件
语法、50 个 Shell 文件语法、两份来源清单的 SHA-256 校验通过。工具安装器在临时目录
演练通过，重复安装保留已有配置；没有把测试安装到现场目录。

已有代码审查指出 RTC 阻塞返回后的观测复查、左右手仲裁、保护退出重试分类、抬高参考点、
无抬升超时与稳定确认的竞争，以及第六步张爪后的释放确认仍需单独处理。
本次整理未修复这些控制行为，也不以离线通过替代现场可达性、碰撞、释放或成功率验证。
