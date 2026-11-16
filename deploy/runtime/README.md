# B70 推理、客户端与数据转换运行源码快照

本目录是 2026-11-16 从 B70 实际挂载的工作树读取的源码，不含模型权重、演示数据、镜像归档、驱动二进制或登录凭据。采集过程没有启动模型推理、机器人动作或修改运行服务。

## 内容与来源

| 本目录路径 | B70 源路径 / 用途 |
| --- | --- |
| `deploy/server/openpi/` | `/data/ta2/software/deploy/server/openpi/`；服务端 `src`、`packages`、转换/服务脚本及依赖说明 |
| `deploy/server/openpi-xpu-env/` | `/data/ta2/software/deploy/server/openpi-xpu-env/`；XPU 环境镜像构建、锁文件和 UMD 清单 |
| `deploy/client/openpi/` | `/data/ta2/software/deploy/client/openpi/`；实际客户端包、RTC runtime、TeleAvatar V2 ROS/相机接口及测试源码 |
| `deploy/client/` 其余文件 | `/data/ta2/software/deploy/client/`；ROS2 Humble 客户端镜像、桥接、VA 解码构建与运行脚本 |
| `convert/rosbag_to_dataset_TA2/` | `/data/ta2/software/convert/rosbag_to_dataset_TA2/`；ROSbag→LeRobot/RLDS/HDF5 及检查/合并/切分脚本 |
| `recipes/step20000/` | `/data/ta2/model_exports/grasp_v3_eval/step20000_deployment/`；v3 20000 步转换、独立服务、离线验证、网页切换的历史执行脚本 |

上游 OpenPI 项目为 <https://github.com/Physical-Intelligence/openpi>；各快照保留其原 `LICENSE`。转换项目的 `LICENSE` 与 `.gitmodules` 原样保留。LeRobot 子模块未复制；`.gitmodules` 指向 `zhou-yh19/lerobot` 的 `rosbag_to_lerobot_TA2` 分支。

现场软件包没有 `.git`，因此不能仅凭当前目录声明其 Git 提交。`container-recipes.json` 记录已加载镜像的来源标签：

- 服务镜像：`openpi-xpu-env:v0.1.0`，镜像 ID `sha256:6ae851138457fb6c5e068c5e3f6f844fe2c0820a2c65794fe794f3ec3f46ffbb`，来源标签 `22a391be9d45b45e96a3a6818de16df589c23ff3`，Python 3.11.17 / Torch 2.12.1+xpu。
- 客户端：`openpi-xpu-client:v0.2.1`，镜像 ID `sha256:a429bf1494e62175e75458b4eeeaf03a64b99b1986796ce88b3523f5c5ac47c3`，来源标签 `29506182db83183feef7d0e560925dbcd5ebcdf2`。

镜像标签表示构建来源，不能替代本目录源码逐文件 SHA-256；实际采集内容以 `capture-manifest.json` 为准。服务端和客户端的 `src` 源码在采集时一致，因此客户端副本只保留实际运行需要的 `packages` 和 `examples/teleavatar_v2`。客户端 `packages/openpi-client/runtime/runtime.py` 与服务端版本不同，分别保留。

## 当前 20000 步模型与推理服务

当前网页选中 `grasp_v3_step20000_rtc`，训练结束检查点目录为 `19999`，并非 200000 步。网页 registry 的默认值可能仍是 v4 10000 步，当前选择应从 `8088/status` 读取。

- JAX 推理检查点：`/data/ta2/model_exports/grasp_v3_step20000/`。
- 转换后的 PyTorch：`/data/ta2/model_exports/grasp_v3_step20000_pytorch/`。
- 统计参数：`assets/grasp_20261115_valid/norm_stats.json`；兼容别名 `assets/checkout_224_plus_no_scan_224 → grasp_20261115_valid`。
- 模型配置：`pi05_teleavatar_v2_chassis`；action_dim 32、action_horizon 30、bfloat16。
- 服务容器 `grasp-v3-step20000-policy`，GPU1 / `ZE_AFFINITY_MASK=1`，宿主 `127.0.0.1:8005` 映射容器 8000，RTC `TRAINED`。
- 提示词：`抓取红色沙包`。

`container-recipes.json` 提供实际 entrypoint、command、白名单环境变量、挂载、设备与端口。为还原服务，先准备原模型/统计参数和 `/data/ta2/sim_pipeline/openpi-assets`，再将本快照的服务源码恢复到配方指定路径，最后按配方启动。只启动策略服务不会发布真机动作。

服务端关键命令为：

```bash
python scripts/serve_policy.py --env TELEAVATAR --port 8000 \
  --default-prompt '抓取红色沙包' --warmup \
  --rtc.enabled --rtc.mode TRAINED \
  policy:checkpoint --policy.config pi05_teleavatar_v2_chassis \
  --policy.dir /ckpt --policy.pytorch-device xpu \
  --policy.pytorch-compile-mode default
```

该命令在配方对应的 `openpi-xpu-env:v0.1.0` 容器内、`/app` 工作目录运行，不是在宿主直接运行。

## JAX → PyTorch 转换

在已经加载的服务端环境镜像内，设置 `JAX_PLATFORMS=cpu`、将源码只读挂到 `/app`，将导出目录挂到 `/outputs`，使用：

```bash
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /outputs/grasp_v3_step20000 \
  --config-name pi05_teleavatar_v2_chassis \
  --output-path /outputs/grasp_v3_step20000_pytorch \
  --precision bfloat16
```

转换需要约 44 GiB 内存限额的环境，不能在本地 2 GiB VM 上执行。历史执行记录采用 B70 CPU；仅在明确安排转换时运行。完整原始编排在 `recipes/step20000/deploy_step20000.py`，有目标目录和容器名称冲突检查。它同时负责启动预热和离线 5 帧验证，因此不是通用只读检查器。

`recipes/step20000/cutover_step20000.py` 是历史网页切换脚本，会改 registry 和重启网页；保留用于解释部署来源，不作为重建当前 registry 的默认步骤。当前 registry 中还有后续 v4，不应被这份旧脚本覆盖。

## 客户端与机器人连接

ROS 在 `openpi-xpu-client` 中，宿主不另装 ROS、不另外启动 zenoh 桥。实际 ROS_DOMAIN_ID=29，RMW 为 CycloneDDS，`ROS_LOCALHOST_ONLY=1`，`ZENOH_ENDPOINT=tcp/192.168.31.218:9000`。桥由已有 `run_client.sh` 的 supervisor 管理。

```bash
cd /data/ta2/software/deploy
export OPENPI_DIR="$PWD/client/openpi"
ZENOH_ENDPOINT=tcp/192.168.31.218:9000 client/run_client.sh status
# 确需启动或停止时，使用同一脚本的 start / stop 子命令。
```

实际 guard、姿态、编排、网页遥控在主仓库 `vla/`、`arm/`、`mission.py`、`web_move.py`；此快照保留其依赖的官方 RTC、ROS2 和视频实现。

## 构建与还原边界

- `server/openpi/pyproject.toml` 是上游源码依赖，含 CUDA 默认依赖。真实 XPU 环境的锁位于 `server/openpi-xpu-env/deps/`；请按该目录的 `README.md`、`deps/NOTE.md` 和构建脚本操作，不在源码目录直接 `uv sync` 替换已验证环境。
- 镜像 tar 和 UMD `.deb` 不入 Git。`deploy/server/openpi-xpu-env/deps/umd-host/SHA256SUMS`、`convert/docker/SHA256SUMS`、客户端 `SHA256SUMS` 保留原始软件包校验清单；重新构建需要组织方提供这些离线依赖或原有镜像。
- 转换项目的 LeRobot 自定义分支源码没有采集，且现场没有 Git 元数据可核对 commit。已有 `rosbag2lerobot` 镜像可运行现有转换脚本；仅本仓库无法证明从空机器联网构建得到逐字节相同镜像。
- 采集后的静态检查只能确认源码语法与快照完整性，不能替代驱动、推理结果或机器人现场验收。
