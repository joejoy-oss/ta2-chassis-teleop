# Docker 使用说明（B70 / Intel VA-API）

本仓库的转换脚本依赖 ROS2 Humble、Python 3.11、LeRobot、ffmpeg 和 VA-API 驱动。这些依赖全部打包在镜像 `rosbag2lerobot:0.2.0-20261030` 里；宿主机不需要安装 ROS2、conda、LeRobot 或 ffmpeg。代码通过 volume 挂载进容器，改代码不需要重建镜像。

本说明对应 **B70 分支**：在 Intel GPU 上通过 VA-API 做 HEVC 硬件编码。脚本按 NVIDIA NVENC → Intel/AMD VA-API → CPU 的顺序自动选择后端。

## 1. 文件准备

| 文件 | 说明 |
|---|---|
| `rosbag2lerobot_0.2.0-20261030.tar` | 转换镜像，约 5.97 GB |
| `SHA256SUMS` | 镜像校验文件，与 tar 放在同一目录 |
| 本仓库（B70 分支） | 转换脚本和 `lerobot` 子模块 |

U 盘里镜像 tar 和 `SHA256SUMS` 在 `convert/docker/` 下，本仓库在 `convert/rosbag_to_dataset_TA2/`（已包含 `lerobot` 子模块，不需要再下载）。从其他途径拿到镜像时，tar 和 `SHA256SUMS` 要放在同一目录。镜像加载后约 21.2 GB；输出数据集按输入 bag 的约 6 倍预留空间。使用 `960x480` / `1280x400` 时，视频体积约为默认分辨率的 1/4。

### 1.1 从 GitHub 获取代码

不用 U 盘里的代码时，可以直接克隆 B70 分支：

```bash
git clone -b B70 https://github.com/dexteleop/rosbag_to_dataset_TA2.git
cd rosbag_to_dataset_TA2
git config --local submodule.lerobot.url https://github.com/zhou-yh19/lerobot.git
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive --depth 1
```

## 2. 前置条件

- Linux x86_64，Docker 可用。
- 宿主机能看到 Intel GPU 的 render 节点，例如：
  ```bash
  ls /dev/dri
  # card0
  # renderD128
  ```
- 宿主机内核驱动正常。容器只借用 `/dev/dri`，libva 和 intel-media-driver 已在镜像内。
- 当前用户有权限访问 Docker，或使用 `sudo`。

## 3. 加载镜像

在镜像 tar 和 `SHA256SUMS` 所在目录（U 盘里是 `convert/docker/`）执行：

```bash
sha256sum -c SHA256SUMS
docker load -i rosbag2lerobot_0.2.0-20261030.tar
docker image ls rosbag2lerobot
```

看到 `rosbag2lerobot:0.2.0-20261030` 即加载成功。

## 4. 目录约定

```bash
PROJECT_DIR=/path/to/rosbag_to_dataset_TA2   # 本仓库代码目录
BAG_DIR=/path/to/rosbags                     # bag 目录
OUT_DIR=/path/to/output                      # 输出目录
mkdir -p "$OUT_DIR"
```

每个 bag 必须是完整的 rosbag2 目录，包含 `metadata.yaml` 和 `.db3` / `.mcap`：

```text
$BAG_DIR/
├── bag_001/
│   ├── metadata.yaml
│   └── bag_001_0.mcap
└── bag_002/
    ├── metadata.yaml
    └── bag_002_0.mcap
```

单独复制的 `.mcap` 文件没有 `metadata.yaml` 时无法读取。

## 5. 运行转换

在 B70 上推荐使用 VA-API：

```bash
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e LIBVA_DRIVER_NAME=iHD \
  -e R2L_VIDEO_BACKEND=vaapi \
  --shm-size=8g \
  -w /workspace/rosbag_to_dataset_TA2 \
  -v "$PROJECT_DIR":/workspace/rosbag_to_dataset_TA2 \
  -v "$BAG_DIR":/data/rosbags:ro \
  -v "$OUT_DIR":/workspace/hf_lerobot \
  rosbag2lerobot:0.2.0-20261030 \
  python scripts/convert_rosbag_to_lerobot.py \
    --multibag \
    --input_directory /data/rosbags \
    --output move_test \
    --fps 45 \
    --head_resolution 960x480 \
    --wrist_resolution 1280x400 \
    --task "my task description"
```

说明：

- `-v "$PROJECT_DIR":...` 不要加 `:ro`：日志会写到代码目录的 `scripts/logs/`。
- `--device /dev/dri` 传入所有 render 节点；`--group-add` 需要与 render 节点的组 ID 一致。如果实际使用的不是 `renderD128`，把 `stat` 的路径换成对应节点。
- 镜像默认以 `ubuntu` 用户（uid 1000）运行。`--user` 让容器改用当前用户的 uid/gid，输出数据集和日志都归当前用户所有；当前用户不是 uid 1000 时，不加它会写不进 `$OUT_DIR` 和 `scripts/logs/`，或写出当前用户改不了的文件。`-e HOME=/tmp` 给这个 uid 一个可写的家目录。
- `--shm-size=8g` 给 PyAV/多进程留出共享内存。

输出保存在 `$OUT_DIR/move_test`，日志在 `$PROJECT_DIR/scripts/logs/convert_rosbag_to_lerobot_<时间戳>.log`。

### 5.1 先试转几条

只转换前 2 个 episode，确认链路通：

```bash
# 在上面的命令末尾加上
--max_episodes 2
```

### 5.2 没有 X/Y 标记时

整个 bag 作为一条 episode：

```bash
# 把 --multibag 换成
--whole_bag
```

## 6. 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--multibag` | 输入目录包含多个 bag 时使用，会递归扫描 | 关闭 |
| `--input_directory` | 容器内 bag 目录 | `./data/rosbags` |
| `--output`, `-o` | 输出数据集名，例如 `move_test` 或 `命名空间/数据集名` | `output/dataset` |
| `--fps` | 数据集帧率 | `45` |
| `--task` | 任务描述，所有 episode 共用 | `task description` |
| `--head_resolution WxH` | 头部相机输出分辨率；`source` 保持源尺寸 | `1920x960` |
| `--wrist_resolution WxH` | 左右腕部相机输出分辨率；不写则保持源尺寸 | `2560x800` |
| `--whole_bag` | 整个 bag 作为一条 episode，不需要 X/Y | 关闭 |
| `--max_episodes N` | 只转换前 N 条，便于测试 | 不限制 |
| `--enforce_all_video_topics` | 缺少任意一路相机视频时跳过该 bag | 关闭 |
| `--crf N` | 编码质量，越低质量越好、文件越大；GPU 上是 QP | `23` |
| `--nvenc_sessions N` | NVIDIA 并发编码数 | `7` |
| `--vaapi_sessions N` | 每个 render 节点的 VA-API 并发编码数 | `4` |
| `--verbose` | 日志直接打印到终端，不写日志文件 | 关闭 |
| `--log_file PATH` | 指定日志文件路径 | `scripts/logs/` 下自动命名 |

分辨率会写入数据集的 `meta/info.json`，创建后不可修改。换参数重跑时必须使用新的输出目录：`LeRobotDataset.create` 在目录已存在时会报 `FileExistsError`。

## 7. 编码后端

脚本按 **NVIDIA NVENC → Intel/AMD VA-API → libx265 CPU** 的顺序自动检测，也可以用环境变量强制指定：

```bash
-e R2L_VIDEO_BACKEND=vaapi    # 强制 VA-API
-e R2L_VIDEO_BACKEND=nvidia   # 强制 NVENC
-e R2L_VIDEO_BACKEND=cpu      # 强制 CPU，仅作临时方案
```

B70 的启动日志里应出现：

```text
Video backend: VA-API (/dev/dri/renderD128)
```

如果显示 `CPU (libx265 ...)`，按第 10 节检查。

在容器内单独验证 VA-API：

```bash
docker run --rm -it \
  --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e LIBVA_DRIVER_NAME=iHD \
  rosbag2lerobot:0.2.0-20261030 \
  vainfo --display drm --device /dev/dri/renderD128
```

能看到 `VAProfileHEVCMain` 和 `VAEntrypointEncSlice` 说明硬件编码可用。

如果有多个 render 节点或需要调整并发，改 `--vaapi_sessions`；默认 4 路通常能跑满 Arc B 系列 GPU。

## 8. X/Y 分段规则

- 手柄 **X** 开始一条 episode；录制中再按 X 会丢弃当前缓冲并重新开始。
- **Y** 结束并保存。
- 按 Y 停止后，紧接着再按 2 次 Y（连同停止那次共 3 次，相邻两次间隔不超过 6 秒），表示该条失败，直接丢弃。一共只按了 2 次 Y 时，日志会提示歧义，但这条仍然保留。
- 少于 30 帧的 episode 直接丢弃。
- bag 结束时仍未按 Y 的 episode 直接丢弃。

## 9. 输出与检查

输出是 LeRobot v2.1 数据集：

```text
move_test/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/
│   └── episode_000000.parquet
└── videos/chunk-000/
    └── observation.images.*/
        └── episode_000000.mp4
```

检查 episode 数和帧数：

```bash
python3 -c "import json; d=json.load(open('$OUT_DIR/move_test/meta/info.json')); print('episodes', d['total_episodes'], 'frames', d['total_frames'])"
```

检查三路视频分辨率（`shape` 顺序为 `[高, 宽, 通道]`）：

```bash
python3 -c "import json; d=json.load(open('$OUT_DIR/move_test/meta/info.json')); [print(k, v['shape']) for k, v in d['features'].items() if 'images' in k]"
```

使用 `--head_resolution 960x480 --wrist_resolution 1280x400` 时应输出：

```text
observation.images.head_camera [480, 960, 3]
observation.images.left_color [400, 1280, 3]
observation.images.right_color [400, 1280, 3]
```

转换日志中确认：

- `Video backend: VA-API (...)`，而不是 `CPU (libx265 ...)`。
- 每条 episode 有 `Saved episode ... with N frames`。
- `total_episodes` / `total_frames` 不为 0。

### 9.1 用于训练

- 把 `$OUT_DIR/move_test` 整个目录拷到训练机，训练配置里的 `repo_id` 填它在训练机上的**绝对路径**（见训练 README）。
- `observation.state` 和 `action` 都是 76 维。其中下标 73–75（从 0 开始数）是底盘目标速度 `chassis_target_vel_x/y/yaw`，`pi05_teleavatar_v2_chassis` 训练底盘用的就是这三维；不带底盘的配置只取前面的手臂和夹爪部分，同一份数据都能用。

## 10. 常见问题

### 无法连接 Docker daemon

```text
permission denied while trying to connect to the Docker daemon socket
```

把当前用户加入 docker 组，然后重新登录：

```bash
sudo usermod -aG docker $USER
```

### 输出目录没有写权限

确认 `docker run` 带了：

```bash
--user "$(id -u):$(id -g)" -e HOME=/tmp
```

并确认 `$OUT_DIR` 对当前用户可写。

### `No module named 'lerobot'`

镜像里的 lerobot 是以可编辑方式安装的，指向 `/workspace/rosbag_to_dataset_TA2/lerobot/src`，所以先确认：

- `-v "$PROJECT_DIR":/workspace/rosbag_to_dataset_TA2` 挂载的是仓库根目录；
- 宿主机上 `$PROJECT_DIR/lerobot/src/lerobot` 存在。

U 盘里的代码已经带了 `lerobot/`，出现这个报错一般是挂载路径不对。从 GitHub 克隆的代码如果 `lerobot/` 是空目录，在仓库根目录执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive --depth 1
```

### VA-API 不可用 / 回退到 CPU

依次检查：

```bash
ls -l /dev/dri
stat -c '%g' /dev/dri/renderD128
```

- `docker run` 是否带了 `--device /dev/dri`。
- `--group-add` 的组 ID 是否与 render 节点一致。
- 是否设置了 `-e LIBVA_DRIVER_NAME=iHD`。
- 容器内 `vainfo` 是否能看到 `VAProfileHEVCMain` + `VAEntrypointEncSlice`。
- 是否误设了 `R2L_VIDEO_BACKEND=nvidia`。

CPU 回退能跑通，但速度慢很多，只适合临时排查。

### 没有生成 episode

- bag 目录是否有 `metadata.yaml` 和 `.mcap` / `.db3`。
- 是否包含约定的相机话题和 `/xr/left_hand_inputs` 标记话题。
- 没有 X/Y 标记时是否加了 `--whole_bag`。
- 具体报错看 `$PROJECT_DIR/scripts/logs/` 下对应日志。

### `FileExistsError`

输出数据集已存在。删除旧的 `$OUT_DIR/move_test`，或换一个 `--output` 名字。

### 编码很慢 / GPU 没跑满

- 确认日志使用 VA-API，而不是 CPU。
- 多 render 节点时，适当调大 `--vaapi_sessions`。
- 确认输出目录和输入 bag 都在本地磁盘，不是网络盘。

### 镜像和输出占空间

- 镜像 tar 约 5.97 GB，加载后约 21.2 GB。
- 输出数据集按输入 bag 的约 6 倍预留；用 `960x480` / `1280x400` 可以明显减小视频体积。
