# openpi-xpu-client：在 Intel B70 上运行 TeleAvatar V2 客户端

本镜像用于在无 NVIDIA GPU 的 B70 机器上运行 `examples/teleavatar_v2/main.py` 和 `zero.py`。
镜像仅包含运行环境（ROS 2 Humble、打过补丁的 `zenoh-bridge-ros2dds`、编译支持 B70 的 Intel VA-API 驱动栈及 Python 虚拟环境），**不含 openpi 代码**。代码在运行时只读挂载到 `/app`（需使用 `feat/teleavatar-v2-client-b70` 分支，本目录下的 `openpi/` 就是该分支的代码，挂载方法见 2.2）。

- **硬件解码（推荐）**：`--video-decode va`（机器人真实码流无 WPP，必须靠显卡硬解）。
- **CPU 解码**：`--video-decode cpu`（仅对开启了 WPP 的码流够快）。

> `openpi/` 里的 README（包括 `openpi/examples/teleavatar_v2/README.md`）只作参考，可以在里面查话题、相机格式、夹爪映射和 `main.py` 的参数。其中建 conda 环境、在宿主机上启动 `zenoh-bridge-ros2dds` 的步骤已由本镜像和 `run_client.sh` 代替，**不要照做**：宿主机上另起的 zenoh 桥会和容器里的桥冲突。

## 1. 构建与加载

**在 B70 上加载并验证显卡：**

使用我们提供的client docker镜像，运行以下命令：
```bash
sha256sum -c openpi-xpu-client_v0.2.1.tar.gz.sha256
docker load -i openpi-xpu-client_v0.2.1.tar.gz

# 验证 VA-API 硬件加速
docker run --rm --device /dev/dri/renderD128 --group-add 993 --entrypoint bash openpi-xpu-client:v0.2.1 \
    -c '/opt/intel-va/bin/vainfo | head -8; gst-inspect-1.0 vah265dec | head -3'
```

## 2. 在 B70 上运行客户端

### 2.1 运行前提
1. **唯一控制源**：其他子网内的控制端无运行；B70 宿主机的 `zenoh-bridge-ros2dds.service` 已停止（TCP 7447 无监听）。
2. **机器人切到 API 模式**（一次性设置，重启后保留）：在 remoteApp（或接在机器人上的显示器）的 **System Config** 面板中，把 **Run mode** 设为 API（默认是 VR），启用左右臂并把控制模式设为 **joint**；先点 **Save config** 保存，再点 **Apply** 生效。不切到 API 模式，机器人不会执行策略发出的指令。
3. **网络与视频**：机器人视频推流目标指向 B70 的 UDP 8890，在推流remote APP中设置B70的IP地址并启动。
4. **推理服务端**：已在 `127.0.0.1:8000` 启动，且参数与客户端配套（详见服务端 README 2.4 节）：
   - **RTC**：服务端以 `--rtc.enabled` 启动时，`main.py` 才能加 `--rtc`，否则报 `Policy server did not return an RTC chunk id`。
   - **底盘**：服务端用 `pi05_teleavatar_v2_chassis`（输出 19 维动作）时，`main.py` 必须加 `--enable-chassis`；其他配置输出 16 维，不能加。两端不一致时第一个动作就会报错。
5. **以 root 运行**：`run_client.sh` 默认把日志写到 `/root/openpi-xpu/client/outputs`，启动时还要修改显卡频率，普通用户执行 `start` 会失败。请先 `sudo -i` 切到 root，再执行 2.2 的全部命令（包括 `export OPENPI_DIR`）。直接用 `sudo client/run_client.sh …` 时，`export` 的变量不会传进去，`OPENPI_DIR` 等会变回默认值。

### 2.2 常用命令

以下命令都在 `client/` 的上一级目录执行。启动前先用 `OPENPI_DIR` 指定 openpi 代码目录（即本目录下的 `openpi/`），`start` 时会把它只读挂载到容器的 `/app`：

```bash
cd /path/to/deploy                        # client/ 的上一级目录
export OPENPI_DIR=$(pwd)/client/openpi    # 必须是绝对路径
```

> **注意**：不设置 `OPENPI_DIR` 时，默认挂载 `/root/openpi-xpu/openpi-client`。该目录不存在时，docker 会挂上一个空目录，之后 `zero.py`、`main.py` 都会报找不到文件。`OPENPI_DIR` 只在 `start` 时生效，容器已经启动的话，需要先 `stop` 再 `start`。

```bash
# 1. 启动常驻容器（自动拉起 zenoh 桥，并将显卡解码引擎最低频率提至 1500MHz）
ZENOH_ENDPOINT=tcp/<ROBOT_IP>:9000 client/run_client.sh start

# 2. 检查状态（确认 bridge.log 正常，且能看到 /left_arm/joint_states 等话题）
client/run_client.sh status

# 3. 运行回零与主程序（硬解）
client/run_client.sh exec python examples/teleavatar_v2/zero.py
client/run_client.sh exec python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --remote-port 8000 \
    --rtc --video-decode va --prompt "<任务描述>"
# 服务端没开 RTC 时去掉 --rtc；服务端用底盘配置时加 --enable-chassis（见 2.1 第 4 条）

# 若使用 CPU 解码，请将 --video-decode va 替换为以下固定参数：
# --video-decode cpu --video-cpu-thread-type slice --video-cpu-threads 8 --video-convert-threads 4

# 4. 停止客户端与容器（安全停机，切勿直接 docker stop！）
client/run_client.sh stop
```
> **注**：容器运行期间会通过 `client/gpu_minfreq.sh` 锁频以保证解码延迟，容器退出时自动恢复（可通过 `GT1_MIN_MHZ=off` 禁用，或用 `client/gpu_minfreq.sh status/restore` 手动查看和恢复）。

### 2.3 网络连接方式

*   **局域网直连**：B70 直连机器人时，启动命令是加下列的环境变量一键启动：`ZENOH_ENDPOINT=tcp/<ROBOT_IP>:9000 client/run_client.sh start`。
 *   **直连安全检查**：跑脚本前必须先用 `ros2 topic hz /left_arm/joint_states` 确认有数据，且用 `ros2 topic info` 确认左右臂话题的 Publisher count 均为 1。

### 2.4 ⚠️ 安全停机与底盘操作守则（必读）
1. **如何停止 `main.py`**：
   * 在当前终端按 `Ctrl-C`；
   * 或在另一终端执行 `client/run_client.sh exec pkill -INT -f examples/teleavatar_v2/main.py`；
   * 或执行 `client/run_client.sh stop`（会自动先发 SIGINT 等其收尾再停容器）。
2. **严禁直接 `docker stop / kill / rm -f`**：这会导致进程被直接 SIGKILL 而无法发送零速度指令，底盘会继续移动约 1 秒！
3. **SSH 断连不会停止程序**：`docker exec` 进程在终端断开后不会收到 SIGHUP，机器人会继续运动！断网时请**先拍现场急停**，重连后再用 `pkill -INT` 停止。
4. **开启底盘（`--enable-chassis`）前的检查**：
   启动 `main.py` 前，必须执行：
   ```bash
   client/run_client.sh exec ros2 topic info -v /api/chassis/velocity
   ```
   确认存在节点名为 `zenoh_bridge_ros2dds` 且 `Reliability: RELIABLE` 的订阅者，否则严禁直接让真机开始运动。首次部署需专人守急停，确认各轴方向及 `Ctrl-C` 能否立即刹停。




