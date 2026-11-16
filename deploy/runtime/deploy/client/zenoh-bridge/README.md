# 打了补丁的 zenoh-bridge-ros2dds 1.9.0（镜像 v0.2.1）

## 为什么要打补丁

B3 的 M4 发现，假机器人收到的 `/api/*/joint_cmd` 有多份：main.py 发 200 Hz，假机器人收到 400、1000、1200 Hz，倍数一轮比一轮大。

原因在 zenoh-bridge-ros2dds 的 `RoutePublisher`，也就是把本机 ROS 话题转发到 zenoh 的那一段：

- route 删除时，只删了当时的 DDS Reader。
- "有没有订阅者"的 matching listener 用 `.background()` 注册，回调里持有 zenoh publisher，所以 route 删掉后它还在。
- 之后机器人一侧每出现一次订阅者，每个残留的监听器都会再建一个 DDS Reader，每条命令就多发一份。

`route_service_cli.rs` 也是同样的写法。1.9.0、1.10.2 和上游 main 分支里，这几处代码都相同，所以换版本修不好。

本机复现、补丁和验证见 b70_client 的 `logs/pB3/41`–`46`（PLAN B28）。

## 补丁

- `route-lifecycle-p1.patch`（sha256 `fd2e037d…`，+53 / −6 行）由 `patch_route_lifecycle.py` 生成。它改了两处：
  - route 自己持有 matching listener。
  - route 删除时，先把"已删除"标志置上、注销监听器，再删 DDS Reader；之后来的回调见到这个标志就什么也不做。
- 补丁不改协议，也不改版本号：`--version` 仍然是 v1.9.0。与机器人一侧没打补丁的 1.9.0 互通，本机实测是 1 倍。
- 只修客户端这一侧，也就是从客户端转发到机器人的方向。机器人一侧的桥若也是 1.9.0，反方向（`joint_states`）照样会出现多份。那不归我们改，只报告。

## 已知限制

补丁没改 `dds_utils.rs`（补丁版与原版逐字节相同），那里还有一处泄漏（b70_client 的 `reports/phaseB3_review2.md`，复审建-4）：

- `delete_dds_entity`（第 71–79 行）只调 `dds_delete`。`create_dds_reader`（第 270 行起）用 `Box::into_raw` 把读者回调交给 DDS listener，这个回调没有人释放；补丁版 `route_publisher.rs` 交给它的回调持有 zenoh publisher 的 `Arc`（第 458 行 `publisher.clone()`）。所以每删一个 route，它的 zenoh publisher 与回调就留着，publisher 也不撤销（undeclare）。
- 删掉的 Reader 不再收数据，所以不会多出副本：打了补丁的桥在本机复现（`logs/pB3/42` 的 p1c、p1b 两组）与 fix1 的 M4（`logs/pB3/70`）里几十次 route 的建与删，都没有多出副本。
- 长时间、频繁重连下的内存没有测过；会话长时可以记一下桥进程的 RSS。

## 编译

```bash
WORK=/some/dir client/zenoh-bridge/build_bridge.sh      # 得到 $WORK/out/zenoh-bridge-ros2dds-1.9.0-p1
```

- 在 `openpi-xpu-client:v0.2.0` 的一次性容器里编，系统是 Ubuntu 22.04，所以二进制要求的 glibc 不比镜像里的新。
- 1.9.0 源码包和 Rust 1.93.0 安装包都按 sha256 核对。Rust 版本是源码里 `rust-toolchain.toml` 指定的。
- 编译命令是 `cargo build --release --locked -p zenoh-bridge-ros2dds`，缺省特性，与 Eclipse 编 .deb 的方式相同。
- 产物 `778d784a…`，19,856,568 字节，只依赖 libgcc_s、libm、libc。
- 用这个脚本在 B3 `45` 的工作目录里重编了一次（`63`，容器内路径相同），得到的字节完全相同。

## 镜像 v0.2.1

```bash
BRIDGE=…/zenoh-bridge-ros2dds-1.9.0-p1 client/zenoh-bridge/build_image.sh   # 本机 buildx
docker save openpi-xpu-client:v0.2.1 | gzip -1 > openpi-xpu-client_v0.2.1.tar.gz
sha256sum openpi-xpu-client_v0.2.1.tar.gz > openpi-xpu-client_v0.2.1.tar.gz.sha256
```

- `Dockerfile.v0.2.1` 的 `FROM` 用摘要钉住 v0.2.0（`openpi-xpu-client:v0.2.0@sha256:bdf07374…`）。
- 它只换 `/usr/bin/zenoh-bridge-ros2dds` 这一个文件，按 sha256 核对，再往 `/opt/openpi-client/TOOLCHAIN` 追加一行。
- 加的 label：`org.opencontainers.image.version=v0.2.1`、`zenoh.bridge-ros2dds=1.9.0+q16p1`、`zenoh.bridge-ros2dds.sha256`、`zenoh.bridge-ros2dds.patch-sha256`。
- 其余的层、库、ENV、入口都和 v0.2.0 相同，`openpi.commit` 也照旧（镜像里没有 openpi 代码）。
- 不在 B70 上建，因为 B70 的 Docker 没有 buildx。做法同 v0.2.0：本机建好、存档，传过去 `docker load`，两边的镜像 ID 相同。
- B3 `64` 的结果：
  - 镜像 ID `sha256:a429bf1494e62175e75458b4eeeaf03a64b99b1986796ce88b3523f5c5ac47c3`，层数 24，即 v0.2.0 的 22 层加 COPY、RUN 两层。
  - 归档 1,227,684,709 字节。
- 镜像里 dpkg 的记录仍是 Eclipse 的 .deb，所以 `dpkg --verify zenoh-bridge-ros2dds` 会报这个文件改过。这是预期的。

## 上游

上游修好以后换回官方版本，再重跑 route 核对和 M4 一轮。要不要向 eclipse-zenoh/zenoh-plugin-ros2dds 报 issue 或提 PR，属于对外动作，要先问负责人（PLAN §7 Q20）。
