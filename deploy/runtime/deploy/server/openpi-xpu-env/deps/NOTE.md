# deps/ 的来源说明（阶段 1，10-29）

- `pyproject.toml`、`uv.lock`：以 fork 提交 `5631db9` 的依赖文件为底（fork 的这两个文件从 `5631db9` 到
  `22a391b` 没有改过），**手工加了 `cuda` / `xpu` 两个依赖组**（PR #1009 风格，见 `reports/phase1_executor.md` §5），
  **不回写 fork**：负责人 10-31 21:58 选了 av 方案 (c)，xpu 组只留在这里，D6 回写到此结束（PROGRESS 决策记录）。
- `OPENPI_COMMIT` 里的 `22a391b…` 只表示同步来源；当前的 `pyproject.toml` / `uv.lock` 不对应任何一个 openpi 提交。
- **依赖在这里手工更新**：改 `pyproject.toml` 后，按阶段 1 的方法重新生成 `uv.lock`（uv 0.13.5，在 `deps/` 目录里联网执行 `uv lock`；
  阶段 1 的命令见 `reports/logs/phase1/06_deps.txt` 第 34–38 行，用的是 openpi-env 镜像里的 uv 和本机代理），再改 `VERSION`、重建镜像。
  `scripts/sync_deps.sh` 会用 fork 的文件覆盖 `pyproject.toml` / `uv.lock` / `OPENPI_COMMIT`，xpu 组就丢了；
  它现在会检查来源有没有 xpu 依赖组，没有就拒绝（10-31 23:57 加，阶段 5 审查 S3），所以按 (c) 对 fork 永远不会执行。
  备份：`reports/logs/phase1/fix1/deps_snapshot/`。
- 镜像入口脚本的"uv.lock differs"提示因此会一直出现：挂载代码的锁不带 xpu 组，镜像的锁带。
- `umd-host/`、`umd-ppa/`：两组 UMD 的 `SHA256SUMS`；`.deb` 不纳入版本管理。`umd-ppa/` 的 .deb 是 A 组唯一的本地副本，
  不要删；构建只用 `umd-host/`（`.dockerignore` 已排除 `umd-ppa/`）。
