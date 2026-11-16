#!/usr/bin/env bash
# openpi-xpu-env 容器入口：把宿主机挂载进来的代码和镜像内的环境对接，然后执行用户命令。
set -euo pipefail

APP_DIR="${OPENPI_APP_DIR:-/app}"
META_DIR=/opt/openpi-xpu-env
# 显式使用镜像内 venv 的解释器，避免用户覆盖 PATH 后找错 python。
VENV_PY="${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python"
log() { echo "[openpi-xpu-env] $*" >&2; }

# 0) 以任意 UID 运行（docker run --user，容器内无对应 passwd 条目）时 HOME 通常为 "/" 且不可写，
#    HF / wandb / matplotlib 等会把缓存与配置写到 $HOME 下而失败；此时回退到 /tmp。
if [ ! -w "${HOME:-/}" ]; then
    log "HOME=${HOME:-<unset>} is not writable for uid $(id -u); using HOME=/tmp"
    export HOME=/tmp
fi

# 1) transformers_replace：openpi 的 PyTorch 路径要求用仓库内的 modeling 文件覆盖 transformers 包
#    (见 README_openpi.md "PyTorch" 一节)。这里每次启动都从挂载的代码目录同步，所以修改这些补丁
#    文件同样无需重建镜像。设 OPENPI_SKIP_TRANSFORMERS_PATCH=1 可跳过。
TR_SRC="$APP_DIR/src/openpi/models_pytorch/transformers_replace"
if [ -d "$TR_SRC" ] && [ "${OPENPI_SKIP_TRANSFORMERS_PATCH:-0}" != "1" ]; then
    TR_DST="$("$VENV_PY" - <<'PY' 2>/dev/null || true
import importlib.util
spec = importlib.util.find_spec("transformers")
print(spec.submodule_search_locations[0] if spec and spec.submodule_search_locations else "")
PY
)"
    if [ -n "$TR_DST" ] && [ -d "$TR_DST" ]; then
        # 以非 root（docker run --user）启动时 site-packages 不可写：只告警，不阻断用户命令。
        if cp -r "$TR_SRC"/. "$TR_DST"/ 2>/dev/null; then
            log "applied transformers_replace -> $TR_DST"
        else
            log "WARNING: failed to apply transformers_replace -> $TR_DST (not writable?);" \
                "PyTorch models may not work. JAX paths are unaffected."
        fi
    else
        log "WARNING: transformers not found in image venv; transformers_replace not applied."
    fi
fi

# 2) 依赖一致性提示（只提示，不阻断）：挂载代码的 uv.lock 与镜像构建时的 uv.lock 不同，
#    说明代码仓库的依赖已经变化，可能需要用新的 uv.lock 重新构建镜像。
if [ -f "$APP_DIR/uv.lock" ]; then
    if [ -f "$META_DIR/uv.lock" ] && ! cmp -s "$APP_DIR/uv.lock" "$META_DIR/uv.lock"; then
        log "WARNING: $APP_DIR/uv.lock differs from the lock this image was built from" \
            "(openpi commit $(cat "$META_DIR/OPENPI_COMMIT" 2>/dev/null || echo unknown))." \
            "If imports fail, rebuild the image with the new uv.lock."
    fi
elif [ ! -d "$APP_DIR/src/openpi" ]; then
    log "NOTE: no openpi checkout at $APP_DIR; mount it with: docker run -v /path/to/openpi:/app ..."
fi

# 3) GPU 提示（只提示，不阻断）：没有 render 节点说明容器没拿到 Intel GPU，torch.xpu 会不可用。
if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
    log "NOTE: no /dev/dri/renderD* in this container; torch.xpu will be unavailable." \
        "Run with: --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path"
fi

exec "$@"
