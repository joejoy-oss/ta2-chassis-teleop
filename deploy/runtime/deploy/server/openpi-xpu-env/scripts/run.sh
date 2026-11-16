#!/usr/bin/env bash
# 以「代码在宿主机、环境在镜像」的方式启动容器。默认镜像 openpi-xpu-env（Intel GPU 推理）；
# 设 IMAGE=openpi-env 也能启动 CUDA 训练镜像 openpi-env（此时按 CUDA 方式传 --gpus）。
#   用法: scripts/run.sh [命令...]      不带命令则进入 bash
#   例:   scripts/run.sh python -c "import torch; print(torch.xpu.device_count(), torch.xpu.get_device_name(0))"
#         # 在 Intel GPU 上服务一个 PyTorch checkpoint（checkpoint 目录用 DOCKER_RUN_EXTRA 挂进来）：
#         DOCKER_RUN_EXTRA="-p 8000:8000 -v /data/ckpt/pi05_teleavatar_v2_pytorch:/ckpt:ro" \
#             scripts/run.sh python scripts/serve_policy.py --env TELEAVATAR --warmup \
#             policy:checkpoint --policy.config pi05_teleavatar_v2 --policy.dir /ckpt \
#             --policy.pytorch-device xpu --policy.pytorch-compile-mode default
#   环境变量:
#     XPU               auto（默认）| 1 | 0。1：按 Intel GPU 启动容器——--device /dev/dri 并挂 /dev/dri/by-path，
#                       --group-add render 节点的属组（以 RUN_AS_USER=1 非 root 运行时才打得开 renderD*），
#                       设 ZE_AFFINITY_MASK，挂 torch.compile 缓存目录。auto：镜像名含 "xpu" 时为 1。
#     ZE_AFFINITY_MASK  XPU 时选卡 (默认 0，即第 0 张卡；多卡机器上可设 1 等)
#     XPU_CACHE_DIR     XPU 时 torch.compile 的缓存（inductor、triton）在宿主机上的目录
#                       (默认 ../xpu-cache，相对本仓库；不存在时自动创建) -> 挂到 /xpu-cache。
#                       不挂的话每次启动都要冷编译（全模型约 1 分钟以上）；换 torch / 驱动版本后可整个删掉重建。
#     OPENPI_DIR        宿主机 openpi 代码目录 (默认 ../openpi，即与本仓库放在同一级目录) -> 挂到 /app（读写）；
#                       目录不存在时报错退出
#     OUTPUT_DIR        训练输出目录           (默认 ../outputs，相对本仓库；不存在时自动创建) -> 挂到 /outputs（读写）
#                       DOCKER_RUN_EXTRA 里已用 -v/--volume 把目录挂到 /outputs（X:/outputs、X:/outputs/、X:/outputs:ro 等）
#                       时不再挂 OUTPUT_DIR，并在 stderr 提示一行（兼容旧写法，docker 不允许重复挂载点）。
#                       root 运行写过的输出目录归 root，之后改用 RUN_AS_USER=1 前先 chown，或另设 OUTPUT_DIR。
#     OPENPI_DATA_HOME  openpi 资产/缓存目录   (默认 ~/.cache/openpi)        -> 挂到 /openpi_assets
#     HF_HOME           HuggingFace 缓存       (默认 ~/.cache/huggingface)   -> 挂到 /hf_home（容器内 HF_HOME=/hf_home）
#     IMAGE / TAG       镜像名与 tag
#     GPUS              --gpus 参数，CUDA 用 (默认：XPU 时 none，否则 all；无 GPU 机器设为 none)
#     XLA_PYTHON_CLIENT_MEM_FRACTION  JAX 预分配显存占比，透传进容器 (默认 0.9，与 openpi README 一致)
#     SHM_SIZE          共享内存 (默认 8g，dataloader 多进程需要)
#     RUN_AS_USER=1     以当前宿主机用户 (--user $(id -u):$(id -g)) 运行，并设 HOME=/tmp：
#                       容器写到挂载目录的文件（checkpoints 等）归当前用户而不是 root。
#                       代价：镜像内 site-packages 属 root，transformers_replace 补丁打不上，
#                       PyTorch 路径的模型不可用（JAX 路径不受影响）。默认 0（root 运行）。
#     PROXY=1           使用 host 网络并透传宿主机代理变量（HTTPS_PROXY/HTTP_PROXY/NO_PROXY 及小写形式），
#                       供需要联网的命令（下载 checkpoint / GCS 资产、联网测试）使用。默认 0。
#     DOCKER_RUN_EXTRA  追加给 docker run 的额外参数。注意：按空白切分，不支持带空格/引号的参数值；
#                       需要带空格的参数请直接用 docker run。
#     WANDB_API_KEY / OPENPI_SKIP_TRANSFORMERS_PATCH  宿主机设置了就透传进容器
#                       （OPENPI_SKIP_TRANSFORMERS_PATCH=1：入口脚本跳过 transformers_replace 补丁）。
#   始终设置 PYTHONDONTWRITEBYTECODE=1（镜像 ENV 已有，这里双保险），避免往挂载的代码目录写 __pycache__。
#   默认设置 WANDB_DIR=/outputs：wandb 的本地文件写到输出目录，而不是 /app/wandb（代码目录）；
#   可用 DOCKER_RUN_EXTRA="-e WANDB_DIR=..." 覆盖（DOCKER_RUN_EXTRA 排在后面，同名 -e 以后者为准）。
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-openpi-xpu-env}"
TAG="${TAG:-$(cat "${here}/VERSION")}"
if [ -n "${OPENPI_DIR:-}" ]; then openpi_src="${OPENPI_DIR}"; else openpi_src="${here}/../openpi"; fi
if [ ! -d "${openpi_src}" ]; then
    echo "[run] ERROR: openpi code directory not found: ${openpi_src}" >&2
    if [ -z "${OPENPI_DIR:-}" ]; then echo "[run]        (OPENPI_DIR is not set; the default is ../openpi next to the openpi-xpu-env repository)" >&2; fi
    echo "[run]        set OPENPI_DIR=/path/to/openpi" >&2
    exit 1
fi
OPENPI_DIR="$(cd "${openpi_src}" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${here}/../outputs}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
XPU="${XPU:-auto}"
if [ "${XPU}" = "auto" ]; then case "${IMAGE}" in *xpu*) XPU=1 ;; *) XPU=0 ;; esac; fi
if [ "${XPU}" = "1" ]; then GPUS="${GPUS:-none}"; else GPUS="${GPUS:-all}"; fi
SHM_SIZE="${SHM_SIZE:-8g}"
RUN_AS_USER="${RUN_AS_USER:-0}"
PROXY="${PROXY:-0}"
mkdir -p "${OPENPI_DATA_HOME}" "${HF_HOME}"

args=( --rm --init --shm-size="${SHM_SIZE}" -w /app
       -v "${OPENPI_DIR}:/app"
       -v "${OPENPI_DATA_HOME}:/openpi_assets"
       -v "${HF_HOME}:/hf_home" -e HF_HOME=/hf_home
       -e PYTHONDONTWRITEBYTECODE=1 -e WANDB_DIR=/outputs
       -e "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" )
# shellcheck disable=SC2206  # 有意按空白切分，见上方 DOCKER_RUN_EXTRA 说明
extra=( ${DOCKER_RUN_EXTRA:-} )
# 输出目录：DOCKER_RUN_EXTRA 已用 -v/--volume 挂载到 /outputs 时跳过（否则 docker 报 Duplicate mount point）。
# 只看 -v/--volume 的参数值，目标路径为 /outputs 或 /outputs/（可带 :ro 等选项）才算；-e X=…:/outputs 之类不算。
extra_outputs=""
for ((i = 0; i < ${#extra[@]}; i++)); do
    a="${extra[i]}"; v=""
    case "${a}" in
        -v | --volume) v="${extra[i+1]:-}" ;;
        --volume=*)    v="${a#--volume=}" ;;
        -v=*)          v="${a#-v=}" ;;
        -v?*)          v="${a#-v}" ;;
    esac
    case "${v}" in
        *:/outputs | *:/outputs/ | *:/outputs:* | *:/outputs/:*) extra_outputs="${v}" ;;
    esac
done
if [ -n "${extra_outputs}" ]; then
    echo "[run] NOTE: DOCKER_RUN_EXTRA mounts /outputs (${extra_outputs}); OUTPUT_DIR (${OUTPUT_DIR}) is not mounted" >&2
else
    mkdir -p "${OUTPUT_DIR}"; args+=( -v "$(cd "${OUTPUT_DIR}" && pwd):/outputs" )
fi
if [ "${GPUS}" != "none" ]; then args+=( --gpus "${GPUS}" ); fi
if [ "${XPU}" = "1" ]; then
    XPU_CACHE_DIR="${XPU_CACHE_DIR:-${here}/../xpu-cache}"
    mkdir -p "${XPU_CACHE_DIR}"
    args+=( --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path
            -e "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}"
            -v "$(cd "${XPU_CACHE_DIR}" && pwd):/xpu-cache"
            -e TORCHINDUCTOR_CACHE_DIR=/xpu-cache/inductor -e TRITON_CACHE_DIR=/xpu-cache/triton )
    for node in /dev/dri/renderD*; do
        if [ -e "${node}" ]; then args+=( --group-add "$(stat -c %g "${node}")" ); break; fi
    done
fi
if [ "${RUN_AS_USER}" = "1" ]; then args+=( --user "$(id -u):$(id -g)" -e HOME=/tmp ); fi
if [ "${PROXY}" = "1" ]; then
    https_p="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
    http_p="${HTTP_PROXY:-${http_proxy:-${https_p}}}"
    no_p="${NO_PROXY:-${no_proxy:-localhost,127.0.0.1}}"
    if [ -z "${https_p}" ]; then
        echo "[run] PROXY=1 but none of HTTPS_PROXY/https_proxy/HTTP_PROXY/http_proxy is set" >&2; exit 1
    fi
    args+=( --network=host
            -e "HTTPS_PROXY=${https_p}" -e "https_proxy=${https_p}"
            -e "HTTP_PROXY=${http_p}"   -e "http_proxy=${http_p}"
            -e "NO_PROXY=${no_p}"       -e "no_proxy=${no_p}" )
fi
if [ -n "${WANDB_API_KEY:-}" ]; then args+=( -e WANDB_API_KEY ); fi
if [ -n "${OPENPI_SKIP_TRANSFORMERS_PATCH:-}" ]; then args+=( -e OPENPI_SKIP_TRANSFORMERS_PATCH ); fi
if [ -t 0 ]; then args+=( -it ); fi
if [ ${#extra[@]} -gt 0 ]; then args+=( "${extra[@]}" ); fi

exec docker run "${args[@]}" "${IMAGE}:${TAG}" "$@"
