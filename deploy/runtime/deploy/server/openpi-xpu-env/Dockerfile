# =============================================================================
# openpi-xpu-env: Intel GPU（Arc Pro B70 / Xe2）推理镜像，只包含「环境」——Python + uv + OpenPI 依赖
# + Intel GPU 用户态驱动，不包含项目代码。代码与 openpi-env 一样通过 `docker run -v <openpi>:/app` 挂载。
#
# 阶段 1 起草（10-29），阶段 5（10-31）在 B70 上用本文件构建出 v0.1.0 并通过验收：镜像里没有代理残留、
# 没有项目代码，`uv sync --frozen --dry-run` 无变更，冒烟全过（任务记录 reports/logs/phase5/04、07）。
#
# 与 openpi-env（CUDA 训练镜像）的差别
#   * 基础镜像 ubuntu:24.04，与 B70 宿主机一致（Intel GPU 用户态驱动只发布 noble 包）。
#   * Intel GPU 用户态驱动（UMD）：Level Zero loader、compute-runtime、IGC、gmmlib、OpenCL ICD，
#     与宿主机离线安装包 26.18.8.2 的 graphics 目录同版本（compute-runtime 26.09.37435.12、IGC 2.30.3、
#     Level Zero 1.28.0）。阶段 1 的 A/B 对照中 PPA 新版（26.35）同样全部通过，按 PLAN 选与宿主一致的这组。
#     .deb 由构建方放到 deps/umd-host/（来源：宿主 /root/multi-arc-bmg-offline-installer-26.18.8.2-combo/
#     ubuntu-24.04.4-server/graphics/），构建时按 deps/umd-host/SHA256SUMS 校验，只以 bind mount 进入构建。
#   * torch.compile（inductor + triton-xpu）运行期要现编 C++ launcher、用 Level Zero 头文件、调用 ocloc，
#     因此装 build-essential、libze-dev 与 intel-ocloc。阶段 1 冒烟依次因为缺 C 编译器、ze_api.h、ocloc 而失败；
#     冒烟里补装的是 gcc/g++/libc6-dev，build-essential 在这几样之外还多装了 make、dpkg-dev。
#   * 依赖：`uv sync --no-default-groups --group xpu --group dev`。torch/torchvision/triton-xpu/torchcodec
#     来自 pytorch-xpu 索引；jax[cuda12] 照旧安装（D7），运行期 JAX_PLATFORMS=cpu。
#   * 不设 NVIDIA_*。GPU 用 `--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path` 进入容器，选卡用
#     ZE_AFFINITY_MASK，由运行脚本设置，不写死在镜像里。
#
# 在 B70 上构建
#   * dockerd 配的代理不可达：基础镜像与 uv 镜像要先在能联网的机器上 pull，再 `docker save | docker load`
#     传过去（FROM 都钉了 digest，本地已有时不会再访问 registry）。
#   * B70 没有 buildx 插件，用宿主已有的 docker-compose-v2 驱动 BuildKit：scripts/build.sh 调
#     `docker compose build`（compose 文件 docker-compose.yml）。RUN 期间的下载走 host 网络和代理；
#     代理与 APT_MIRROR 都以 build-arg 传入，构建完要核对镜像里没有残留代理设置。
#   * 经代理访问 archive.ubuntu.com 超时：传 `--build-arg APT_MIRROR=http://mirrors.ustc.edu.cn/ubuntu`。
#     镜像源只在构建期生效，结束前恢复原始 sources，不残留到镜像。
#
# 阶段 5 定下的几项：
#   * compile 缓存不放进镜像：scripts/run.sh 把宿主机目录挂到 /xpu-cache，并设 TORCHINDUCTOR_CACHE_DIR、
#     TRITON_CACHE_DIR。不设 SYCL_CACHE_PERSISTENT：torch 的 XPU kernel 是为本卡预编译好的（阶段 1 H1），
#     运行期现编的只有 triton kernel，已由 TRITON_CACHE_DIR 缓存。
#   * 保留 dev 组（pytest 等），镜像里可以直接跑测试。
#   * 体积：`docker images` 显示 22.1GB（B70，containerd 镜像存储），其中 venv 14G。
# =============================================================================
ARG BASE_IMAGE=ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ARG UV_VERSION=0.13.5
# 与 UV_VERSION 配套的镜像 index digest；改 UV_VERSION 时必须同步修改（构建末尾会校验 uv 版本）。
ARG UV_IMAGE_DIGEST=sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff

FROM ghcr.io/astral-sh/uv:${UV_VERSION}@${UV_IMAGE_DIGEST} AS uv-bin

FROM ${BASE_IMAGE}

COPY --from=uv-bin /uv /uvx /bin/

# 仅构建期生效（ARG），不残留到运行期 ENV。
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=
# 与 deps/umd-host/ 里的 .deb 一致；装完后逐个核对，标签也用这几个值。
ARG COMPUTE_RUNTIME_VERSION=26.09.37435.12-1~24.04~ppa1
ARG IGC_VERSION=2.30.3-1249~24.04
ARG LEVEL_ZERO_VERSION=1.28.0-1~24.04~ppa1
ARG GMMLIB_VERSION=22.9.0-1~24.04~ppa1

# 1) 系统包 + Intel GPU 用户态驱动，同一层装完：
#    git/git-lfs: lerobot 是 git 依赖；build-essential: triton-xpu 运行期现编 launcher，个别 sdist 也要编译；
#    ffmpeg: torchcodec（dlopen FFmpeg 库）与 lerobot 视频解码；libgl1/libglib2.0-0 等: opencv-python 运行时依赖；
#    clinfo: 排查 GPU 用。
#    本地 .deb 直接交给 apt-get install，由 apt 从 Ubuntu 源补齐它们的依赖（如 ocl-icd-libopencl1）。
#    重试循环沿用 openpi-env：代理偶发 502，且 Acquire::Retries 对 502 不生效；docker-clean 会在 dpkg 运行后
#    删掉已下载的 .deb，所以先 --download-only 下全，再离线安装。
RUN --mount=type=bind,source=deps/umd-host,target=/tmp/umd-host \
    set -eu; \
    (cd /tmp/umd-host && sha256sum -c SHA256SUMS); \
    src=/etc/apt/sources.list.d/ubuntu.sources; \
    cp "$src" /tmp/ubuntu.sources.orig; \
    if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s#http://archive.ubuntu.com/ubuntu#${APT_MIRROR}#g; s#http://security.ubuntu.com/ubuntu#${APT_MIRROR}#g" "$src"; \
    fi; \
    pkgs="ca-certificates curl git git-lfs build-essential ffmpeg clinfo \
          libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
          /tmp/umd-host/libze1_${LEVEL_ZERO_VERSION}_amd64.deb \
          /tmp/umd-host/libze-dev_${LEVEL_ZERO_VERSION}_amd64.deb \
          /tmp/umd-host/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION}_amd64.deb \
          /tmp/umd-host/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION}_amd64.deb \
          /tmp/umd-host/intel-ocloc_${COMPUTE_RUNTIME_VERSION}_amd64.deb \
          /tmp/umd-host/libigc2_${IGC_VERSION}_amd64.deb \
          /tmp/umd-host/libigdfcl2_${IGC_VERSION}_amd64.deb \
          /tmp/umd-host/libigdgmm12_${GMMLIB_VERSION}_amd64.deb"; \
    updated=0; \
    for i in 1 2 3 4 5; do \
        if [ "$updated" = 0 ] && apt-get -o Acquire::Retries=5 update; then updated=1; fi; \
        if [ "$updated" = 1 ] && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --download-only $pkgs; then break; fi; \
        [ "$i" -lt 5 ] || exit 1; \
        echo "apt attempt $i failed (updated=$updated), retrying in 10s ..." >&2; sleep 10; \
    done; \
    apt-get install -y --no-install-recommends $pkgs; \
    for p in "libze1 ${LEVEL_ZERO_VERSION}" "libze-dev ${LEVEL_ZERO_VERSION}" \
             "libze-intel-gpu1 ${COMPUTE_RUNTIME_VERSION}" "intel-opencl-icd ${COMPUTE_RUNTIME_VERSION}" \
             "intel-ocloc ${COMPUTE_RUNTIME_VERSION}" "libigc2 ${IGC_VERSION}" "libigdfcl2 ${IGC_VERSION}" \
             "libigdgmm12 ${GMMLIB_VERSION}"; do \
        actual="$(dpkg-query -W -f='${Package} ${Version}' "${p% *}")"; \
        if [ "$actual" != "$p" ]; then echo "UMD version mismatch: got '$actual', want '$p'" >&2; exit 1; fi; \
    done; \
    test -f /usr/include/level_zero/ze_api.h; \
    command -v ocloc; \
    cp /tmp/ubuntu.sources.orig "$src"; \
    rm -rf /var/lib/apt/lists/* /tmp/ubuntu.sources.orig

# uv 相关（构建期 + 运行期）：Python 与 venv 都放在 /opt 下，避免被 /app 挂载覆盖。
# UV_PYTHON_PREFERENCE=only-managed：只用 uv 托管的 Python，不会误用系统 python3（官方指南推荐）。
# UV_LINK_MODE / UV_COMPILE_BYTECODE / GIT_LFS_SKIP_SMUDGE 主要服务于构建期，留在运行期无害。
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    GIT_LFS_SKIP_SMUDGE=1

# 放在 apt 层之后声明：ARG 会进入其后所有 RUN 的环境并参与缓存键，改 Python 版本不应让 apt 层重跑。
ARG PYTHON_VERSION=3.11.17

# 2) 固定版本的 Python（uv 托管的 python-build-standalone）
#    uv 运行时会在 /tmp 留下 0 字节的进程锁文件，在同一层内删除，避免进入镜像。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install --no-bin ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} ${UV_PROJECT_ENVIRONMENT} && \
    rm -f /tmp/uv-*.lock

WORKDIR /app

# 3) 只装依赖，不装项目本身。依赖文件只在构建期以 bind mount 方式提供，不进入镜像层。
#    --locked：断言 uv.lock 与 pyproject.toml 一致。--no-default-groups --group xpu：不装默认的 cuda 组
#    （两组在 pyproject 里声明为互斥）；dev 组带 pytest，镜像内可直接跑测试。
#    torch xpu wheel 很大，经隧道代理下载时把 HTTP 超时放宽到 300 s（只在这一步生效）。
#    workspace 成员（packages/*）逐个挂载，目前只有 openpi-client。
ARG UV_GROUPS="--group xpu --group dev"
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=deps/uv.lock,target=uv.lock \
    --mount=type=bind,source=deps/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=deps/.python-version,target=.python-version \
    --mount=type=bind,source=deps/packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    UV_HTTP_TIMEOUT=300 uv sync --locked --no-default-groups ${UV_GROUPS} --no-install-project --no-install-workspace && \
    rm -f /tmp/uv-*.lock

# 构建元数据参数在依赖安装之后才声明：ARG 会隐式进入其后所有 RUN 的环境并参与缓存键，
# 若在前面声明，仅改 VERSION / OPENPI_COMMIT 也会让 apt 与 uv sync 层全部失效重建。
ARG BASE_IMAGE
ARG UV_VERSION
ARG OPENPI_COMMIT=unknown
ARG IMAGE_VERSION=dev
ARG TORCH_VERSION=2.12.1+xpu

# 4) 记录构建所用的依赖快照与驱动版本（溯源 / 启动时一致性提示），并校验：
#    uv 版本与 UV_VERSION 一致（防止只改 tag 不改 digest）；venv 里的 torch 就是 TORCH_VERSION。
# 依赖快照连同 .python-version 和 workspace 成员的 pyproject 一起放进元数据目录，镜像里可直接核对 venv 与锁一致：
#   cd /opt/openpi-xpu-env && uv sync --frozen --dry-run --no-default-groups --group xpu --group dev \
#       --no-install-project --no-install-workspace      （应输出 "Would make no changes"）
COPY deps/pyproject.toml deps/uv.lock deps/.python-version /opt/openpi-xpu-env/
COPY deps/packages/openpi-client/pyproject.toml /opt/openpi-xpu-env/packages/openpi-client/pyproject.toml
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/openpi-xpu-entrypoint.sh
RUN set -eu; \
    actual_uv="$(uv --version | cut -d' ' -f2)"; \
    if [ "${actual_uv}" != "${UV_VERSION}" ]; then \
        echo "uv version mismatch: binary ${actual_uv} != UV_VERSION ${UV_VERSION} (check UV_IMAGE_DIGEST)" >&2; exit 1; \
    fi; \
    actual_torch="$(/opt/venv/bin/python -c 'import importlib.metadata as m; print(m.version("torch"))')"; \
    if [ "${actual_torch}" != "${TORCH_VERSION}" ]; then \
        echo "torch version mismatch: venv ${actual_torch} != TORCH_VERSION ${TORCH_VERSION}" >&2; exit 1; \
    fi; \
    echo "${OPENPI_COMMIT}" > /opt/openpi-xpu-env/OPENPI_COMMIT; \
    printf 'uv=%s python=%s torch=%s base=%s\n' "${UV_VERSION}" "${PYTHON_VERSION}" "${TORCH_VERSION}" "${BASE_IMAGE}" \
        > /opt/openpi-xpu-env/TOOLCHAIN; \
    dpkg-query -W -f='${Package} ${Version}\n' 'libze*' 'intel-opencl-icd' 'intel-ocloc' 'libigc*' 'libigdfcl*' 'libigdgmm*' \
        | sort > /opt/openpi-xpu-env/UMD_VERSIONS

# 运行期设置（必须放在 uv sync 之后：UV_CACHE_DIR 若提前设置会改变构建期 cache mount 的路径）：
#   PATH/VIRTUAL_ENV        -> 直接 `python` 即镜像内环境；`uv run` 亦复用 /opt/venv
#   PYTHONPATH              -> import 挂载在 /app 的 openpi / openpi_client 源码
#   PYTHONDONTWRITEBYTECODE -> 不往挂载的代码目录写 __pycache__（以 root 运行时会留下 root 属主文件）
#   UV_CACHE_DIR            -> 放在任何用户都可写的 /tmp，非 root（--user）运行 `uv run` 也能用
#   UV_NO_SYNC              -> `uv run` 不再尝试同步依赖/联网
#   UV_PYTHON_DOWNLOADS     -> 运行期绝不下载 Python
#   JAX_PLATFORMS=cpu       -> jax[cuda12] 在没有 NVIDIA GPU 的机器上只用 CPU 后端（serve_policy 的导入链需要 jax）
ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/app/src:/app/packages/openpi-client/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_SYNC=1 \
    UV_PYTHON_DOWNLOADS=never \
    OPENPI_DATA_HOME=/openpi_assets \
    IS_DOCKER=true \
    JAX_PLATFORMS=cpu

LABEL org.opencontainers.image.title="openpi-xpu-env" \
      org.opencontainers.image.description="Environment-only image (Python+uv+deps+Intel GPU UMD) for openpi PyTorch inference on Intel XPU; mount code at /app" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      openpi.commit="${OPENPI_COMMIT}" \
      openpi.python="${PYTHON_VERSION}" \
      openpi.uv="${UV_VERSION}" \
      openpi.torch="${TORCH_VERSION}" \
      intel.compute-runtime="${COMPUTE_RUNTIME_VERSION}" \
      intel.igc="${IGC_VERSION}" \
      intel.level-zero="${LEVEL_ZERO_VERSION}" \
      intel.gmmlib="${GMMLIB_VERSION}"

ENTRYPOINT ["/usr/local/bin/openpi-xpu-entrypoint.sh"]
CMD ["/bin/bash"]
