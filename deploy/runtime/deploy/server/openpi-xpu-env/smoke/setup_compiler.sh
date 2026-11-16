#!/usr/bin/env bash
# 阶段 1 冒烟：triton-xpu（torch.compile/inductor）在运行时要现编 launcher（main.cpp），需要：
#   - C/C++ 编译器（gcc/g++，最小 ubuntu 镜像里没有）；
#   - Level Zero 头文件 level_zero/ze_api.h（libze-dev，版本与已装的 libze1 一致）；
#   - ocloc（intel-ocloc，把 SPIR-V 编成 zebin；版本与已装的 compute-runtime/IGC 一致）。
#   用法（容器内，root，已跑过 setup_umd.sh，apt 源已切到镜像站）：setup_compiler.sh ppa|host
#     ppa : libze-dev、intel-ocloc 从 kobuk PPA 装，.deb 追加存到 /work/umd-ppa/（并追加 sha256）
#     host: 用宿主离线安装包里的同版本 deb（/work/umd-host/{libze-dev,intel-ocloc}_*.deb）
set -euxo pipefail
VARIANT="${1:?usage: setup_compiler.sh ppa|host}"
export DEBIAN_FRONTEND=noninteractive
APT="apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60"
${APT} update
${APT} install -y --no-install-recommends gcc g++ libc6-dev
gcc --version | head -n 1

case "${VARIANT}" in
  ppa)
    rm -f /var/cache/apt/archives/*.deb
    # 同 setup_umd.sh：先 --download-only 留档（docker-clean 会在 dpkg 运行后删缓存），再安装
    ${APT} install -y --no-install-recommends --download-only libze-dev intel-ocloc
    for p in libze-dev intel-ocloc; do
      if ! ls /work/umd-ppa/${p}_*.deb >/dev/null 2>&1; then
        cp -v /var/cache/apt/archives/${p}_*.deb /work/umd-ppa/
        (cd /work/umd-ppa && sha256sum ./${p}_*.deb >> SHA256SUMS)
      fi
    done
    ${APT} install -y --no-install-recommends libze-dev intel-ocloc
    ;;
  host)
    dpkg -i /work/umd-host/libze-dev_*.deb /work/umd-host/intel-ocloc_*.deb
    ;;
  *) echo "unknown variant ${VARIANT}" >&2; exit 2 ;;
esac

dpkg-query -W -f='${Package} ${Version}\n' libze1 libze-dev intel-ocloc libigc2
ls -l /usr/include/level_zero/ze_api.h
command -v ocloc
