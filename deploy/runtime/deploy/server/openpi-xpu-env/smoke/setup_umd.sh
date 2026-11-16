#!/usr/bin/env bash
# 阶段 1 冒烟：在 ubuntu:24.04 容器内安装 Intel GPU 用户态驱动（UMD）。
#   用法（容器内，root）：setup_umd.sh ppa|host
#     ppa : 从 kobuk-team/intel-graphics PPA 安装当前版本，并把实际下载的 .deb 存到 /work/umd-ppa/（附 sha256）
#     host: 安装宿主机同款 26.09 UMD（/work/umd-host/*.deb，取自宿主离线安装包），缺的依赖从 Ubuntu 源补
#   依赖环境变量：http_proxy/https_proxy（经反向隧道的代理）
set -euxo pipefail
VARIANT="${1:?usage: setup_umd.sh ppa|host}"
export DEBIAN_FRONTEND=noninteractive
APT="apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60"
# 经隧道代理访问 archive.ubuntu.com 超时（10-29 实测 20s 无响应），国内镜像 <0.5s；只改容器内的源
APT_MIRROR="${APT_MIRROR:-http://mirrors.ustc.edu.cn/ubuntu}"
sed -i "s#http://archive.ubuntu.com/ubuntu#${APT_MIRROR}#g; s#http://security.ubuntu.com/ubuntu#${APT_MIRROR}#g" \
  /etc/apt/sources.list.d/ubuntu.sources
command grep -E "^URIs" /etc/apt/sources.list.d/ubuntu.sources

${APT} update
${APT} install -y --no-install-recommends ca-certificates curl gnupg git xz-utils

case "${VARIANT}" in
  ppa)
    install -d -m 0755 /etc/apt/keyrings
    # kobuk-team/intel-graphics 签名密钥（指纹经 Launchpad API 核对：0C0E6AF955CE463C03FC51574D098D70AFBE5E1F）
    curl -fsSL --retry 5 "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x0C0E6AF955CE463C03FC51574D098D70AFBE5E1F" \
      | gpg --dearmor -o /etc/apt/keyrings/kobuk-intel-graphics.gpg
    gpg --show-keys --with-fingerprint /etc/apt/keyrings/kobuk-intel-graphics.gpg
    echo "deb [signed-by=/etc/apt/keyrings/kobuk-intel-graphics.gpg] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
      > /etc/apt/sources.list.d/kobuk-intel-graphics.list
    ${APT} update
    apt-cache policy libze-intel-gpu1 libze1 intel-opencl-icd libigc2 libigdgmm12
    rm -f /var/cache/apt/archives/*.deb
    # ubuntu 镜像的 /etc/apt/apt.conf.d/docker-clean 会在每次 dpkg 运行后删掉缓存的 .deb，
    # 所以先 --download-only（不触发 dpkg）拷出留档，再正式安装
    ${APT} install -y --no-install-recommends --download-only libze1 libze-intel-gpu1 intel-opencl-icd libigdgmm12 clinfo
    mkdir -p /work/umd-ppa
    cp -v /var/cache/apt/archives/*.deb /work/umd-ppa/
    (cd /work/umd-ppa && sha256sum ./*.deb > SHA256SUMS)
    ${APT} install -y --no-install-recommends libze1 libze-intel-gpu1 intel-opencl-icd libigdgmm12 clinfo
    ;;
  host)
    ls -l /work/umd-host/*.deb
    # dpkg 先装，依赖缺口由 apt 从 Ubuntu 源补齐（不引入 PPA）
    dpkg -i /work/umd-host/libze1_*.deb /work/umd-host/libigdgmm12_*.deb /work/umd-host/libigc2_*.deb \
            /work/umd-host/libigdfcl2_*.deb /work/umd-host/libze-intel-gpu1_*.deb \
            /work/umd-host/intel-opencl-icd_*.deb || true
    ${APT} install -y -f --no-install-recommends
    ${APT} install -y --no-install-recommends clinfo
    ;;
  *) echo "unknown variant ${VARIANT}" >&2; exit 2 ;;
esac

echo "=== installed UMD versions (${VARIANT})"
# 有模式匹配不到（如未安装的 intel-ocloc）时 dpkg-query 返回 1，在 pipefail 下会让整个脚本失败
dpkg-query -W -f='${Package} ${Version}\n' 'libze*' 'intel-opencl-icd' 'libigc*' 'libigdfcl*' 'libigdgmm*' 'intel-ocloc' 2>/dev/null | sort || true
echo "=== clinfo -l"
clinfo -l || true
