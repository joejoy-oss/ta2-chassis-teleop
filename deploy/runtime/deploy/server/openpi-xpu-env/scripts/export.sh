#!/usr/bin/env bash
# 导出镜像为归档文件（供 Nextcloud 分发）。
#   用法: scripts/export.sh            -> dist/<IMAGE>_<TAG>.tar.gz (+ .sha256)
#         FORMAT=tar scripts/export.sh -> dist/<IMAGE>_<TAG>.tar     (不压缩)
#   IMAGE 中的 "/" 在文件名里替换为 "_"（如 registry/openpi-xpu-env -> registry_openpi-xpu-env）。
#   归档同时包含 IMAGE:TAG 与 IMAGE:latest 两个 tag（仅当 latest 与 TAG 指向同一镜像时；
#   否则只导出 IMAGE:TAG 并给出提示），docker load 后两个 tag 都会恢复。
# docker load 对 .tar 和 .tar.gz 都可直接读取。
# 先写到 <归档>.partial，成功后再改名为最终文件名并生成 .sha256：中途失败（磁盘满、Ctrl-C 等）
# 不会留下看似完整的归档，临时文件由 trap 删除；同名旧归档在新归档成功之前保持不变。
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-openpi-xpu-env}"
TAG="${TAG:-$(cat VERSION)}"
FORMAT="${FORMAT:-tar.gz}"

refs=( "${IMAGE}:${TAG}" )
id_tag="$(docker image inspect --format '{{.Id}}' "${IMAGE}:${TAG}")"
if [ "${TAG}" != "latest" ]; then
    id_latest="$(docker image inspect --format '{{.Id}}' "${IMAGE}:latest" 2>/dev/null || true)"
    if [ "${id_latest}" = "${id_tag}" ]; then
        refs+=( "${IMAGE}:latest" )
    else
        echo "[export] NOTE: ${IMAGE}:latest is missing or points to a different image; exporting ${IMAGE}:${TAG} only" >&2
    fi
fi

mkdir -p dist
out="dist/${IMAGE//\//_}_${TAG}.${FORMAT}"
tmp="${out}.partial"
trap 'rm -f "${tmp}" "${out}.sha256.partial"' EXIT
rm -f "${tmp}"
echo "[export] ${refs[*]} (${id_tag}) -> ${out}" >&2
case "${FORMAT}" in
    tar)    docker save "${refs[@]}" -o "${tmp}"
            chmod 0644 "${tmp}" ;;   # docker save -o 生成的文件权限是 0600，改为 0644 便于分发
    tar.gz)
        if command -v pigz >/dev/null 2>&1; then gz=( pigz -p "$(nproc)" ); else gz=( gzip -6 ); fi
        docker save "${refs[@]}" | "${gz[@]}" > "${tmp}" ;;
    *) echo "unknown FORMAT=${FORMAT}" >&2; exit 1 ;;
esac
# 归档完整写出后才改名；sha256 按最终文件名记录，可在 dist/ 下直接 sha256sum -c
sum="$(sha256sum "${tmp}" | cut -d' ' -f1)"
rm -f "${out}.sha256"
mv -f "${tmp}" "${out}"
printf '%s  %s\n' "${sum}" "$(basename "${out}")" > "${out}.sha256.partial"
mv -f "${out}.sha256.partial" "${out}.sha256"
ls -lh "${out}" "${out}.sha256"
