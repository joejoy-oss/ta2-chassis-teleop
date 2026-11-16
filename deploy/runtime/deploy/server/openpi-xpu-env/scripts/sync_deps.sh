#!/usr/bin/env bash
# 从一份 openpi 代码检出同步依赖定义到 deps/，并记录来源 commit。
#   用法: scripts/sync_deps.sh /path/to/openpi
#   流程：先在临时目录完成全部校验，全部通过后才写入 deps/；任一步失败都不改动 deps/。
#     1) 来源必须是 git 检出的根目录（记录 commit 用于溯源；嵌套在别的 git 仓库里的非 git 目录会被拒绝，
#        否则 git 会向上找到外层仓库、记录成外层仓库的 commit）；
#     2) pyproject.toml / uv.lock / .python-version 齐全；
#     3) 按 pyproject.toml 的 [tool.uv.workspace] members 展开 workspace 成员。Dockerfile 的
#        bind mount 是逐个写死的（目前只有 packages/openpi-client），成员集合变化时报错，
#        提示同步修改 Dockerfile 与本脚本的 KNOWN_MEMBERS；
#     4) lock 与 pyproject 一致性检查：用现有镜像内的 uv（与构建同版本）执行 `uv lock --check`。
#        先 --network none 离线检查；离线失败（不一致，或需要拉取 git 依赖元数据才能判断）时，
#        再以 host 网络 + 宿主机代理变量联网复查，以联网结果为准。联网复查失败时区分两种情况：
#        uv 报 "needs to be updated" -> lock 与 pyproject 不一致；其他失败（网络/代理等）-> 无法判断，
#        提示查看上方 uv 输出。两种情况都不改动 deps/。
#        镜像不存在时跳过并提示（构建时 `uv sync --locked` 仍会检查）。SKIP_LOCK_CHECK=1 可显式跳过。
#   依赖变更后：执行本脚本 -> 改 VERSION -> scripts/build.sh 重建镜像。
set -euo pipefail
cd "$(dirname "$0")/.."
src="$(cd "${1:?usage: $0 /path/to/openpi}" && pwd)"
IMAGE="${IMAGE:-openpi-xpu-env}"
TAG="${TAG:-$(cat VERSION)}"
KNOWN_MEMBERS=( packages/openpi-client )
die() { echo "[sync_deps] ERROR: $*" >&2; echo "[sync_deps] deps/ NOT modified." >&2; exit 1; }

# 1) git commit：src 必须是 git 仓库的根目录。git -C 会向上查找仓库，嵌套在别的仓库里的非 git 目录
#    会拿到外层仓库的 HEAD，所以先比较 --show-toplevel 与 src（都取物理路径）。
top="$(git -C "${src}" rev-parse --show-toplevel 2>/dev/null)" || die "${src} is not a git checkout (need a commit to record)"
[ "$(cd "${top}" && pwd -P)" = "$(cd "${src}" && pwd -P)" ] \
    || die "${src} is not the root of a git checkout (it lies inside the git repository ${top}); pass the openpi repository root"
commit="$(git -C "${src}" rev-parse --verify HEAD 2>/dev/null)" || die "cannot resolve HEAD in ${src} (repository without commits?)"

# 2) 固定文件
for f in pyproject.toml uv.lock .python-version; do
    [ -f "${src}/${f}" ] || die "missing ${src}/${f}"
done

# 2b) xpu 依赖组：按负责人 10-31 的决定（av 方案 (c)），xpu 组只在 deps/ 里，不回写 openpi。来源没有 xpu 组时，
#     同步会把 deps/ 里的 xpu 组连同锁一起丢掉，所以拒绝。依赖要在 deps/ 里手工更新，见 deps/NOTE.md。
#     只认 [dependency-groups] 段里行首的 `xpu =`（deps/pyproject.toml 的写法）；别的 TOML 写法（键缩进或带引号、
#     inline table、dotted key）也会被拒绝，提示里说明了。grep 不用 -q：-q 提前退出会让 sed 收到 SIGPIPE，
#     在 pipefail 下把整段判成失败。
sed -n '/^\[dependency-groups\]/,/^\[/p' "${src}/pyproject.toml" | grep -E '^xpu[[:space:]]*=' >/dev/null \
    || die "${src}/pyproject.toml has no xpu dependency group (only a line starting with 'xpu =' under" \
        "[dependency-groups] is recognized); syncing would drop the one in deps/. Update deps/ by hand instead, see deps/NOTE.md"

# 3) workspace 成员（只支持单行 members = [...] 写法，与上游一致；其他写法报错）
ws="$(sed -n '/^\[tool\.uv\.workspace\]/,/^\[/p' "${src}/pyproject.toml" | grep -E '^[[:space:]]*members[[:space:]]*=' || true)"
members=()
if [ -n "${ws}" ]; then
    case "${ws}" in *"]"*) ;; *) die "unsupported multi-line [tool.uv.workspace] members in ${src}/pyproject.toml; update this script" ;; esac
    while IFS= read -r pat; do
        pat="${pat//\"/}"
        for d in "${src}"/${pat}; do
            if [ -f "${d}/pyproject.toml" ]; then members+=( "${d#"${src}"/}" ); fi
        done
    done < <(grep -o '"[^"]*"' <<<"${ws#*=}")
fi
if [ "$(printf '%s\n' "${members[@]}" | sort)" != "$(printf '%s\n' "${KNOWN_MEMBERS[@]}" | sort)" ]; then
    die "workspace members changed: found [${members[*]}], expected [${KNOWN_MEMBERS[*]}]." \
        "Update the bind mounts in Dockerfile and KNOWN_MEMBERS in $0 together."
fi

# 暂存到临时目录
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT
for f in pyproject.toml uv.lock .python-version; do cp "${src}/${f}" "${stage}/${f}"; done
for m in "${members[@]}"; do mkdir -p "${stage}/${m}"; cp "${src}/${m}/pyproject.toml" "${stage}/${m}/pyproject.toml"; done

# 4) lock 一致性
if [ "${SKIP_LOCK_CHECK:-0}" = "1" ]; then
    echo "[sync_deps] SKIP_LOCK_CHECK=1: lock consistency check skipped" >&2
elif docker image inspect "${IMAGE}:${TAG}" >/dev/null 2>&1; then
    lock_check() { docker run --rm "$@" --entrypoint uv -v "${stage}:/w:ro" -w /w "${IMAGE}:${TAG}" lock --check; }
    echo "[sync_deps] uv lock --check (offline, in ${IMAGE}:${TAG})" >&2
    if ! lock_check --network none -e UV_OFFLINE=1; then
        p="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}"
        net=( --network=host )
        if [ -n "${p}" ]; then net+=( -e "HTTPS_PROXY=${p}" -e "HTTP_PROXY=${HTTP_PROXY:-${http_proxy:-${p}}}" ); fi
        echo "[sync_deps] offline check failed; re-checking with network (proxy: ${p:-none}) ..." >&2
        rc=0; out="$(lock_check "${net[@]}" 2>&1)" || rc=$?
        if [ -n "${out}" ]; then printf '%s\n' "${out}" >&2; fi
        if [ "${rc}" -ne 0 ]; then
            if grep -q 'needs to be updated' <<<"${out}"; then
                die "uv.lock is not consistent with pyproject.toml in ${src}; run 'uv lock' there first (or SKIP_LOCK_CHECK=1)"
            fi
            die "online lock check failed (exit ${rc}) without reporting an outdated lock -- most likely a network/proxy" \
                "problem, see the uv output above; consistency could not be determined. Fix the network and retry (or SKIP_LOCK_CHECK=1)"
        fi
    fi
else
    echo "[sync_deps] WARNING: image ${IMAGE}:${TAG} not found; lock check skipped (build will still run uv sync --locked)" >&2
fi

# GIT_OPTIONAL_LOCKS=0：只读查询，不让 git status 顺手刷新并写回来源仓库的 .git/index
dirty="$(GIT_OPTIONAL_LOCKS=0 git -C "${src}" status --porcelain -- pyproject.toml uv.lock .python-version "${members[@]/%//pyproject.toml}")"

# 全部通过：写入 deps/（清掉已不存在的成员目录）
rm -rf deps/packages
mkdir -p deps
for f in pyproject.toml uv.lock .python-version; do cp "${stage}/${f}" "deps/${f}"; done
for m in "${members[@]}"; do mkdir -p "deps/${m}"; cp "${stage}/${m}/pyproject.toml" "deps/${m}/pyproject.toml"; done
echo "${commit}" > deps/OPENPI_COMMIT
echo "[sync_deps] deps/ <- ${src} @ ${commit}"
if [ -n "${dirty}" ]; then
    echo "[sync_deps] WARNING: dependency files in ${src} have uncommitted changes (OPENPI_COMMIT does not fully describe them):"
    echo "${dirty}"
fi
git status --short deps/ 2>/dev/null || true  # openpi-xpu-env 尚未 git init 时静默
