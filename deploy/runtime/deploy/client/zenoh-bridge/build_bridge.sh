#!/usr/bin/env bash
# Build zenoh-bridge-ros2dds 1.9.0 with route-lifecycle-p1.patch (Q16: a removed route keeps no matching listener, so
# a robot-side subscriber that goes and comes back does not get every command once more per removed route), the way
# the binary in openpi-xpu-client:v0.2.1 was built (B3 45): in a throwaway container of openpi-xpu-client:v0.2.0
# (Ubuntu 22.04, so the binary needs no newer glibc than the image has), Rust 1.93.0 as the source pins it
# (rust-toolchain.toml; the standalone installer, sha256 checked), `cargo build --release --locked -p
# zenoh-bridge-ros2dds` with the default features, as Eclipse builds the .deb.
#   WORK=/some/dir [VARIANTS="p1"] client/zenoh-bridge/build_bridge.sh
#     WORK      work directory (about 2.5 GB: source trees, the toolchain, the cargo registry and target directory);
#               the binaries end up in WORK/out as zenoh-bridge-ros2dds-1.9.0-<variant> with WORK/out/SHA256SUMS
#     VARIANTS  p1 (patched; default) and/or orig (the release as it is, a control)
#   Downloads (source tarball, Rust installer, apt packages for bindgen) go through the proxy HTTPS_PROXY (default
#   http://127.0.0.1:7897), TLS at most 1.2; apt only from the Ubuntu archive.
# The same paths inside the container as in 45 (/w/src-<variant>, /w/cache/...): with them a rebuild gave the same bytes
# (see README.md); the image takes the binary by its sha256 (Dockerfile.v0.2.1), not whatever a rebuild gives.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${WORK:?set WORK to a work directory}"
VARIANTS="${VARIANTS:-p1}"
PX="${HTTPS_PROXY:-http://127.0.0.1:7897}"
BASE=openpi-xpu-client:v0.2.0
BASE_ID=sha256:bdf073743a5526899f47dfe653fc06caeac3c26e72cff98d178dc07c7cc9534c
SRC_URL=https://codeload.github.com/eclipse-zenoh/zenoh-plugin-ros2dds/tar.gz/refs/tags/1.9.0
SRC_SHA256=d9b1694fb287fcfe72348eb4900f81d16ec4c419a34cdfc95a4655d42216de36
PATCH_SHA256=fd2e037d7fcf5d2515476b460bb362a4941939edc2a19f6736aebf51c2a80063
RUST=rust-1.93.0-x86_64-unknown-linux-gnu
RUST_SHA256=b9d9f01a96a2542852ccfddd82194276ba1c86bc76353309ff636b737fc0a772
NAME=openpi-xpu-client-bridge-build

[ "$(docker image inspect "$BASE" --format '{{.Id}}')" = "$BASE_ID" ] || { echo "$BASE is not $BASE_ID" >&2; exit 1; }
echo "$PATCH_SHA256  $HERE/route-lifecycle-p1.patch" | sha256sum -c
mkdir -p "$WORK/cache" "$WORK/out"
WORK="$(cd "$WORK" && pwd)"
if [ ! -f "$WORK/src-1.9.0.tar.gz" ]; then
    curl --tls-max 1.2 -x "$PX" -fsSL -o "$WORK/src-1.9.0.tar.gz.part" "$SRC_URL"
    mv "$WORK/src-1.9.0.tar.gz.part" "$WORK/src-1.9.0.tar.gz"
fi
echo "$SRC_SHA256  $WORK/src-1.9.0.tar.gz" | sha256sum -c
for v in $VARIANTS; do
    case "$v" in p1 | orig) ;; *) echo "unknown variant $v" >&2; exit 1 ;; esac
    rm -rf "$WORK/src-$v"
    mkdir -p "$WORK/src-$v"
    tar xzf "$WORK/src-1.9.0.tar.gz" -C "$WORK/src-$v" --strip-components=1
    if [ "$v" = p1 ]; then
        patch -p1 --forward --no-backup-if-mismatch -d "$WORK/src-$v" < "$HERE/route-lifecycle-p1.patch"
    fi
done
docker run --rm --name "$NAME" --network host --user 0:0 --cpus 10 -m 16g --memory-swap 16g \
    -e PX="$PX" -e VARIANTS="$VARIANTS" -e RUST="$RUST" -e RUST_SHA256="$RUST_SHA256" -e OWNER="$(id -u):$(id -g)" \
    -v "$WORK:/w" --entrypoint bash "$BASE" -c '
set -eux
mkdir -p /tmp/nosrc
APT="-o Acquire::http::Proxy=$PX -o Dir::Etc::SourceParts=/tmp/nosrc -o APT::Get::List-Cleanup=0"
apt-get $APT update -qq
DEBIAN_FRONTEND=noninteractive apt-get $APT install -y -qq --no-install-recommends build-essential clang libclang-dev
cd /w/cache
if [ ! -x /w/cache/rust/bin/cargo ]; then
    curl --tls-max 1.2 -x "$PX" -fsSLO "https://static.rust-lang.org/dist/$RUST.tar.xz"
    echo "$RUST_SHA256  $RUST.tar.xz" | sha256sum -c
    tar xJf "$RUST.tar.xz"
    "./$RUST/install.sh" --prefix=/w/cache/rust --components=rustc,cargo,rust-std-x86_64-unknown-linux-gnu > /dev/null
    rm -rf "$RUST" "$RUST.tar.xz"
fi
export PATH=/w/cache/rust/bin:$PATH CARGO_HOME=/w/cache/cargo CARGO_TARGET_DIR=/w/cache/target
mkdir -p "$CARGO_HOME"
printf "[http]\nproxy = \"%s\"\nssl-version.max = \"tlsv1.2\"\n[net]\nretry = 5\n" "$PX" > "$CARGO_HOME/config.toml"
rustc --version
cargo --version
for v in $VARIANTS; do
    cd "/w/src-$v"
    cargo build --release --locked -p zenoh-bridge-ros2dds 2>&1 | grep -vE "^\s+(Compiling|Downloaded|Downloading) " | tail -5
    cp "$CARGO_TARGET_DIR/release/zenoh-bridge-ros2dds" "/w/out/zenoh-bridge-ros2dds-1.9.0-$v"
    "/w/out/zenoh-bridge-ros2dds-1.9.0-$v" --version | tail -1
    ldd "/w/out/zenoh-bridge-ros2dds-1.9.0-$v"
done
chown -R "$OWNER" /w
'
(cd "$WORK/out" && sha256sum zenoh-bridge-ros2dds-1.9.0-* | tee SHA256SUMS)
