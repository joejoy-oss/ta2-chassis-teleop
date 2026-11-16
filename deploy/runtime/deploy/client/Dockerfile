# =============================================================================
# openpi-xpu-client: TeleAvatar V2 robot client (examples/teleavatar_v2/main.py, zero.py) for running next to the policy
# server on the B70 machine. Environment only: the openpi working tree is mounted at /app, as in openpi-xpu-env.
#
# Based on the owner's original client container (/home/liyike/2026_fall/ros2-humble: osrf/ros:humble-desktop,
# Ubuntu 22.04, apt ROS 2 Humble + GStreamer 1.20, Python 3.10 uv venv, zenoh-bridge-ros2dds 1.9.0, cyclonedds), with:
#   * gstreamer1.0-libav: CPU H.265 decode (avdec_h265); the B70 has no NVIDIA GPU;
#   * no NVIDIA libraries or settings, no server venv;
#   * the client venv baked in from the hash-locked client/requirements.lock (client/lock.sh regenerates it);
#   * iproute2 for socket checks (`ss`) inside the container;
#   * Intel GPU (VA-API) H.265 decode for the B70's Arc Pro B70 (Battlemage, PCI ID e223): libva 2.23.0, gmmlib 22.9.0,
#     the iHD media driver 26.1.4 and vainfo built from the release tarballs into /opt/intel-va (stage va-builder;
#     Ubuntu 22.04's iHD and Intel's 22.04 packages, up to 25.2.4, do not know e223), plus libvapp.so (va_vpp/vapp.c:
#     VA video processing of vah265dec's frames straight into the client's memory, see
#     examples/teleavatar_v2/va_frame.py). Used with `docker run --device /dev/dri/renderD12x --group-add <render gid>`;
#   * no proxy ARG declared: proxies come only as predefined build args, which docker keeps out of the history.
#
# Build with client/build.sh (APT_MIRROR / ROS_APT_MIRROR swap the Ubuntu and ROS sources for the build only; the
# original sources are restored in the same layer). A ROS mirror also drops the ros2 deb-src line: the USTC mirror
# has no source indices (404, 11-03 build log).
# =============================================================================
ARG BASE_IMAGE=osrf/ros:humble-desktop@sha256:b624d8bcea33796d32e0dbd85326722188485759bbd6b538e4785750a5c88a7b
ARG UV_VERSION=0.13.5
# Index digest that goes with UV_VERSION (same as openpi-xpu-env); change both together.
ARG UV_IMAGE_DIGEST=sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff
# The Intel VA stack (stage va-builder); the tarballs' SHA256s are next to the versions there.
ARG LIBVA_VERSION=2.23.0
ARG GMMLIB_VERSION=22.9.0
ARG MEDIA_DRIVER_VERSION=26.1.4

FROM ghcr.io/astral-sh/uv:${UV_VERSION}@${UV_IMAGE_DIGEST} AS uv-bin

# 0) The Intel VA stack, built in a throwaway stage on the same base; only /opt/intel-va is copied into the image. The
#    tarballs come from GitHub through the build proxy and are checked against the SHA256s below (client/SHA256SUMS).
#    The libraries are stripped (the iHD driver is ~555 MB with its debug info). libvapp.so is compiled last, so a
#    change to vapp.c does not rebuild the driver.
FROM ${BASE_IMAGE} AS va-builder
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=
ARG LIBVA_VERSION
ARG GMMLIB_VERSION
ARG MEDIA_DRIVER_VERSION
ARG LIBVA_SHA256=b10aceb30e93ddf13b2030eb70079574ba437be9b3b76065caf28a72c07e23e7
ARG GMMLIB_SHA256=6ae262b95540681ad5478c63affb880135f772c95410b092009a6f7dfc04a5ba
ARG MEDIA_DRIVER_SHA256=dd9132c69d68702b87520d9f4a4ee51796d67cb164e8befbcd024257369edba7
ARG LIBVA_UTILS_SHA256=fa7ff29847b55010fbbb775b35382f97f29b7b97abe9a2f6fb3e22b36db5440a
ARG VA_JOBS=10
RUN set -eu; \
    if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s#http://archive.ubuntu.com/ubuntu#${APT_MIRROR}#g; s#http://security.ubuntu.com/ubuntu#${APT_MIRROR}#g" /etc/apt/sources.list; \
    fi; \
    rm -f /etc/apt/sources.list.d/ros2*; \
    pkgs="build-essential cmake ninja-build meson pkg-config curl ca-certificates libdrm-dev libx11-dev libxext-dev \
          libxfixes-dev libx11-xcb-dev libxcb-dri3-dev libwayland-dev libwayland-bin libgstreamer1.0-dev"; \
    for i in 1 2 3 4 5; do \
        if apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends $pkgs; then break; fi; \
        [ "$i" -lt 5 ] || exit 1; echo "apt attempt $i failed, retrying in 10s ..." >&2; sleep 10; \
    done; \
    rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    mkdir -p /src /build; cd /src; \
    fetch() { \
        for i in 1 2 3 4 5; do \
            if curl -sSL --tls-max 1.2 --max-time 900 -o "$2" "https://github.com/intel/$1"; then break; fi; \
            [ "$i" -lt 5 ] || exit 1; echo "download of $1 failed (attempt $i), retrying in 10s ..." >&2; sleep 10; \
        done; \
        echo "$3  $2" | sha256sum -c; \
    }; \
    fetch "libva/archive/refs/tags/${LIBVA_VERSION}.tar.gz" libva.tar.gz "${LIBVA_SHA256}"; \
    fetch "gmmlib/archive/refs/tags/intel-gmmlib-${GMMLIB_VERSION}.tar.gz" gmmlib.tar.gz "${GMMLIB_SHA256}"; \
    fetch "media-driver/archive/refs/tags/intel-media-${MEDIA_DRIVER_VERSION}.tar.gz" media-driver.tar.gz "${MEDIA_DRIVER_SHA256}"; \
    fetch "libva-utils/archive/refs/tags/${LIBVA_VERSION}.tar.gz" libva-utils.tar.gz "${LIBVA_UTILS_SHA256}"; \
    cd /build; \
    for t in /src/*.tar.gz; do tar xf "$t"; done; \
    export PKG_CONFIG_PATH=/opt/intel-va/lib/pkgconfig; \
    meson setup libva-build "libva-${LIBVA_VERSION}" --prefix=/opt/intel-va --libdir=lib --buildtype=release \
        -Ddriverdir=/opt/intel-va/lib/dri -Dwith_x11=yes -Dwith_glx=no -Dwith_wayland=yes; \
    ninja -C libva-build install; \
    cmake -S "gmmlib-intel-gmmlib-${GMMLIB_VERSION}" -B gmm-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/intel-va -DCMAKE_INSTALL_LIBDIR=lib; \
    ninja -C gmm-build -j"${VA_JOBS}" install; \
    cmake -S "media-driver-intel-media-${MEDIA_DRIVER_VERSION}" -B md-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/intel-va -DCMAKE_INSTALL_LIBDIR=lib -DLIBVA_DRIVERS_PATH=/opt/intel-va/lib/dri \
        -DINSTALL_DRIVER_SYSCONF=OFF -DMEDIA_BUILD_FATAL_WARNINGS=OFF -DENABLE_NONFREE_KERNELS=ON; \
    ninja -C md-build -j"${VA_JOBS}" install; \
    meson setup utils-build "libva-utils-${LIBVA_VERSION}" --prefix=/opt/intel-va --libdir=lib --buildtype=release \
        -Ddrm=true -Dx11=false -Dwayland=false -Dtests=false; \
    ninja -C utils-build install; \
    find /opt/intel-va -type f \( -name "*.so*" -o -path "/opt/intel-va/bin/*" \) -exec strip --strip-unneeded {} +; \
    rm -rf /src /build; \
    du -sh /opt/intel-va /opt/intel-va/lib/dri/iHD_drv_video.so
COPY va_vpp/vapp.c /opt/intel-va/src/vapp.c
RUN set -eu; \
    gcc -O2 -Wall -Wextra -Werror -shared -fPIC /opt/intel-va/src/vapp.c -o /opt/intel-va/lib/libvapp.so \
        $(pkg-config --cflags --libs gstreamer-1.0) -I/opt/intel-va/include -L/opt/intel-va/lib -lva; \
    strip --strip-unneeded /opt/intel-va/lib/libvapp.so; \
    ldd /opt/intel-va/lib/libvapp.so

FROM ${BASE_IMAGE}

COPY --from=uv-bin /uv /uvx /bin/

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=
ARG ROS_APT_MIRROR=
ARG ZENOH_BRIDGE_VERSION=1.9.0
# SHA256 of zenoh-bridge-ros2dds_1.9.0_amd64.deb in the Eclipse repository's Packages index.
ARG ZENOH_DEB_SHA256=ebd95cb7ba2ea148ab87c7d83c7802b75f3f7cddfc6e85d7b4344ec49967e73a

# 1) System packages in one layer: the owner's list, plus gstreamer1.0-libav and iproute2. zenoh comes from the Eclipse
#    repository as in the owner's Dockerfile; its .deb is checked against ZENOH_DEB_SHA256 before installing. Everything
#    is downloaded first (retried: the proxy occasionally answers 502), then installed offline; docker-clean only
#    deletes the cached .deb files when dpkg runs, so the downloaded zenoh .deb is still there to check.
RUN set -eu; \
    cp /etc/apt/sources.list /tmp/sources.list.orig; \
    cp /etc/apt/sources.list.d/ros2.sources /tmp/ros2.sources.orig; \
    if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s#http://archive.ubuntu.com/ubuntu#${APT_MIRROR}#g; s#http://security.ubuntu.com/ubuntu#${APT_MIRROR}#g" /etc/apt/sources.list; \
    fi; \
    if [ -n "${ROS_APT_MIRROR}" ]; then \
        sed -i "s#http://packages.ros.org/ros2/ubuntu#${ROS_APT_MIRROR}#g; s#^Types: deb deb-src\$#Types: deb#" /etc/apt/sources.list.d/ros2.sources; \
    fi; \
    for i in 1 2 3 4 5; do \
        if apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends ca-certificates wget gnupg; then break; fi; \
        [ "$i" -lt 5 ] || exit 1; echo "apt attempt $i failed, retrying in 10s ..." >&2; sleep 10; \
    done; \
    install -d -m 0755 /etc/apt/keyrings; \
    wget -qO- https://download.eclipse.org/zenoh/debian-repo/zenoh-public-key | gpg --dearmor --yes -o /etc/apt/keyrings/zenoh-public-key.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/zenoh-public-key.gpg] https://download.eclipse.org/zenoh/debian-repo/ /" > /etc/apt/sources.list.d/zenoh.list; \
    pkgs="zenoh-bridge-ros2dds=${ZENOH_BRIDGE_VERSION} ros-humble-rmw-cyclonedds-cpp ros-humble-rosbag2-storage-mcap \
          iputils-ping iproute2 python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base \
          gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav python3-opencv python3-numpy \
          python3-venv build-essential pkg-config"; \
    updated=0; \
    for i in 1 2 3 4 5; do \
        if [ "$updated" = 0 ] && apt-get -o Acquire::Retries=5 update; then updated=1; fi; \
        if [ "$updated" = 1 ] && apt-get -o Acquire::Retries=5 install -y --no-install-recommends --download-only $pkgs; then break; fi; \
        [ "$i" -lt 5 ] || exit 1; echo "apt attempt $i failed (updated=$updated), retrying in 10s ..." >&2; sleep 10; \
    done; \
    echo "${ZENOH_DEB_SHA256}  /var/cache/apt/archives/zenoh-bridge-ros2dds_${ZENOH_BRIDGE_VERSION}_amd64.deb" | sha256sum -c; \
    apt-get install -y --no-install-recommends $pkgs; \
    zenoh-bridge-ros2dds --version | grep -F "v${ZENOH_BRIDGE_VERSION}"; \
    gst-inspect-1.0 avdec_h265 > /dev/null; \
    cp /tmp/sources.list.orig /etc/apt/sources.list; \
    cp /tmp/ros2.sources.orig /etc/apt/sources.list.d/ros2.sources; \
    rm -rf /var/lib/apt/lists/* /tmp/sources.list.orig /tmp/ros2.sources.orig; \
    install -d /opt/openpi-client; \
    dpkg-query -W -f='${Package} ${Version}\n' | sort > /opt/openpi-client/APT_VERSIONS

# 2) Client venv on the system Python 3.10, seeing the apt rclpy, gi and cv2 (--system-site-packages, as in the
#    owner's ros2-humble.sh). Only the locked, hashed requirements are installed (--no-deps: the lock is complete).
COPY requirements.lock /opt/openpi-client/requirements.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    uv venv --python /usr/bin/python3 --system-site-packages /opt/openpi-client-venv; \
    uv pip install --python /opt/openpi-client-venv/bin/python --require-hashes --no-deps \
        -r /opt/openpi-client/requirements.lock; \
    rm -f /tmp/uv-*.lock

# 3) User `ros` (uid/gid 1000, as in the owner's image) for local runs with `--user 1000:1000`. The image itself stays
#    root: on the B70 the mounted directories are root-owned.
RUN groupadd --gid 1000 ros && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash ros

# 4) The VA stack. /etc/ld.so.conf.d/00-intel-va.conf sorts before libc.conf, so the dynamic linker takes
#    /opt/intel-va/lib before the system's libva 2.14 (still installed: gst-plugins-bad, libavcodec and OpenCV depend on
#    the package) for every process, cv2, gst-libav, gst-va and `docker exec` shells alike, without LD_LIBRARY_PATH.
#    Checked here: each libva soname resolves to /opt/intel-va/lib first, and the libraries that load libva get it from
#    there. The element vah265dec itself only appears with a render node (`--device`), so it is checked at run time.
COPY --from=va-builder /opt/intel-va /opt/intel-va
RUN set -eu; \
    echo /opt/intel-va/lib > /etc/ld.so.conf.d/00-intel-va.conf; \
    ldconfig; \
    for so in libva.so.2 libva-drm.so.2 libva-x11.so.2 libva-wayland.so.2; do \
        first="$(ldconfig -p | awk -v so="$so" '$1 == so { print $NF; exit }')"; \
        [ "$first" = "/opt/intel-va/lib/$so" ] || { echo "$so resolves to '$first' first" >&2; exit 1; }; \
    done; \
    cv2_so="$(/usr/bin/python3 -c 'import cv2; print(cv2.__file__)')"; \
    for lib in /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstva.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstlibav.so \
               "$cv2_so" /opt/intel-va/lib/libvapp.so; do \
        echo "$lib:"; ldd "$lib" | grep -E 'libva' | tee /tmp/ldd.txt; \
        if grep -v -F '=> /opt/intel-va/lib/' /tmp/ldd.txt | grep -q .; then echo "$lib loads libva from elsewhere" >&2; exit 1; fi; \
        if grep -q 'libva-glx' /tmp/ldd.txt; then echo "$lib needs libva-glx (not built)" >&2; exit 1; fi; \
    done; \
    rm -f /tmp/ldd.txt; \
    ls -l /opt/intel-va/lib/dri/iHD_drv_video.so /opt/intel-va/lib/libvapp.so /opt/intel-va/bin/vainfo; \
    /usr/bin/python3 -c "import cv2"; \
    gst-inspect-1.0 avdec_h265 videoconvert x265enc > /dev/null; \
    rm -rf /root/.cache/gstreamer-1.0

# Build metadata after the expensive layers (ARGs declared here do not invalidate them).
ARG BASE_IMAGE
ARG UV_VERSION
ARG OPENPI_COMMIT=unknown
ARG IMAGE_VERSION=dev
ARG LOCK_SHA256=unknown
ARG LIBVA_VERSION
ARG GMMLIB_VERSION
ARG MEDIA_DRIVER_VERSION
ARG VAPP_SHA256=unknown

COPY --chmod=755 entrypoint.sh /usr/local/bin/openpi-client-entrypoint.sh
COPY --chmod=644 zenoh-bridge-client.json5 /opt/openpi-client/zenoh-bridge-client.json5
RUN set -eu; \
    actual_uv="$(uv --version | cut -d' ' -f2)"; \
    if [ "${actual_uv}" != "${UV_VERSION}" ]; then echo "uv ${actual_uv} != ${UV_VERSION}" >&2; exit 1; fi; \
    /opt/openpi-client-venv/bin/python -c "import numpy, msgpack, websockets, tree, PIL, tyro, yaml, pandas, typing_extensions, cv2, gi; assert numpy.__version__.startswith('1.'), numpy.__version__"; \
    echo "${OPENPI_COMMIT}" > /opt/openpi-client/OPENPI_COMMIT; \
    printf 'uv=%s python=%s base=%s zenoh=%s lock=%s\n' "${UV_VERSION}" "$(/usr/bin/python3 -c 'import platform; print(platform.python_version())')" \
        "${BASE_IMAGE}" "$(zenoh-bridge-ros2dds --version 2>/dev/null | tail -1)" "${LOCK_SHA256}" > /opt/openpi-client/TOOLCHAIN

# Runtime:
#   PATH/VIRTUAL_ENV         -> `python` is the client venv
#   PYTHONPATH               -> openpi_client from the mounted code (the entrypoint adds ROS's paths in front)
#   PYTHONDONTWRITEBYTECODE  -> no __pycache__ in the mounted code
#   RMW / ROS_LOCALHOST_ONLY -> cyclonedds as in the owner's container; DDS discovery and traffic stay on 127.0.0.1
#                               (unicast peers), so nothing leaks onto the host network with --network host
#   ROS_DOMAIN_ID=29         -> the value in examples/teleavatar_v2/README.md; only has to match the bridge in the
#                               same container (zenoh routes by topic name)
#   LIBVA_DRIVERS_PATH/NAME  -> libva loads /opt/intel-va's iHD driver (harmless without a render node)
ENV PATH=/opt/openpi-client-venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/openpi-client-venv \
    PYTHONPATH=/app/packages/openpi-client/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_LOCALHOST_ONLY=1 \
    ROS_DOMAIN_ID=29 \
    LIBVA_DRIVERS_PATH=/opt/intel-va/lib/dri \
    LIBVA_DRIVER_NAME=iHD

LABEL org.opencontainers.image.title="openpi-xpu-client" \
      org.opencontainers.image.description="TeleAvatar V2 robot client env (ROS 2 Humble, GStreamer 1.20 with CPU and Intel VA-API H.265 decode, zenoh-bridge-ros2dds); mount code at /app" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      openpi.commit="${OPENPI_COMMIT}" \
      openpi.uv="${UV_VERSION}" \
      openpi.client.lock-sha256="${LOCK_SHA256}" \
      zenoh.bridge-ros2dds="${ZENOH_BRIDGE_VERSION}" \
      openpi.intel-va="libva ${LIBVA_VERSION} / gmmlib ${GMMLIB_VERSION} / iHD ${MEDIA_DRIVER_VERSION}" \
      openpi.vapp-sha256="${VAPP_SHA256}"

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/openpi-client-entrypoint.sh"]
CMD ["bash"]
