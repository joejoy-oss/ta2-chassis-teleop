#!/usr/bin/env bash
# TA2 chassis small steps (run on B70)
#
# Measured axes (2026-11-14): +vx = right, +vy = forward, +wz = CCW (seen from above)
#
# Usage:
#   ./move.sh forward [cm]   move forward, default 10 cm
#   ./move.sh back [cm]
#   ./move.sh left [cm]
#   ./move.sh right [cm]
#   ./move.sh all            forward, back, left, right 10 cm each, 3 s pause between, ends where it started
#   ./move.sh stop           shut down the client container safely (never use docker stop)
# Short forms: f / b / l / r.
#
# Speed is fixed at 0.10 m/s; distance 1-50 cm. 3 s countdown before each move (Ctrl+C cancels); Ctrl+C while moving stops at once.
# From another machine use -t so Ctrl+C gets through:  ssh -t root@192.168.31.228 ./move.sh forward
set -e
SELF=$(readlink -f "$0")

DEPLOY=${DEPLOY:-/data/ta2/software/deploy}
ROBOT=192.168.31.218
SPEED=0.10
TEST=/outputs/tools/chassis_test.py      # path inside the container = host /root/openpi-xpu/client/outputs/tools/

cd "$DEPLOY"
export OPENPI_DIR=$DEPLOY/client/openpi
R=client/run_client.sh

ensure_client() {
    if ! docker ps --format '{{.Names}}' | grep -qx openpi-xpu-client; then
        echo "Starting the client container (robot $ROBOT)..."
        ZENOH_ENDPOINT=tcp/$ROBOT:9000 $R start > /dev/null
        sleep 8
    fi
}

step() {
    local dir=$1 cm=${2:-10} v
    case $dir in
        forward|f) v="0 $SPEED 0" ;;
        back|b)    v="0 -$SPEED 0" ;;
        left|l)    v="-$SPEED 0 0" ;;
        right|r)   v="$SPEED 0 0" ;;
        *) echo "Direction must be forward/back/left/right (or all, stop)" >&2; exit 2 ;;
    esac
    if ! [[ $cm =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v c="$cm" 'BEGIN{exit !(c >= 1 && c <= 50)}'; then
        echo "Distance must be 1-50 cm" >&2; exit 2
    fi
    # acceleration + braking lose about 0.2 cm; add 0.04 s to make up for it
    local dur
    dur=$(awk -v c="$cm" -v s="$SPEED" 'BEGIN{printf "%.2f", c / 100 / s + 0.04}')
    echo "=== $dir $cm cm ($SPEED m/s x $dur s)"
    # FAST=1 (used by the web page): 1 s countdown and 1 s observation instead of 3 s
    local fast=()
    [ "${FAST:-0}" = 1 ] && fast=(--countdown 1 --observe 1)
    $R exec python3 $TEST $v --dur "$dur" "${fast[@]}"
}

case ${1:-} in
    stop)
        $R stop ;;
    all)
        ensure_client
        step forward 10; sleep 3
        step back 10;    sleep 3
        step left 10;    sleep 3
        step right 10 ;;
    "")
        sed -n '2,16p' "$SELF"; exit 2 ;;
    *)
        ensure_client
        step "$1" "${2:-10}" ;;
esac
