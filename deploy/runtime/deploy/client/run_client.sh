#!/usr/bin/env bash
# The long-running robot-client container on the B70 (or any host without an NVIDIA GPU): its main process keeps
# zenoh-bridge-ros2dds running in client mode towards ZENOH_ENDPOINT, the robot's zenoh router: by default through the
# local end of the SSH tunnel to it (README 3.1), or directly with ZENOH_ENDPOINT=tcp/<ROBOT_IP>:9000 (README 3.4,
# conditions D1-D3); zero.py and main.py run inside it with `exec`.
#   client/run_client.sh start            start the container (refuses if a zenoh router or bridge already runs here),
#                                         then raise the card's gt1 minimum clock (gpu_minfreq.sh set) and leave a
#                                         watcher that puts it back whenever the container ends
#   ZENOH_ENDPOINT=tcp/<ROBOT_IP>:9000 client/run_client.sh start
#                                         the same, the bridge connected to the robot directly (no tunnel)
#   client/run_client.sh stop             first SIGINT (as Ctrl+C) to main.py and zero.py running in it, then wait up to
#                                         STOP_WAIT s for them to end (main.py commands zero chassis velocity first);
#                                         then stop and remove it, put the gt1 minimum clock back (gpu_minfreq.sh
#                                         restore) and remove STATE_DIR if nothing but its lock is left. docker stop
#                                         alone sends main.py no signal and SIGKILLs it: no zero velocity (README 3.5)
#   client/run_client.sh status           container state, the end of the bridge log, the card (xpu-smi, clocks), the
#                                         ROS topics it sees
#   client/run_client.sh exec CMD [ARGS]  run CMD in it with the ROS environment, e.g.
#       client/run_client.sh exec python examples/teleavatar_v2/zero.py
#       client/run_client.sh exec python examples/teleavatar_v2/main.py --remote-host 127.0.0.1 --remote-port 8000 \
#           --rtc --video-decode va --prompt "..."
#     or, decoding on the CPU (WPP streams only):
#       ... --video-decode cpu --video-cpu-thread-type slice --video-cpu-threads 8 --video-convert-threads 4 ...
# Environment (defaults):
#   OPENPI_DIR=/root/openpi-xpu/openpi-client   openpi code, mounted read-only at /app
#   OUTPUTS=/root/openpi-xpu/client/outputs      mounted at /outputs: bridge.log, ROS logs (ros_log/); gpu_minfreq.log
#   IMAGE=openpi-xpu-client:v0.2.1  NAME=openpi-xpu-client  CPUSET=4-7,28-31  MEMORY=8g
#                                v0.2.1: v0.2.0 with zenoh-bridge-ros2dds 1.9.0 patched for the duplicated commands
#                                (zenoh-bridge/README.md)
#   ZENOH_ENDPOINT=tcp/127.0.0.1:19000   the tunnel's local end; tcp/<ROBOT_IP>:9000 connects directly
#   ROS_DOMAIN_ID=29
#   RENDER=/dev/dri/renderD128   the Intel GPU render node for --video-decode va (renderD128: the policy server's card;
#                                renderD129 the other one); none: no GPU in the container, no clock change
#   RENDER_GID=993               the host's render group
#   RUST_LOG                     unset: the bridge's own default (info); set: passed to the bridge, e.g.
#                                RUST_LOG=info,zenoh_plugin_ros2dds=debug,zenoh_plugin_ros2dds::route_publisher=trace,zenoh_plugin_ros2dds::route_subscriber=trace
#                                logs each route's matching changes and every message it routes (bridge.log grows by
#                                about a line per message)
#   GT1_MIN_MHZ=1500             gt1 minimum clock while the container runs (gpu_minfreq.sh; a positive integer in
#                                MHz); off: leave it. Anything else stops start before the container is created
#   STOP_WAIT=10                 stop: seconds (a whole number) to wait for main.py and zero.py to end after the
#                                SIGINT; after that the container is stopped all the same
#   STATE_DIR=/run/openpi-xpu-client   gpu_minfreq.sh's record and lock
# The container uses the host network (the robot pushes the video to UDP 8890 of this host; the bridge and the policy
# server are on 127.0.0.1); DDS stays on 127.0.0.1 (ROS_LOCALHOST_ONLY=1 in the image). It gets the render node, no
# other device and no extra privileges.
set -euo pipefail
OPENPI_DIR="${OPENPI_DIR:-/root/openpi-xpu/openpi-client}"
OUTPUTS="${OUTPUTS:-/root/openpi-xpu/client/outputs}"
IMAGE="${IMAGE:-openpi-xpu-client:v0.2.1}"
NAME="${NAME:-openpi-xpu-client}"
CPUSET="${CPUSET:-4-7,28-31}"
MEMORY="${MEMORY:-8g}"
ZENOH_ENDPOINT="${ZENOH_ENDPOINT:-tcp/127.0.0.1:19000}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-29}"
RENDER="${RENDER:-/dev/dri/renderD128}"
RENDER_GID="${RENDER_GID:-993}"
STOP_WAIT="${STOP_WAIT:-10}"
STATE_DIR="${STATE_DIR:-/run/openpi-xpu-client}"
GPU_MINFREQ="$(dirname "$(readlink -f "$0")")/gpu_minfreq.sh"
export RENDER OUTPUTS STATE_DIR GT1_MIN_MHZ="${GT1_MIN_MHZ:-1500}"
# main.py and zero.py in the container, for pgrep -f / pkill -f. [e]: the pattern does not match its own text, so not
# the shell that runs pgrep or pkill with it either
CLIENTS='[e]xamples/teleavatar_v2/(main|zero)\.py'

# Main process of the container: restart the bridge whenever it exits; stop it on SIGTERM (docker stop).
SUPERVISOR='
trap "kill \$child 2> /dev/null; wait \$child 2> /dev/null; exit 0" TERM INT
while true; do
    echo "$(date "+%F %T %Z") starting zenoh-bridge-ros2dds client -e $ZENOH_ENDPOINT -d $ROS_DOMAIN_ID"
    zenoh-bridge-ros2dds client -c /opt/openpi-client/zenoh-bridge-client.json5 -e "$ZENOH_ENDPOINT" \
        --no-multicast-scouting --ros-localhost-only -d "$ROS_DOMAIN_ID" &
    child=$!
    wait $child
    echo "$(date "+%F %T %Z") bridge exited with $?; restarting in 2 s"
    sleep 2
done >> /outputs/bridge.log 2>&1
'

running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" = true ]; }

# stop, before the container goes: main.py and zero.py run in it under docker exec, so neither docker stop (they are
# SIGKILLed once the main process has exited) nor a closed terminal or SSH session sends them a signal. SIGINT, as
# Ctrl+C in their terminal, makes main.py command zero chassis velocity before anything else; wait up to STOP_WAIT s.
stop_clients() {
    local found rc=0
    mapfile -t found < <(docker exec "$NAME" pgrep -af "$CLIENTS")
    [ ${#found[@]} -gt 0 ] || return 0  # none running
    echo "SIGINT to these in $NAME, waiting up to $STOP_WAIT s for them to end:"
    printf '  %s\n' "${found[@]}"
    # shellcheck disable=SC2016  # $1, $2: bash -c's arguments, expanded in the container
    docker exec "$NAME" bash -c 'pkill -INT -f "$1"
        for ((i = 0; i < $2 * 10; i++)); do
            pgrep -f "$1" > /dev/null || { echo "they ended after about $((i / 10)).$((i % 10)) s"; exit 0; }
            sleep 0.1
        done
        exit 3' stop_clients "$CLIENTS" "$STOP_WAIT" || rc=$?
    case $rc in
        0) ;;
        3) echo "warning: still running after $STOP_WAIT s; stopping the container all the same, which SIGKILLs them" \
               "(main.py's log says whether its zero chassis velocity went out)" >&2 ;;
        *) echo "warning: docker exec failed ($rc); stopping the container all the same" >&2 ;;
    esac
}

# stop, at the end: gpu_minfreq.sh leaves STATE_DIR behind with only its lock in it once the record is gone. Remove it
# then, holding that lock (fd 9, as gpu_minfreq.sh does), so that nothing is recorded meanwhile. A failure is noted,
# not an error.
remove_state_dir() {
    local i
    # The watcher restores once the container has ended and creates STATE_DIR and the lock again on the way: first
    # let it finish.
    for ((i = 0; i < 50; i++)); do
        pgrep -f "[g]pu_minfreq\.sh watch $NAME\$" > /dev/null || break
        sleep 0.1
    done
    if pgrep -f "[g]pu_minfreq\.sh watch $NAME\$" > /dev/null; then
        echo "note: gpu_minfreq.sh watch $NAME still runs after 5 s; $STATE_DIR left as it is" >&2
        return 0
    fi
    [ -d "$STATE_DIR" ] || return 0
    (
        shopt -s nullglob dotglob
        { exec 9> "$STATE_DIR/lock" && flock -w 30 9; } || exit 1
        entries=("$STATE_DIR"/*)
        if [ ${#entries[@]} -ne 1 ] || [ "${entries[0]}" != "$STATE_DIR/lock" ]; then
            echo "kept $STATE_DIR (holds ${entries[*]##*/})"
            exit 0
        fi
        rm -f "$STATE_DIR/lock" && rmdir "$STATE_DIR" && echo "removed $STATE_DIR (nothing but its lock was left)"
    ) || echo "note: could not remove $STATE_DIR; left as it is" >&2
}

xpu_id() {  # xpu-smi device id of RENDER's card (matched by PCI address)
    local pci
    pci=$(basename "$(readlink -f "/sys/class/drm/$(basename "$RENDER")/device")")
    xpu-smi discovery 2> /dev/null | awk -F'|' -v pci="$pci" '$2 ~ /[0-9]/ {id = $2; gsub(/ /, "", id)} index($3, pci) {print id; exit}'
}

case "${1:-}" in
    start)
        if docker inspect "$NAME" > /dev/null 2>&1; then
            echo "container $NAME exists already (run_client.sh stop first)" >&2
            exit 1
        fi
        # A zenoh router on this host (e.g. the host's own bridge service, router mode by default) would give the
        # robot's topics a second path; only one may exist.
        if systemctl is-active --quiet zenoh-bridge-ros2dds.service 2> /dev/null; then
            echo "the host's zenoh-bridge-ros2dds.service is active; stop it first" >&2
            exit 1
        fi
        if ss -tln | grep -qE ':7447\b'; then
            echo "something listens on TCP 7447 (a zenoh router?); stop it first" >&2
            exit 1
        fi
        if [[ "$ZENOH_ENDPOINT" == tcp/127.0.0.1:* ]]; then
            ss -tln | grep -qE "127.0.0.1:${ZENOH_ENDPOINT##*:}\b" ||
                echo "note: nothing listens on ${ZENOH_ENDPOINT#tcp/} yet (tunnel down?); the bridge keeps retrying" >&2
        else
            # Not probed from here: the bridge alone connects to the robot.
            echo "note: direct connection to ${ZENOH_ENDPOINT#tcp/} (no tunnel; README 3.4, D1-D3); the bridge keeps" \
                "retrying until it answers" >&2
        fi
        ss -uln | grep -qE ':8890\b' && echo "warning: UDP 8890 is in use; main.py will not get the video" >&2
        gpu=()
        if [ "$RENDER" != none ]; then
            [ -c "$RENDER" ] || { echo "$RENDER is not a device (RENDER=none for no GPU)" >&2; exit 1; }
            gpu=(--device "$RENDER" --group-add "$RENDER_GID")
            # Before anything starts: a GT1_MIN_MHZ that gpu_minfreq.sh would refuse (or a card without the clock files).
            "$GPU_MINFREQ" check || { echo "fix GT1_MIN_MHZ (a positive integer in MHz, or off) or RENDER" >&2; exit 1; }
        fi
        rl=()
        [ -n "${RUST_LOG:-}" ] && rl=(-e "RUST_LOG=$RUST_LOG")
        mkdir -p "$OUTPUTS"
        docker run -d --init --name "$NAME" --network host --cpuset-cpus "$CPUSET" \
            --memory "$MEMORY" --memory-swap "$MEMORY" "${gpu[@]}" \
            -v "$OPENPI_DIR:/app:ro" -v "$OUTPUTS:/outputs" \
            -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" -e ZENOH_ENDPOINT="$ZENOH_ENDPOINT" "${rl[@]}" \
            "$IMAGE" bash -c "$SUPERVISOR"
        # The clock goes back when the container ends, however it ends: the watcher waits for it (docker wait).
        if "$GPU_MINFREQ" set; then
            setsid nohup "$GPU_MINFREQ" watch "$NAME" > /dev/null 2>> "$OUTPUTS/gpu_minfreq.log" < /dev/null &
        else
            echo "warning: could not raise the gt1 minimum clock (gpu_minfreq.sh set); decoding is slower" >&2
        fi
        ;;
    stop)
        [[ "$STOP_WAIT" =~ ^(0|[1-9][0-9]*)$ ]] ||
            { echo "STOP_WAIT must be a whole number of seconds, not '$STOP_WAIT'" >&2; exit 2; }
        if running; then
            stop_clients
        fi
        docker stop -t 10 "$NAME" > /dev/null && docker rm "$NAME" > /dev/null && echo "stopped $NAME"
        "$GPU_MINFREQ" restore
        remove_state_dir
        ;;
    status)
        docker ps -a --filter "name=^${NAME}$" --format '{{.Names}} | {{.Image}} | {{.Status}}'
        tail -n 5 "$OUTPUTS/bridge.log" 2> /dev/null || true
        if [ "$RENDER" != none ]; then
            id=$(xpu_id)
            [ -n "$id" ] && xpu-smi dump -d "$id" -m 0,18,24 -n 1
            "$GPU_MINFREQ" status || true
        fi
        if running; then
            docker exec "$NAME" /usr/local/bin/openpi-client-entrypoint.sh ros2 topic list --no-daemon
        fi
        ;;
    exec)
        shift
        [ $# -gt 0 ] || { echo "usage: $0 exec CMD [ARGS...]" >&2; exit 2; }
        running || { echo "container $NAME is not running (run_client.sh start)" >&2; exit 1; }
        tty=()
        [ -t 0 ] && tty=(-it)
        # docker exec skips the image entrypoint; run it explicitly for the ROS environment.
        exec docker exec "${tty[@]}" "$NAME" /usr/local/bin/openpi-client-entrypoint.sh "$@"
        ;;
    *)
        sed -n '2,/^[^#]/{/^#/p;}' "$0"  # the comment block at the top, however long it gets
        exit 2
        ;;
esac
