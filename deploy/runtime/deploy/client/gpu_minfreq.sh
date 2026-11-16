#!/usr/bin/env bash
# Raises the minimum clock of the media GT (gt1: video decode) of the client's Intel GPU while the client runs, and puts
# it back afterwards (owner, 11-03: only during client sessions). At the default minimum (400 MHz) the GT idles low
# between frames and decoding the robot's stream takes about 4 ms longer per frame. Run as root on the host;
# run_client.sh calls it (start: set, then a watcher; stop: restore).
#   client/gpu_minfreq.sh set           record the current gt1 min_freq (unless already recorded), write GT1_MIN_MHZ
#   client/gpu_minfreq.sh restore       write the recorded value back if gt1 min_freq still is what set left there
#                                       (someone else's change is left alone and logged); forget the record. Safe to
#                                       repeat.
#   client/gpu_minfreq.sh watch NAME    wait until container NAME exits (docker wait), then restore
#   client/gpu_minfreq.sh status        the card, both GTs' min/max/actual clocks, the record, running watchers
#   client/gpu_minfreq.sh check         exit 2 if GT1_MIN_MHZ is not usable, 1 if the card's clock files are missing
# Environment (defaults): RENDER=/dev/dri/renderD128 (the card is found through /sys/class/drm/renderD12x/device, not
# by cardN; RENDER=none: nothing to do), GT1_MIN_MHZ=1500 (a positive integer in MHz, capped at the GT's max_freq;
# off, in any case: nothing to do; anything else is refused before the record or the clock is touched),
# OUTPUTS=/root/openpi-xpu/client/outputs (gpu_minfreq.log: one line per action), STATE_DIR=/run/openpi-xpu-client
# (tmpfs: gone after a reboot, as the sysfs value is).
# By hand: `gpu_minfreq.sh status` shows the record; `gpu_minfreq.sh restore` puts the value back; without a record,
# echo 400 > /sys/class/drm/renderD128/device/tile0/gt1/freq0/min_freq (400 MHz was the value on 11-03).
set -euo pipefail
RENDER="${RENDER:-/dev/dri/renderD128}"
GT1_MIN_MHZ="${GT1_MIN_MHZ:-1500}"
OUTPUTS="${OUTPUTS:-/root/openpi-xpu/client/outputs}"
STATE_DIR="${STATE_DIR:-/run/openpi-xpu-client}"
LOG="$OUTPUTS/gpu_minfreq.log"

log() {
    mkdir -p "$OUTPUTS"
    echo "$(date '+%F %T %Z') $*" | tee -a "$LOG"
}

disabled() {
    [ "$RENDER" = none ] || [ "${GT1_MIN_MHZ,,}" = off ]
}

check_target() {  # GT1_MIN_MHZ must be a positive integer (MHz) when not off: a typo must not raise the clock
    case "$GT1_MIN_MHZ" in
        '' | *[!0-9]* | 0*)
            echo "gpu_minfreq: GT1_MIN_MHZ must be off or a positive integer (MHz), not '$GT1_MIN_MHZ'" >&2
            exit 2
            ;;
    esac
}

resolve() {  # sets NODE (renderD12x), PCI (0000:xx:00.0), FREQ (gt1 freq0 directory), STATE (record file)
    NODE=$(basename "$RENDER")
    local device=/sys/class/drm/$NODE/device
    [ -e "$device" ] || { echo "no $device (RENDER=$RENDER)" >&2; exit 1; }
    PCI=$(basename "$(readlink -f "$device")")
    FREQ=$device/tile0/gt1/freq0
    [ -e "$FREQ/min_freq" ] || { echo "no $FREQ/min_freq" >&2; exit 1; }
    STATE=$STATE_DIR/gt1_min_freq.$PCI
}

locked() {  # run "$@" holding the lock (set, restore and the watcher may run at the same time)
    mkdir -p "$STATE_DIR"
    exec 9> "$STATE_DIR/lock"
    flock -w 30 9
    "$@"
}

do_set() {
    local now max target original prev_set=
    now=$(cat "$FREQ/min_freq")
    max=$(cat "$FREQ/max_freq")
    target=$GT1_MIN_MHZ
    [ "$target" -le "$max" ] || target=$max
    if [ -e "$STATE" ]; then  # keep the first original; the record follows the value written now
        original=$(sed -n 's/^original=//p' "$STATE")
        prev_set=$(sed -n 's/^set=//p' "$STATE")
        log "set $PCI ($NODE) gt1 min_freq: original $original already recorded, now $now"
    else
        original=$now
    fi
    record "$original" "$target"
    if ! echo "$target" > "$FREQ/min_freq"; then  # the value did not change: the record goes back to what it was
        if [ -n "$prev_set" ]; then record "$original" "$prev_set"; else rm -f "$STATE"; fi
        log "set $PCI ($NODE): writing $target to gt1 min_freq failed; it is $(cat "$FREQ/min_freq")"
        exit 1
    fi
    got=$(cat "$FREQ/min_freq")
    if [ "$got" != "$target" ]; then  # e.g. rounded by the driver: record what is there, so restore knows it as ours
        record "$original" "$got"
        log "set $PCI ($NODE): gt1 min_freq reads back $got, not $target; recorded set=$got"
    fi
    log "set $PCI ($NODE) gt1 min_freq $now -> $got (original $original, max $max)"
}

record() {  # record ORIGINAL SET
    printf 'original=%s\nset=%s\nnode=%s\nat=%s\n' "$1" "$2" "$NODE" "$(date '+%F %T %Z')" > "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}

do_restore() {
    local now original wrote
    if [ ! -e "$STATE" ]; then
        log "restore $PCI ($NODE): nothing recorded, gt1 min_freq $(cat "$FREQ/min_freq") left as it is"
        return 0
    fi
    original=$(sed -n 's/^original=//p' "$STATE")
    wrote=$(sed -n 's/^set=//p' "$STATE")
    now=$(cat "$FREQ/min_freq")
    if [ "$now" = "$wrote" ]; then
        echo "$original" > "$FREQ/min_freq"
        log "restore $PCI ($NODE) gt1 min_freq $now -> $(cat "$FREQ/min_freq") (recorded original $original)"
    else
        log "restore $PCI ($NODE): gt1 min_freq is $now, not the $wrote set left (changed by someone else): left as it is"
    fi
    rm -f "$STATE"
}

case "${1:-}" in
    set)
        if disabled; then echo "gpu_minfreq: nothing to do (RENDER=$RENDER, GT1_MIN_MHZ=$GT1_MIN_MHZ)"; exit 0; fi
        check_target
        resolve
        locked do_set
        ;;
    restore)  # whatever GT1_MIN_MHZ says now: a record left by set is acted on
        if [ "$RENDER" = none ]; then echo "gpu_minfreq: nothing to do (RENDER=none)"; exit 0; fi
        resolve
        locked do_restore
        ;;
    check)
        if disabled; then exit 0; fi
        check_target
        resolve
        ;;
    watch)
        [ $# -ge 2 ] || { echo "usage: $0 watch CONTAINER" >&2; exit 2; }
        if disabled; then exit 0; fi
        resolve
        log "watch $PCI ($NODE): waiting for container $2 (pid $$)"
        rc=$(exec docker wait "$2" 2>&1) || true  # exec: no second "gpu_minfreq.sh watch" process meanwhile
        log "watch $PCI ($NODE): container $2 ended ($rc)"
        locked do_restore
        ;;
    status)
        if [ "$RENDER" = none ]; then echo "RENDER=none: no card"; exit 0; fi
        resolve
        echo "$NODE = $PCI; GT1_MIN_MHZ=$GT1_MIN_MHZ"
        for g in gt0 gt1; do
            f=/sys/class/drm/$NODE/device/tile0/$g/freq0
            echo "$g: min $(cat "$f/min_freq") max $(cat "$f/max_freq") act $(cat "$f/act_freq") cur $(cat "$f/cur_freq") MHz"
        done
        if [ -e "$STATE" ]; then echo "record $STATE:"; sed 's/^/  /' "$STATE"; else echo "no record ($STATE)"; fi
        pgrep -af "[g]pu_minfreq.sh watch" || echo "no watcher running"
        tail -n 3 "$LOG" 2> /dev/null || true
        ;;
    *)
        sed -n '2,19p' "$0"
        exit 2
        ;;
esac
