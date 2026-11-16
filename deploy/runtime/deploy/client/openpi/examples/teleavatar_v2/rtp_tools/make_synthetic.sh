#!/bin/bash
# Synthetic stand-ins for the robot's RTP/H.265 camera stream, as .rtpcap files for rtp_replay.py.
#
# Usage, inside the client image (openpi-xpu-client) with the openpi tree at /app:
#   bash /app/examples/teleavatar_v2/rtp_tools/make_synthetic.sh OUT_DIR [WORK_DIR]
# Writes per variant OUT_DIR/synthetic_{nowpp,wpp}_{10,20,30}mbps.rtpcap, its stream_info.py summary (.json) and
# sha256sum line (.rtpcap.sha256); WORK_DIR (default OUT_DIR) holds a temporary RFC 4571 stream per variant.
# Exit status 1 if any variant fails its checks. Environment: FRAMES (default 2700 = 60 s), VARIANTS (default
# "nowpp:10 nowpp:20 nowpp:30 wpp:10 wpp:20 wpp:30", mode:Mbit/s), POOL_THREADS (x265 pool, default 10),
# FRAME_THREADS (x265 frame threads, default 4).
#
# The picture is the robot's composite: width 1280, height 2720 (the README's "2720x1280" is H x W: the six views of
# rtp_video_interface.TELEAVATAR_SPLIT_REGIONS are then the 960x960 head eyes and 640x400 wrist eyes it lists), 45 fps,
# 8-bit 4:2:0. Each view is a moving videotestsrc pattern, and the whole frame gets snow (random noise) blended in at
# alpha 0.15 so the encoder really spends the bitrate (about QP 35 at 20 Mbit/s); neither flat nor pure noise.
#
# Encoding is offline (as fast as it goes), x265 3.5 like a camera's hardware encoder: one slice per picture, no B
# frames, one reference, closed GOP with an IDR every 45 frames (1 s), no scene-cut keyframes, CTU 32, SAO and
# deblocking on, no weighted prediction, ABR at the target with VBV maxrate = target and a 1 s buffer, no lookahead
# (tune zerolatency). Only frame-threads is raised from zerolatency's 1 to make encoding faster; the bitstream still
# has one slice per picture, and WPP only in the wpp variants. repeat-headers=1 puts VPS/SPS/PPS before every IDR;
# x265enc also prepends one copy at the very start, so the first IDR carries two (stream_info reports it).
#
# Packetizing is GStreamer's rtph265pay (MTU 1400, payload type 96; parameter sets as single NAL unit packets, slices
# as fragmentation units), not a packetizer of our own: the analysis tools are then checked against an independent
# implementation, the one the receiver's rtph265depay pairs with. rtpstreampay (RFC 4571 framing) hands the packets to
# rtpcap.py without gi. Sequence numbers start at 60000 and RTP timestamps at 4294000000 so that both wrap inside every
# fixture. Send times are synthesized (rtpcap.synthesize_send_times): access unit i at i/45 s, its packets back to
# back at 1 Gbit/s.
set -euo pipefail

OUT=${1:?usage: make_synthetic.sh OUT_DIR [WORK_DIR]}
WORK=${2:-$OUT}
HERE=$(cd "$(dirname "$0")" && pwd)
FRAMES=${FRAMES:-2700}
VARIANTS=${VARIANTS:-"nowpp:10 nowpp:20 nowpp:30 wpp:10 wpp:20 wpp:30"}
POOL_THREADS=${POOL_THREADS:-10}
FRAME_THREADS=${FRAME_THREADS:-4}
WIDTH=1280
HEIGHT=2720
FPS=45
SNOW_ALPHA=0.15
LINE_RATE_BPS=1e9
mkdir -p "$OUT" "$WORK"

raw() {  # caps of a raw video source of $1 x $2 pixels
    echo "video/x-raw,format=I420,width=$1,height=$2,framerate=$FPS/1"
}
# The six views at their TELEAVATAR_SPLIT_REGIONS positions (head eyes 960x960 with 160-pixel black bars at the sides,
# wrist eyes 640x400 in a 2x2 grid below), then the snow layer over everything.
SOURCE="compositor name=mix background=black \
    sink_0::xpos=160 sink_0::ypos=0 sink_1::xpos=160 sink_1::ypos=960 \
    sink_2::xpos=0 sink_2::ypos=1920 sink_3::xpos=640 sink_3::ypos=1920 \
    sink_4::xpos=0 sink_4::ypos=2320 sink_5::xpos=640 sink_5::ypos=2320 \
    sink_6::xpos=0 sink_6::ypos=0 sink_6::alpha=$SNOW_ALPHA \
    ! $(raw $WIDTH $HEIGHT) ! ENCODE_AND_PACK \
    videotestsrc num-buffers=$FRAMES pattern=smpte horizontal-speed=6 ! $(raw 960 960) ! mix.sink_0 \
    videotestsrc num-buffers=$FRAMES pattern=pinwheel horizontal-speed=4 ! $(raw 960 960) ! mix.sink_1 \
    videotestsrc num-buffers=$FRAMES pattern=colors horizontal-speed=3 ! $(raw 640 400) ! mix.sink_2 \
    videotestsrc num-buffers=$FRAMES pattern=ball motion=sweep ! $(raw 640 400) ! mix.sink_3 \
    videotestsrc num-buffers=$FRAMES pattern=checkers-8 horizontal-speed=2 ! $(raw 640 400) ! mix.sink_4 \
    videotestsrc num-buffers=$FRAMES pattern=circular horizontal-speed=5 ! $(raw 640 400) ! mix.sink_5 \
    videotestsrc num-buffers=$FRAMES pattern=snow ! $(raw $WIDTH $HEIGHT) ! mix.sink_6"

failed=0
index=0
for variant in $VARIANTS; do
    mode=${variant%%:*}
    mbps=${variant##*:}
    index=$((index + 1))
    case $mode in
        nowpp) wpp=0 ;;
        wpp) wpp=1 ;;
        *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    name=synthetic_${mode}_${mbps}mbps
    kbps=$((mbps * 1000))
    ssrc=$(printf "0x5eed%04x" "$index")
    x265_options="ctu=32:keyint=45:min-keyint=45:scenecut=0:open-gop=0:bframes=0:ref=1:weightp=0:wpp=$wpp:slices=1"
    x265_options+=":sao=1:vbv-maxrate=$kbps:vbv-bufsize=$kbps:frame-threads=$FRAME_THREADS:pools=$POOL_THREADS"
    x265_options+=":info=0:repeat-headers=1"
    x265enc="x265enc bitrate=$kbps key-int-max=45 speed-preset=veryfast tune=zerolatency log-level=info"
    pay="rtph265pay mtu=1400 pt=96 config-interval=0 ssrc=$ssrc seqnum-offset=60000 timestamp-offset=4294000000"
    tail="$x265enc option-string=$x265_options ! h265parse config-interval=-1"
    tail+=" ! video/x-h265,stream-format=byte-stream,alignment=au ! $pay ! rtpstreampay"
    tail+=" ! filesink location=$WORK/$name.rtpstream"
    echo "== $name: $(date '+%F %T %Z')"
    echo "gst-launch-1.0 -q ${SOURCE/ENCODE_AND_PACK/$tail}"
    start=$(date +%s)
    # x265 logs to stderr; keep its summary lines (the set_mempolicy lines are harmless NUMA noise in a container).
    eval "gst-launch-1.0 -q ${SOURCE/ENCODE_AND_PACK/$tail}" 2>&1 | grep -v set_mempolicy | tee "$WORK/$name.x265.log"
    echo "encoded in $(($(date +%s) - start)) s"
    x265_version=$(sed -n 's/.*HEVC encoder version //p' "$WORK/$name.x265.log" | head -1)
    python3 -B "$HERE/rtpcap.py" from-rfc4571 "$WORK/$name.rtpstream" "$OUT/$name.rtpcap" \
        --fps "$FPS" --line-rate-bps "$LINE_RATE_BPS" \
        --meta "generator=make_synthetic.sh" --meta "name=$name" --meta "frames=$FRAMES" \
        --meta "width=$WIDTH" --meta "height=$HEIGHT" --meta "fps=$FPS" --meta "wpp=$wpp" \
        --meta "bitrate_kbps=$kbps" --meta "snow_alpha=$SNOW_ALPHA" --meta "x265_version=$x265_version" \
        --meta "x265enc=$x265enc" --meta "x265_options=$x265_options" --meta "rtph265pay=$pay" \
        --meta "layout=head eyes 960x960 at (160,0),(160,960); wrist eyes 640x400 at (0,1920),(640,1920),(0,2320),(640,2320); snow over all"
    rm -f "$WORK/$name.rtpstream" "$WORK/$name.x265.log"
    python3 -B "$HERE/stream_info.py" "$OUT/$name.rtpcap" --json "$OUT/$name.json"
    (cd "$OUT" && sha256sum "$name.rtpcap" > "$name.rtpcap.sha256" && cat "$name.rtpcap.sha256")
    # Does the fixture have what was asked for?
    if ! python3 -B - "$OUT/$name.json" "$wpp" "$mbps" "$FRAMES" "$WIDTH" "$HEIGHT" "$FPS" <<'EOF'
import json
import sys

path, wpp, mbps, frames, width, height, fps = sys.argv[1:]
info = json.load(open(path))
s, rtp = info["stream"], info["rtp"]
checks = {
    f"{width}x{height}, Main, 8-bit 4:2:0": (s["width"], s["height"], s["profile"], s["bit_depth_luma"], s["chroma_format"])
    == (int(width), int(height), "Main", 8, "4:2:0"),
    f"{frames} access units, {fps} fps from the RTP timestamps": s["access_units"] == int(frames)
    and rtp["access_units"] == int(frames)
    and rtp["fps_from_timestamps"] == float(fps),
    f"WPP {wpp == '1'}, no tiles": s["wpp"] == (wpp == "1") and s["tiles"] is None,
    "one slice per picture": s["slice_segments_per_au"] == {"1": int(frames)} and s["dependent_slice_segments"] == 0,
    "no B slices, no reordering": s["b_slices"] == 0 and s["reordered_pictures"] == 0
    and s["sps_max_num_reorder_pics"] == [0],
    "an IDR every 45 frames from the first": s["idr_count"] == s["irap_count"] == -(-int(frames) // 45)
    and s["first_irap_au"] == 0 and set(s["irap_interval_frames"]) <= {"45"},
    "VPS/SPS/PPS in every IDR access unit": all("0" not in key.split("/") for key in s["parameter_sets"]["per_irap_au_vps_sps_pps"])
    and s["parameter_sets"]["in_other_aus"] == 0,
    "CTB 32, SAO on, deblocking on": s["ctb_size"] == 32 and s["sao"] and not s["deblocking_disabled_in_pps"],
    f"ES bitrate within 5% of {mbps} Mbit/s": abs(s["es_bitrate_mbps"] - float(mbps)) <= 0.05 * float(mbps),
    "payload type 96, packets <= 1400 bytes": rtp["payload_types"] == {"96": rtp["packets"]}
    and rtp["packet_bytes"]["max"] <= 1400,
    "no sequence gaps; sequence numbers wrap (when there are more than 5536 packets)": rtp["seq_missing"] == 0
    and rtp["seq_out_of_order"] == 0
    and (rtp["seq_wraps"] >= 1 or rtp["packets"] <= 65536 - 60000),
    "RTP timestamps wrap (when the fixture is longer than 10.7 s)": rtp["timestamp_last"] < rtp["timestamp_first"]
    or int(frames) * 90000 // int(fps) < 2**32 - 4294000000,
    "depacketized and parsed without errors": info["depacketizer"]["fu_dropped"] == 0 and not s["errors"]
    and info["depacketizer"]["access_units_without_marker"] == 0,
}
for label, ok in checks.items():
    print(("OK   " if ok else "FAIL ") + label)
print(f"es_bitrate_mbps {s['es_bitrate_mbps']} packets {rtp['packets']} packets_sha256 {info['packets_sha256']}")
sys.exit(0 if all(checks.values()) else 1)
EOF
    then
        failed=1
    fi
done
ls -l "$OUT"
exit $failed
