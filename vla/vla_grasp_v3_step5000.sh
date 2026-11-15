#!/usr/bin/env bash
# Current web grasp model: v3 checkpoint 5000, prompt 抓取红色沙包, policy port 8002.
# Run only when ready for robot motion: resets both arms and opens the grippers first.
# Usage: vla_grasp_v3_step5000.sh <unique_tag> [seconds=60] [max_speed=0.5]
set -euo pipefail
TAG=${1:?provide a unique run tag}; SEC=${2:-60}; SPD=${3:-0.5}
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid run tag" >&2; exit 2; }
python3 - "$SEC" "$SPD" <<'CHECK'
import math, re, sys
for value in sys.argv[1:]:
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) or not math.isfinite(float(value)) or float(value) <= 0:
        sys.exit("seconds and max_speed must be positive numbers")
CHECK
[ "$(docker inspect -f '{{.State.Running}}' grasp-v3-step5000-policy 2>/dev/null)" = true ] || {
  echo "New v3 step-5000 policy container is not running" >&2; exit 1;
}
# Check the listener before resetting the arms. Sensor/mode/policy checks remain in guarded_grasp.py.
python3 - <<'CHECK'
import socket, sys
try:
    with socket.create_connection(("127.0.0.1", 8002), timeout=3):
        pass
except OSError as exc:
    sys.exit(f"New v3 step-5000 policy port 8002 is unavailable: {exc}")
CHECK
exec env PORT=8002 PROMPT=抓取红色沙包 RESET=1 /root/vla_run.sh "$TAG" "$SEC" "$SPD"
