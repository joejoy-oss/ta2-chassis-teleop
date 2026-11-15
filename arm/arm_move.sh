#!/bin/bash
# Usage: arm_move.sh <left|right> <forward_cm> <left_cm> <up_cm> [--dry-run]
# Straight-line move of one arm's end effector in its shoulder frame (x forward, y left, z up; confirmed by the
# MuJoCo fit: rotation < 0.4 deg, shoulder offsets +/-0.22 m), orientation unchanged, grippers held.
# Each component -15..15 cm. Calibrated MuJoCo IK (ik_offset.py) refuses unreachable targets, joint-limit hits
# and paths that leave the straight line; downward moves stop at EE z = -0.40 m.
# Reuses read_arm_now.py -> ik_offset.py (ta2-sim-pipeline) -> arm.sh goto.
set -euo pipefail
SIDE=${1:-}; DRY=${5:-}
[[ $# -ge 4 && $# -le 5 && ( "$SIDE" = left || "$SIDE" = right ) && ( -z "$DRY" || "$DRY" = --dry-run ) ]] || {
  echo 'usage: arm_move.sh <left|right> <forward_cm> <left_cm> <up_cm> [--dry-run]'; exit 2;
}
read -r DX DY DZ < <(python3 - "$2" "$3" "$4" <<'PY'
import math, sys
try:
    v = [float(x) for x in sys.argv[1:4]]
    assert all(math.isfinite(x) and abs(x) <= 15 for x in v)
except (ValueError, AssertionError):
    sys.exit('each distance must be a number from -15 to 15 cm')
if max(abs(x) for x in v) < 0.1:
    sys.exit('nothing to do: all distances are below 0.1 cm')
print(*(x / 100 for x in v))
PY
)
A=/root/openpi-xpu/client/outputs/arm
SIM=ta2-sim-pipeline
docker ps --format '{{.Names}}' | grep -qx "$SIM" || { echo "IK container $SIM is not running"; exit 2; }
# read_arm_now.py prints its reason on stdout and exits 1 when the arm stream is stale; say so instead of
# exiting silently under pipefail.
NOW=$(docker exec openpi-xpu-client bash -lc "source /opt/ros/humble/setup.bash; python3 /outputs/arm/read_arm_now.py $SIDE 2>/dev/null" \
  | tail -1) || true
python3 -c 'import json, sys; d = json.loads(sys.argv[1]); assert isinstance(d, dict) and "q" in d and "ee" in d' "$NOW" 2>/dev/null || {
  echo "cannot read the current $SIDE arm state: ${NOW:-no output} (is /${SIDE}_arm/joint_states publishing?)"; exit 1;
}
printf '%s\n' "$NOW" > "$A/${SIDE}_now.json"
for f in ik_offset.py "${SIDE}_fk_fit.json" "${SIDE}_now.json"; do docker cp "$A/$f" "$SIM:/tmp/$f"; done
docker exec "$SIM" /workspace/.venv/bin/python /tmp/ik_offset.py --arm "$SIDE" \
  --fit "/tmp/${SIDE}_fk_fit.json" --now "/tmp/${SIDE}_now.json" --dx "$DX" --dy "$DY" --dz "$DZ" --out "/tmp/goto_${SIDE}_move.json"
docker cp "$SIM:/tmp/goto_${SIDE}_move.json" "$A/goto_${SIDE}_move.json"
args=(); [[ -z "$DRY" ]] || args+=("$DRY")
/root/arm.sh goto --target "/outputs/arm/goto_${SIDE}_move.json" --time 8 "${args[@]}"
