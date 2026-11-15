#!/bin/bash
# Usage: arm_down.sh <left|right> <cm> [--dry-run]
# Shoulder-frame vertical descent, orientation unchanged, EE z >= -0.40 m.
# Reuses read_arm_now.py -> calibrated MuJoCo IK -> arm.sh goto (holds grippers).
set -euo pipefail
SIDE=${1:-}; CM=${2:-}; DRY=${3:-}
[[ $# -ge 2 && $# -le 3 && ( "$SIDE" = left || "$SIDE" = right ) && ( -z "$DRY" || "$DRY" = --dry-run ) ]] || {
  echo 'usage: arm_down.sh <left|right> <cm> [--dry-run]'; exit 2;
}
DZ=$(python3 - "$CM" <<'PY'
import math, sys
try:
    cm = float(sys.argv[1])
    assert math.isfinite(cm) and cm > 0
except (ValueError, AssertionError):
    sys.exit('cm must be finite and positive')
print(-cm / 100)
PY
)
A=/root/openpi-xpu/client/outputs/arm
SIM=ta2-sim-pipeline
docker ps --format '{{.Names}}' | grep -qx "$SIM" || { echo "IK container $SIM is not running"; exit 2; }
docker exec openpi-xpu-client bash -lc "source /opt/ros/humble/setup.bash; python3 /outputs/arm/read_arm_now.py $SIDE 2>/dev/null" \
  | tail -1 > "$A/${SIDE}_now.json"
for f in ik_offset.py "${SIDE}_fk_fit.json" "${SIDE}_now.json"; do docker cp "$A/$f" "$SIM:/tmp/$f"; done
docker exec "$SIM" /workspace/.venv/bin/python /tmp/ik_offset.py --arm "$SIDE" \
  --fit "/tmp/${SIDE}_fk_fit.json" --now "/tmp/${SIDE}_now.json" --dz "$DZ" --out "/tmp/goto_${SIDE}_down.json"
docker cp "$SIM:/tmp/goto_${SIDE}_down.json" "$A/goto_${SIDE}_down.json"
args=(); [[ -z "$DRY" ]] || args+=("$DRY")
/root/arm.sh goto --target "/outputs/arm/goto_${SIDE}_down.json" --time 8 "${args[@]}"
