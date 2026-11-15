#!/bin/bash
# Usage: arm_down.sh <left|right> <cm> [--dry-run]
# Shoulder-frame vertical descent, orientation unchanged, EE z >= -0.40 m.
# Reuses read_arm_now.py -> calibrated MuJoCo IK -> arm.sh goto (holds grippers).
set -euo pipefail
SIDE=${1:-}; CM=${2:-}; DRY=${3:-}
[[ $# -ge 2 && $# -le 3 && ( "$SIDE" = left || "$SIDE" = right ) && ( -z "$DRY" || "$DRY" = --dry-run ) ]] || {
  echo 'usage: arm_down.sh <left|right> <cm> [--dry-run]'; exit 2;
}
python3 - "$CM" <<'PY'
import math, sys
try:
    cm = float(sys.argv[1])
    assert math.isfinite(cm) and cm > 0
except (ValueError, AssertionError):
    sys.exit('cm must be finite and positive')
PY
# Same IK path as every other straight-line move: forward 0, left 0, up -cm.
exec "$(dirname "$0")/arm_move.sh" "$SIDE" 0 0 "$(python3 -c "import sys; print(-float(sys.argv[1]))" "$CM")" ${DRY:+"$DRY"}
