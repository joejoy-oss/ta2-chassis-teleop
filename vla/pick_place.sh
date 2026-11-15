#!/bin/bash
# Usage: pick_place.sh <tag> [place=vla|script] [grasp_seconds=60] [place_seconds=60]
#   1. grasp: vla_run.sh (start pose, then the policy with 抓取沙包)
#   2. place: vla  -> vla_place.sh (policy with 放下沙包, no reset)
#             script -> arm.sh place (grippers held closed, move low over the table, then open)
#   Stops after step 1 if the grasp run was aborted (non-zero exit). Ctrl+C / 急停 stop any step.
TAG=${1:?tag}; MODE=${2:-vla}; G=${3:-60}; P=${4:-60}
case $MODE in vla|script) ;; *) echo "place must be vla or script"; exit 2;; esac
echo "== 1/2 grasp ($G s)"
/root/vla_run.sh "${TAG}_grasp" "$G" 1.0 || { echo "grasp run did not finish normally: not placing"; exit 1; }
echo "== 2/2 place ($MODE)"
if [ "$MODE" = vla ]; then /root/vla_place.sh "${TAG}_place" "$P" 1.0; else /root/arm.sh place --time 8; fi
