#!/bin/bash
# Usage: [PROMPT=..] [RESET=0] [PORT=8001] vla_run.sh <tag> [seconds=120] [max_speed=0.5]
#   Reset arms to the training start pose (grippers open), then run the VLA policy on the robot.
#   PROMPT  instruction sent to the policy (default 抓取沙包)
#   RESET=0 skip the start-pose reset (keeps whatever the grippers hold; used by vla_place.sh)
#   PORT    policy server port on B70 (default 8000 = sandbag-step4999-policy)
# Stops the camera web viewer (ta2_vision_server.py) first if it holds the robot video port 8890.
# Exit code: 0 = ran to the time limit, non-zero = aborted or failed (see /root/openpi-xpu/client/outputs/<tag>/report.json).
set -o pipefail
TAG=${1:?tag}; SEC=${2:-120}; SPD=${3:-0.5}
PROMPT=${PROMPT:-抓取沙包}; RESET=${RESET:-1}; PORT=${PORT:-8000}
# The robot pushes one RTP video stream to UDP 8890 and only one program can receive it. The camera web
# viewer (ta2_vision_server.py, :8097) also listens there: stop it (it is not restarted afterwards).
if pgrep -f ta2_vision_server.py >/dev/null; then
  echo "stopping the camera web viewer (ta2_vision_server.py) to free UDP 8890 for the VLA"
  /data/ta2/tools/ta2_vision_ops.sh stop >/dev/null 2>&1
  for i in $(seq 1 10); do ss -lunH sport = :8890 | grep -q . || break; sleep 0.5; done
fi
if ss -lunpH sport = :8890 | grep -q .; then
  echo "UDP 8890 (robot video) is held by another program, VLA cannot get images:"; ss -lunpH sport = :8890; exit 1
fi
if [ "$RESET" = 1 ]; then /root/arm.sh start --time 6 || exit 1; fi
TTY=-i; [ -t 0 ] && TTY=-it            # the web remote runs this without a terminal
docker exec $TTY openpi-xpu-client bash -lc "source /opt/ros/humble/setup.bash; cd /outputs/tools && \
python3 live_observation_probe.py --output-dir /outputs/${TAG}_probe | tail -1 && \
python3 guarded_grasp.py --observation /outputs/${TAG}_probe/observation.npz --output-dir /outputs/${TAG} --execute \
  --prompt '$PROMPT' --port $PORT --max-seconds $SEC --max-actions 5400 --max-speed $SPD 2>&1 | grep -v -E 'INFO|multicast'; exit \${PIPESTATUS[0]}"
