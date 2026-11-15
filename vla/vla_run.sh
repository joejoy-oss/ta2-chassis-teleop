#!/bin/bash
# Usage: vla_run.sh <tag> [seconds] [max_speed]   -- reset arms to training start pose, then run step-4999 VLA
TAG=${1:?tag}; SEC=${2:-120}; SPD=${3:-0.5}
/root/arm.sh start --time 6 || exit 1
TTY=-i; [ -t 0 ] && TTY=-it            # the web remote runs this without a terminal
docker exec $TTY openpi-xpu-client bash -lc "source /opt/ros/humble/setup.bash; cd /outputs/tools && \
python3 live_observation_probe.py --output-dir /outputs/${TAG}_probe | tail -1 && \
python3 guarded_grasp.py --observation /outputs/${TAG}_probe/observation.npz --output-dir /outputs/${TAG} --execute \
  --prompt 抓取沙包 --max-seconds $SEC --max-actions 5400 --max-speed $SPD 2>&1 | grep -v -E 'INFO|multicast'"
