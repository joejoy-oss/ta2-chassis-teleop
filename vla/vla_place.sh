#!/bin/bash
# Usage: vla_place.sh <tag> [seconds=60] [max_speed=1.0] [prompt=放下沙包]
#   Ask the VLA policy to put down what the gripper holds. No start-pose reset, grippers are not opened
#   by the script. Experimental: the step-4999 model was trained on grasping only and largely ignores
#   the prompt in offline tests (2026-11-15), so expect it to keep holding. Fallback: /root/arm.sh place
TAG=${1:?tag}; SEC=${2:-60}; SPD=${3:-1.0}; P=${4:-放下沙包}
RESET=0 PROMPT="$P" exec /root/vla_run.sh "$TAG" "$SEC" "$SPD"
