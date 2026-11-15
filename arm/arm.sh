#!/usr/bin/env bash
# TA2 arms, standalone (separate from the chassis scripts). Run on B70.
#
#   ./arm.sh check            check mode/state and print the plan, send nothing
#   ./arm.sh raise [--time 8] both arms to the official zero.py "raised" pose
#   ./arm.sh lower [--time 8] back to the pose saved by the last raise
#   ./arm.sh start [--time 8] both arms to the VLA training start pose (arm/train_start_pose.json)
#   ./arm.sh start_hold [--time 8] like start, but a gripper that is holding something stays closed
#   ./arm.sh goto --target /outputs/arm/<file>.json [--time 8]  move to given joint angles; holding grippers stay closed
#   ./arm.sh open_right       open the right gripper (full opening force); arms hold their pose
#   ./arm.sh close_right      close the right gripper (full closing force); arms hold their pose
#   ./arm.sh open_left / close_left  left gripper only; arms hold their pose
#   ./arm.sh place [--time 8] keep grippers closed, move to the training start pose (low over the table), then open them
#
# Needs: run mode API, both arms enabled and in JOINT control mode (dev config -> Save -> Apply -> reboot).
# From another machine use -t so Ctrl+C reaches the robot:  ssh -t root@192.168.31.228 ./arm.sh raise
set -e
DEPLOY=/data/ta2/software/deploy
ROBOT=192.168.31.218
SCRIPT=/outputs/arm/arm_pose.py          # container path = host /root/openpi-xpu/client/outputs/arm/
cd "$DEPLOY"
export OPENPI_DIR=$DEPLOY/client/openpi
R=client/run_client.sh
if ! docker ps --format '{{.Names}}' | grep -qx openpi-xpu-client; then
    echo "Starting the client container (robot $ROBOT)..."
    ZENOH_ENDPOINT=tcp/$ROBOT:9000 $R start > /dev/null
    sleep 8
fi
case ${1:-} in
    check)       $R exec python3 $SCRIPT raise --dry-run ;;
    raise|lower|start|place|start_hold|goto|open_right|close_right|open_left|close_left) $R exec python3 $SCRIPT "$@" ;;
    *) sed -n '2,15p' "$(readlink -f "$0")"; exit 2 ;;
esac
