#!/usr/bin/env bash
# Tunnel (B): run on a third machine that reaches both a machine in the robot's LAN (TUNNEL_HOST) and the B70, when
# neither of those may be changed. Two hops: -L to the robot's zenoh router through TUNNEL_HOST, then -R of that port
# to 19000 on the B70's loopback. Both stay in the foreground of this script; Ctrl-C closes them.
#   ROBOT_IP=192.168.1.40 TUNNEL_HOST="-p 2222 ubuntu@127.0.0.1" B70="-p 2224 root@127.0.0.1" client/tunnel_b.sh
# TUNNEL_HOST and B70 are ssh arguments (port and login), here the forwarded ports this project's machines use.
set -euo pipefail
ROBOT_IP="${ROBOT_IP:?set ROBOT_IP to the robot address in its LAN}"
TUNNEL_HOST="${TUNNEL_HOST:?set TUNNEL_HOST, e.g. \"-p 2222 ubuntu@127.0.0.1\"}"
B70="${B70:-root@10.1.100.166}"
MIDDLE_PORT="${MIDDLE_PORT:-19001}"
opts=(-N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
# shellcheck disable=SC2086  # TUNNEL_HOST and B70 hold several ssh arguments
ssh "${opts[@]}" -L "127.0.0.1:${MIDDLE_PORT}:${ROBOT_IP}:9000" $TUNNEL_HOST &
first=$!
trap 'kill $first 2> /dev/null' EXIT
sleep 2
# shellcheck disable=SC2086
ssh "${opts[@]}" -R "19000:127.0.0.1:${MIDDLE_PORT}" $B70
