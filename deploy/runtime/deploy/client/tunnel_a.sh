#!/usr/bin/env bash
# Tunnel (A): run on a machine in the robot's LAN that can ssh to the B70. Forwards the robot's zenoh router
# (ROBOT_IP:9000) to port 19000 on the B70's loopback, where the client container's bridge connects
# (run_client.sh, ZENOH_ENDPOINT=tcp/127.0.0.1:19000). autossh reopens it when the connection drops.
#   ROBOT_IP=192.168.1.40 client/tunnel_a.sh          (foreground; Ctrl-C closes the tunnel)
# One-time setup: a key for this forward only, accepted by the B70's root for nothing else:
#   ssh-keygen -t ed25519 -f ~/.ssh/b70_tunnel -N '' -C b70-tunnel        (on this machine)
#   restrict,port-forwarding,permitlisten="localhost:19000",permitopen="127.0.0.1:1",command="/bin/false" ssh-ed25519 AAAA... b70-tunnel
#                                                       (one line in /root/.ssh/authorized_keys on the B70)
#   and the B70's host key in this machine's known_hosts (checked against its fingerprint), so autossh never asks.
# restrict alone would still let the key run commands as root: command="/bin/false" stops that (-N opens no session,
# so the forward does not need one); permitopen="127.0.0.1:1" leaves no useful -L forward (an authorized_keys
# permitopen has no "none"); permitlisten allows this -R only. The forward's own traffic is not a permitopen matter.
# The -R spec names no bind address: ssh then asks for "localhost", which is what permitlisten allows (an explicit
# 127.0.0.1 would not match it), and the B70's sshd (GatewayPorts no) binds the loopback only.
set -euo pipefail
ROBOT_IP="${ROBOT_IP:?set ROBOT_IP to the robot address in its LAN}"
B70="${B70:-root@10.1.100.166}"
KEY="${KEY:-$HOME/.ssh/b70_tunnel}"
exec autossh -M 0 -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o IdentitiesOnly=yes -i "$KEY" -R "19000:${ROBOT_IP}:9000" "$B70"
