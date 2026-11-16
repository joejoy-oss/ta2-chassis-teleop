#!/bin/bash
# Start the TA2 voice service on this Mac (run it in Terminal.app so the microphone is allowed).
# 1) reverse tunnel: B70 127.0.0.1:18096 -> this Mac 127.0.0.1:8096 (reconnects forever)
# 2) the voice service in the foreground (Ctrl-C stops both)
cd "$(dirname "$0")"
export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH
B70=${B70:-root@192.168.31.228}
pkill -f "18096:127.0.0.1:8096" 2>/dev/null
pkill -f "ta2-voice/mac_voice.py" 2>/dev/null
( while true; do
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        -R 127.0.0.1:18096:127.0.0.1:8096 "$B70"
    sleep 3
  done ) > tunnel.log 2>&1 &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null; pkill -f "18096:127.0.0.1:8096"' EXIT
"$PWD/.venv/bin/python" -u "$PWD/mac_voice.py" 2>&1 | tee -a mac_voice.log
