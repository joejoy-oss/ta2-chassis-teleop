#!/bin/bash
set -euo pipefail
SOURCE=/data/ta2/tools/ta2_perception
STATE=/data/ta2/perception
mkdir -p "$STATE"
if [ ! -f "$STATE/config.json" ]; then cp "$SOURCE/config.json" "$STATE/config.json"; fi
"$STATE/venv/bin/python" "$SOURCE/test_perception.py"
"$STATE/venv/bin/python" "$SOURCE/calibrate.py" --config "$STATE/config.json" board --out "$STATE/board"
install -m 644 "$SOURCE/ta2-perception.service" /etc/systemd/system/ta2-perception.service
systemctl daemon-reload
systemctl enable --now ta2-perception.service
