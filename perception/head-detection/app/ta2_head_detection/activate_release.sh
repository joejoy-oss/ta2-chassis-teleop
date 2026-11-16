#!/usr/bin/env bash
set -eu

root=/data/ta2/head-detection
release="$1"
python="$root/venv-cpu/bin/python"
name=ta2-head-detection.service
unit="/etc/systemd/system/$name"
"$python" -m py_compile "$release/ta2_head_detection/"*.py "$release/ta2_grasp/frames.py"
test -r "$root/models/yolo26n.pt"
previous=$(readlink "$root/app" || true)
if test -f "$unit"; then cp -a "$unit" "$release/previous.service"; fi

restore() {
  if test -n "$previous"; then
    ln -s "$previous" "$release/previous-app"
    mv -Tf "$release/previous-app" "$root/app"
  elif test "$(readlink "$root/app")" = "$release"; then
    unlink "$root/app"
  fi
  if test -f "$release/previous.service"; then
    install -m 644 "$release/previous.service" "$unit"
    systemctl daemon-reload
    systemctl restart "$name" || true
  else
    systemctl disable --now "$name" || true
    unlink "$unit"
    systemctl daemon-reload
  fi
}

ln -s "$release" "$release/app-link"
mv -Tf "$release/app-link" "$root/app"
install -m 644 "$release/ta2_head_detection/$name" "$unit"
systemctl daemon-reload
if systemctl restart "$name" && "$python" - <<'PY'
import json, time, urllib.request
for _ in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8104/api/state', timeout=1) as response:
            assert len(json.load(response)['classes']) == 4
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit('Workbench never became ready')
PY
then
  systemctl enable "$name"
else
  journalctl -u "$name" -n 20 --no-pager || true
  restore
  exit 1
fi
