#!/bin/bash
set -euo pipefail
STATE=/data/ta2/perception
SERVER=/data/ta2/tools/ta2_vision_server.py
ENVFILE=/data/ta2/vision/env
EXPECTED=a37373842aef5a9db633610d826f09be572e9cfc4b5c3a31552b5fdf5897433c
if [ "$(sha256sum "$SERVER" | cut -d' ' -f1)" != "$EXPECTED" ]; then
  echo 'Existing vision server changed; inspect before replacing.' >&2; exit 2
fi
BACKUP="$STATE/vision-backup-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP"
cp "$SERVER" "$BACKUP/ta2_vision_server.py"
cp "$ENVFILE" "$BACKUP/env"
printf '%s\n' "$BACKUP" > "$STATE/vision-backup-path"
python3 -m py_compile "$STATE/ta2_vision_server.new.py"
MARK=/data/ta2/vision/.maintenance
OWN_MARK=0
if [ ! -f "$MARK" ]; then touch "$MARK"; OWN_MARK=1; fi
rollback() {
  cp "$BACKUP/ta2_vision_server.py" "$SERVER"
  cp "$BACKUP/env" "$ENVFILE"
  systemctl restart ta2-vision.service
  if [ "$OWN_MARK" = 1 ]; then rm -f "$MARK"; fi
}
trap rollback ERR
systemctl stop ta2-vision.service
cp "$STATE/ta2_vision_server.new.py" "$SERVER"
python3 - <<'PY'
from pathlib import Path
p=Path('/data/ta2/vision/env')
lines=p.read_text().splitlines()
values={'TA2V_WIDTH':'1280','TA2V_QUALITY':'2','TA2V_FPS':'12'}
result=[]
for line in lines:
    k=line.split('=',1)[0]
    result.append(k+'='+values.pop(k) if k in values else line)
result.extend(k+'='+v for k,v in values.items())
p.write_text('\n'.join(result)+'\n')
PY
systemctl start ta2-vision.service
python3 - <<'PY'
import urllib.request,json,time
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(30):
    try:
        with opener.open('http://127.0.0.1:8097/api/frame',timeout=2) as r: p=json.load(r)
        if p['width']==1280 and p['height']==2720 and time.time()-p['received_at'] < 1:
            print('Verified original-size live frame',p['seq'],p['width'],p['height']);break
    except Exception: pass
    time.sleep(1)
else: raise SystemExit('No fresh original-size frame; rolling back')
PY
trap - ERR
if [ "$OWN_MARK" = 1 ]; then rm -f "$MARK"; fi
echo "Backup: $BACKUP"
