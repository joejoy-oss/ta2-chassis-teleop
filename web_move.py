#!/usr/bin/env python3
"""Tiny web remote for the TA2 chassis. Runs on B70.

Start:  nohup python3 /root/web_move.py > /root/web_move.log 2>&1 &
Open:   http://192.168.31.228:8088  (any browser on the robot's router network)

Click a direction (release within 0.3 s): move 10 cm, or turn 15 deg for CCW/CW.
Press and hold: keep moving at 0.05 m/s (turning at 0.2 rad/s) until release (dead-man: stops
0.4 s after the browser stops sending heartbeats; a single hold is capped at 10 s).

Trajectory: Start / Stop recording, Save (CSV download + copy in /root/ta2_trails/), Reset
(odometry origin = current pose, trail cleared). CSV columns: t_s, x_m (right), y_m (forward),
heading_deg (CCW), all relative to the origin.

Paths: CSV files of that format in /root/ta2_paths/ are listed on the page (Load adds more from
the browser). Run follows one path from the current pose; Run all follows them in name order,
each starting where the previous one ended. At every corner of a path (turn > 45 deg) the robot
stands still for CORNER_PAUSE seconds. A path stops if the page stops polling for 1 s.

Arms: "Raise arms" / "Lower arms" / "Start pose" run arm/arm_pose.py (standalone, /outputs/arm/ in the container)
as a separate process; its output goes to the log. Arms and chassis never move at the same time.
STOP also interrupts the arm script, which then holds where it is and pauses the FSM.

Motion is done by teleop.py, a long-running ROS node inside the openpi-xpu-client
container, fed over stdin. If this server exits, teleop.py sees EOF, sends zero and exits.
"""
import collections
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8088"))
DEPLOY = os.environ.get("DEPLOY", "/data/ta2/software/deploy")
ROBOT = "192.168.31.218"
CONTAINER = "openpi-xpu-client"
TELEOP = ["docker", "exec", "-i", CONTAINER, "/usr/local/bin/openpi-client-entrypoint.sh",
          "python3", "-u", "/outputs/tools/teleop.py"]
if os.environ.get("TELEOP_CMD"):              # for offline testing
    TELEOP = os.environ["TELEOP_CMD"].split()
DIRS = ("forward", "back", "left", "right", "ccw", "cw")
TRAIL_DIR = os.environ.get("TRAIL_DIR", "/root/ta2_trails")
PATH_DIR = os.environ.get("PATH_DIR", "/root/ta2_paths")
CORNER_PAUSE = float(os.environ.get("CORNER_PAUSE", "10"))   # seconds to stand still at each corner
ARM_CMD = os.environ.get("ARM_CMD", f"docker exec {CONTAINER} /usr/local/bin/openpi-client-entrypoint.sh "
                                    "python3 -u /outputs/arm/arm_pose.py").split()
ARM_STOP = os.environ.get("ARM_STOP", f"docker exec {CONTAINER} pkill -INT -f arm_pose.py").split()
ARM_TIME = {"raise": 8, "lower": 15, "start": 10}    # seconds; lower is slower on purpose
# start = VLA training start pose (arm/train_start_pose.json, first frame of a training episode)
paths = {}                  # name -> {"points": [[x, y, th], ...], "length": m, "end": [dx, dy]}
seq = {"names": [], "i": -1, "abort": False}
path_done = threading.Event()
path_result = {"why": None}
last_poll = [0.0]

state = {"connected": False, "moving": None, "mode_ok": None, "last": "starting", "clicks": 0, "holds": 0,
         "pose": [0.0, 0.0, 0.0], "epoch": int(time.time()),    # epoch changes on restart and on trail reset
         "recording": False, "rec_t0": None, "active_path": None, "path_anchor": [0.0, 0.0, 0.0],
         "arm": None, "arm_last": "arms: not moved yet"}
trail = []                  # [[x, y, th, t], ...]; metres / rad / s; x = right, y = forward of the origin
TRAIL_MAX = 20000
log = collections.deque(maxlen=60)
lock = threading.Lock()
proc = {"p": None}


def note(msg):
    log.append(time.strftime("%H:%M:%S ") + msg)


def parse_path_csv(text):
    pts = []
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    head = [h.strip() for h in lines[0].split(",")]
    ix, iy = head.index("x_m"), head.index("y_m")
    ih = head.index("heading_deg") if "heading_deg" in head else None
    for ln in lines[1:]:
        c = ln.split(",")
        th = float(c[ih]) * 0.0174532925 if ih is not None else 0.0
        pts.append([float(c[ix]), float(c[iy]), th])
    if len(pts) < 2:
        raise ValueError("needs at least 2 points")
    length = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:]))
    return {"points": pts, "length": round(length, 3),
            "end": [round(pts[-1][0] - pts[0][0], 3), round(pts[-1][1] - pts[0][1], 3)]}


def clean_name(n):
    n = os.path.basename(n).rsplit(".", 1)[0]
    n = "".join(ch for ch in n if ch.isalnum() or ch in "-_")[:40]
    return n or "path"


def load_path_dir():
    if not os.path.isdir(PATH_DIR):
        return
    for f in sorted(os.listdir(PATH_DIR)):
        if f.lower().endswith(".csv"):
            try:
                with open(os.path.join(PATH_DIR, f)) as fh:
                    paths[clean_name(f)] = parse_path_csv(fh.read())
            except (ValueError, OSError, IndexError) as e:
                note(f"skip {f}: {e}")
    note(f"loaded {len(paths)} paths from {PATH_DIR}")


def run_sequence(names):
    seq.update(names=names, i=-1, abort=False)
    note("run all: " + " -> ".join(names))
    for i, n in enumerate(names):
        if seq["abort"]:
            break
        seq["i"] = i
        path_done.clear()
        path_result["why"] = None
        send("path_cmd", n)
        if not path_done.wait(timeout=3 * paths[n]["length"] / 0.05 + 30 + 10 * CORNER_PAUSE):
            send("stop")
            note(f"run all: {n} did not finish, stopped")
            break
        if path_result["why"] != "path done":
            note(f"run all: stopped at {n} ({path_result['why']})")
            break
        time.sleep(1.0)
    else:
        note("run all: finished")
    seq.update(names=[], i=-1)


def run_arm(action):
    try:
        ensure_container()
        p = subprocess.Popen(ARM_CMD + [action, "--time", str(ARM_TIME[action])], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        last = ""
        for line in p.stdout:
            line = line.rstrip()
            if line and "multicast" not in line and not line.startswith("Current mode"):
                note("arm: " + line)
                last = line
        rc = p.wait()
        state["arm_last"] = f"arms {action}: " + ("done" if rc == 0 else f"stopped ({last})")
    except Exception as e:  # noqa: BLE001
        state["arm_last"] = f"arms {action}: error {e}"
        note(state["arm_last"])
    finally:
        state["arm"] = None


def alive_loop():
    while True:
        time.sleep(0.25)
        if state["active_path"] and time.monotonic() - last_poll[0] < 1.0:
            send("alive")


def add_point(x, y, th, force=False):
    moved = not trail or (x - trail[-1][0]) ** 2 + (y - trail[-1][1]) ** 2 > 0.005 ** 2 \
        or abs(th - trail[-1][2]) > 0.035
    if (force or moved) and len(trail) < TRAIL_MAX:
        trail.append([x, y, th, round(time.monotonic() - state["rec_t0"], 2)])


def trail_csv():
    rows = ["t_s,x_m,y_m,heading_deg"]
    rows += [f"{t:.2f},{x:.4f},{y:.4f},{th * 57.29578:.2f}" for x, y, th, t in trail]
    return "\n".join(rows) + "\n"


def ensure_container():
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    if CONTAINER in r.stdout.split():
        return
    note("starting client container")
    env = {**os.environ, "OPENPI_DIR": f"{DEPLOY}/client/openpi", "ZENOH_ENDPOINT": f"tcp/{ROBOT}:9000"}
    r = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True)
    if CONTAINER in r.stdout.split():          # left in Exited state (e.g. after a B70 reboot): start refuses it
        subprocess.run([f"{DEPLOY}/client/run_client.sh", "stop"], cwd=DEPLOY, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    subprocess.run([f"{DEPLOY}/client/run_client.sh", "start"], cwd=DEPLOY, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(8)


def reader(p):
    for line in p.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        e = ev.get("event")
        if e == "ready":
            state["connected"] = True
            state["last"] = "ready"
            note("teleop ready")
        elif e == "mode":
            state["mode_ok"] = ev.get("ok")
            note("robot mode " + ("OK (API, chassis enabled)" if ev.get("ok") else "NOT OK"))
        elif e == "moving":
            state["moving"] = f"{ev['kind']} {ev['dir']}"
            if ev["kind"] == "path":
                state["active_path"] = ev["dir"]
                state["path_anchor"] = list(state["pose"])
                note(f"path {ev['dir']}: {ev.get('length')} m, {ev.get('corners')} corner(s), "
                     f"{ev.get('dwell', 0):.0f} s pause at each")
            else:
                note(f"moving: {ev['kind']} {ev['dir']}")
        elif e == "path":
            if ev.get("dwell"):
                state["moving"] = f"path {ev['name']} - corner pause {ev['dwell']:.0f} s"
                note(f"path {ev['name']}: at corner, pausing {ev['dwell']:.0f} s")
            elif ev.get("resume"):
                state["moving"] = f"path {ev['name']}"
                note(f"path {ev['name']}: pause over, continuing")
            else:
                note(f"path {ev['name']}: waypoint {ev['waypoint']}/{ev['of']}")
        elif e == "idle":
            state["moving"] = None
            if ev.get("kind") == "path":
                state["active_path"] = None
                path_result["why"] = ev.get("why")
                path_done.set()
            state["last"] = f"{ev.get('kind')} {ev.get('dir')}: {ev.get('why')}"
            note(state["last"])
        elif e == "error":
            state["last"] = "error: " + ev.get("msg", "")
            note(state["last"])
        elif e == "pose":
            x, y, th = ev["x"], ev["y"], ev["th"]
            state["pose"] = [x, y, th]
            if ev.get("reset"):
                trail.clear()
                state["epoch"] += 1
                if state["recording"]:
                    state["rec_t0"] = time.monotonic()
            if state["recording"]:
                add_point(x, y, th)
        elif e == "exit":
            note("teleop exited")
    state["connected"] = False
    state["moving"] = None


def teleop():
    """Return a running teleop process, starting it if needed."""
    with lock:
        p = proc["p"]
        if p is not None and p.poll() is None:
            return p
        ensure_container()
        note("starting teleop")
        p = subprocess.Popen(TELEOP, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=reader, args=(p,), daemon=True).start()
        proc["p"] = p
        return p


def send(cmd, d=None):
    p = teleop()
    if cmd == "path_cmd":
        msg = {"cmd": "path", "name": d, "points": paths[d]["points"], "dwell": CORNER_PAUSE}
    else:
        msg = {"cmd": cmd} if d is None else {"cmd": cmd, "dir": d}
    try:
        p.stdin.write(json.dumps(msg) + "\n")
        p.stdin.flush()
        return True
    except (BrokenPipeError, OSError):
        return False


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>TA2 Chassis</title>
<style>
:root{--bg:#f4f6f7;--fg:#15212a;--muted:#5d6d78;--panel:#fff;--line:#cfd9de;--grid:#e6ecef;--accent:#0d6a88;--stop:#c0392b;--trail:#d9822b}
@media (prefers-color-scheme:dark){:root{--bg:#0f1519;--fg:#e2eaee;--muted:#93a5b0;--panel:#18232a;--line:#2a3a44;--grid:#1f2c34;--accent:#5cb7d6;--stop:#e5584a;--trail:#f0a050}}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
.app{height:100%;display:grid;grid-template-columns:minmax(260px,360px) minmax(0,1fr);gap:16px;padding:16px}
.left{display:flex;flex-direction:column;gap:14px;min-width:0}
.right{display:grid;grid-template-rows:minmax(120px,1fr) minmax(220px,2fr);gap:16px;min-width:0;min-height:0}
@media (max-width:760px){.app{height:auto;grid-template-columns:minmax(0,1fr)}.right{grid-template-rows:200px 360px}}
h1{font-size:20px;margin:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;min-height:0;display:flex;flex-direction:column}
.panel h2{font-size:13px;font-weight:600;color:var(--muted);margin:0;padding:8px 12px;border-bottom:1px solid var(--line);
  display:flex;justify-content:space-between;align-items:center;gap:8px}
.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.rot{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.tools{display:flex;gap:6px;flex-wrap:wrap}
#rec{color:var(--stop);font-weight:700;visibility:hidden}
.paths{padding:0}
.plist{display:flex;flex-direction:column}
.prow{display:flex;align-items:center;gap:8px;padding:6px 12px;border-top:1px solid var(--line);font-size:13px;cursor:pointer}
.prow:first-child{border-top:none}
.prow.sel{background:var(--grid)}
.prow.run{color:var(--accent);font-weight:700}
.prow .nm{font-weight:600;min-width:42px}
.prow .info{color:var(--muted);flex:1;font-variant-numeric:tabular-nums}
.left{overflow:auto}
button{font:inherit;font-size:17px;padding:22px 0;border-radius:10px;border:1px solid var(--line);
  background:var(--panel);color:var(--fg);cursor:pointer;touch-action:none;user-select:none;-webkit-user-select:none}
button:hover:not(:disabled){border-color:var(--accent)}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
#stop{background:var(--stop);border-color:var(--stop);color:#fff;font-weight:700}
.small{font-size:12px;padding:3px 10px;border-radius:6px}
#status{font-weight:600}
.muted{color:var(--muted);font-size:13px}
pre{margin:0;padding:10px 12px;font-size:12px;overflow:auto;white-space:pre-wrap;flex:1;min-height:0}
.canvas-wrap{flex:1;min-height:0;position:relative}
canvas{position:absolute;inset:0;width:100%;height:100%}
#pose{font-variant-numeric:tabular-nums;font-weight:400}
</style></head><body><div class="app">
<div class="left">
  <h1>TA2 Chassis</h1>
  <div id="status">connecting...</div>
  <div class="pad">
    <span></span><button data-d="forward">Forward</button><span></span>
    <button data-d="left">Left</button><button id="stop">STOP</button><button data-d="right">Right</button>
    <span></span><button data-d="back">Back</button><span></span>
  </div>
  <div class="rot"><button data-d="ccw">&#10226; CCW</button><button data-d="cw">CW &#10227;</button></div>
  <div class="panel"><h2><span>Arms</span><span class="tools">
    <button class="small" id="armraise">Raise arms</button><button class="small" id="armlower">Lower arms (slowly)</button><button class="small" id="armstart">Start pose</button></span></h2>
    <div class="muted" id="armst" style="padding:6px 12px">arms: idle</div>
  </div>
  <div class="panel paths"><h2><span>Paths</span><span class="tools">
    <button class="small" id="load">Load files</button><button class="small" id="runall">Run all in order</button></span></h2>
    <input type="file" id="file" accept=".csv,text/csv" multiple hidden>
    <div class="plist" id="plist"><div class="prow muted">No paths loaded</div></div>
  </div>
  <div class="muted">Arms: raise takes 8 s, lower 15 s (back to the pose saved before raising), Start pose 10 s (VLA training start pose, use before inference); arms need joint control mode. Paths stop for 10 s at every corner. Click: move 10 cm / turn 15&deg; in place. Press and hold: keeps moving at 0.05 m/s (turning at 11.5&deg;/s) until you let go, max 10 s per hold. Keep someone at the e-stop.</div>
</div>
<div class="right">
  <div class="panel"><h2>Log</h2><pre id="log"></pre></div>
  <div class="panel"><h2><span>Trajectory <span id="rec">&#9679; REC</span> <span id="pose"></span></span>
    <span class="tools"><button class="small" id="recstart">Start</button><button class="small" id="recstop">Stop</button>
    <button class="small" id="save">Save</button><button class="small" id="reset">Reset</button></span></h2>
    <div class="canvas-wrap"><canvas id="cv"></canvas></div>
    <div class="muted" style="padding:4px 12px 8px">Wheel odometry estimate (not tape-calibrated). Up = robot forward at the origin, grid 10 cm. Start/Stop records the trail, Save downloads a CSV (copy kept on B70 in /root/ta2_trails/), Reset sets the origin to the current pose.</div>
  </div>
</div>
</div><script>
const st=document.getElementById('status'),lg=document.getElementById('log'),poseEl=document.getElementById('pose');
const moves=[...document.querySelectorAll('button[data-d]')];
let press=null;
function post(u){return fetch(u,{method:'POST',keepalive:true}).then(r=>r.json()).catch(e=>({error:String(e)}))}
function begin(b){
  if(press||b.disabled)return;
  press={d:b.dataset.d,btn:b,hb:null};
  b.classList.add('on');
  press.timer=setTimeout(()=>{press.timer=null;post('/hold/'+press.d);
    press.hb=setInterval(()=>post('/hold/'+press.d),100)},300);
}
function end(){
  if(!press)return;
  const p=press;press=null;p.btn.classList.remove('on');
  if(p.timer){clearTimeout(p.timer);post('/step/'+p.d)}else{clearInterval(p.hb);post('/release')}
}
moves.forEach(b=>{
  b.addEventListener('pointerdown',e=>{e.preventDefault();b.setPointerCapture(e.pointerId);begin(b)});
  b.addEventListener('pointerup',end);b.addEventListener('pointercancel',end);
  b.addEventListener('lostpointercapture',end);b.addEventListener('contextmenu',e=>e.preventDefault());
});
window.addEventListener('blur',end);
document.addEventListener('visibilitychange',()=>{if(document.hidden)end()});
document.getElementById('stop').onclick=()=>{end();post('/stop')};
document.getElementById('reset').onclick=()=>post('/reset');
document.getElementById('recstart').onclick=()=>post('/rec/start');
document.getElementById('recstop').onclick=()=>post('/rec/stop');
document.getElementById('save').onclick=async()=>{
  const r=await post('/rec/save');if(r.error){st.textContent=r.error;return}
  const a=document.createElement('a');a.href='/trail.csv';a.download='';document.body.appendChild(a);a.click();a.remove();
};
const recEl=document.getElementById('rec'),recStart=document.getElementById('recstart'),recStop=document.getElementById('recstop'),saveBtn=document.getElementById('save');
let recording=false;

// paths
let paths={},sel=null,active=null,anchor=[0,0,0],seqNames=[],seqI=-1;
const plist=document.getElementById('plist'),fileIn=document.getElementById('file');
async function loadPaths(){try{const r=await (await fetch('/paths')).json();paths={};r.paths.forEach(p=>paths[p.name]=p);renderPaths()}catch(e){}}
function renderPaths(){
  const names=Object.keys(paths).sort();
  if(!names.length){plist.innerHTML='<div class="prow muted">No paths loaded</div>';return}
  plist.innerHTML='';
  names.forEach(n=>{const p=paths[n],row=document.createElement('div');
    row.className='prow'+(n===sel?' sel':'')+(n===active?' run':'');
    const q=seqNames.length?(seqNames.indexOf(n)<seqI?' done':''):'';
    row.innerHTML='<span class="nm"></span><span class="info"></span><button class="small">Run</button>';
    row.querySelector('.nm').textContent=n;
    row.querySelector('.info').textContent=p.length.toFixed(2)+' m → ('+(p.end[0]*100).toFixed(0)+', '+(p.end[1]*100).toFixed(0)+') cm'+q;
    row.onclick=()=>{sel=(sel===n?null:n);renderPaths();draw()};
    const b=row.querySelector('button');b.disabled=!!active||seqNames.length>0;
    b.onclick=async e=>{e.stopPropagation();const r=await post('/paths/run/'+encodeURIComponent(n));if(r.error)st.textContent=r.error};
    plist.appendChild(row)});
  document.getElementById('runall').disabled=!!active||seqNames.length>0;
}
document.getElementById('load').onclick=()=>fileIn.click();
fileIn.onchange=async()=>{
  const fs=[...fileIn.files].sort((a,b)=>a.name.localeCompare(b.name));
  for(const f of fs){const r=await fetch('/paths/upload',{method:'POST',body:JSON.stringify({name:f.name,csv:await f.text()})}).then(r=>r.json()).catch(e=>({error:String(e)}));
    if(r.error)st.textContent=f.name+': '+r.error}
  fileIn.value='';loadPaths();
};
const armRaise=document.getElementById('armraise'),armLower=document.getElementById('armlower'),armStart=document.getElementById('armstart'),armSt=document.getElementById('armst');
armRaise.onclick=async()=>{const r=await post('/arm/raise');if(r.error)st.textContent=r.error};
armLower.onclick=async()=>{const r=await post('/arm/lower');if(r.error)st.textContent=r.error};
armStart.onclick=async()=>{const r=await post('/arm/start');if(r.error)st.textContent=r.error};
document.getElementById('runall').onclick=async()=>{const r=await post('/paths/runall');if(r.error)st.textContent=r.error};
loadPaths();

// trajectory
const cv=document.getElementById('cv');let trail=[],epoch=-1,pose=[0,0,0];
function anchored(p,a){  // path points in the path's own frame -> odometry frame, first point at pose a
  const [x0,y0,t0]=p.points[0],r=a[2]-t0,c=Math.cos(r),s=Math.sin(r);
  return p.points.map(q=>[a[0]+(q[0]-x0)*c-(q[1]-y0)*s,a[1]+(q[0]-x0)*s+(q[1]-y0)*c]);
}
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function draw(){
  const dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;
  if(!W||!H)return;
  cv.width=W*dpr;cv.height=H*dpr;const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,W,H);
  const show=active&&paths[active]?anchored(paths[active],anchor):(sel&&paths[sel]?anchored(paths[sel],pose):null);
  const pts=trail.concat([[pose[0],pose[1]]]).concat(show||[]);
  let x0=Math.min(...pts.map(p=>p[0])),x1=Math.max(...pts.map(p=>p[0]));
  let y0=Math.min(...pts.map(p=>p[1])),y1=Math.max(...pts.map(p=>p[1]));
  const cx=(x0+x1)/2,cy=(y0+y1)/2,half=Math.max(0.6,(x1-x0)/2+0.45,(y1-y0)/2+0.45);
  const s=Math.min(W,H)/(2*half);
  const X=x=>W/2+(x-cx)*s,Y=y=>H/2-(y-cy)*s;
  // grid every 10 cm, stronger every 50 cm
  g.lineWidth=1;
  const gx0=Math.floor((cx-W/2/s)*10),gx1=Math.ceil((cx+W/2/s)*10),gy0=Math.floor((cy-H/2/s)*10),gy1=Math.ceil((cy+H/2/s)*10);
  for(let i=gx0;i<=gx1;i++){g.strokeStyle=i%5?css('--grid'):css('--line');g.beginPath();g.moveTo(X(i/10),0);g.lineTo(X(i/10),H);g.stroke()}
  for(let i=gy0;i<=gy1;i++){g.strokeStyle=i%5?css('--grid'):css('--line');g.beginPath();g.moveTo(0,Y(i/10));g.lineTo(W,Y(i/10));g.stroke()}
  // origin
  g.fillStyle=css('--muted');g.beginPath();g.arc(X(0),Y(0),4,0,7);g.fill();
  g.font='11px system-ui';g.fillText('start',X(0)+6,Y(0)-6);
  // trail
  if(trail.length>0){g.strokeStyle=css('--trail');g.lineWidth=2.5;g.lineJoin='round';g.beginPath();
    trail.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
    if(recording)g.lineTo(X(pose[0]),Y(pose[1]));g.stroke()}
  // planned path (selected preview, or the one running)
  if(show){g.setLineDash([6,5]);g.strokeStyle=css('--accent');g.lineWidth=2;g.beginPath();
    show.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();g.setLineDash([]);
    const e=show[show.length-1];g.fillStyle=css('--accent');g.beginPath();g.arc(X(e[0]),Y(e[1]),4,0,7);g.fill();
    g.fillText(active||sel,X(e[0])+6,Y(e[1])-6)}
  // robot footprint 0.62 x 0.625 m, forward = +y rotated by th (CCW)
  const [px,py,th]=pose,c=Math.cos(th),sn=Math.sin(th);
  const R=(dx,dy)=>[X(px+dx*c-dy*sn),Y(py+dx*sn+dy*c)];
  const hw=0.31,hl=0.3125;
  g.strokeStyle=css('--accent');g.lineWidth=2;g.fillStyle=css('--accent')+'22';g.beginPath();
  [[-hw,-hl],[hw,-hl],[hw,hl],[-hw,hl]].forEach((q,i)=>{const r=R(q[0],q[1]);i?g.lineTo(r[0],r[1]):g.moveTo(r[0],r[1])});
  g.closePath();g.fill();g.stroke();
  const a=R(0,0),b=R(0,hl+0.08),l=R(-0.06,hl-0.02),r=R(0.06,hl-0.02);
  g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke();
  g.fillStyle=css('--accent');g.beginPath();g.moveTo(b[0],b[1]);g.lineTo(l[0],l[1]);g.lineTo(r[0],r[1]);g.closePath();g.fill();
}
window.addEventListener('resize',draw);

async function poll(){try{
  const s=await (await fetch('/status?epoch='+epoch+'&since='+trail.length)).json();
  if(s.epoch!==epoch||s.trail_from===0){trail=[];epoch=s.epoch}
  trail=trail.concat(s.trail);pose=s.pose;recording=s.recording;
  const pn=(s.path_names||[]).join(',');if(pn!==Object.keys(paths).sort().join(','))loadPaths();
  if(active!==s.active_path||seqI!==s.seq_i||seqNames.length!==s.seq.length){active=s.active_path;anchor=s.path_anchor;seqNames=s.seq;seqI=s.seq_i;renderPaths()}
  recEl.style.visibility=recording?'visible':'hidden';recStart.disabled=recording;recStop.disabled=!recording;saveBtn.disabled=!trail.length;
  poseEl.textContent='  x '+(pose[0]*100).toFixed(1)+' cm, y '+(pose[1]*100).toFixed(1)+' cm, heading '+(pose[2]*180/Math.PI).toFixed(1)+'°';
  let t=!s.connected?'connecting to robot...':s.moving?('moving: '+s.moving):s.last;
  if(s.seq.length)t='run all: '+s.seq.map((n,i)=>i===s.seq_i?'['+n+']':n).join(' → ')+(s.moving?'  ('+s.moving+')':'');
  if(s.arm)t='arms: '+s.arm+' in progress (STOP holds them where they are)';
  if(s.mode_ok===false)t+='  |  robot NOT in API mode / chassis disabled';
  armSt.textContent=s.arm?('arms: '+s.arm+'...'):s.arm_last;
  const still=!s.moving&&!s.seq.length&&!s.arm;
  armRaise.disabled=!still;armLower.disabled=!still;armStart.disabled=!still;
  st.textContent=t+'  (clicks '+s.clicks+', holds '+s.holds+')';
  const busyStep=s.moving&&(s.moving.startsWith('step')||s.moving.startsWith('path'))||s.seq.length>0||!!s.arm;
  const pathBusy=!!s.active_path||s.seq.length>0||!!s.arm;
  document.querySelectorAll('.prow button').forEach(b=>b.disabled=pathBusy);
  document.getElementById('runall').disabled=pathBusy;
  moves.forEach(x=>x.disabled=!press&&(busyStep||!s.connected));
  lg.textContent=s.log.join('\n');lg.scrollTop=lg.scrollHeight;draw();
}catch(e){st.textContent='B70 not reachable';if(!press)moves.forEach(x=>x.disabled=true)}}
poll();setInterval(poll,300);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/":
            self._send(200, PAGE, "text/html")
        elif path == "/trail.csv":
            data = trail_csv().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="ta2_trail_{time.strftime("%Y%m%d_%H%M%S")}.csv"')
            self.end_headers()
            self.wfile.write(data)
        elif path == "/paths":
            self._send(200, {"paths": [{"name": n, **paths[n]} for n in sorted(paths)]})
        elif path == "/status":
            last_poll[0] = time.monotonic()
            q = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
            try:
                since = int(q.get("since", 0)) if int(q.get("epoch", -1)) == state["epoch"] else 0
            except ValueError:
                since = 0
            since = min(since, len(trail))
            self._send(200, {**state, "log": list(log), "trail_from": since, "trail": trail[since:],
                             "seq": seq["names"], "seq_i": seq["i"], "path_names": sorted(paths)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        if parts[0] in ("step", "hold", "paths") and state["arm"] and parts[:2] != ["paths", "upload"]:
            return self._send(409, {"error": "arms are moving"})
        if parts[0] == "arm" and len(parts) == 2 and parts[1] in ARM_TIME:
            if state["arm"] or state["moving"] or seq["names"]:
                return self._send(409, {"error": "busy: wait until the robot is still"})
            state["arm"] = parts[1]
            note(f"arms: {parts[1]} ({ARM_TIME[parts[1]]} s)")
            threading.Thread(target=run_arm, args=(parts[1],), daemon=True).start()
            return self._send(200, {"ok": True})
        if parts[0] in ("step", "hold") and len(parts) == 2:
            d = parts[1]
            if d not in DIRS:
                return self._send(400, {"error": "bad direction"})
            if parts[0] == "step":
                state["clicks"] += 1
            elif state["moving"] != f"hold {d}":
                state["holds"] += 1
            ok = send(parts[0], d)
            self._send(200 if ok else 503, {"ok": ok})
        elif parts[0] == "paths" and len(parts) >= 2:
            if parts[1] == "upload":
                n = int(self.headers.get("Content-Length", 0))
                if n > 2_000_000:
                    return self._send(413, {"error": "file too large"})
                try:
                    body = json.loads(self.rfile.read(n))
                    name = clean_name(body["name"])
                    paths[name] = parse_path_csv(body["csv"])
                    os.makedirs(PATH_DIR, exist_ok=True)
                    with open(os.path.join(PATH_DIR, name + ".csv"), "w") as f:
                        f.write(body["csv"])
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    return self._send(400, {"error": f"cannot read file: {e}"})
                note(f"loaded path {name}: {paths[name]['length']} m")
                return self._send(200, {"ok": True, "name": name})
            if state["moving"] or seq["names"]:
                return self._send(409, {"error": "busy"})
            if parts[1] == "run" and len(parts) == 3 and parts[2] in paths:
                last_poll[0] = time.monotonic()
                ok = send("path_cmd", parts[2])
                return self._send(200 if ok else 503, {"ok": ok})
            if parts[1] == "runall" and paths:
                last_poll[0] = time.monotonic()
                threading.Thread(target=run_sequence, args=(sorted(paths),), daemon=True).start()
                return self._send(200, {"ok": True})
            self._send(400, {"error": "unknown path"})
        elif parts[0] == "reset":
            ok = send("reset_pose")
            note("trail reset, origin = current pose")
            self._send(200 if ok else 503, {"ok": ok})
        elif parts[0] == "rec" and len(parts) == 2 and parts[1] in ("start", "stop", "save"):
            if parts[1] == "start" and not state["recording"]:
                if not trail:
                    state["rec_t0"] = time.monotonic()
                state["recording"] = True
                add_point(*state["pose"], force=True)
                note("recording started")
            elif parts[1] == "stop" and state["recording"]:
                state["recording"] = False
                note(f"recording stopped ({len(trail)} points)")
            elif parts[1] == "save":
                if not trail:
                    return self._send(400, {"error": "trail is empty"})
                os.makedirs(TRAIL_DIR, exist_ok=True)
                path = os.path.join(TRAIL_DIR, time.strftime("trail_%Y%m%d_%H%M%S.csv"))
                with open(path, "w") as f:
                    f.write(trail_csv())
                note(f"saved {len(trail)} points to {path}")
                return self._send(200, {"ok": True, "file": path})
            self._send(200, {"ok": True})
        elif parts[0] in ("release", "stop"):
            if parts[0] == "stop":
                seq["abort"] = True
                if state["arm"]:
                    subprocess.run(ARM_STOP, capture_output=True)
                    note("STOP: arm script interrupted")
            ok = send(parts[0])
            if parts[0] == "stop":
                note("STOP pressed")
            self._send(200 if ok else 503, {"ok": ok})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"TA2 chassis web remote on http://0.0.0.0:{PORT}", flush=True)
    load_path_dir()
    threading.Thread(target=teleop, daemon=True).start()   # connect at startup
    threading.Thread(target=alive_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
