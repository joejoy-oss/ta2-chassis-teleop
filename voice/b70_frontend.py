"""B70 voice interaction module page (8099) = step 7 of full flow v2 on the 8088 control server.

The module itself runs inside the 8088 mission (mission.py, flow "v2" = steps 1-7, flow "voice" = step 7 only).
This page only shows sub-steps 7.1-7.8 and offers two controls, all through one interface:
  GET  /api/voice/status   step 7 state (from 8088 /api/v1/state), selected model, Mac voice service state
  POST /api/voice/run      run step 7 alone   (8088 mission.run {flow: "voice"})
  POST /api/voice/estop    emergency stop     (8088 /api/v1/estop)
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HOST = os.environ.get("TA2_VOICE_HOST", "192.168.31.228:8099")
BIND, PORT = HOST.rsplit(":", 1)
CONTROL = os.environ.get("TA2_CONTROL", "http://127.0.0.1:8088/api/v1")
MAC_BRIDGE = os.environ.get("TA2_MAC_BRIDGE", "http://127.0.0.1:18096")
ALLOWED_CLIENT = os.environ.get("TA2_ALLOWED_CLIENT", "192.168.31.54")
# Shown before step 7 has run once in this 8088 session; the live list comes from the mission state.
FALLBACK_STEPS = [("7.1", "识别口令"), ("7.2", "真机抓取（抓住即停）"), ("7.3", "收臂（保持夹紧）"),
                  ("7.4", "前伸 10 cm"), ("7.5", "下降 10 cm"), ("7.6", "松开夹爪"), ("7.7", "确认交接"), ("7.8", "完成播报")]
PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TA2 · 语音交互模块</title><style>
body{background:#101923;color:#edf2f7;font:16px/1.6 system-ui}
main{max-width:720px;margin:6vh auto;padding:24px}
section{border:1px solid #394b5e;border-radius:12px;background:#192635;padding:24px}
button{padding:10px 16px;font:inherit;border:0;border-radius:7px;background:#a8c8ff;color:#101923}
button:disabled{opacity:.5} #state,#mission{color:#aed0ff}small{color:#9fb0c3}
#estop{background:#e53935;color:#fff;font-weight:700;float:right}
ol{list-style:none;padding-left:0} li{margin:6px 0} li.running{color:#ffd479;font-weight:700}
li.succeeded{color:#9be39b} li.failed{color:#ff8a80;font-weight:700}
#listen{display:none;align-items:center;gap:20px;margin:18px 0;padding:18px;border-radius:12px;background:#2a1d22;border:1px solid #e57373}
#listen.on{display:flex} #listen.ok{background:#1d2a20;border-color:#81c784}
#listen .mic{position:relative;flex:none;width:64px;height:64px;border-radius:50%;background:#e53935;display:grid;place-items:center;font-size:30px}
#listen.ok .mic{background:#43a047}
#listen.on:not(.ok) .mic::before,#listen.on:not(.ok) .mic::after{content:'';position:absolute;inset:0;border-radius:50%;border:3px solid #e57373;animation:ring 1.6s ease-out infinite}
#listen .mic::after{animation-delay:.8s}
#listen b{display:block;font-size:20px} #listen .bars{display:flex;gap:4px;align-items:center;height:28px;margin-top:6px}
#listen .bars i{width:5px;height:100%;background:#ff8a80;border-radius:3px;animation:bar 1s ease-in-out infinite}
#listen.ok .bars{display:none}
#listen .bars i:nth-child(2){animation-delay:.15s}#listen .bars i:nth-child(3){animation-delay:.3s}#listen .bars i:nth-child(4){animation-delay:.45s}#listen .bars i:nth-child(5){animation-delay:.6s}
@keyframes ring{from{transform:scale(1);opacity:.8}to{transform:scale(2.2);opacity:0}}
@keyframes bar{0%,100%{transform:scaleY(.25)}50%{transform:scaleY(1)}}
@media (prefers-reduced-motion:reduce){#listen .mic::before,#listen .mic::after,#listen .bars i{animation:none}}
</style><main><h1>语音交互模块 <button id="estop">急停</button></h1><section>
<p>全流程 v2 的第 7 步。口令：「小度小度，请把沙包给我」。</p>
<p id="model">VLA 模型：读取中…</p>
<p id="state" role="status">Mac 语音端：读取中…</p>
<p id="mission" role="status"></p>
<div id="listen" role="status"><span class="mic">🎤</span><div><b id="listentext">正在听…</b><span id="listenhint">请说「小度小度，请把沙包给我」</span><span class="bars"><i></i><i></i><i></i><i></i><i></i></span></div></div>
<ol id="steps"></ol>
<button id="run">单独运行第 7 步</button>
<p><small>完整的全流程 v2（1–7）在 8088 控制页的「全流程」卡片里启动；这里与 8088 显示的是同一份第 7 步状态。
机器人会实际动作，急停随时可用。</small></p>
</section></main><script>
const HAND={left:'左手',right:'右手'}, MARK={pending:'',running:' …',succeeded:' ✓',failed:' ✗',skipped:' –'};
const runBtn=document.querySelector('#run');
let data=null, busy=false;
function render(){
 if(!data)return;
 const m=data.mission||{};
 document.querySelector('#model').textContent='VLA 模型：'+(data.model||'未知')+'（跟随 8088 设置）';
 const v=data.voice_service||{};
 const labels={idle:'空闲',listening:'正在听',recognized:'已识别口令',playing:'正在播报',done:'播报完成',failed:'失败'};
 document.querySelector('#state').textContent='Mac 语音端：'+(v.error||labels[v.state]||v.state||'未知');
 const flows={v1:'全流程',v2:'全流程 v2',voice:'单独运行第 7 步',grasp:'仅抓取',handoff:'仅放下（7.4–7.7）'}, res={idle:'尚未运行',running:'运行中',succeeded:'成功',failed:'失败',aborted:'已急停中止'};
 document.querySelector('#mission').textContent=(flows[m.flow]||'')+(m.flow?' · ':'')+(res[m.status]||'尚未运行')+
   (data.hand?' · 抓取手：'+(HAND[data.hand]||data.hand):'')+(m.result?' · '+m.result:'');
 const s1=(data.steps||[])[0]||{}, lst=document.querySelector('#listen');
 const listening=s1.status==='running'||v.state==='listening', heard=s1.status==='succeeded'&&((data.steps||[])[1]||{}).status==='running';
 lst.classList.toggle('on',listening||heard);lst.classList.toggle('ok',heard&&!listening);
 document.querySelector('#listentext').textContent=heard&&!listening?'已识别口令':'正在听…';
 document.querySelector('#listenhint').textContent=heard&&!listening?'开始抓取沙包':'请说「小度小度，请把沙包给我」';
 const ol=document.querySelector('#steps');ol.textContent='';
 data.steps.forEach(s=>{const li=document.createElement('li');li.className=s.status||'';
  li.textContent=s.id+' '+s.name+(MARK[s.status]||'')+(s.detail?' · '+s.detail:'');
  if(s.uses){const u=document.createElement('small');u.textContent='　调用：'+s.uses;li.appendChild(u);}
  ol.appendChild(li);});
 runBtn.disabled=busy||!!m.active;
}
async function post(path){
 busy=true;render();
 try{const r=await fetch(path,{method:'POST'});const d=await r.json();
  if(!r.ok)throw Error(typeof d.error==='string'?d.error:(d.error&&d.error.message)||JSON.stringify(d));}
 catch(e){alert(e.message);}
 finally{busy=false;update();}
}
runBtn.onclick=()=>post('/api/voice/run');
document.querySelector('#estop').onclick=async()=>{
 try{const r=await fetch('/api/voice/estop',{method:'POST'});document.querySelector('#mission').textContent=r.ok?'已发送急停':'急停请求失败 HTTP '+r.status}
 catch(e){document.querySelector('#mission').textContent='急停请求失败：'+e.message}
};
async function update(){
 try{data=await (await fetch('/api/voice/status')).json();}
 catch(e){document.querySelector('#mission').textContent='8088 或本服务连接失败';}
 render();
}
update();setInterval(update,700);
</script></html>"""


def request_json(url, data=None, *, mac=False, timeout=4):
    headers = {"Content-Type": "application/json"}
    if mac:
        headers["Host"] = "127.0.0.1:8096"
    payload = None if data is None else json.dumps(data).encode()
    with urlopen(Request(url, data=payload, headers=headers), timeout=timeout) as response:
        return json.load(response)


def voice_status():
    mission = request_json(CONTROL + "/state").get("mission") or {}
    voice = mission.get("voice")
    steps = voice["steps"] if voice else [{"id": i, "name": n, "status": "pending", "detail": ""} for i, n in FALLBACK_STEPS]
    try:
        models = request_json(CONTROL + "/models")
        entry = next((m for m in models.get("models", []) if m.get("id") == models.get("selected")), {})
        model = entry.get("name", models.get("selected"))
    except (OSError, ValueError):
        model = None
    try:
        service = request_json(MAC_BRIDGE + "/status", mac=True)
    except (OSError, ValueError) as exc:
        service = {"error": f"连接失败（{exc}）"}
    return {"module": "语音交互模块",
            "mission": {k: mission.get(k) for k in ("id", "flow", "status", "active", "step", "result")},
            "hand": voice.get("hand") if voice else None,
            "steps": steps,
            "model": model, "voice_service": service}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, data, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def route(self, method):
        if (self.client_address[0] != ALLOWED_CLIENT or self.headers.get("Host") != HOST or
                self.headers.get("Origin") not in (None, "http://" + HOST)):
            return self.send(403, b'{}', 'application/json')
        try:
            if method == "GET" and self.path == "/":
                return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if method == "GET" and self.path == "/api/voice/status":
                return self.json_reply(200, voice_status())
            if method == "POST" and self.path == "/api/voice/estop":
                return self.json_reply(200, request_json(CONTROL + "/estop", {}))
            if method == "POST" and self.path == "/api/voice/run":
                return self.json_reply(202, request_json(CONTROL + "/actions",
                                                         {"type": "mission.run", "params": {"flow": "voice"}}))
            return self.json_reply(404, {"error": "未知接口"})
        except HTTPError as exc:
            return self.send(exc.code, exc.read(8192), "application/json")
        except (OSError, ValueError, KeyError):
            return self.json_reply(503, {"error": "8088 控制服务未确认；请先在 8088 核对运行状态"})

    def json_reply(self, status, data):
        return self.send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")


if __name__ == "__main__":
    ThreadingHTTPServer((BIND, int(PORT)), Handler).serve_forever()
