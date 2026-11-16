"""TA2 voice service on the operator's MacBook: microphone keyword spotting + speaker for full-flow step 7.

Serves 127.0.0.1:8096; B70 reaches it as 127.0.0.1:18096 through `ssh -R` (see start_mac_voice.sh).
  POST /rehearse  start listening for the command (restarts a listen already running)  -> listening
  GET  /status    {"state": idle|listening|recognized|playing|done|failed, "detail": ..., "level": mic RMS dBFS}
  POST /confirm   play the completion announcement                                    -> playing -> done
Keyword spotting reuses the B70 voice_handoff model (sherpa-onnx zipformer KWS) and its WAV prompts.
Must run inside Terminal.app (not over ssh): macOS only gives the microphone to an app the user allowed.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
BIND = os.environ.get("VOICE_BIND", "127.0.0.1")
PORT = int(os.environ.get("VOICE_PORT", "8096"))
HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
MIC = os.environ.get("VOICE_MIC", ":MacBook Air麦克风")      # ffmpeg avfoundation device
LISTEN_MAX = float(os.environ.get("VOICE_LISTEN_MAX", "125"))  # s; the mission gives up after 120 s (7.1)
RATE = 16000
KEYWORDS = HERE / "keywords_mac.txt"
lock = threading.Lock()
state = {"state": "idle", "detail": "", "level": None, "heard": None, "since": time.time()}
generation = [0]
capture = [None]


def set_state(name, detail="", **extra):
    with lock:
        state.update(state=name, detail=detail, since=time.time(), **extra)
    print(time.strftime("%H:%M:%S"), name, detail, flush=True)


def spotter():
    import sherpa_onnx
    root = HERE / "model"
    return sherpa_onnx.KeywordSpotter(
        tokens=str(root / "tokens.txt"),
        **{part: str(next(root.glob(f"{part}*-chunk-16-left-64.onnx"))) for part in ("encoder", "decoder", "joiner")},
        keywords_file=str(KEYWORDS), num_threads=1, provider="cpu")


KWS = None


def stop_capture():
    p = capture[0]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(2)
        except subprocess.TimeoutExpired:
            p.kill()


def listen(gen):
    stream = KWS.create_stream()
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "avfoundation", "-i", MIC,
           "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return set_state("failed", f"无法启动录音：{exc}")
    capture[0] = p
    end, silent_since = time.monotonic() + LISTEN_MAX, time.monotonic()
    try:
        while generation[0] == gen and time.monotonic() < end:
            chunk = p.stdout.read(3200)          # 0.1 s
            if not chunk:
                err = p.stderr.read().decode(errors="replace").strip()
                if generation[0] == gen:
                    set_state("failed", "录音中断：" + (err[-200:] or f"ffmpeg 退出 {p.poll()}"))
                return
            x = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            with lock:
                state["level"] = round(20 * math.log10(max(rms, 1e-6)), 1)
            if rms > 0:
                silent_since = time.monotonic()
            elif time.monotonic() - silent_since > 3:       # exact zeros = macOS refused the microphone
                return set_state("failed", "麦克风没有声音：请在 系统设置→隐私与安全性→麦克风 允许「终端」，再重启服务")
            stream.accept_waveform(RATE, x)
            while KWS.is_ready(stream):
                KWS.decode_stream(stream)
                heard = KWS.get_result(stream)
                if heard and generation[0] == gen:
                    set_state("recognized", "已识别：" + heard, heard=heard)
                    subprocess.Popen(["afplay", str(HERE / "reply.wav")])
                    return
        if generation[0] == gen:
            set_state("failed", f"{LISTEN_MAX:g} 秒内没有听到口令")
    finally:
        stop_capture()


def announce(gen):
    rc = subprocess.run(["afplay", str(HERE / "completion.wav")]).returncode
    if generation[0] == gen:
        set_state("done", "播报完成") if rc == 0 else set_state("failed", f"播报失败（afplay {rc}）")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, data):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def allowed(self):
        if self.headers.get("Host") not in HOSTS or self.headers.get("Origin"):
            self.reply(403, {"error": "forbidden"})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        if self.path == "/status":
            with lock:
                return self.reply(200, dict(state))
        self.reply(404, {"error": "unknown"})

    def do_POST(self):
        if not self.allowed():
            return
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/rehearse":
            generation[0] += 1
            stop_capture()
            set_state("listening", "正在听", heard=None)
            threading.Thread(target=listen, args=(generation[0],), daemon=True).start()
        elif self.path == "/confirm":
            generation[0] += 1
            stop_capture()
            set_state("playing", "正在播报")
            threading.Thread(target=announce, args=(generation[0],), daemon=True).start()
        else:
            return self.reply(404, {"error": "unknown"})
        with lock:
            self.reply(200, dict(state))


if __name__ == "__main__":
    KWS = spotter()
    print(f"TA2 voice service on {BIND}:{PORT}, microphone {MIC}", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
