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
# ffmpeg avfoundation devices, "|"-separated; all listen at once and the first one to hear the command wins.
# The iPhone microphone (Continuity, same Apple ID, Wi-Fi + Bluetooth on) shows up as its own device.
MICS = [m for m in os.environ.get("VOICE_MIC", ":MacBook Air麦克风|:“iPhone”的麦克风").split("|") if m.strip()]
LISTEN_MAX = float(os.environ.get("VOICE_LISTEN_MAX", "125"))  # s; the mission gives up after 120 s (7.1)
RATE = 16000
KEYWORDS = HERE / "keywords_mac.txt"
lock = threading.Lock()
kws_lock = threading.Lock()          # one spotter, one stream per microphone
state = {"state": "idle", "detail": "", "level": None, "heard": None, "mic": None, "mics": {}, "since": time.time()}
generation = [0]
captures = []


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
    for p in list(captures):
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                p.kill()
    captures.clear()


def label(mic):
    return mic.lstrip(":")


def still_listening(gen):
    return generation[0] == gen and state["state"] == "listening"


def mic_loop(gen, mic, end):
    """Listen on one microphone until the command is heard (by any microphone), time runs out or it fails.
    Returns an error text when this microphone is unusable, else None."""
    with kws_lock:
        stream = KWS.create_stream()
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "avfoundation", "-i", mic,
           "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return f"无法启动录音：{exc}"
    captures.append(p)
    try:
        return read_mic(gen, mic, end, p, stream)
    finally:                              # each microphone closes its own capture when it stops listening
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(2)
            except subprocess.TimeoutExpired:
                p.kill()
        if p in captures:
            captures.remove(p)


def read_mic(gen, mic, end, p, stream):
    silent_since = time.monotonic()
    while still_listening(gen) and time.monotonic() < end:
        chunk = p.stdout.read(3200)          # 0.1 s
        if not chunk:
            if not still_listening(gen):
                return None
            err = p.stderr.read().decode(errors="replace").strip()
            return "录音中断：" + (err[-160:] or f"ffmpeg 退出 {p.poll()}")
        x = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        db = round(20 * math.log10(max(rms, 1e-6)), 1)
        with lock:
            state["mics"][label(mic)] = db
            state["level"] = max(v for v in state["mics"].values() if isinstance(v, float))
        if rms > 0:
            silent_since = time.monotonic()
        elif time.monotonic() - silent_since > 3:       # exact zeros = no permission or no device behind it
            return "没有声音（终端没有麦克风权限，或设备没连上）"
        with kws_lock:
            stream.accept_waveform(RATE, x)
            heard = None
            while KWS.is_ready(stream):
                KWS.decode_stream(stream)
                heard = KWS.get_result(stream) or heard
        if heard:
            with lock:
                first = generation[0] == gen and state["state"] == "listening"
            if first:
                set_state("recognized", f"已识别（{label(mic)}）：{heard}", heard=heard, mic=label(mic))
                subprocess.Popen(["afplay", str(HERE / "reply.wav")])
            return None
    return None


def listen(gen):
    end, errors = time.monotonic() + LISTEN_MAX, {}

    def run(mic):
        err = mic_loop(gen, mic, end)
        if err:
            errors[mic] = err
            with lock:
                state["mics"][label(mic)] = "不可用：" + err
            print(time.strftime("%H:%M:%S"), label(mic), err, flush=True)

    threads = [threading.Thread(target=run, args=(m,), daemon=True) for m in MICS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not still_listening(gen):
        return
    if len(errors) == len(MICS):
        set_state("failed", "所有麦克风都不可用：" + "；".join(f"{label(m)} {e}" for m, e in errors.items()))
    else:
        set_state("failed", f"{LISTEN_MAX:g} 秒内没有听到口令")


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
            set_state("listening", "正在听", heard=None, mic=None, mics={})
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
    print(f"TA2 voice service on {BIND}:{PORT}, microphones {[label(m) for m in MICS]}", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
