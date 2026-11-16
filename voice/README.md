# Voice interaction module (full flow v2, step 7)

- `b70_frontend.py` — the 8099 page on B70 (`/root/ta2-mission/voice_handoff/`): a view of step 7.1–7.8 of the
  8088 mission, plus "run step 7 alone" and E-STOP. Test: `python3 -m unittest test_frontend`.
- `mac_voice.py` + `start_mac_voice.sh` — the voice service on the operator's MacBook (`~/ta2-voice/`):
  microphone keyword spotting and speaker on 127.0.0.1:8096, reached from B70 as 127.0.0.1:18096 via `ssh -R`.
  Run `start_mac_voice.sh` in Terminal.app (macOS only gives the microphone to an app the user allowed).
  The microphone opens only at 7.1 (`POST /rehearse`) and closes on recognition or after 125 s.
- `keywords_mac.txt` — accepted commands (小度小度（请）把（红色）沙包给我 …). The KWS model
  (`model/`, sherpa-onnx zipformer) and `reply.wav` / `completion.wav` come from B70 `voice_handoff/`.
