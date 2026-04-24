"""
server.py
Flask backend for the resurgence experiment.

Server → Client : polling  GET /poll?since=<seq>  (browser polls every 200 ms)
Client → Server : HTTP POST /cmd/<action>

No WebSocket, no SSE, no streaming — works with any plain HTTP server.
"""

from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import threading
import traceback
import yaml

BASE_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE_DIR / "frontend" / "static"))

_runner      = None
_model       = None
_model_ready = False

# ── Event log ─────────────────────────────────────────────────────────────────
# Each entry: {"seq": int, "event": str, "data": dict}
_event_log  = []
_event_seq  = 0
_log_lock   = threading.Lock()


def _push(event: str, data: dict = None):
    """Append an event to the log. Thread-safe."""
    global _event_seq
    with _log_lock:
        _event_seq += 1
        _event_log.append({"seq": _event_seq, "event": event, "data": data or {}})
        if len(_event_log) > 500:          # keep only recent events
            _event_log.pop(0)


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.route("/")
def experimenter():
    return send_from_directory(str(BASE_DIR / "frontend"), "experimenter.html")


@app.route("/participant")
def participant():
    return send_from_directory(str(BASE_DIR / "frontend"), "participant.html")


@app.route("/config")
def get_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return jsonify({
        "participant_id": cfg["experiment"]["participant_id"],
        "group":          cfg["experiment"]["group"],
        "window_seconds": cfg["trial"]["window_seconds"],
        "model_ready":    _model_ready,
    })


@app.route("/poll")
def poll():
    """Return all events with seq > since."""
    since = int(request.args.get("since", 0))
    with _log_lock:
        new = [e for e in _event_log if e["seq"] > since]
    return jsonify(new)


@app.route("/cmd/<action>", methods=["POST"])
def cmd(action):
    global _runner

    if action == "start":
        if not _model_ready:
            return jsonify({"error": "Whisper model not ready — please wait."}), 503
        if _runner is not None and _runner.running:
            return jsonify({"error": "Session already running."}), 409

        from trial_loop import ExperimentRunner
        _runner = ExperimentRunner("config.yaml", model=_model)

        _runner.on_status = lambda phase, cycle, last_response, remaining: _push("status_update", {
            "phase": phase, "cycle": cycle, "last_response": last_response, "remaining": remaining,
        })
        _runner.on_iti         = lambda active: _push("iti_change",   {"active": active})
        _runner.on_pause       = lambda paused: _push("pause_change", {"paused": paused})
        _runner.on_instruction = lambda text:   _push("instruction",  {"text": text})
        _runner.on_complete    = lambda s:      _push("session_complete", {"summary": s})
        _runner.on_cycle_start = lambda secs:   _push("cycle_start", {"window_seconds": secs})
        _runner.on_cycle_end   = lambda:        _push("cycle_end")
        _runner.on_reinforcement = lambda info: _push("reinforcement", info)

        def _run():
            global _runner
            print("[runner] started")
            try:
                _runner.run()
                print("[runner] finished normally")
            except Exception:
                traceback.print_exc()
                _push("server_error", {"message": "Experiment crashed — see terminal."})
            finally:
                if _runner is not None:
                    _runner.running = False
                _runner = None
                _push("session_ended")

        threading.Thread(target=_run, daemon=True).start()
        _push("session_started")
        return jsonify({"ok": True})

    elif action == "stop":
        if _runner:
            _runner.stop()
        _push("session_stopped")
        return jsonify({"ok": True})

    elif action == "pause":
        if _runner:
            if _runner.is_paused:
                _runner.resume()
            else:
                _runner.pause()
        return jsonify({"ok": True})

    elif action == "next_phase":
        if _runner:
            _runner.force_next_phase()
        return jsonify({"ok": True})

    elif action == "acknowledge":
        if _runner:
            _runner.acknowledge_instruction()
        _push("instruction_done")
        return jsonify({"ok": True})

    elif action == "test_chime":
        with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from audio import play_reinforcement, generate_reward_wav
        reward_path = BASE_DIR / cfg["audio"]["reinforcement_sound"]
        if not reward_path.exists():
            generate_reward_wav(str(reward_path))
        play_reinforcement(str(reward_path))
        _push("reinforcement", {"test": True, "word": "test chime"})
        return jsonify({"ok": True})

    return jsonify({"error": "Unknown action"}), 400


# ── Model preload ─────────────────────────────────────────────────────────────

def _preload_model():
    global _model, _model_ready
    try:
        with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        from faster_whisper import WhisperModel
        _model = WhisperModel(cfg["asr"]["model_size"], device="cpu")
        _model_ready = True
        print("  Whisper model ready.")
        _push("model_status", {"ready": True})
    except Exception:
        traceback.print_exc()


def start(host="127.0.0.1", port=5000):
    threading.Thread(target=_preload_model, daemon=True).start()
    print(f"\n  Experimenter panel : http://{host}:{port}/")
    print(f"  Participant screen : http://{host}:{port}/participant")
    print("  Press Ctrl+C to quit.\n")
    app.run(host=host, port=port, debug=False, threaded=True)
