"""
main.py
Entry point for the resurgence experiment.
Run this file to start the session:

    python main.py

Before each session, update config.yaml:
    - experiment.participant_id
    - experiment.group (2 or 3)
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def check_dependencies():
    missing = []
    for pkg in ["faster_whisper", "sounddevice", "soundfile", "numpy", "yaml",
                "flask", "flask_socketio"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Run: pip install faster-whisper sounddevice soundfile pygame numpy pyyaml flask flask-socketio")
        sys.exit(1)


def check_files():
    required = [
        "config.yaml",
        "list/list1_animals.txt",
        "list/list2_professions.txt",
    ]
    missing = [f for f in required if not (BASE_DIR / f).exists()]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f"  - {BASE_DIR / f}")
        sys.exit(1)


def check_practice(cfg):
    practice = cfg.get("practice", {})
    if practice.get("enabled", False):
        path = BASE_DIR / practice["list"]
        if not path.exists():
            print(f"Warning: practice.enabled=true but list not found: {path}")
            print("  Place words in that file or set practice.enabled: false")
            sys.exit(1)


def check_group_3(cfg):
    if cfg["experiment"]["group"] == 3:
        path = BASE_DIR / cfg["lists"]["path"] / cfg["lists"]["list3"]
        if not path.exists():
            print(f"Group is set to 3 but list3 file not found: {path}")
            sys.exit(1)


if __name__ == "__main__":
    print("=" * 50)
    print("  Resurgence Verbal Behavior Experiment")
    print("=" * 50)

    check_dependencies()
    check_files()

    import yaml
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    check_practice(cfg)
    check_group_3(cfg)

    pid = cfg["experiment"]["participant_id"]
    group = cfg["experiment"]["group"]
    print(f"\n  Participant : {pid}")
    print(f"  Group       : {group}-list")
    print(f"  Window      : {cfg['trial']['window_seconds']}s")
    print(f"  Streak limit: {cfg['switching']['no_response_streak']} cycles")
    print()

    confirm = input("Start server? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    # Open browser windows automatically after server starts
    import threading, time, webbrowser
    def _open_browsers():
        time.sleep(2.0)
        webbrowser.open("http://127.0.0.1:5000/participant")
        time.sleep(0.3)
        webbrowser.open("http://127.0.0.1:5000/")
    threading.Thread(target=_open_browsers, daemon=True).start()

    from server import start
    start()
