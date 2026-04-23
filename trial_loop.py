"""
trial_loop.py
Core experiment loop. Orchestrates each cycle:
  1. Wait for window to open
  2. Record microphone until speech ends or window expires
  3. Transcribe with Whisper
  4. Match response against current phase's lists
  5. Play reinforcement if criteria met
  6. Log result
  7. Check phase-switching criteria
"""

import time
import threading
import yaml
import os
from pathlib import Path
from faster_whisper import WhisperModel

from matcher import ListManager
from logger import SessionLogger
from audio import record_until_silence, play_reinforcement, cleanup_audio_file, generate_reward_wav

# All relative paths in config are resolved against this directory
BASE_DIR = Path(__file__).parent


# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    resolved = BASE_DIR / path
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative_path: str) -> str:
    """Resolve a config-relative path to an absolute path."""
    return str(BASE_DIR / relative_path)


# ─── ASR ──────────────────────────────────────────────────────────────────────
import re

# Common English filler words to skip when extracting the participant's response
_FILLERS = {
    "um", "uh", "er", "ah", "like", "hmm", "hm", "oh", "okay", "ok",
    "so", "well", "yeah", "yes", "no", "the", "a", "an",
}

def transcribe(model: WhisperModel, wav_path: str, language: str = "en") -> tuple[str, str]:
    """
    Run Whisper on a wav file.
    Return:
    - the first non-filler word spoken
    - the full transcript text for logging / later review

    Uses Regex to replace non-alphabet characters with spaces, preventing 
    Whisper from merging words (e.g., "um...apple" -> "um apple").
    """
    segments, _ = model.transcribe(wav_path, language=language)

    transcript_parts = []
    words = []
    for seg in segments:
        seg_text = (seg.text or "").strip()
        if seg_text:
            transcript_parts.append(seg_text)

        # 1. Replace all punctuation and special characters with spaces
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', seg.text or "")

        # 2. Split the cleaned text into individual words
        words.extend(clean_text.split())

    transcript = " ".join(transcript_parts).strip()
    for w in words:
        w_lower = w.lower()
        if w_lower and w_lower not in _FILLERS:
            return w_lower, transcript

    return "", transcript


# ─── Phase Logic ──────────────────────────────────────────────────────────────

def should_reinforce(match_result: dict, reinforced_list: int | None) -> bool:
    """
    Return True if this response earns reinforcement.
    Rules:
    - Phase 3 (reinforced_list=None): never reinforce
    - Must match the reinforced list
    - Must not be a repeat
    """
    if reinforced_list is None:
        return False
    return (
        match_result["list"] == reinforced_list
        and not match_result["repeat"]
        and not match_result["novel"]
    )


def check_phase_switch(
    no_response_streak: int,
    streak_threshold: int,
    list_manager: ListManager,
    reinforced_list: int | None,
) -> tuple[bool, str]:
    """
    Return (should_switch, reason).
    Two criteria:
    1. Consecutive no-response cycles >= threshold
    2. All words on the reinforced list have been said
    """
    if no_response_streak >= streak_threshold:
        return True, f"{no_response_streak} consecutive cycles with no response"
    if reinforced_list and list_manager.is_exhausted(reinforced_list):
        return True, f"All words on List {reinforced_list} exhausted"
    return False, ""


# ─── Main Experiment Loop ─────────────────────────────────────────────────────

class ExperimentRunner:

    def __init__(self, config_path: str = "config.yaml", model=None):
        self.cfg = load_config(config_path)
        self.running = False
        self.current_phase_idx = 0
        self._last_printed_phase = -1
        self.cycle = 0
        self.no_response_streak = 0

        self.on_status      = None
        self.on_iti         = None   # callback(active: bool) — True=ITI started, False=ended
        self.on_pause       = None   # callback(paused: bool)
        self.on_instruction = None   # callback(text: str) — show instruction screen
        self.on_complete    = None   # callback(summary: dict) — session finished
        self.on_cycle_start = None   # callback(window_seconds: float)
        self.on_cycle_end   = None   # callback()

        self._pause_event = threading.Event()
        self._pause_event.set()  # starts in "not paused" (set = go)
        self._force_phase_switch = False

        self._instruction_event = threading.Event()
        self._instruction_event.set()  # starts cleared by show_instruction()

        # Use pre-loaded model if provided
        if model is not None:
            self.model = model
        else:
            asr_cfg = self.cfg["asr"]
            print(f"Loading Whisper model ({asr_cfg['model_size']})...")
            self.model = WhisperModel(asr_cfg["model_size"], device="cpu")
            print("Model ready.")

        # Load word lists
        list_cfg = self.cfg["lists"]
        group = self.cfg["experiment"]["group"]
        list_paths = {
            1: resolve(os.path.join(list_cfg["path"], list_cfg["list1"])),
            2: resolve(os.path.join(list_cfg["path"], list_cfg["list2"])),
        }
        if group == 3:
            list_paths[3] = resolve(os.path.join(list_cfg["path"], list_cfg["list3"]))
        self.list_manager = ListManager(list_paths)

        # Setup logger
        data_cfg = self.cfg["data"]
        exp_cfg = self.cfg["experiment"]
        self.logger = SessionLogger(
            db_path=resolve(data_cfg["db_path"]),
            participant_id=exp_cfg["participant_id"],
            group_num=group,
        )

        # Ensure reward sound exists
        reward_path = resolve(self.cfg["audio"]["reinforcement_sound"])
        if not os.path.exists(reward_path):
            generate_reward_wav(reward_path)
        self.reward_path = reward_path

    @property
    def current_phase(self) -> dict:
        return self.cfg["phases"][self.current_phase_idx]

    @property
    def phase_number(self) -> int:
        return self.current_phase_idx + 1

    def stop(self):
        self.running = False
        self._pause_event.set()  # unblock if currently paused
        self._instruction_event.set()  # unblock if waiting on an instruction screen

    def pause(self):
        self._pause_event.clear()
        if self.on_pause:
            self.on_pause(True)

    def resume(self):
        self._pause_event.set()
        if self.on_pause:
            self.on_pause(False)

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def _wait_if_paused(self):
        """Block here while paused. Returns immediately if not paused."""
        self._pause_event.wait()

    def _show_instruction(self, text: str):
        """Display instruction screen and block until experimenter clicks Continue."""
        if self.on_instruction:
            self._instruction_event.clear()
            self.on_instruction(text)
            self._instruction_event.wait()  # blocks until acknowledge_instruction()

    def _record_response(self) -> tuple[str | None, float | None]:
        """Record one response window using the experiment config."""
        window_seconds = self.cfg["trial"]["window_seconds"]
        if self.on_cycle_start:
            self.on_cycle_start(window_seconds)
        try:
            return record_until_silence(
                window_seconds=window_seconds,
                sample_rate=self.cfg["audio"]["sample_rate"],
                channels=self.cfg["audio"].get("channels", 1),
                silence_cutoff=self.cfg["trial"].get("min_response_gap", 0.4),
            )
        finally:
            if self.on_cycle_end:
                self.on_cycle_end()

    def acknowledge_instruction(self):
        """Called by GUI when experimenter clicks Continue after an instruction screen."""
        self._instruction_event.set()

    def force_next_phase(self):
        """
        Immediately advance to the next phase (thread-safe).
        Called from the GUI thread via experimenter panel button.
        """
        self._force_phase_switch = True

    def _run_practice(self):
        """
        Run a practice round before Phase 1.
        Results are not logged. Word list is separate from experiment lists.
        """
        practice_cfg = self.cfg.get("practice", {})
        if not self.running or not practice_cfg.get("enabled", False):
            return

        list_path = resolve(practice_cfg["list"])
        if not os.path.exists(list_path):
            print(f"  [Practice] List not found: {list_path} - skipping practice.")
            return

        streak_limit = practice_cfg.get("no_response_streak", 3)
        trial_cfg    = self.cfg["trial"]

        # Show practice instruction
        prac_instr = practice_cfg.get("instruction", "")
        if prac_instr:
            self._show_instruction(prac_instr)
            if not self.running:
                return

        # Load practice words into a temporary ListManager
        practice_manager = ListManager({99: list_path})  # list 99 = practice

        print("\n--- Practice Round ---")
        streak = 0
        cycle  = 0

        while self.running and streak < streak_limit and not practice_manager.is_exhausted(99):
            self._wait_if_paused()
            if not self.running:
                break

            cycle += 1
            wav_path, response_time = self._record_response()

            if wav_path is None:
                streak += 1
                print(f"  [Practice] Cycle {cycle:02d} | No response (streak: {streak})")
            else:
                word, transcript = transcribe(self.model, wav_path, self.cfg["asr"]["language"])
                cleanup_audio_file(wav_path)

                if not word:
                    streak += 1
                    print(f"  [Practice] Cycle {cycle:02d} | Unintelligible (streak: {streak})")
                else:
                    streak = 0
                    result = practice_manager.match(word, raw_response=transcript)
                    reinforce = (result["list"] == 99 and not result["repeat"])
                    if reinforce:
                        play_reinforcement(self.reward_path)
                    tag = "[REINFORCED]" if reinforce else f"[{result.get('list', 'novel')}]"
                    print(f"  [Practice] Cycle {cycle:02d} | '{word}' {tag}")

            iti = trial_cfg.get("iti_seconds", 0)
            if iti > 0 and self.running:
                if self.on_iti:
                    self.on_iti(True)
                time.sleep(iti)
                if self.on_iti:
                    self.on_iti(False)

        print("--- Practice complete ---\n")

    def run(self):
        """Start the experiment. Blocks until all phases complete or stop() is called."""
        self.running = True
        trial_cfg = self.cfg["trial"]
        streak_threshold = self.cfg["switching"]["no_response_streak"]

        print(f"\n=== Session started: {self.logger.session_id} ===")
        print(f"Group: {self.cfg['experiment']['group']}-list\n")

        instr_cfg = self.cfg.get("instructions", {})
        intro_text = instr_cfg.get("intro", "")
        between_text = instr_cfg.get("between_phases", "")

        # Show intro instruction before Phase 1
        if intro_text:
            self._show_instruction(intro_text)

        # Run practice round (no-op if disabled)
        if self.running:
            self._run_practice()

        while self.running and self.current_phase_idx < len(self.cfg["phases"]):
            phase = self.current_phase
            reinforced_list = phase["reinforced_list"]

            if self.current_phase_idx != self._last_printed_phase:
                print(f"--- {phase['name']} | Reinforcing List {reinforced_list} ---")
                self._last_printed_phase = self.current_phase_idx

            # ── Pause check ───────────────────────────────────────────────────
            self._wait_if_paused()
            if not self.running:
                break

            # ── Single Cycle ──────────────────────────────────────────────────
            cycle_start = time.perf_counter()
            self.cycle += 1

            # Record
            wav_path, response_time = self._record_response()

            if wav_path is None:
                # No response this cycle
                self.no_response_streak += 1
                self.logger.log_no_response(
                    phase=self.phase_number,
                    cycle=self.cycle,
                    timestamp=cycle_start,
                )
                print(f"  Cycle {self.cycle:03d} | No response (streak: {self.no_response_streak})")
                self._update_status(response="-")

            else:
                # Transcribe
                word, transcript = transcribe(self.model, wav_path, self.cfg["asr"]["language"])
                cleanup_audio_file(wav_path)

                if not word:
                    # Whisper returned empty (noise, breath, etc.)
                    self.no_response_streak += 1
                    self.logger.log_trial(
                        phase=self.phase_number,
                        cycle=self.cycle,
                        timestamp=cycle_start,
                        match_result={
                            "raw": transcript or None,
                            "word": None,
                            "list": None,
                            "novel": False,
                            "repeat": False,
                        },
                        reinforced=False,
                        response_time=response_time,
                    )
                    print(f"  Cycle {self.cycle:03d} | Unintelligible audio (streak: {self.no_response_streak})")
                    self._update_status(response="?")
                else:
                    self.no_response_streak = 0
                    match_result = self.list_manager.match(word, raw_response=transcript)
                    reinforced = should_reinforce(match_result, reinforced_list)

                    if reinforced:
                        play_reinforcement(self.reward_path)

                    self.logger.log_trial(
                        phase=self.phase_number,
                        cycle=self.cycle,
                        timestamp=cycle_start,
                        match_result=match_result,
                        reinforced=reinforced,
                        response_time=response_time,
                    )

                    tag = ""
                    if match_result["novel"]:
                        tag = "[NOVEL]"
                    elif match_result["repeat"]:
                        tag = "[REPEAT]"
                    elif reinforced:
                        tag = "[REINFORCED]"

                    print(
                        f"  Cycle {self.cycle:03d} | '{word}' -> "
                        f"List {match_result['list']} {tag} | rt={response_time:.2f}s"
                    )
                    self._update_status(response=word)

            # ── Inter-Trial Interval ──────────────────────────────────────────
            iti = trial_cfg.get("iti_seconds", 0)
            if iti > 0 and self.running:
                if self.on_iti:
                    self.on_iti(True)
                time.sleep(iti)
                if self.on_iti:
                    self.on_iti(False)

            # ── Check Phase Switch ────────────────────────────────────────────
            if self._force_phase_switch:
                self._force_phase_switch = False
                switch, reason = True, "Manual override by experimenter"
            else:
                switch, reason = check_phase_switch(
                    no_response_streak=self.no_response_streak,
                    streak_threshold=streak_threshold,
                    list_manager=self.list_manager,
                    reinforced_list=reinforced_list,
                )
            if switch:
                print(f"\n  -> Phase switch: {reason}")
                self.no_response_streak = 0
                self.current_phase_idx += 1
                # Show between-phase instruction if there is a next phase
                if (between_text
                        and self.running
                        and self.current_phase_idx < len(self.cfg["phases"])):
                    msg = between_text.format(phase=self.current_phase_idx + 1)
                    self._show_instruction(msg)

        # ── Session End ───────────────────────────────────────────────────────
        self.running = False  # ensure flag is cleared after natural completion
        export = self.cfg["data"].get("export_csv", True)
        csv_path = self.logger.close(export_csv=export)
        print(f"\n=== Session complete ===")
        if csv_path:
            print(f"CSV exported: {csv_path}")

        summary = self.logger.get_summary()
        if self.on_complete:
            self.on_complete(summary)

        end_text = instr_cfg.get("end", "")
        if end_text:
            self._show_instruction(end_text)

    def _update_status(self, response: str):
        """Send status update to GUI if callback is set."""
        if self.on_status:
            reinforced_list = self.current_phase.get("reinforced_list")
            remaining = (
                self.list_manager.remaining_count(reinforced_list)
                if reinforced_list else 0
            )
            self.on_status(
                phase=self.phase_number,
                cycle=self.cycle,
                last_response=response,
                remaining=remaining,
            )


# ─── Quick Test (no GUI) ──────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = ExperimentRunner("config.yaml")

    def print_status(phase, cycle, last_response, remaining):
        print(f"  [GUI] Phase {phase} | Cycle {cycle} | Last: '{last_response}' | Remaining: {remaining}")

    runner.on_status = print_status
    runner.run()
