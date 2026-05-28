"""
trial_loop.py
Core experiment loop. Orchestrates each cycle:
  1. Wait for window to open
  2. Record microphone until speech ends or window expires
  3. Transcribe with ASR
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

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from matcher import ListManager
from logger import SessionLogger
from audio import record_until_silence, play_reinforcement, cleanup_audio_file, generate_reward_wav

BASE_DIR = Path(__file__).parent


# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    resolved = BASE_DIR / path
    with open(resolved, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve(relative_path: str) -> str:
    return str(BASE_DIR / relative_path)

# ─── ASR ──────────────────────────────────────────────────────────────────────
import re

_FILLERS = {
    "um", "uh", "er", "ah", "like", "hmm", "hm", "oh", "okay", "ok",
    "so", "well", "yeah", "yes", "no", "the", "a", "an",
}

def _build_asr_prompt(vocabulary: list[str] | None) -> str | None:
    if not vocabulary:
        return None
    words = ", ".join(vocabulary)
    return (
        "The speaker says one clear English word in a behavioral experiment. "
        "Transcribe the spoken word exactly. "
        f"The closed-set vocabulary is: {words}."
    )

def _clean_words(text: str) -> list[str]:
    clean_text = re.sub(r"[^a-zA-Z\s]", " ", text or "")
    return [w.lower() for w in clean_text.split()]

def _extract_response_word(
    transcript: str,
    vocabulary: list[str] | None = None,
    prefer_vocabulary_matches: bool = True,
) -> str:
    words = [w for w in _clean_words(transcript) if w not in _FILLERS]
    if not words:
        return ""

    if prefer_vocabulary_matches and vocabulary:
        vocab = {re.sub(r"[^a-zA-Z]", "", word).lower(): word.lower() for word in vocabulary}
        for word in words:
            if word in vocab:
                return vocab[word]
        for span in (2, 3):
            for i in range(0, len(words) - span + 1):
                joined = "".join(words[i:i + span])
                if joined in vocab:
                    return vocab[joined]
        joined_all = "".join(words)
        if joined_all in vocab:
            return vocab[joined_all]

    return words[0]

def _transcribe_local(
    model,
    wav_path: str,
    language: str = "en",
    vocabulary: list[str] | None = None,
    asr_cfg: dict | None = None,
) -> str:
    if model is None:
        raise RuntimeError("Local ASR selected, but faster-whisper is not available.")

    asr_cfg = asr_cfg or {}
    prompt = _build_asr_prompt(vocabulary)
    transcribe_kwargs = {
        "language": language,
        "condition_on_previous_text": False,
        "beam_size": int(asr_cfg.get("beam_size", 5)),
    }
    if prompt:
        transcribe_kwargs["initial_prompt"] = prompt
        transcribe_kwargs["hotwords"] = " ".join(vocabulary)

    segments, _ = model.transcribe(wav_path, **transcribe_kwargs)
    transcript_parts = []
    for seg in segments:
        seg_text = (seg.text or "").strip()
        if seg_text:
            transcript_parts.append(seg_text)

    return " ".join(transcript_parts).strip()

def _transcribe_openai(
    wav_path: str,
    language: str = "en",
    vocabulary: list[str] | None = None,
    asr_cfg: dict | None = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI ASR selected, but the openai package is not installed.") from exc

    asr_cfg = asr_cfg or {}
    model_name = asr_cfg.get("openai_model", "gpt-4o-transcribe")
    prompt = _build_asr_prompt(vocabulary)
    client = OpenAI()
    with open(wav_path, "rb") as audio_file:
        request = {
            "model": model_name,
            "file": audio_file,
            "response_format": "text",
        }
        if prompt:
            request["prompt"] = prompt
        if language:
            request["language"] = language
        result = client.audio.transcriptions.create(**request)

    if isinstance(result, str):
        return result.strip()
    return getattr(result, "text", str(result)).strip()

def transcribe(
    model,
    wav_path: str,
    language: str = "en",
    vocabulary: list[str] | None = None,
    asr_cfg: dict | None = None,
) -> tuple[str, str]:
    asr_cfg = asr_cfg or {}
    provider = str(asr_cfg.get("provider", "local")).lower()

    if provider in {"local", "whisper", "faster-whisper"}:
        transcript = _transcribe_local(model, wav_path, language, vocabulary, asr_cfg)
    elif provider in {"openai", "api", "cloud"}:
        transcript = _transcribe_openai(wav_path, language, vocabulary, asr_cfg)
    else:
        raise RuntimeError(f"Unsupported ASR provider: {provider}")

    word = _extract_response_word(
        transcript,
        vocabulary=vocabulary,
        prefer_vocabulary_matches=bool(asr_cfg.get("prefer_vocabulary_matches", True)),
    )
    return word, transcript


# ─── Phase Logic ──────────────────────────────────────────────────────────────

def should_reinforce(match_result: dict, reinforced_list: int | None) -> bool:
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
    if no_response_streak >= streak_threshold:
        return True, f"{no_response_streak} consecutive cycles with no response"
    if reinforced_list and list_manager.is_exhausted(reinforced_list):
        return True, f"All words on List {reinforced_list} exhausted"
    return False, ""


# ─── Main Experiment Loop ─────────────────────────────────────────────────────

class ExperimentRunner:

    def __init__(self, config_path: str = "config.yaml", model=None, practice_enabled: bool | None = None):
        self.cfg = load_config(config_path)
        if practice_enabled is not None:
            self.cfg.setdefault("practice", {})["enabled"] = bool(practice_enabled)
        self.running = False
        self.current_phase_idx = 0
        self._last_printed_phase = -1
        self.cycle = 0
        self.no_response_streak = 0
        self.phase_attempts = {
            idx + 1: 1 for idx in range(len(self.cfg.get("phases", [])))
        }

        self.on_status      = None
        self.on_iti         = None   
        self.on_pause       = None   
        self.on_instruction = None   
        self.on_complete    = None   
        self.on_cycle_start = None   
        self.on_cycle_end   = None   
        self.on_reinforcement = None 
        self.on_phase_change = None  

        self._pause_event = threading.Event()
        self._pause_event.set()
        self._force_phase_switch = False
        self._skip_practice = False
        self._restart_practice = False
        self._restart_phase = False
        self.in_practice = False
        self.in_formal_phase = False

        self._instruction_event = threading.Event()
        self._instruction_event.set()

        if model is not None:
            self.model = model
        else:
            asr_cfg = self.cfg.get("asr", {})
            provider = str(asr_cfg.get("provider", "local")).lower()
            if provider in {"local", "whisper", "faster-whisper"}:
                if WhisperModel is None:
                    raise RuntimeError("Local ASR selected, but faster-whisper is not installed.")
                print(f"Loading Whisper model ({asr_cfg['model_size']})...")
                self.model = WhisperModel(asr_cfg["model_size"], device="cpu")
                print("Model ready.")
            else:
                self.model = None
                print(f"Using hosted ASR provider: {provider}")

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

    @property
    def current_phase_attempt(self) -> int:
        return self.phase_attempts.get(self.phase_number, 1)

    def stop(self):
        self.running = False
        self._pause_event.set()
        self._instruction_event.set()

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
        self._pause_event.wait()

    def _show_instruction(self, text: str, words_to_flash: list[str] = None, needs_countdown: bool = False):
        if self.on_instruction:
            self._instruction_event.clear()
            self.on_instruction(text, words_to_flash or [], needs_countdown)
            self._instruction_event.wait() 

    def _record_response(self) -> tuple[str | None, float | None]:
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

    def _phase_context(self, phase: dict | None = None) -> dict:
        phase = phase or self.current_phase
        reinforced_list = phase.get("reinforced_list")
        if reinforced_list is None:
            target = "Extinction: no chime for any word"
        else:
            target = f"List {reinforced_list} earns chime"
        return {
            "mode": "experiment",
            "phase": self.phase_number,
            "phase_attempt": self.current_phase_attempt,
            "name": phase.get("name", f"Phase {self.phase_number}"),
            "reinforced_list": reinforced_list,
            "target": target,
        }

    def _experiment_vocabulary(self) -> list[str]:
        return self.list_manager.vocabulary()

    def _notify_phase_change(self, info: dict):
        if self.on_phase_change:
            self.on_phase_change(info)

    def acknowledge_instruction(self):
        self._instruction_event.set()

    def force_next_phase(self):
        if self.in_practice:
            self._skip_practice = True
            self._instruction_event.set()
        elif self.in_formal_phase:
            self._force_phase_switch = True

    def restart_practice(self):
        if self.in_practice:
            self._restart_practice = True
            self._instruction_event.set()

    def restart_current_phase(self):
        if self.in_practice:
            self.restart_practice()
        elif self.in_formal_phase:
            self._restart_phase = True
            self._instruction_event.set()

    def _apply_phase_restart(self, reason: str = "Manual restart by experimenter"):
        phase_num = self.phase_number
        attempt = self.current_phase_attempt
        self.logger.mark_phase_attempt_discarded(phase=phase_num, phase_attempt=attempt)
        self.phase_attempts[phase_num] = attempt + 1
        self.no_response_streak = 0
        self._force_phase_switch = False
        self._last_printed_phase = -1

        reinforced_list = self.current_phase.get("reinforced_list")
        if reinforced_list:
            self.list_manager.reset_list(reinforced_list)

        print(f"\n  -> Restart {self.current_phase['name']}: {reason}; attempt {attempt + 1}")
        self._notify_phase_change(self._phase_context(self.current_phase))

    def _run_practice(self):
        practice_cfg = self.cfg.get("practice", {})
        if not self.running or not practice_cfg.get("enabled", False):
            return

        self.in_practice = True
        self._skip_practice = False

        list_path = resolve(practice_cfg["list"])
        if not os.path.exists(list_path):
            print(f"  [Practice] List not found: {list_path} - skipping practice.")
            self.in_practice = False
            return

        streak_limit = practice_cfg.get("no_response_streak", 3)
        trial_cfg    = self.cfg["trial"]
        practice_manager = ListManager({99: list_path})

        prac_instr = practice_cfg.get("instruction", "")
        if prac_instr:
            self._show_instruction(prac_instr, practice_manager.vocabulary([99]), needs_countdown=True)
            if not self.running:
                self.in_practice = False
                return

        practice_attempt = 0
        while self.running and not self._skip_practice:
            practice_attempt += 1
            self._restart_practice = False

            print(f"\n--- Practice Round attempt {practice_attempt} ---")
            self._notify_phase_change({
                "mode": "practice",
                "phase": "practice",
                "phase_attempt": practice_attempt,
                "name": "Practice Round",
                "reinforced_list": "practice",
                "target": "Practice list earns chime",
            })
            streak = 0
            cycle  = 0

            while (self.running
                   and not self._skip_practice
                   and not self._restart_practice
                   and streak < streak_limit
                   and not practice_manager.is_exhausted(99)):
                self._wait_if_paused()
                if not self.running:
                    break

                cycle += 1
                wav_path, response_time = self._record_response()

                if wav_path is None:
                    streak += 1
                    print(f"  [Practice] Cycle {cycle:02d} | No response (streak: {streak})")
                else:
                    word, transcript = transcribe(
                        self.model,
                        wav_path,
                        self.cfg["asr"]["language"],
                        vocabulary=practice_manager.vocabulary([99]),
                        asr_cfg=self.cfg.get("asr", {}),
                    )
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
                            if self.on_reinforcement:
                                self.on_reinforcement({
                                    "phase": "practice",
                                    "cycle": cycle,
                                    "word": result["word"],
                                    "matched_list": result["list"],
                                })
                        tag = "[REINFORCED]" if reinforce else f"[{result.get('list', 'novel')}]"
                        print(f"  [Practice] Cycle {cycle:02d} | '{word}' {tag}")

                iti = trial_cfg.get("iti_seconds", 0)
                if iti > 0 and self.running:
                    if self.on_iti:
                        self.on_iti(True)
                    time.sleep(iti)
                    if self.on_iti:
                        self.on_iti(False)

            if not self._restart_practice:
                break

        print("--- Practice complete ---\n")
        self.in_practice = False
        self._skip_practice = False
        self._restart_practice = False

    def run(self):
        self.running = True
        trial_cfg = self.cfg["trial"]
        streak_threshold = self.cfg["switching"]["no_response_streak"]
        instr_cfg = self.cfg.get("instructions", {})

        print(f"\n=== Session started: {self.logger.session_id} ===")
        print(f"Group: {self.cfg['experiment']['group']}-list\n")

        intro_text = instr_cfg.get("intro", "")
        if intro_text:
            self._show_instruction(intro_text, [], needs_countdown=False)

        if self.running:
            self._run_practice()

        self.in_formal_phase = True
        while self.running and self.current_phase_idx < len(self.cfg["phases"]):
            phase = self.current_phase
            reinforced_list = phase.get("reinforced_list")

            if self.current_phase_idx != self._last_printed_phase:
                print(f"--- {phase.get('name', 'Phase')} | Reinforcing List {reinforced_list} ---")
                self._last_printed_phase = self.current_phase_idx
                self._notify_phase_change(self._phase_context(phase))

                lists_to_extract = []
                if reinforced_list is not None:
                    lists_to_extract.append(reinforced_list)
                lists_to_extract.extend(phase.get("extinction_lists", []))
                phase_words = self.list_manager.vocabulary(lists_to_extract) if lists_to_extract else []

                if self.current_phase_idx == 0:
                    msg = "Phase 1 is coming next.\n\nThe real test is about to begin.\n\nPlease get ready."
                else:
                    between_text = instr_cfg.get("between_phases", "Please wait. Phase {phase} will begin shortly.")
                    msg = between_text.format(phase=self.current_phase_idx + 1)

                if self.running:
                    self._show_instruction(msg, phase_words, needs_countdown=True)

            if self._restart_phase:
                self._restart_phase = False
                self._apply_phase_restart()
                continue

            self._wait_if_paused()
            if not self.running:
                break

            cycle_start = time.perf_counter()
            self.cycle += 1

            wav_path, response_time = self._record_response()

            if wav_path is None:
                self.no_response_streak += 1
                self.logger.log_no_response(
                    phase=self.phase_number,
                    phase_attempt=self.current_phase_attempt,
                    cycle=self.cycle,
                    timestamp=cycle_start,
                )
                print(f"  Cycle {self.cycle:03d} | No response (streak: {self.no_response_streak})")
                self._update_status(response="-")
            else:
                word, transcript = transcribe(
                    self.model,
                    wav_path,
                    self.cfg["asr"]["language"],
                    vocabulary=self._experiment_vocabulary(),
                    asr_cfg=self.cfg.get("asr", {}),
                )
                cleanup_audio_file(wav_path)

                if not word:
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
                        phase_attempt=self.current_phase_attempt,
                    )
                    print(f"  Cycle {self.cycle:03d} | Unintelligible audio (streak: {self.no_response_streak})")
                    self._update_status(response="?")
                else:
                    self.no_response_streak = 0
                    match_result = self.list_manager.match(word, raw_response=transcript)
                    reinforced = should_reinforce(match_result, reinforced_list)

                    if reinforced:
                        play_reinforcement(self.reward_path)
                        if self.on_reinforcement:
                            self.on_reinforcement({
                                "phase": self.phase_number,
                                "cycle": self.cycle,
                                "word": match_result["word"],
                                "matched_list": match_result["list"],
                            })

                    self.logger.log_trial(
                        phase=self.phase_number,
                        cycle=self.cycle,
                        timestamp=cycle_start,
                        match_result=match_result,
                        reinforced=reinforced,
                        response_time=response_time,
                        phase_attempt=self.current_phase_attempt,
                    )

                    tag = ""
                    if match_result["novel"]:
                        tag = "[NOVEL]"
                    elif match_result["repeat"]:
                        tag = "[REPEAT]"
                    elif reinforced:
                        tag = "[REINFORCED]"

                    print(f"  Cycle {self.cycle:03d} | '{word}' -> List {match_result['list']} {tag} | rt={response_time:.2f}s")
                    self._update_status(response=word)

            iti = trial_cfg.get("iti_seconds", 0)
            if iti > 0 and self.running:
                if self.on_iti:
                    self.on_iti(True)
                time.sleep(iti)
                if self.on_iti:
                    self.on_iti(False)

            if self._restart_phase:
                self._restart_phase = False
                self._apply_phase_restart()
                continue

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

        self.in_formal_phase = False
        self.running = False 
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
            self._show_instruction(end_text, [], needs_countdown=False)

    def _update_status(self, response: str):
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

if __name__ == "__main__":
    runner = ExperimentRunner("config.yaml")

    def print_status(phase, cycle, last_response, remaining):
        print(f"  [GUI] Phase {phase} | Cycle {cycle} | Last: '{last_response}' | Remaining: {remaining}")

    runner.on_status = print_status
    runner.run()