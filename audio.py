"""
audio.py
Handles microphone recording with early cutoff on silence,
and reinforcement sound playback via sounddevice (no pygame required).

Recording logic:
- Opens mic and records in small chunks
- After the first non-silent chunk (speech detected), starts watching for silence
- If silence persists for `silence_cutoff` seconds → stop early and return audio
- If window_seconds elapses with no speech → return None (no response)
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import threading
import time


# ─── Recording ────────────────────────────────────────────────────────────────

def record_until_silence(
    window_seconds: float = 3.0,
    sample_rate: int = 16000,
    silence_cutoff: float = 0.4,     # stop after this many seconds of silence post-speech
    silence_threshold: float = 0.01, # RMS amplitude below this = silence
    chunk_duration: float = 0.05,    # process audio in 50ms chunks
) -> tuple[str | None, float | None]:
    """
    Record from microphone. Stop early if silence is detected after speech.

    Returns:
        (wav_path, response_time) if speech was detected
        (None, None)              if no speech within window_seconds
    """
    chunk_size = int(sample_rate * chunk_duration)
    max_chunks = int(window_seconds / chunk_duration)
    silence_chunks_needed = int(silence_cutoff / chunk_duration)

    frames = []
    speech_detected = False
    silence_count = 0
    speech_start_chunk = None

    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32")
    stream.start()

    try:
        for i in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_silent = rms < silence_threshold

            if not speech_detected:
                if not is_silent:
                    speech_detected = True
                    speech_start_chunk = i
                    silence_count = 0
            else:
                if is_silent:
                    silence_count += 1
                    if silence_count >= silence_chunks_needed:
                        break
                else:
                    silence_count = 0

    finally:
        stream.stop()
        stream.close()

    if not speech_detected:
        return None, None

    response_time = round(speech_start_chunk * chunk_duration, 3)
    audio = np.concatenate(frames[speech_start_chunk:], axis=0)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)

    return tmp.name, response_time


def cleanup_audio_file(path: str):
    """Delete temporary wav file after Whisper has processed it."""
    try:
        os.unlink(path)
    except Exception:
        pass


# ─── Reinforcement Sound ──────────────────────────────────────────────────────

def play_reinforcement(sound_path: str):
    """
    Play the reinforcement sound in a background thread (non-blocking).
    Uses sounddevice — no pygame required.
    Falls back to a generated tone if the wav file is missing.
    """
    def _play():
        try:
            if os.path.exists(sound_path):
                data, sr = sf.read(sound_path, dtype="float32")
                sd.play(data, sr)
                sd.wait()
            else:
                _play_tone()
        except Exception as e:
            print(f"  [audio] play_reinforcement error: {e}")

    threading.Thread(target=_play, daemon=True).start()


def _play_tone(frequency: float = 880, duration: float = 0.3, volume: float = 0.5):
    """Generate and play a sine wave tone directly via sounddevice."""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = (np.sin(2 * np.pi * frequency * t) * volume).astype(np.float32)
    sd.play(wave, sample_rate)
    sd.wait()


def generate_reward_wav(path: str):
    """
    Generate a default reward .wav file (pleasant two-tone chime).
    Uses numpy + soundfile only — no pygame required.
    """
    sample_rate = 44100
    duration = 0.4

    def tone(freq, dur):
        n = int(sample_rate * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        fade = np.linspace(1.0, 0.0, n)
        return (np.sin(2 * np.pi * freq * t) * fade).astype(np.float32)

    wave = np.concatenate([tone(880, duration / 2), tone(1100, duration / 2)])
    wave *= 0.5

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, wave, sample_rate)
    print(f"  Generated reward sound: {path}")


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    reward_path = str(Path(__file__).parent / "audio" / "reward.wav")

    if not os.path.exists(reward_path):
        generate_reward_wav(reward_path)

    print("Testing reinforcement sound...")
    play_reinforcement(reward_path)
    time.sleep(1)

    print("\nTesting recording (speak something within 3 seconds)...")
    wav_path, rt = record_until_silence(window_seconds=3.0)

    if wav_path:
        print(f"Speech detected! Response time: {rt:.2f}s")
        cleanup_audio_file(wav_path)
    else:
        print("No speech detected.")
