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
import math

_play_lock = threading.Lock()


def _format_for_output(data: np.ndarray) -> np.ndarray:
    """Return mono or duplicated-stereo audio that fits the default output device."""
    if data.ndim == 1:
        mono = data
    elif data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1)

    try:
        output = sd.query_devices(kind="output")
        max_channels = int(output.get("max_output_channels", 1))
    except Exception:
        max_channels = 1

    if max_channels >= 2:
        return np.column_stack((mono, mono)).astype(np.float32)
    return mono.astype(np.float32)


# ─── Recording ────────────────────────────────────────────────────────────────

def record_until_silence(
    window_seconds: float = 3.0,
    sample_rate: int = 16000,
    channels: int = 1,
    silence_cutoff: float = 0.4,     # stop after this many seconds of silence post-speech
    silence_threshold: float = 0.01, # RMS amplitude below this = silence
    chunk_duration: float = 0.05,    # process audio in 50ms chunks
    min_audio_seconds: float = 1.2,  # Whisper is less reliable on very short isolated words
) -> tuple[str | None, float | None]:
    """
    Record from microphone. Stop early if silence is detected after speech.

    Returns:
        (wav_path, response_time) if speech was detected
        (None, None)              if no speech within window_seconds
    """
    chunk_size = int(sample_rate * chunk_duration)
    max_chunks = int(window_seconds / chunk_duration)
    silence_chunks_needed = max(1, math.ceil(silence_cutoff / chunk_duration))

    frames = []
    speech_detected = False
    silence_count = 0
    speech_start_chunk = None

    stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32")
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

    # Keep a short pre-roll so soft initial consonants are not clipped.
    pre_roll_chunks = 2
    safe_start_chunk = max(0, speech_start_chunk - pre_roll_chunks)

    response_time = round(speech_start_chunk * chunk_duration, 3)
    audio = np.concatenate(frames[safe_start_chunk:], axis=0)
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            # Downmix to mono so ASR sees a consistent input format.
            audio = np.mean(audio, axis=1)
    min_samples = int(sample_rate * min_audio_seconds)
    if audio.shape[0] < min_samples:
        audio = np.pad(audio, (0, min_samples - audio.shape[0]))

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Write the numpy array to the temporary wav file
    sf.write(tmp_path, audio, sample_rate)

    return tmp_path, response_time


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
            with _play_lock:
                if os.path.exists(sound_path):
                    data, sr = sf.read(sound_path, dtype="float32")
                    sd.play(_format_for_output(data), sr)
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
