import sounddevice as sd
import numpy as np

print("Recording for 3 seconds. Please speak...")
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32")
sd.wait()
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"RMS volume: {rms:.4f}")
if rms > 0.01:
    print("Microphone is working normally.")
elif rms > 0.001:
    print("Audio detected, but the level is low. Lower silence_threshold.")
else:
    print("Almost no audio detected. Check microphone permissions or the selected device.")
