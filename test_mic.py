import sounddevice as sd
import numpy as np

print("录音3秒，请说话...")
audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="float32")
sd.wait()
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"RMS音量: {rms:.4f}")
if rms > 0.01:
    print("麦克风正常")
elif rms > 0.001:
    print("有声音但很小，silence_threshold需要调低")
else:
    print("几乎无声音，检查麦克风权限或设备")
