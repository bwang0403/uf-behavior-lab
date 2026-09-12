import sounddevice as sd

print("Available audio devices:")
print(sd.query_devices())
print(f"\nCurrent default input device: {sd.query_devices(kind='input')['name']}")
