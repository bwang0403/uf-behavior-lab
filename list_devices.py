import sounddevice as sd

print("可用音频设备：")
print(sd.query_devices())
print(f"\n当前默认输入设备: {sd.query_devices(kind='input')['name']}")
