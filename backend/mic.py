import pvrecorder
devices = pvrecorder.PvRecorder.get_available_devices()
print("\n--- Available Microphones ---")
for index, device in enumerate(devices):
    print(f"[{index}] {device}")
print("-----------------------------\n")