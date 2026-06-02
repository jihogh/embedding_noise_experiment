import whisper

AUDIO_PATH = "data/noisy_wav/voice_01_s001_cafe_0db.wav"

model = whisper.load_model("base")
result = model.transcribe(AUDIO_PATH, language="en")

print("Prediction:")
print(result["text"].strip())