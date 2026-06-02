import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

api_key = os.environ.get("ELEVENLABS_API_KEY")

if not api_key:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")

client = ElevenLabs(api_key=api_key)

OUT_DIR = Path("data/clean_mp3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

voice_id = "UgBBYS2sOqTuMpoF3BR0"

text = "The package arrived before noon."

audio_stream = client.text_to_speech.convert(
    voice_id=voice_id,
    text=text,
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

out_path = OUT_DIR / "test_clip.mp3"

with open(out_path, "wb") as f:
    for chunk in audio_stream:
        f.write(chunk)

print(f"Saved audio to {out_path}")