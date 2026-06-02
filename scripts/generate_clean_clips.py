import csv
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

api_key = os.environ.get("ELEVENLABS_API_KEY")
if not api_key:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in .env")

client = ElevenLabs(api_key=api_key)

SENTENCES_CSV = Path("sentences.csv")
VOICES_CSV = Path("voices.csv")

OUT_DIR = Path("data/clean_mp3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_OUT = Path("metadata_clean.csv")


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


sentences = read_csv(SENTENCES_CSV)
voices = read_csv(VOICES_CSV)

required_sentence_cols = {"sentence_id", "text"}
required_voice_cols = {"voice_id", "elevenlabs_voice_id", "display_name"}

if not required_sentence_cols.issubset(sentences[0].keys()):
    raise ValueError(f"sentences.csv must contain columns: {required_sentence_cols}")

if not required_voice_cols.issubset(voices[0].keys()):
    raise ValueError(f"voices.csv must contain columns: {required_voice_cols}")

metadata_rows = []

total = len(sentences) * len(voices)
count = 0

for voice in voices:
    local_voice_id = voice["voice_id"]
    elevenlabs_voice_id = voice["elevenlabs_voice_id"]
    display_name = voice["display_name"]

    for sentence in sentences:
        count += 1

        sentence_id = sentence["sentence_id"]
        text = sentence["text"]

        clip_id = f"{local_voice_id}_{sentence_id}_clean"
        out_path = OUT_DIR / f"{clip_id}.mp3"

        print(f"[{count}/{total}] {clip_id} | {display_name} | {text}")

        if out_path.exists():
            print(f"  Skipping existing file: {out_path}")
        else:
            try:
                audio_stream = client.text_to_speech.convert(
                    voice_id=elevenlabs_voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )

                with out_path.open("wb") as f:
                    for chunk in audio_stream:
                        f.write(chunk)

                # Be polite to the API and avoid rapid-fire requests.
                time.sleep(0.25)

            except Exception as e:
                print(f"  ERROR generating {clip_id}: {e}")
                continue

        metadata_rows.append({
            "clip_id": clip_id,
            "voice_id": local_voice_id,
            "elevenlabs_voice_id": elevenlabs_voice_id,
            "display_name": display_name,
            "sentence_id": sentence_id,
            "transcript": text,
            "audio_type": "clean",
            "noise_type": "clean",
            "snr_db": "",
            "path": str(out_path),
            "split": ""
        })


fieldnames = [
    "clip_id",
    "voice_id",
    "elevenlabs_voice_id",
    "display_name",
    "sentence_id",
    "transcript",
    "audio_type",
    "noise_type",
    "snr_db",
    "path",
    "split",
]

with METADATA_OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"\nDone. Wrote {len(metadata_rows)} rows to {METADATA_OUT}")