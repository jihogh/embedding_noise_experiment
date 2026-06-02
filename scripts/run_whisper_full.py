import re
from pathlib import Path

import pandas as pd
import torch
import whisper
from jiwer import wer, cer


INPUT_CSV = Path("metadata_noisy.csv")
OUTPUT_CSV = Path("asr_results_full.csv")

MODEL_NAME = "base"


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()
df["snr_db"] = pd.to_numeric(df["snr_db"])

print(f"Total clips to transcribe: {len(df)}")
print(df.groupby(["noise_type", "snr_db"]).size())

model = whisper.load_model(MODEL_NAME, device=device)

rows = []

# If output already exists, resume from where you left off.
if OUTPUT_CSV.exists():
    existing = pd.read_csv(OUTPUT_CSV)
    done_clip_ids = set(existing["clip_id"].astype(str))
    rows = existing.to_dict("records")
    print(f"Resuming. Already completed {len(done_clip_ids)} clips.")
else:
    done_clip_ids = set()

for i, row in df.iterrows():
    clip_id = str(row["clip_id"])

    if clip_id in done_clip_ids:
        continue

    audio_path = row["path"]
    reference_raw = row["transcript"]

    print(f"[{i + 1}/{len(df)}] {audio_path}")

    try:
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=(device == "cuda"),
            verbose=False,
        )

        prediction_raw = result["text"].strip()

        reference = normalize_text(reference_raw)
        prediction = normalize_text(prediction_raw)

        rows.append({
            "clip_id": clip_id,
            "voice_id": row["voice_id"],
            "sentence_id": row["sentence_id"],
            "noise_type": row["noise_type"],
            "snr_db": row["snr_db"],
            "path": audio_path,
            "reference_raw": reference_raw,
            "prediction_raw": prediction_raw,
            "reference_norm": reference,
            "prediction_norm": prediction,
            "wer": wer(reference, prediction),
            "cer": cer(reference, prediction),
        })

    except Exception as e:
        print(f"ERROR on {clip_id}: {e}")

        rows.append({
            "clip_id": clip_id,
            "voice_id": row.get("voice_id", ""),
            "sentence_id": row.get("sentence_id", ""),
            "noise_type": row.get("noise_type", ""),
            "snr_db": row.get("snr_db", ""),
            "path": audio_path,
            "reference_raw": reference_raw,
            "prediction_raw": "",
            "reference_norm": normalize_text(reference_raw),
            "prediction_norm": "",
            "wer": None,
            "cer": None,
            "error": str(e),
        })

    # Save progress every clip so you can stop/restart safely.
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

print(f"Done. Wrote {OUTPUT_CSV}")