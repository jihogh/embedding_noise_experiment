import re
from pathlib import Path

import pandas as pd
import torch
import whisper
from jiwer import wer, cer


INPUT_CSV = Path("metadata_noisy.csv")
OUTPUT_CSV = Path("asr_results_sample.csv")

MODEL_NAME = "base"

# 3 noise types × 6 SNR levels × 30 = 540 clips
SAMPLES_PER_GROUP = 30


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

sample_parts = []

for (noise_type, snr), group in df.groupby(["noise_type", "snr_db"]):
    sample_parts.append(
        group.sample(
            n=min(len(group), SAMPLES_PER_GROUP),
            random_state=42
        )
    )

sample_df = pd.concat(sample_parts, ignore_index=True)

print("Sample counts:")
print(sample_df.groupby(["noise_type", "snr_db"]).size())
print()
print(f"Total clips to transcribe: {len(sample_df)}")

model = whisper.load_model(MODEL_NAME, device=device)

rows = []

for i, row in sample_df.iterrows():
    audio_path = row["path"]
    reference_raw = row["transcript"]

    print(f"[{i + 1}/{len(sample_df)}] {audio_path}")

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
        "clip_id": row["clip_id"],
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

    # Save every clip so progress is not lost if it crashes.
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

print(f"Done. Wrote {OUTPUT_CSV}")