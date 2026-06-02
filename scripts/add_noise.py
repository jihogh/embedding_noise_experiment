import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

CLEAN_DIR = Path("data/clean_wav")
NOISE_DIR = Path("data/noise_sources")
OUT_DIR = Path("data/noisy_wav")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_CLEAN = Path("metadata_clean.csv")
METADATA_NOISY = Path("metadata_noisy.csv")

NOISES = {
    "cafe": NOISE_DIR / "cafe.wav",
    "traffic": NOISE_DIR / "traffic.wav",
    "fan": NOISE_DIR / "machinery.wav",
}

SNR_LEVELS = [20, 10, 5, 0, -5, -10]
SAMPLE_RATE = 16000

random.seed(42)
np.random.seed(42)


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def make_mono(x):
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def get_noise_segment(noise, length):
    if len(noise) < length:
        repeats = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, repeats)

    start = random.randint(0, len(noise) - length)
    return noise[start:start + length], start


def mix_snr(speech, noise, snr_db):
    speech_rms = rms(speech)
    noise_rms = rms(noise)

    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)

    mixed = speech + noise

    # avoid clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    return mixed.astype(np.float32)


# load metadata
with open(METADATA_CLEAN, newline="", encoding="utf-8") as f:
    clean_rows = list(csv.DictReader(f))

# load noise files
noise_audio = {}

for noise_name, noise_path in NOISES.items():
    noise, sr = sf.read(noise_path)
    noise = make_mono(noise)

    if sr != SAMPLE_RATE:
        raise ValueError(f"{noise_path} is {sr} Hz, expected {SAMPLE_RATE} Hz")

    noise_audio[noise_name] = noise
    print(f"Loaded {noise_name}: {len(noise) / sr:.1f} seconds")


out_rows = []
total = len(clean_rows) * len(NOISES) * len(SNR_LEVELS)
count = 0

for row in clean_rows:
    # your metadata path may point to clean_mp3, so rebuild wav path
    wav_name = Path(row["path"]).with_suffix(".wav").name
    clean_path = CLEAN_DIR / wav_name

    speech, sr = sf.read(clean_path)
    speech = make_mono(speech)

    if sr != SAMPLE_RATE:
        raise ValueError(f"{clean_path} is {sr} Hz, expected {SAMPLE_RATE} Hz")

    for noise_name, noise in noise_audio.items():
        for snr_db in SNR_LEVELS:
            count += 1

            clip_id = f"{row['voice_id']}_{row['sentence_id']}_{noise_name}_{snr_db}db"
            out_path = OUT_DIR / f"{clip_id}.wav"

            noise_seg, start = get_noise_segment(noise, len(speech))
            mixed = mix_snr(speech, noise_seg, snr_db)

            sf.write(out_path, mixed, sr)

            print(f"[{count}/{total}] {out_path.name}")

            out_rows.append({
                "clip_id": clip_id,
                "voice_id": row["voice_id"],
                "sentence_id": row["sentence_id"],
                "transcript": row["transcript"],
                "audio_type": "noisy",
                "noise_type": noise_name,
                "snr_db": snr_db,
                "noise_start_sample": start,
                "path": str(out_path),
            })


with open(METADATA_NOISY, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "clip_id",
        "voice_id",
        "sentence_id",
        "transcript",
        "audio_type",
        "noise_type",
        "snr_db",
        "noise_start_sample",
        "path",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Done. Wrote {len(out_rows)} files.")