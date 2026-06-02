from pathlib import Path
from pydub import AudioSegment

IN_DIR = Path("data/clean_mp3")
OUT_DIR = Path("data/clean_wav")

OUT_DIR.mkdir(parents=True, exist_ok=True)

mp3_files = sorted(IN_DIR.glob("*.mp3"))

print(f"Found {len(mp3_files)} MP3 files")

for i, mp3_path in enumerate(mp3_files, start=1):
    wav_path = OUT_DIR / mp3_path.with_suffix(".wav").name

    if wav_path.exists():
        print(f"[{i}/{len(mp3_files)}] Skipping existing {wav_path.name}")
        continue

    audio = AudioSegment.from_mp3(mp3_path)

    # Standardize for ASR/noise experiments
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)

    audio.export(wav_path, format="wav")

    print(f"[{i}/{len(mp3_files)}] Converted {mp3_path.name} -> {wav_path.name}")

print("Done.")