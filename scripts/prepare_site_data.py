from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_PUBLIC = ROOT / "site" / "public"
SITE_DATA = SITE_PUBLIC / "data"
SITE_AUDIO = SITE_PUBLIC / "audio-demo"

RESULT_CANDIDATES = ["asr_results_sample.csv", "asr_results_full.csv"]
SUMMARY_CANDIDATES = ["asr_summary_sample.csv", "asr_summary_full.csv"]
DEMO_ROWS_PER_GROUP = 3
INTRO_CLEAN_OPTIONS = 12
INTRO_NOISY_OPTIONS = 12
INTRO_SNR_PRIORITY = [-5, -10, 0]
NOISE_ORDER = {"cafe": 0, "fan": 1, "traffic": 2}


def first_existing(names: list[str]) -> Path:
    for name in names:
        path = ROOT / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist: {', '.join(names)}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def as_int(value: str | None, default: int = 0) -> int:
    return int(round(as_float(value, default)))


def repo_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    return ROOT / raw_path.replace("\\", "/")


def browser_copy(raw_path: str | None) -> str | None:
    source = repo_path(raw_path)
    if source is None or not source.exists():
        return None
    SITE_AUDIO.mkdir(parents=True, exist_ok=True)
    target = SITE_AUDIO / source.name
    shutil.copy2(source, target)
    return f"audio-demo/{target.name}"


def clean_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    return (
        NOISE_ORDER.get(row.get("noise_type", ""), 99),
        as_int(row.get("snr_db")),
        row.get("clip_id", ""),
    )


def result_payload(row: dict[str, str], audio_path: str) -> dict[str, object]:
    return {
        "clipId": row.get("clip_id", ""),
        "voiceId": row.get("voice_id", ""),
        "sentenceId": row.get("sentence_id", ""),
        "noiseType": row.get("noise_type", ""),
        "snrDb": as_int(row.get("snr_db")),
        "audioPath": audio_path,
        "reference": row.get("reference_raw") or row.get("reference_norm") or "",
        "prediction": row.get("prediction_raw") or row.get("prediction_norm") or "",
        "referenceNorm": row.get("reference_norm", ""),
        "predictionNorm": row.get("prediction_norm", ""),
        "whisperWer": as_float(row.get("wer")),
        "whisperCer": as_float(row.get("cer")),
    }


def summary_payload(row: dict[str, str]) -> dict[str, object]:
    return {
        "noiseType": row.get("noise_type", ""),
        "snrDb": as_int(row.get("snr_db")),
        "meanWer": as_float(row.get("mean_wer")),
        "medianWer": as_float(row.get("median_wer")),
        "meanCer": as_float(row.get("mean_cer")),
        "medianCer": as_float(row.get("median_cer")),
        "n": as_int(row.get("n")),
    }


def find_clean_metadata(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    clean_rows = {}
    for row in rows:
        key = (row.get("voice_id", ""), row.get("sentence_id", ""))
        if key[0] and key[1] and key not in clean_rows:
            clean_rows[key] = row
    return clean_rows


def choose_clean_intro_row(
    clean_by_key: dict[tuple[str, str], dict[str, str]], noisy_row: dict[str, str]
) -> dict[str, str]:
    noisy_key = (noisy_row.get("voice_id", ""), noisy_row.get("sentence_id", ""))
    same_voice = [
        row
        for key, row in clean_by_key.items()
        if key != noisy_key and key[0] == noisy_key[0] and repo_path(row.get("path")) is not None and repo_path(row.get("path")).exists()
    ]
    if same_voice:
        return sorted(same_voice, key=lambda row: row.get("clip_id", ""))[0]

    candidates = [
        row
        for key, row in clean_by_key.items()
        if key != noisy_key and repo_path(row.get("path")) is not None and repo_path(row.get("path")).exists()
    ]
    if not candidates:
        raise RuntimeError("Could not find a clean intro clip with a different sentence.")
    return sorted(candidates, key=lambda row: row.get("clip_id", ""))[0]


def choose_intro_row(
    result_rows: list[dict[str, str]], clean_by_key: dict[tuple[str, str], dict[str, str]]
) -> dict[str, str]:
    for snr in INTRO_SNR_PRIORITY:
        candidates = [
            row
            for row in result_rows
            if row.get("noise_type") == "cafe"
            and as_int(row.get("snr_db")) == snr
            and repo_path(row.get("path")) is not None
            and repo_path(row.get("path")).exists()
            and (row.get("voice_id", ""), row.get("sentence_id", "")) in clean_by_key
        ]
        candidates.sort(key=lambda row: (abs(as_float(row.get("wer")) - 0.6), row.get("clip_id", "")))
        if candidates:
            return candidates[0]
    raise RuntimeError("Could not find a cafe intro row with matching clean audio.")


def clean_intro_payload(row: dict[str, str], audio_path: str) -> dict[str, object]:
    return {
        "clipId": row.get("clip_id", ""),
        "voiceId": row.get("voice_id", ""),
        "sentenceId": row.get("sentence_id", ""),
        "displayName": row.get("display_name", ""),
        "transcript": row.get("transcript") or "",
        "audioPath": audio_path,
    }


def noisy_intro_payload(row: dict[str, str], audio_path: str) -> dict[str, object]:
    return {
        "clipId": row.get("clip_id", ""),
        "voiceId": row.get("voice_id", ""),
        "sentenceId": row.get("sentence_id", ""),
        "noiseType": row.get("noise_type", ""),
        "snrDb": as_int(row.get("snr_db")),
        "transcript": row.get("reference_raw") or row.get("reference_norm") or "",
        "audioPath": audio_path,
        "whisperPrediction": row.get("prediction_raw") or row.get("prediction_norm") or "",
        "whisperWer": as_float(row.get("wer")),
    }


def make_clean_intro_options(clean_by_key: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, object]]:
    rows = [
        row
        for row in clean_by_key.values()
        if repo_path(row.get("path")) is not None and repo_path(row.get("path")).exists()
    ]
    rows.sort(key=lambda row: (row.get("voice_id", ""), row.get("sentence_id", "")))

    selected = []
    used_sentences = set()
    for row in rows:
        sentence_id = row.get("sentence_id", "")
        if sentence_id in used_sentences:
            continue
        audio_path = browser_copy(row.get("path"))
        if audio_path:
            selected.append(clean_intro_payload(row, audio_path))
            used_sentences.add(sentence_id)
        if len(selected) >= INTRO_CLEAN_OPTIONS:
            break

    if not selected:
        raise RuntimeError("Could not find clean intro options.")
    return selected


def make_noisy_intro_options(result_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected = []
    used_sentences = set()
    for snr in INTRO_SNR_PRIORITY:
        candidates = [
            row
            for row in result_rows
            if row.get("noise_type") == "cafe"
            and as_int(row.get("snr_db")) == snr
            and repo_path(row.get("path")) is not None
            and repo_path(row.get("path")).exists()
        ]
        candidates.sort(key=lambda row: (abs(as_float(row.get("wer")) - 0.6), row.get("clip_id", "")))
        for row in candidates:
            sentence_id = row.get("sentence_id", "")
            if sentence_id in used_sentences:
                continue
            audio_path = browser_copy(row.get("path"))
            if audio_path:
                selected.append(noisy_intro_payload(row, audio_path))
                used_sentences.add(sentence_id)
            if len(selected) >= INTRO_NOISY_OPTIONS:
                return selected

    if not selected:
        raise RuntimeError("Could not find noisy intro options.")
    return selected


def make_intro(
    result_rows: list[dict[str, str]], clean_by_key: dict[tuple[str, str], dict[str, str]]
) -> dict[str, object]:
    clean_options = make_clean_intro_options(clean_by_key)
    noisy_options = make_noisy_intro_options(result_rows)
    clean_default = clean_options[0]
    noisy_default = next(
        (option for option in noisy_options if option.get("sentenceId") != clean_default.get("sentenceId")),
        noisy_options[0],
    )

    return {
        "clean": clean_default,
        "noisy": noisy_default,
        "cleanOptions": clean_options,
        "noisyOptions": noisy_options,
    }


def make_demo_results(result_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        path = repo_path(row.get("path"))
        if path and path.exists():
            grouped[(row.get("noise_type", ""), as_int(row.get("snr_db")))].append(row)

    rng = random.Random(7)
    demo_rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (NOISE_ORDER.get(item[0], 99), item[1])):
        rows = grouped[key]
        rows.sort(key=lambda row: row.get("clip_id", ""))
        selected = rng.sample(rows, min(DEMO_ROWS_PER_GROUP, len(rows)))
        selected.sort(key=lambda row: row.get("clip_id", ""))
        for row in selected:
            audio_path = browser_copy(row.get("path"))
            if audio_path:
                demo_rows.append(result_payload(row, audio_path))

    return demo_rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    result_path = first_existing(RESULT_CANDIDATES)
    summary_path = first_existing(SUMMARY_CANDIDATES)
    metadata_path = ROOT / "metadata_clean.csv"
    if not metadata_path.exists():
        raise FileNotFoundError("metadata_clean.csv is required for intro clean clips.")

    SITE_DATA.mkdir(parents=True, exist_ok=True)
    SITE_AUDIO.mkdir(parents=True, exist_ok=True)

    result_rows = read_csv(result_path)
    summary_rows = read_csv(summary_path)
    clean_rows = read_csv(metadata_path)

    clean_by_key = find_clean_metadata(clean_rows)
    intro = make_intro(result_rows, clean_by_key)
    demo_results = make_demo_results(result_rows)
    summary = sorted(
        (summary_payload(row) for row in summary_rows),
        key=lambda row: (NOISE_ORDER.get(str(row["noiseType"]), 99), -int(row["snrDb"])),
    )

    write_json(
        SITE_DATA / "results.json",
        {
            "generatedFrom": result_path.name,
            "rows": demo_results,
        },
    )
    write_json(
        SITE_DATA / "summary.json",
        {
            "generatedFrom": summary_path.name,
            "rows": summary,
        },
    )
    write_json(SITE_DATA / "intro_clips.json", intro)

    print(f"Wrote {SITE_DATA / 'results.json'} with {len(demo_results)} demo clips.")
    print(f"Wrote {SITE_DATA / 'summary.json'} with {len(summary)} summary rows.")
    print(f"Wrote {SITE_DATA / 'intro_clips.json'} and copied demo audio.")


if __name__ == "__main__":
    main()
