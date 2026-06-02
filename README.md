# Noisy Speech ASR Experiment

This repo contains a noisy speech transcription experiment and a static Vite + React site for exploring the results.

The experiment generated synthetic clean speech clips, mixed in controlled background noise, ran Whisper transcription, and measured WER/CER. The website uses only precomputed CSV/JSON data and copied demo audio clips. It does not run Whisper in the browser and does not need a backend.

## Prepare The Site Data

From the repo root:

```powershell
python scripts/prepare_site_data.py
```

The script prefers `asr_results_sample.csv` and `asr_summary_sample.csv` when they exist, falling back to the full CSVs. It writes:

- `site/public/data/results.json`
- `site/public/data/summary.json`
- `site/public/data/intro_clips.json`
- a small audio subset in `site/public/audio-demo/`

## Install Site Dependencies

```powershell
cd site
npm install
```

## Run The Vite Dev Server

```powershell
cd site
npm run dev
```

Vite will print a local URL, usually `http://127.0.0.1:5173/`.

## Build The Site

```powershell
cd site
npm run build
```

The static build is written to `site/dist/`.

## GitHub Pages Deployment

For a custom domain or a user/organization site served from the domain root, build with the default Vite base path:

```powershell
cd site
npm run build
```

For a project page such as `https://USER.github.io/REPO/`, set the Vite base path before building:

```powershell
cd site
$env:VITE_BASE_PATH="/REPO/"
npm run build
```

Then deploy the contents of `site/dist/` with your preferred GitHub Pages flow, such as a Pages action or a `gh-pages` branch.

## Adding Fine-Tuned Results Later

The site is structured so another prepared file such as `asr_results_finetuned.csv` can be added later. A future update can extend `scripts/prepare_site_data.py` to export fine-tuned predictions next to the current Whisper baseline and add a second game mode: Human vs Whisper vs Fine-tuned Whisper.
