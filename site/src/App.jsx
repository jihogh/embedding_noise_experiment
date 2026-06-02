import { useEffect, useMemo, useState } from "react";

const BASE_URL = import.meta.env.BASE_URL || "/";
const NOISE_LABELS = {
  cafe: "Cafe",
  fan: "Fan",
  traffic: "Traffic",
};
const NOISE_COLORS = {
  cafe: "#b42318",
  fan: "#13795b",
  traffic: "#365486",
};

function assetPath(path) {
  return `${BASE_URL}${path.replace(/^\/+/, "")}`;
}

function normalizeWord(word) {
  return word.toLowerCase().replace(/[^a-z0-9']/g, "");
}

function tokenize(text) {
  return text
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((raw) => ({ raw, norm: normalizeWord(raw) }))
    .filter((word) => word.norm.length > 0);
}

function alignWords(reference, hypothesis) {
  const ref = tokenize(reference);
  const hyp = tokenize(hypothesis);
  const rows = ref.length + 1;
  const cols = hyp.length + 1;
  const dp = Array.from({ length: rows }, () => Array(cols).fill(0));
  const back = Array.from({ length: rows }, () => Array(cols).fill(null));

  for (let i = 1; i < rows; i += 1) {
    dp[i][0] = i;
    back[i][0] = "delete";
  }
  for (let j = 1; j < cols; j += 1) {
    dp[0][j] = j;
    back[0][j] = "insert";
  }

  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < cols; j += 1) {
      const same = ref[i - 1].norm === hyp[j - 1].norm;
      const replaceCost = dp[i - 1][j - 1] + (same ? 0 : 1);
      const deleteCost = dp[i - 1][j] + 1;
      const insertCost = dp[i][j - 1] + 1;
      const best = Math.min(replaceCost, deleteCost, insertCost);

      dp[i][j] = best;
      if (best === replaceCost) {
        back[i][j] = same ? "match" : "replace";
      } else if (best === deleteCost) {
        back[i][j] = "delete";
      } else {
        back[i][j] = "insert";
      }
    }
  }

  const ops = [];
  let i = ref.length;
  let j = hyp.length;
  while (i > 0 || j > 0) {
    const step = back[i][j];
    if (step === "match" || step === "replace") {
      ops.push({ type: step, ref: ref[i - 1], hyp: hyp[j - 1] });
      i -= 1;
      j -= 1;
    } else if (step === "delete") {
      ops.push({ type: "delete", ref: ref[i - 1], hyp: null });
      i -= 1;
    } else {
      ops.push({ type: "insert", ref: null, hyp: hyp[j - 1] });
      j -= 1;
    }
  }

  return {
    wer: ref.length === 0 ? (hyp.length > 0 ? 1 : 0) : dp[ref.length][hyp.length] / ref.length,
    ops: ops.reverse(),
  };
}

function formatRate(rate) {
  const value = Number(rate);
  if (!Number.isFinite(value)) return "0.00";
  return value.toFixed(2);
}

function wordScore(reference, transcript) {
  const { wer, ops } = alignWords(reference, transcript);
  const total = tokenize(reference).length;
  const correct = ops.filter((op) => op.type === "match").length;
  return {
    wer,
    correct,
    total,
    label: `${correct}/${total}`,
  };
}

function formatNoise(noiseType) {
  return NOISE_LABELS[noiseType] || noiseType;
}

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function introPairKey(pair) {
  return `${pair.clean?.clipId || ""}|${pair.noisy?.clipId || ""}`;
}

function chooseIntroPair(introData) {
  const cleanOptions = introData.cleanOptions?.length ? introData.cleanOptions : [introData.clean].filter(Boolean);
  const noisyOptions = introData.noisyOptions?.length ? introData.noisyOptions : [introData.noisy].filter(Boolean);

  const candidates = cleanOptions.flatMap((clean) =>
    noisyOptions
      .filter((noisy) => noisy.sentenceId !== clean.sentenceId)
      .map((noisy) => ({ clean, noisy }))
  );

  const pool = candidates.length
    ? candidates
    : cleanOptions.flatMap((clean) => noisyOptions.map((noisy) => ({ clean, noisy })));

  if (!pool.length) return introData;

  let lastKey = "";
  try {
    lastKey = window.localStorage.getItem("introPairKey") || "";
  } catch {
    lastKey = "";
  }

  const freshPool = pool.filter((pair) => introPairKey(pair) !== lastKey);
  const selected = randomItem(freshPool.length ? freshPool : pool);

  try {
    window.localStorage.setItem("introPairKey", introPairKey(selected));
  } catch {
    // The random pair still works if localStorage is unavailable.
  }

  return {
    ...introData,
    clean: selected.clean,
    noisy: selected.noisy,
  };
}

function TranscriptDiff({ reference, transcript }) {
  const { ops } = alignWords(reference, transcript);
  if (!transcript.trim()) {
    return <p className="empty-transcript">No transcript entered.</p>;
  }

  return (
    <p className="diff-line" aria-label="Word comparison">
      {ops.map((op, index) => {
        if (op.type === "match") {
          return (
            <span className="word word-good" key={`${op.type}-${index}`}>
              {op.hyp.raw}
            </span>
          );
        }
        if (op.type === "delete") {
          return (
            <span className="word word-bad" key={`${op.type}-${index}`}>
              missing: {op.ref.raw}
            </span>
          );
        }
        const label = op.hyp?.raw || op.ref?.raw || "";
        return (
          <span className="word word-bad" key={`${op.type}-${index}`}>
            {label}
          </span>
        );
      })}
    </p>
  );
}

function MetricCard({ label, value, sublabel }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      {sublabel ? <small>{sublabel}</small> : null}
    </div>
  );
}

function AudioTranscriptionChallenge({ title, kicker, clip, onResult }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  function handleSubmit(event) {
    event.preventDefault();
    const score = wordScore(clip.transcript, text);
    const nextResult = {
      transcript: text,
      ...score,
    };
    setResult(nextResult);
    onResult?.(nextResult);
  }

  return (
    <section className="article-section challenge-section">
      <p className="kicker">{kicker}</p>
      <h2>{title}</h2>
      <div className="audio-card">
        <audio controls src={assetPath(clip.audioPath)}>
          Your browser does not support audio playback.
        </audio>
        <form onSubmit={handleSubmit} className="transcript-form">
          <label>
            What did you hear?
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows="3"
              placeholder="Type the sentence here..."
            />
          </label>
          <button type="submit">Check transcript</button>
        </form>
      </div>

      {result ? (
        <div className="feedback-card">
          <div className="metric-row">
            <MetricCard label="Your score" value={result.label} sublabel="words correct" />
          </div>
          <h3>True transcript</h3>
          <p className="truth-line">{clip.transcript}</p>
          <h3>Your transcript</h3>
          <TranscriptDiff reference={clip.transcript} transcript={result.transcript} />
        </div>
      ) : null}
    </section>
  );
}

function ScoreComparison({ cleanResult, noisyResult }) {
  if (!cleanResult || !noisyResult) return null;
  return (
    <div className="score-comparison">
      <MetricCard label="Clean clip" value={cleanResult.label} sublabel="words correct" />
      <MetricCard label="Noisy clip" value={noisyResult.label} sublabel="words correct" />
    </div>
  );
}

function ResultsTable({ rows }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Noise</th>
            <th>SNR</th>
            <th>Mean WER</th>
            <th>Mean CER</th>
            <th>Clips</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.noiseType}-${row.snrDb}`}>
              <td>{formatNoise(row.noiseType)}</td>
              <td>{row.snrDb} dB</td>
              <td>{formatRate(row.meanWer)}</td>
              <td>{formatRate(row.meanCer)}</td>
              <td>{row.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResultsChart({ rows }) {
  const width = 760;
  const height = 340;
  const margin = { top: 26, right: 24, bottom: 44, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const snrs = Array.from(new Set(rows.map((row) => row.snrDb))).sort((a, b) => b - a);
  const maxWer = Math.max(1, ...rows.map((row) => row.meanWer));
  const yMax = Math.ceil(maxWer * 10) / 10;
  const groups = Object.entries(
    rows.reduce((acc, row) => {
      acc[row.noiseType] ||= [];
      acc[row.noiseType].push(row);
      return acc;
    }, {})
  );

  function xScale(snr) {
    const left = snrs[0];
    const right = snrs[snrs.length - 1];
    return margin.left + ((snr - left) / (right - left || 1)) * plotWidth;
  }

  function yScale(value) {
    return margin.top + plotHeight - (value / yMax) * plotHeight;
  }

  return (
    <div className="chart-card">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Mean WER by SNR and noise type">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const value = tick * yMax;
          const y = yScale(value);
          return (
            <g key={tick}>
              <line className="grid-line" x1={margin.left} x2={width - margin.right} y1={y} y2={y} />
              <text className="axis-label" x={margin.left - 12} y={y + 4} textAnchor="end">
                {formatRate(value)}
              </text>
            </g>
          );
        })}
        {snrs.map((snr) => {
          const x = xScale(snr);
          return (
            <g key={snr}>
              <line className="tick-line" x1={x} x2={x} y1={height - margin.bottom} y2={height - margin.bottom + 6} />
              <text className="axis-label" x={x} y={height - 16} textAnchor="middle">
                {snr} dB
              </text>
            </g>
          );
        })}
        <line
          className="axis-line"
          x1={margin.left}
          x2={width - margin.right}
          y1={height - margin.bottom}
          y2={height - margin.bottom}
        />
        <line className="axis-line" x1={margin.left} x2={margin.left} y1={margin.top} y2={height - margin.bottom} />
        {groups.map(([noiseType, group]) => {
          const sorted = [...group].sort((a, b) => b.snrDb - a.snrDb);
          const points = sorted.map((row) => `${xScale(row.snrDb)},${yScale(row.meanWer)}`).join(" ");
          return (
            <g key={noiseType}>
              <polyline fill="none" stroke={NOISE_COLORS[noiseType]} strokeWidth="3" points={points} />
              {sorted.map((row) => (
                <circle
                  key={`${noiseType}-${row.snrDb}`}
                  cx={xScale(row.snrDb)}
                  cy={yScale(row.meanWer)}
                  r="4"
                  fill={NOISE_COLORS[noiseType]}
                />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="chart-legend">
        {groups.map(([noiseType]) => (
          <span key={noiseType}>
            <i style={{ background: NOISE_COLORS[noiseType] }} />
            {formatNoise(noiseType)}
          </span>
        ))}
      </div>
    </div>
  );
}

function HumanVsWhisperGame({ rows }) {
  const [noiseType, setNoiseType] = useState("random");
  const [snrDb, setSnrDb] = useState("random");
  const [round, setRound] = useState(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const noiseOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.noiseType))).sort(), [rows]);
  const snrOptions = useMemo(() => Array.from(new Set(rows.map((row) => row.snrDb))).sort((a, b) => b - a), [rows]);

  function pickRound(nextNoise = noiseType, nextSnr = snrDb) {
    const candidates = rows.filter((row) => {
      const noiseOk = nextNoise === "random" || row.noiseType === nextNoise;
      const snrOk = nextSnr === "random" || String(row.snrDb) === String(nextSnr);
      return noiseOk && snrOk;
    });
    const pool = candidates.length ? candidates : rows;
    const next = pool[Math.floor(Math.random() * pool.length)];
    setRound(next);
    setText("");
    setResult(null);
  }

  useEffect(() => {
    if (rows.length) pickRound();
  }, [rows.length]);

  function handleNoiseChange(event) {
    const nextNoise = event.target.value;
    setNoiseType(nextNoise);
    pickRound(nextNoise, snrDb);
  }

  function handleSnrChange(event) {
    const nextSnr = event.target.value;
    setSnrDb(nextSnr);
    pickRound(noiseType, nextSnr);
  }

  function handleSubmit(event) {
    event.preventDefault();
    if (!round) return;
    const userScore = wordScore(round.reference, text);
    const whisperScore = wordScore(round.reference, round.prediction);
    let winner = "Tie";
    if (userScore.wer < round.whisperWer) winner = "You";
    if (userScore.wer > round.whisperWer) winner = "Whisper";
    setResult({
      userWer: userScore.wer,
      userScore: userScore.label,
      whisperScore: whisperScore.label,
      winner,
    });
  }

  if (!round) return null;

  return (
    <section className="article-section game-section">
      <p className="kicker">Human vs Whisper game</p>
      <h2>Can you beat Whisper?</h2>
      <p>
        Pick a noise condition, listen once or twice, and see whether your transcript beats Whisper on the same clip.
      </p>

      <div className="game-controls">
        <label>
          Noise
          <select value={noiseType} onChange={handleNoiseChange}>
            <option value="random">Random</option>
            {noiseOptions.map((option) => (
              <option key={option} value={option}>
                {formatNoise(option)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Difficulty
          <select value={snrDb} onChange={handleSnrChange}>
            <option value="random">Random</option>
            {snrOptions.map((option) => (
              <option key={option} value={option}>
                {option} dB
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="secondary-button" onClick={() => pickRound()}>
          New clip
        </button>
      </div>

      <div className="audio-card">
        <div className="clip-meta">
          <span>{formatNoise(round.noiseType)}</span>
          <span>{round.snrDb} dB SNR</span>
        </div>
        <audio controls src={assetPath(round.audioPath)}>
          Your browser does not support audio playback.
        </audio>
        <form onSubmit={handleSubmit} className="transcript-form">
          <label>
            Your transcript
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows="3"
              placeholder="Type what you heard..."
            />
          </label>
          <button type="submit">Score this round</button>
        </form>
      </div>

      {result ? (
        <div className="feedback-card">
          <div className="winner-line">
            Winner: <strong>{result.winner}</strong>
          </div>
          <div className="score-comparison">
            <MetricCard label="You" value={result.userScore} sublabel="words correct" />
            <MetricCard label="Whisper" value={result.whisperScore} sublabel="words correct" />
          </div>
          <h3>True transcript</h3>
          <p className="truth-line">{round.reference}</p>
          <h3>Your transcript</h3>
          <TranscriptDiff reference={round.reference} transcript={text} />
          <h3>Whisper transcript</h3>
          <TranscriptDiff reference={round.reference} transcript={round.prediction} />
        </div>
      ) : null}
    </section>
  );
}

function App() {
  const [intro, setIntro] = useState(null);
  const [summary, setSummary] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cleanResult, setCleanResult] = useState(null);
  const [noisyResult, setNoisyResult] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [introResponse, summaryResponse, resultsResponse] = await Promise.all([
          fetch(assetPath("data/intro_clips.json")),
          fetch(assetPath("data/summary.json")),
          fetch(assetPath("data/results.json")),
        ]);
        if (!introResponse.ok || !summaryResponse.ok || !resultsResponse.ok) {
          throw new Error("Site data was not found. Run python scripts/prepare_site_data.py first.");
        }
        const [introJson, summaryJson, resultsJson] = await Promise.all([
          introResponse.json(),
          summaryResponse.json(),
          resultsResponse.json(),
        ]);
        setIntro(chooseIntroPair(introJson));
        setSummary(summaryJson.rows || []);
        setResults(resultsJson.rows || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <main>
      <article className="article-shell">
        <header className="hero">
          <p className="eyebrow">Noisy speech ASR experiment</p>
          <h1>Do machines hear like humans?</h1>
          <p className="subtitle">
            I wanted to answer the question: when recorded audio contains background noise, do humans and models fail in the same way?
          </p>
          <p>
            Before we look at the model results, try the task yourself. First you will hear a clean synthetic voice,
            then a different sentence with background noise mixed in.
          </p>
        </header>

        {loading ? <p className="status-card">Loading demo clips...</p> : null}
        {error ? <p className="status-card error-card">{error}</p> : null}

        {intro ? (
          <>
            <AudioTranscriptionChallenge
              title="First, try an easy one."
              clip={intro.clean}
              onResult={setCleanResult}
            />

            <AudioTranscriptionChallenge
              title="Now add background noise."
              clip={intro.noisy}
              onResult={setNoisyResult}
            />

            <ScoreComparison cleanResult={cleanResult} noisyResult={noisyResult} />
          </>
        ) : null}

        {!loading && !error ? (
          <>
            <section className="article-section">
              <p className="kicker">The question</p>
              <h2>When the audio gets worse, do humans and models fail in the same way?</h2>
              <p>
                To make that question concrete, I generated 10 ElevenLabs voices reading 50 short sentences. Then I
                mixed clean clips with cafe, fan, and traffic noise at controlled SNR levels. Whisper transcribed each
                noisy clip, and I measured the errors with WER and CER so the model's failures could be compared across
                noise type and difficulty.
              </p>
            </section>

            <section className="article-section">
              <p className="kicker">What is WER?</p>
              <h2>WER counts how many words the transcript got wrong.</h2>
              <p>
                WER stands for word error rate. It compares a transcript with the true sentence and counts word
                substitutions, missing words, and extra words. Lower is better: 0 means every word matched, while 0.25
                means roughly one word error for every four reference words.
              </p>
              <p>
                <code>WER = (substitutions + deletions + insertions) / reference words</code>
              </p>
            </section>

            <section className="article-section">
              <p className="kicker">Whisper results</p>
              <h2>SNR mattered, but noise type mattered too.</h2>
              <p>
                Cafe noise hurt Whisper much more than traffic noise at the same SNR, so SNR alone does not explain
                transcription difficulty. Fan noise stayed mostly manageable until the very low SNR settings.
              </p>
              <ResultsTable rows={summary} />
              <ResultsChart rows={summary} />
            </section>

            <HumanVsWhisperGame rows={results} />

            <section className="article-section final-section">
              <p className="kicker">Future work</p>
              <h2>Next: fine-tuning.</h2>
              <p>
                LoRA fine-tuning is next. Later I will add a second game mode comparing Human vs Whisper vs Fine-tuned
                Whisper, using another prepared results file such as <code>asr_results_finetuned.csv</code>.
              </p>
            </section>
          </>
        ) : null}
      </article>
    </main>
  );
}

export default App;
