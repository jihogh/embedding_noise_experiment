import pandas as pd

df = pd.read_csv("asr_results_full.csv")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print()

summary = (
    df.groupby(["noise_type", "snr_db"])
    .agg(
        mean_wer=("wer", "mean"),
        median_wer=("wer", "median"),
        mean_cer=("cer", "mean"),
        median_cer=("cer", "median"),
        n=("wer", "count"),
    )
    .reset_index()
    .sort_values(["noise_type", "snr_db"], ascending=[True, False])
)

print(summary.to_string(index=False))

summary.to_csv("asr_summary_sample.csv", index=False)