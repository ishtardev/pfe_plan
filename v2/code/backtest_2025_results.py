"""
Backtest on 2025 using the exact same 4 methods as the HTML app.
Train: 2021-2024 (2026 excluded, partial year).
Outputs:
  - v2/data/03_forecast/04_backtest_2025_detail.csv   (one row per (ligne, method))
  - v2/data/03_forecast/04_backtest_2025_leaderboard.csv
  - v2/data/03_forecast/04_backtest_2025_summary.csv  (KPI cards)
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT = ROOT / "data" / "03_forecast"
OUT.mkdir(parents=True, exist_ok=True)

BACKTEST_YEAR = 2025
PARTIAL_YEARS = {2026}


def smape(a, p):
    d = (abs(a) + abs(p)) / 2
    return 0.0 if d == 0 else 100 * abs(a - p) / d


def naive(hist):
    return hist[-1]


def mean3(hist):
    return float(np.mean(hist[-3:]))


def median_(hist):
    return float(np.median(hist))


def trend(hist):
    years = np.arange(len(hist))
    if len(hist) < 2:
        return float(hist[-1])
    a, b = np.polyfit(years, hist, 1)
    return float(a * len(hist) + b)


METHODS = {"Naive": naive, "Mean3": mean3, "Median": median_, "Trend": trend}


def main():
    df = pd.read_excel(PANEL)
    print(f"Loaded {len(df):,} rows from stable panel")
    df = df[~df["Year"].isin(PARTIAL_YEARS)]
    df = df[df["Level"] == "Ligne"]
    print(f"After partial-year filter + Ligne only: {len(df):,} rows")
    print(f"Years retained: {sorted(df['Year'].unique())}")

    rows = []
    for key, grp in df.groupby("Line_Key"):
        g = grp.sort_values("Year")
        hist = g[["Year", "Total_Engage_Vises"]].dropna()
        train = hist[hist["Year"] < BACKTEST_YEAR]
        actual_row = hist[hist["Year"] == BACKTEST_YEAR]
        if len(train) < 2 or actual_row.empty:
            continue
        actual = float(actual_row["Total_Engage_Vises"].iloc[0])
        train_values = train["Total_Engage_Vises"].astype(float).tolist()
        for name, fn in METHODS.items():
            try:
                pred = fn(train_values)
            except Exception:
                continue
            if not np.isfinite(pred):
                continue
            rows.append({
                "Line_Key": key,
                "Intitule": g["Intitule"].iloc[0],
                "Method": name,
                "Actual_2025": round(actual),
                "Predicted_2025": round(pred),
                "Abs_Error": round(abs(actual - pred)),
                "sMAPE_pct": round(smape(actual, pred), 2),
            })

    res = pd.DataFrame(rows)
    print(f"\nBacktest rows: {len(res):,}")
    print(f"Unique lignes backtested: {res['Line_Key'].nunique():,}")

    # Per-line best method
    best_idx = res.groupby("Line_Key")["sMAPE_pct"].idxmin()
    best = res.loc[best_idx].copy()
    best["Is_Best"] = "Yes"

    res.to_csv(OUT / "04_backtest_2025_detail.csv", index=False)
    best.to_csv(OUT / "04_backtest_2025_best_per_line.csv", index=False)

    # Leaderboard
    lb = (res.groupby("Method")
              .agg(n=("sMAPE_pct", "size"),
                   mean_sMAPE=("sMAPE_pct", "mean"),
                   median_sMAPE=("sMAPE_pct", "median"),
                   mean_AbsErr=("Abs_Error", "mean"))
              .round(2)
              .reset_index())
    wins = best["Method"].value_counts().to_dict()
    lb["wins"] = lb["Method"].map(lambda m: wins.get(m, 0))
    lb = lb.sort_values("mean_sMAPE")
    lb.to_csv(OUT / "04_backtest_2025_leaderboard.csv", index=False)

    # Accuracy KPI summary (using best per ligne)
    sm = best["sMAPE_pct"].values
    summary = pd.DataFrame([{
        "n_lignes_backtest": len(best),
        "mean_sMAPE_best": round(float(np.mean(sm)), 2),
        "median_sMAPE_best": round(float(np.median(sm)), 2),
        "pct_lignes_under_10pct": round(100 * float(np.mean(sm <= 10)), 1),
        "pct_lignes_under_25pct": round(100 * float(np.mean(sm <= 25)), 1),
        "pct_lignes_under_50pct": round(100 * float(np.mean(sm <= 50)), 1),
        "total_actual_2025": int(best["Actual_2025"].sum()),
        "total_predicted_2025_best": int(best["Predicted_2025"].sum()),
        "global_error_pct": round(100 * (best["Predicted_2025"].sum() - best["Actual_2025"].sum()) / best["Actual_2025"].sum(), 2),
    }])
    summary.to_csv(OUT / "04_backtest_2025_summary.csv", index=False)

    print("\n=== LEADERBOARD ===")
    print(lb.to_string(index=False))
    print("\n=== ACCURACY (best method per line) ===")
    print(summary.T)
    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
