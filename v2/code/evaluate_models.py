"""
evaluate_models.py
==================
Comprehensive evaluation of the forecast methods used by the manager pipeline.

Prints (to terminal):
  1) Last-year backtest (sMAPE / MAE / Bias) for every method
  2) Rolling backtest over all eligible years (stability over time)
  3) Per-method leaderboard with win-rates
  4) Comparison vs Naive baseline (skill score)
  5) 90% confidence-interval coverage (pooled conformal residuals)
  6) Error analysis by line SIZE, VOLATILITY, and CHAPTER
  7) Bias check (signed % error) per method and per year

Run:
    python v2/code/evaluate_models.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"

# Import methods + helpers from the existing pipeline so we evaluate exactly
# what ships to the manager.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_pipeline import METHODS, m_xgb, train_xgb, smape  # type: ignore


# ---------- pretty printing ----------
def hr(title=""):
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def fmt_pct(x, nd=1):
    return "  —  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}%"


# ---------- data loading ----------
def load_lignes() -> tuple[pd.DataFrame, list[int]]:
    panel = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str,
                                        "Proj": str, "Lb": str})
    lignes = panel[panel.Level == "Ligne"].copy()
    yearly_total = lignes.groupby("Year")["Total_Engage_Vises"].sum()
    median_total = yearly_total.median()
    complete = sorted(int(y) for y, v in yearly_total.items() if v >= 0.5 * median_total)
    lignes = lignes[lignes.Year.isin(complete)].copy()
    return lignes, complete


# ---------- core: predict every line × method × target year ----------
def backtest_one_year(lignes: pd.DataFrame, target_year: int,
                      min_history: int = 3) -> pd.DataFrame:
    """For every line where ≥ min_history prior years exist, predict target_year
    with every method. Returns long-form dataframe."""
    train_xgb(lignes[lignes.Year < target_year], target_year)
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        hist = sub.loc[sub.index < target_year, "Total_Engage_Vises"]
        if len(hist) < min_history or target_year not in sub.index:
            continue
        actual = float(sub.loc[target_year, "Total_Engage_Vises"])
        if actual == 0 and hist.iloc[-1] == 0:
            continue  # uninformative
        for name, fn in METHODS.items():
            try:
                m_xgb._current_key = key
                pred = float(fn(hist, target_year))
            except Exception:
                continue
            err = pred - actual
            rows.append({
                "Line_Key": key,
                "Year": target_year,
                "Method": name,
                "Actual": actual,
                "Pred": pred,
                "Error": err,
                "AbsError": abs(err),
                "sMAPE": smape(actual, pred),
                "SignedPct": 100 * err / actual if actual > 0 else np.nan,
            })
    return pd.DataFrame(rows)


# ---------- 1. last-year backtest ----------
def report_last_year(lignes: pd.DataFrame, last_year: int):
    hr(f"1. BACKTEST ON LAST FULL YEAR ({last_year})")
    df = backtest_one_year(lignes, last_year)
    if df.empty:
        print("  (no eligible lines)")
        return df
    nb_lines = df["Line_Key"].nunique()
    print(f"  Lines evaluated      : {nb_lines}")
    total_actual = df[df.Method == "Naive"]["Actual"].sum()
    print(f"  Aggregate actual {last_year}: {total_actual:>18,.0f}")

    agg = df.groupby("Method").agg(
        sMAPE_mean=("sMAPE", "mean"),
        sMAPE_median=("sMAPE", "median"),
        MAE=("AbsError", "mean"),
        Bias_pct=("SignedPct", "mean"),
        Bias_med_pct=("SignedPct", "median"),
    ).round(2)
    print()
    print(agg.to_string())
    return df


# ---------- 2. rolling backtest ----------
def report_rolling(lignes: pd.DataFrame, complete_years: list[int]) -> pd.DataFrame:
    hr("2. ROLLING BACKTEST (stability across years)")
    # Need at least 3 prior years -> start from index 3
    rolling_years = [y for i, y in enumerate(complete_years) if i >= 3]
    if not rolling_years:
        print("  Not enough years for rolling backtest")
        return pd.DataFrame()
    print(f"  Years backtested     : {rolling_years}")
    all_rows = []
    for ty in rolling_years:
        df = backtest_one_year(lignes, ty)
        df["TargetYear"] = ty
        all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True)
    pivot = (full.groupby(["TargetYear", "Method"])["sMAPE"].mean()
                 .unstack("Method").round(2))
    print("\n  sMAPE moyen par année cible × méthode :")
    print(pivot.to_string())

    print("\n  Méthode gagnante par année (sMAPE moyen le plus bas) :")
    for ty in rolling_years:
        row = pivot.loc[ty]
        print(f"    {ty}: {row.idxmin():<8} ({row.min():.2f}%)")
    return full


# ---------- 3. leaderboard & win rates ----------
def report_leaderboard(rolling: pd.DataFrame):
    hr("3. PER-METHOD LEADERBOARD (all rolling years pooled)")
    if rolling.empty:
        return
    agg = rolling.groupby("Method").agg(
        n_predictions=("sMAPE", "size"),
        sMAPE_mean=("sMAPE", "mean"),
        sMAPE_median=("sMAPE", "median"),
        MAE=("AbsError", "mean"),
        Bias_pct=("SignedPct", "mean"),
    ).round(2).sort_values("sMAPE_mean")
    print(agg.to_string())

    # Win counts per method
    print("\n  Wins = nb (Line_Key, Year) where this method has the lowest sMAPE :")
    idx = rolling.groupby(["Line_Key", "TargetYear"])["sMAPE"].idxmin()
    winners = rolling.loc[idx, "Method"].value_counts()
    total = winners.sum()
    for m, n in winners.items():
        print(f"    {m:<8} : {n:>6}  ({100*n/total:.1f}%)")


# ---------- 4. naive baseline comparison ----------
def report_vs_naive(rolling: pd.DataFrame):
    hr("4. SKILL VS NAIVE BASELINE")
    if rolling.empty:
        return
    naive_mae = rolling[rolling.Method == "Naive"].groupby(
        ["Line_Key", "TargetYear"])["AbsError"].first()
    out = []
    for m in rolling["Method"].unique():
        if m == "Naive":
            continue
        meth_mae = rolling[rolling.Method == m].groupby(
            ["Line_Key", "TargetYear"])["AbsError"].first()
        joined = pd.concat([naive_mae, meth_mae], axis=1, keys=["Naive", m]).dropna()
        # Skill: 1 - MAE_method / MAE_naive  (positive = better than naive)
        skill = 1 - joined[m].sum() / joined["Naive"].sum()
        wins = (joined[m] < joined["Naive"]).mean() * 100
        out.append({"Method": m,
                    "Skill_vs_Naive": f"{skill*100:+.1f}%",
                    "Wins_vs_Naive_pct": f"{wins:.1f}%"})
    print(pd.DataFrame(out).to_string(index=False))
    print("\n  Skill > 0  =>  better than copy-paste-last-year.")
    print("  Wins %     =>  share of lines where the method beats Naive.")


# ---------- 5. 90% IC coverage ----------
def report_ic_coverage(rolling: pd.DataFrame):
    hr("5. 90% CONFIDENCE-INTERVAL COVERAGE (pooled conformal)")
    if rolling.empty:
        return
    # Build per-line "best method" using all but the latest rolling year as calibration,
    # then test coverage on the latest year.
    last_year = rolling["TargetYear"].max()
    calib = rolling[rolling.TargetYear < last_year]
    test  = rolling[rolling.TargetYear == last_year]
    if calib.empty or test.empty:
        print("  Not enough rolling years to evaluate coverage")
        return

    # Best method per line based on calibration sMAPE
    best = (calib.groupby(["Line_Key", "Method"])["sMAPE"].mean()
                 .reset_index().sort_values("sMAPE")
                 .drop_duplicates("Line_Key", keep="first")
                 [["Line_Key", "Method"]]
                 .rename(columns={"Method": "Best"}))

    # Calib residuals (pct of actual) of the BEST method only
    calib_best = calib.merge(best, on="Line_Key")
    calib_best = calib_best[calib_best.Method == calib_best.Best]
    # symmetric pct residual
    residuals = (calib_best["Pred"] - calib_best["Actual"]).abs() / calib_best["Actual"].replace(0, np.nan)
    residuals = residuals.dropna()
    if residuals.empty:
        print("  No usable residuals")
        return
    q90 = residuals.quantile(0.90)
    print(f"  Pooled |residual|/actual  90th percentile : {q90*100:.1f}%")
    print(f"  (Implies IC bounds = pred × (1 ± {q90:.3f}))")

    test_best = test.merge(best, on="Line_Key")
    test_best = test_best[test_best.Method == test_best.Best].copy()
    lo = test_best["Pred"] * (1 - q90)
    hi = test_best["Pred"] * (1 + q90)
    inside = ((test_best["Actual"] >= lo) & (test_best["Actual"] <= hi)).mean() * 100
    print(f"  Empirical coverage on {last_year:<4}     : {inside:.1f}%  (target ≥ 90%)")
    print(f"  Lines tested                          : {len(test_best)}")


# ---------- 6. error analysis by segment ----------
def report_segments(rolling: pd.DataFrame, lignes: pd.DataFrame):
    hr("6. ERROR ANALYSIS BY SEGMENT")
    if rolling.empty:
        return

    # Segment each line by its mean size and its volatility (CV)
    summary = (lignes.groupby("Line_Key")["Total_Engage_Vises"]
                     .agg(mean_val="mean", std_val="std")
                     .assign(CV=lambda d: d.std_val / d.mean_val.replace(0, np.nan)))
    summary["Size_Bucket"] = pd.qcut(summary["mean_val"], 3,
                                     labels=["Petite", "Moyenne", "Grande"],
                                     duplicates="drop")
    summary["Vol_Bucket"] = pd.qcut(summary["CV"].fillna(0), 3,
                                    labels=["Stable", "Modérée", "Volatile"],
                                    duplicates="drop")
    summary["Chap"] = summary.index.str.split("-").str[0]

    # Best-method-per-line (full rolling pool) — what the manager actually gets
    best = (rolling.groupby(["Line_Key", "Method"])["sMAPE"].mean()
                   .reset_index().sort_values("sMAPE")
                   .drop_duplicates("Line_Key", keep="first")
                   .rename(columns={"Method": "Best", "sMAPE": "Best_sMAPE"}))
    merged = best.merge(summary.reset_index(), on="Line_Key")

    print("\n  By SIZE bucket (mean engagement):")
    print(merged.groupby("Size_Bucket", observed=True)["Best_sMAPE"]
                .agg(["count", "mean", "median"]).round(2).to_string())

    print("\n  By VOLATILITY bucket (coefficient of variation):")
    print(merged.groupby("Vol_Bucket", observed=True)["Best_sMAPE"]
                .agg(["count", "mean", "median"]).round(2).to_string())

    print("\n  By CHAPTER (top 10 by line count):")
    by_chap = (merged.groupby("Chap")["Best_sMAPE"]
                     .agg(["count", "mean", "median"]).round(2)
                     .sort_values("count", ascending=False).head(10))
    print(by_chap.to_string())


# ---------- 7. bias check ----------
def report_bias(rolling: pd.DataFrame):
    hr("7. BIAS CHECK (signed % error — positive = over-shoot)")
    if rolling.empty:
        return
    pivot = (rolling.groupby(["TargetYear", "Method"])["SignedPct"].mean()
                    .unstack("Method").round(2))
    print("  Mean signed error % par année × méthode :")
    print(pivot.to_string())
    print("\n  Aggregate bias (all years pooled):")
    print(rolling.groupby("Method")["SignedPct"].mean().round(2).to_string())


# ---------- main ----------
def main():
    lignes, complete_years = load_lignes()
    print(f"[data] years complete : {complete_years}")
    print(f"[data] nb lines       : {lignes['Line_Key'].nunique()}")
    print(f"[data] methods        : {list(METHODS.keys())}")

    last_year = complete_years[-1]
    _ = report_last_year(lignes, last_year)
    rolling = report_rolling(lignes, complete_years)
    report_leaderboard(rolling)
    report_vs_naive(rolling)
    report_ic_coverage(rolling)
    report_segments(rolling, lignes)
    report_bias(rolling)

    hr("DONE")


if __name__ == "__main__":
    main()
