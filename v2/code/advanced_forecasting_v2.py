"""
Round 2 of forecasting experiments.

Round 1 (advanced_forecasting.py) showed:
  - EnsembleMedianTop3 (DriftNaive + Naive + ExecRate) -> median sMAPE 3.95%
  - DriftNaive alone is the strongest single method (median 4.92%)
  - Oracle ceiling is 0.48% -> per-line selection has huge room
  - Mean sMAPE (36%) is dominated by a handful of blow-ups (p75 around 50%)

This round attacks the tail: prediction clipping, volatility-aware routing,
sign-of-trend filtering and a bias-corrected ensemble.

Outputs to v2/data/03_forecast/
    v2_strategies.csv
    v2_per_line.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
ANOM_EXTRACTION = ROOT / "data" / "03_forecast" / "anomalies_extraction.csv"
OUT = ROOT / "data" / "03_forecast"
BACKTEST_YEAR = 2025
PARTIAL_YEARS = {2026}


def smape_one(a, p):
    d = (abs(a) + abs(p)) / 2.0
    return 0.0 if d == 0 else 100.0 * abs(a - p) / d


# ----------------------------------------------------------------------
# Base predictors (reused from round 1)
# ----------------------------------------------------------------------
def m_naive(h): return float(h[-1])
def m_mean3(h): return float(np.mean(h[-3:]))
def m_median(h): return float(np.median(h))
def m_trend(h):
    if len(h) < 2: return float(h[-1])
    a, b = np.polyfit(np.arange(len(h)), h, 1)
    return float(a * len(h) + b)


def m_ses(h):
    arr = np.asarray(h, dtype=float)
    if len(arr) < 2: return float(arr[-1])
    best_alpha, best_err = 0.5, float("inf")
    if len(arr) >= 3:
        for a in np.linspace(0.1, 0.9, 9):
            L = arr[0]
            for v in arr[1:-1]:
                L = a * v + (1 - a) * L
            pred_last = L
            err = (pred_last - arr[-1]) ** 2
            if err < best_err:
                best_err, best_alpha = err, a
    L = arr[0]
    for v in arr[1:]:
        L = best_alpha * v + (1 - best_alpha) * L
    return float(L)


def m_theta(h):
    if len(h) < 2: return float(h[-1])
    return 0.5 * (m_ses(h) + m_trend(h))


def m_robust_mean(h):
    arr = np.asarray(h[-3:], dtype=float)
    if len(arr) <= 1: return float(arr[-1])
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) or 1.0
    keep = arr[np.abs(arr - med) <= 3 * mad]
    return float(np.mean(keep)) if len(keep) else float(med)


def m_drift_naive(hist, agg_growth):
    if not hist: return None
    g = float(np.clip(agg_growth, 0.5, 2.0))
    return float(hist[-1] * g)


def m_exec_rate(g_line, target_year):
    train = g_line[g_line["Year"] < target_year]
    rates = []
    for _, r in train.iterrows():
        c = r.get("Total_Credits_Vises") or r.get("Credits_Ouverts_Vises")
        e = r.get("Total_Engage_Vises")
        if c and e is not None and c > 0:
            rates.append(e / c)
    if not rates: return None
    rate = float(np.median(rates[-3:]) if len(rates) >= 3 else np.mean(rates))
    tgt = g_line[g_line["Year"] == target_year]
    cred = None
    if not tgt.empty:
        c = tgt.iloc[0].get("Total_Credits_Vises") or tgt.iloc[0].get("Credits_Ouverts_Vises")
        if c and c > 0: cred = float(c)
    if cred is None:
        last = train.iloc[-1]
        cred = last.get("Total_Credits_Vises") or last.get("Credits_Ouverts_Vises") or 0
    return float(rate * cred) if cred else None


# ----------------------------------------------------------------------
# Decorators
# ----------------------------------------------------------------------
def clip_pred(pred, hist, lo=0.3, hi=3.0):
    """Bound a prediction to [lo, hi] x last observed.  Stops a single bad
    forecast from wrecking the global error."""
    if pred is None or hist is None or not hist:
        return pred
    last = hist[-1]
    if last <= 0:
        return pred
    return float(np.clip(pred, lo * last, hi * last))


def coef_var(hist):
    arr = np.asarray(hist, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 2:
        return 0.0
    mean = arr.mean()
    return float(arr.std(ddof=0) / mean) if mean > 0 else 0.0


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    df = pd.read_excel(PANEL)
    df = df[~df["Year"].isin(PARTIAL_YEARS)]
    df = df[df["Level"] == "Ligne"].copy()
    if ANOM_EXTRACTION.exists():
        susp = set(pd.read_csv(ANOM_EXTRACTION)["Line_Key"].dropna().astype(str))
        df = df[~df["Line_Key"].astype(str).isin(susp)]
    print(f"Working with {df['Line_Key'].nunique()} lignes after anomaly filter")

    # Aggregate YoY growth for DriftNaive
    agg = df[df["Year"] < BACKTEST_YEAR].groupby("Year")["Total_Engage_Vises"].sum()
    agg_growth = float(agg.iloc[-1] / agg.iloc[-2]) if len(agg) >= 2 and agg.iloc[-2] > 0 else 1.0
    print(f"Aggregate engage growth 2023->2024 = {agg_growth:.3f}")

    eligible = []
    for key, grp in df.groupby("Line_Key"):
        hist = grp[grp["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].dropna()
        actual = grp[grp["Year"] == BACKTEST_YEAR]["Total_Engage_Vises"].dropna()
        if len(hist) >= 3 and len(actual) == 1:
            eligible.append(key)
    print(f"Eligible lignes: {len(eligible)}")

    # Per-line context
    ctx = {}
    for key in eligible:
        g = df[df["Line_Key"] == key].sort_values("Year")
        hist = g[g["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].astype(float).tolist()
        actual = float(g[g["Year"] == BACKTEST_YEAR]["Total_Engage_Vises"].iloc[0])
        ctx[key] = {"g": g, "hist": hist, "actual": actual, "cv": coef_var(hist)}

    # Build base predictions
    methods = ["Naive", "Mean3", "Median", "Trend", "SES", "Theta",
               "RobustMean", "DriftNaive", "ExecRate"]
    base = {m: {} for m in methods}
    for key, c in ctx.items():
        h = c["hist"]
        base["Naive"][key] = m_naive(h)
        base["Mean3"][key] = m_mean3(h)
        base["Median"][key] = m_median(h)
        base["Trend"][key] = m_trend(h)
        base["SES"][key] = m_ses(h)
        base["Theta"][key] = m_theta(h)
        base["RobustMean"][key] = m_robust_mean(h)
        base["DriftNaive"][key] = m_drift_naive(h, agg_growth)
        base["ExecRate"][key] = m_exec_rate(c["g"], BACKTEST_YEAR)

    # ------------------------------------------------------------------
    # Strategies to compare
    # ------------------------------------------------------------------
    strategies: dict[str, dict] = {}

    # Plain reference
    for name in ["Naive", "DriftNaive", "Theta", "RobustMean"]:
        strategies[name] = dict(base[name])

    # Clipped variants
    for name in ["Naive", "DriftNaive", "Theta", "RobustMean"]:
        strategies[f"{name}_Clip"] = {
            k: clip_pred(base[name].get(k), ctx[k]["hist"]) for k in eligible
        }

    # EnsembleMedianTop3 (round-1 winner) and its clipped variant
    def ensemble_median(top, suffix=""):
        out = {}
        for k in eligible:
            preds = [base[m].get(k) for m in top]
            preds = [p for p in preds if p is not None and np.isfinite(p)]
            if not preds: continue
            p = float(np.median(preds))
            if suffix == "_Clip":
                p = clip_pred(p, ctx[k]["hist"])
            out[k] = p
        return out

    strategies["EnsMedianTop3"] = ensemble_median(["DriftNaive", "Naive", "ExecRate"])
    strategies["EnsMedianTop3_Clip"] = ensemble_median(["DriftNaive", "Naive", "ExecRate"], "_Clip")
    strategies["EnsMedianTop5"] = ensemble_median(["DriftNaive", "Naive", "ExecRate", "SES", "RobustMean"])
    strategies["EnsMedianTop5_Clip"] = ensemble_median(["DriftNaive", "Naive", "ExecRate", "SES", "RobustMean"], "_Clip")

    # Volatility-aware router (rule of thumb learned from round 1 stats):
    #   CV  < 0.15 -> Naive (very stable, recent value is best)
    #   CV in [0.15, 0.40) -> EnsembleMedianTop3 (mixed)
    #   CV >= 0.40 -> RobustMean (drop outlier years)
    router = {}
    for k in eligible:
        cv = ctx[k]["cv"]
        if cv < 0.15:
            p = base["Naive"].get(k)
        elif cv < 0.40:
            preds = [base[m].get(k) for m in ["DriftNaive", "Naive", "ExecRate"]]
            preds = [pp for pp in preds if pp is not None and np.isfinite(pp)]
            p = float(np.median(preds)) if preds else base["Naive"].get(k)
        else:
            p = base["RobustMean"].get(k)
        router[k] = clip_pred(p, ctx[k]["hist"])
    strategies["VolatilityRouter"] = router

    # Bias-corrected ensemble: ensemble median, then rescale so total matches
    # historical aggregate growth applied to 2024 total
    base_total_2024 = sum(c["hist"][-1] for c in ctx.values())
    target_total = base_total_2024 * agg_growth
    raw = strategies["EnsMedianTop3_Clip"]
    s = sum(v for v in raw.values() if v is not None)
    scale = target_total / s if s else 1.0
    strategies["EnsMedianTop3_BiasCorr"] = {k: v * scale for k, v in raw.items() if v is not None}

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    rows = []
    for name, preds in strategies.items():
        errs, total_a, total_p = [], 0.0, 0.0
        for k, p in preds.items():
            if p is None or not np.isfinite(p): continue
            errs.append(smape_one(ctx[k]["actual"], p))
            total_a += ctx[k]["actual"]; total_p += p
        if not errs: continue
        e = np.array(errs)
        rows.append({
            "Strategy": name,
            "n": len(e),
            "mean_sMAPE": round(e.mean(), 2),
            "median_sMAPE": round(float(np.median(e)), 2),
            "p75": round(float(np.percentile(e, 75)), 2),
            "p90": round(float(np.percentile(e, 90)), 2),
            "max": round(float(e.max()), 1),
            "pct_le_10": round(100 * np.mean(e <= 10), 1),
            "pct_le_25": round(100 * np.mean(e <= 25), 1),
            "bias_pct": round(100 * (total_p - total_a) / total_a, 2) if total_a else 0,
        })

    lb = pd.DataFrame(rows).sort_values(["median_sMAPE", "mean_sMAPE"])
    print("\n=== ROUND 2 STRATEGY LEADERBOARD ===")
    print(lb.to_string(index=False))

    lb.to_csv(OUT / "v2_strategies.csv", index=False)

    # Per-line detail for the best strategy
    best_name = lb.iloc[0]["Strategy"]
    print(f"\nBest strategy: {best_name}")
    best_preds = strategies[best_name]
    detail = []
    for k, p in best_preds.items():
        if p is None: continue
        detail.append({
            "Line_Key": k,
            "Intitule": ctx[k]["g"]["Intitule"].iloc[0],
            "CV_train": round(ctx[k]["cv"], 3),
            "Actual_2025": round(ctx[k]["actual"]),
            "Predicted": round(p),
            "sMAPE": round(smape_one(ctx[k]["actual"], p), 2),
        })
    pd.DataFrame(detail).sort_values("sMAPE", ascending=False).to_csv(
        OUT / "v2_per_line.csv", index=False
    )
    print(f"Files written to {OUT}")


if __name__ == "__main__":
    main()
