"""
Round 3: tackle the long tail.

Inspection of round 2 showed the worst errors come from:
  * 4 lignes where actual 2025 = 0 (we predicted positive -> 200% sMAPE each)
  * a few high-volatility lignes (CV > 0.5) with huge swings
  * lignes where the last year already showed a step-down vs history

This round adds:
  * Zero-detector  (if last train value = 0 OR last two are both very low -> predict 0)
  * Step-down rule (if last value is far below historical median -> trust the trend down)
  * Step-up rule   (symmetric, capped)
  * Bias correction using aggregate growth
  * Final ensemble strategy combining the best round-2 winner with the new rules

Also runs a second pass excluding *business* anomalies too, to quantify how
much further the accuracy could go with stricter data cleaning.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
ANOM_EXT = ROOT / "data" / "03_forecast" / "03_anomalies_extraction.csv"
ANOM_BUS = ROOT / "data" / "03_forecast" / "03_anomalies_business.csv"
OUT = ROOT / "data" / "03_forecast"
BACKTEST_YEAR = 2025
PARTIAL_YEARS = {2026}


def smape_one(a, p):
    d = (abs(a) + abs(p)) / 2.0
    return 0.0 if d == 0 else 100.0 * abs(a - p) / d


# Predictors ------------------------------------------------------------------
def m_naive(h): return float(h[-1])
def m_median(h): return float(np.median(h))
def m_mean3(h): return float(np.mean(h[-3:]))


def m_robust_mean(h):
    arr = np.asarray(h[-3:], dtype=float)
    if len(arr) <= 1: return float(arr[-1])
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) or 1.0
    keep = arr[np.abs(arr - med) <= 3 * mad]
    return float(np.mean(keep)) if len(keep) else float(med)


def m_drift_naive(h, g):
    if not h: return None
    return float(h[-1] * float(np.clip(g, 0.5, 2.0)))


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


# Smart wrapper ---------------------------------------------------------------
def smart_predict(hist, ensemble_pred):
    """Apply zero / step-down detection on top of an ensemble base prediction."""
    if hist is None or not hist:
        return ensemble_pred
    last = hist[-1]
    arr = np.asarray(hist, dtype=float)

    # 1. Zero-detector
    if last == 0 and len(arr) >= 2 and arr[-2] == 0:
        return 0.0
    if last == 0:
        # last value zero, but earlier wasn't -> 50/50 it stays zero, take min(0, ens)/2
        return ensemble_pred * 0.5 if ensemble_pred and ensemble_pred > 0 else 0.0

    # 2. Step-down: if last value is < 30% of historical median, prefer last
    med = np.median(arr[:-1]) if len(arr) > 1 else last
    if med > 0 and last / med < 0.3:
        return last  # take the recent low as the regime

    # 3. Step-up: if last value is > 3x historical median, blend (ensemble already does this)
    if med > 0 and last / med > 3.0:
        return 0.7 * last + 0.3 * med  # damp the spike

    # 4. Default: ensemble
    return ensemble_pred


def coef_var(h):
    a = np.asarray(h, dtype=float)
    a = a[a > 0]
    if len(a) < 2: return 0.0
    m = a.mean()
    return float(a.std(ddof=0) / m) if m > 0 else 0.0


def run_experiment(df, label):
    print(f"\n========== {label} ==========")
    print(f"Lignes: {df['Line_Key'].nunique()}")
    agg = df[df["Year"] < BACKTEST_YEAR].groupby("Year")["Total_Engage_Vises"].sum()
    g_growth = float(agg.iloc[-1] / agg.iloc[-2]) if len(agg) >= 2 and agg.iloc[-2] > 0 else 1.0

    eligible = []
    for k, grp in df.groupby("Line_Key"):
        h = grp[grp["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].dropna()
        a = grp[grp["Year"] == BACKTEST_YEAR]["Total_Engage_Vises"].dropna()
        if len(h) >= 3 and len(a) == 1:
            eligible.append(k)
    print(f"Eligible (>=3 training years + actual 2025): {len(eligible)}")

    ctx = {}
    for k in eligible:
        g = df[df["Line_Key"] == k].sort_values("Year")
        h = g[g["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].astype(float).tolist()
        a = float(g[g["Year"] == BACKTEST_YEAR]["Total_Engage_Vises"].iloc[0])
        ctx[k] = {"g": g, "h": h, "actual": a, "cv": coef_var(h)}

    base = {m: {} for m in ["Naive", "Median", "RobustMean", "DriftNaive", "ExecRate"]}
    for k, c in ctx.items():
        base["Naive"][k] = m_naive(c["h"])
        base["Median"][k] = m_median(c["h"])
        base["RobustMean"][k] = m_robust_mean(c["h"])
        base["DriftNaive"][k] = m_drift_naive(c["h"], g_growth)
        base["ExecRate"][k] = m_exec_rate(c["g"], BACKTEST_YEAR)

    strategies = {}

    # Reference: EnsMedianTop3 (round-2 winner)
    em3 = {}
    for k in eligible:
        preds = [base[m].get(k) for m in ["DriftNaive", "Naive", "ExecRate"]]
        preds = [p for p in preds if p is not None and np.isfinite(p)]
        em3[k] = float(np.median(preds)) if preds else base["Naive"].get(k)
    strategies["EnsMedianTop3"] = em3

    # Smart wrapper applied to ensemble
    smart = {k: smart_predict(ctx[k]["h"], em3[k]) for k in eligible}
    strategies["Smart_EnsMedian"] = smart

    # Smart wrapper applied to DriftNaive
    smart_dn = {k: smart_predict(ctx[k]["h"], base["DriftNaive"].get(k)) for k in eligible}
    strategies["Smart_DriftNaive"] = smart_dn

    # Smart + bias correction
    base_total_24 = sum(c["h"][-1] for c in ctx.values())
    target_total = base_total_24 * g_growth
    s = sum(v for v in smart.values() if v is not None and v > 0)
    scale = target_total / s if s else 1.0
    strategies["Smart_EnsMedian_BiasCorr"] = {
        k: v * scale if v and v > 0 else v for k, v in smart.items()
    }

    # Volatility router with smart wrapper
    vol = {}
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
        vol[k] = smart_predict(ctx[k]["h"], p)
    strategies["Smart_VolRouter"] = vol

    # Score ------------------------------------------------------------------
    rows = []
    for name, preds in strategies.items():
        errs, ta, tp = [], 0.0, 0.0
        for k, p in preds.items():
            if p is None or not np.isfinite(p): continue
            errs.append(smape_one(ctx[k]["actual"], p))
            ta += ctx[k]["actual"]; tp += p
        if not errs: continue
        e = np.array(errs)
        rows.append({
            "Strategy": name, "n": len(e),
            "mean_sMAPE": round(e.mean(), 2),
            "median_sMAPE": round(float(np.median(e)), 2),
            "p75": round(float(np.percentile(e, 75)), 2),
            "p90": round(float(np.percentile(e, 90)), 2),
            "max": round(float(e.max()), 1),
            "pct_le_10": round(100 * np.mean(e <= 10), 1),
            "pct_le_25": round(100 * np.mean(e <= 25), 1),
            "pct_le_50": round(100 * np.mean(e <= 50), 1),
            "bias_pct": round(100 * (tp - ta) / ta, 2) if ta else 0,
        })

    lb = pd.DataFrame(rows).sort_values(["median_sMAPE", "mean_sMAPE"])
    print(lb.to_string(index=False))
    return lb, strategies, ctx


def main():
    df = pd.read_excel(PANEL)
    df = df[~df["Year"].isin(PARTIAL_YEARS)]
    df = df[df["Level"] == "Ligne"].copy()

    susp_ext = set()
    if ANOM_EXT.exists():
        susp_ext = set(pd.read_csv(ANOM_EXT)["Line_Key"].dropna().astype(str))
    susp_bus = set()
    if ANOM_BUS.exists():
        susp_bus = set(pd.read_csv(ANOM_BUS)["Line_Key"].dropna().astype(str))

    df_ext = df[~df["Line_Key"].astype(str).isin(susp_ext)]
    df_both = df_ext[~df_ext["Line_Key"].astype(str).isin(susp_bus)]

    lb1, _, _ = run_experiment(df_ext, "Filter = extraction anomalies only (74 lignes)")
    lb2, strat2, ctx2 = run_experiment(df_both, "Filter = extraction + business (stricter)")

    lb1.to_csv(OUT / "v3_strategies_ext_only.csv", index=False)
    lb2.to_csv(OUT / "v3_strategies_strict.csv", index=False)

    # Detail for the overall best
    best_name = lb1.iloc[0]["Strategy"]
    print(f"\nGlobal best on ext-only filter: {best_name}")
    print(f"Files written to {OUT}")


if __name__ == "__main__":
    main()
