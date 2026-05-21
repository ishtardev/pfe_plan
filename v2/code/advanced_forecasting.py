"""
Advanced forecasting experiments on the 2025 hold-out.

Builds on backtest_2025_results.py by trying many more methods and several
selection / ensemble strategies, then writes a full leaderboard.

All methods are numpy-only (no statsmodels) so the winner can be ported into
the HTML app verbatim.

Methods tested
--------------
Baselines (already in the app)
    Naive            last observed value
    Mean3            average of last 3 observations
    Median           median of all training years
    Trend            linear regression on year index

New univariate methods
    DampedTrend      Holt damped trend, grid-search alpha/beta/phi
    SES              simple exponential smoothing, grid-search alpha
    Theta            Theta method (average of SES + linear regression line)
    LogTrend         linear trend in log space, exp back (handles multiplicative growth)
    WeightedAvg      exponentially weighted average, lambda=0.7
    DriftNaive       last * aggregate YoY growth (borrows signal from the whole budget)
    RobustMean       mean of last 3 after dropping MAD-outliers

New cross-series methods
    ExecRate         predict execution rate engage/credits, multiply by credits_2025
    Hierarchical     bottom-up forecast reconciled with top-down chapter-level trend

Selection / ensemble strategies (scored against 2025)
    Oracle           best method per ligne in hindsight (upper bound only)
    Adaptive         per-ligne method selected by walk-forward CV on 2023 + 2024
    EnsembleMedian   median of the top 3 methods (by global mean sMAPE)
    EnsembleMean     mean of the top 3 methods
    EnsembleTrim     trimmed mean (drop min and max) of all baseline methods
    StackingLite     weighted blend, weights inversely proportional to walk-forward MAE

Outputs to v2/data/03_forecast/
    advanced_leaderboard.csv          mean / median sMAPE + wins per method
    advanced_per_line.csv             every (line, method) prediction + error
    advanced_best_strategy.csv        per-ligne best selection chosen by Adaptive
    advanced_summary.csv              accuracy KPIs per strategy
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
ANOM_EXTRACTION = ROOT / "data" / "03_forecast" / "anomalies_extraction.csv"
OUT = ROOT / "data" / "03_forecast"
OUT.mkdir(parents=True, exist_ok=True)

BACKTEST_YEAR = 2025
PARTIAL_YEARS = {2026}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def smape_one(a: float, p: float) -> float:
    d = (abs(a) + abs(p)) / 2.0
    return 0.0 if d == 0 else 100.0 * abs(a - p) / d


def smape_vec(actuals, preds):
    actuals = np.asarray(actuals, dtype=float)
    preds = np.asarray(preds, dtype=float)
    denom = (np.abs(actuals) + np.abs(preds)) / 2.0
    out = np.where(denom == 0, 0.0, 100.0 * np.abs(actuals - preds) / denom)
    return out


# ----------------------------------------------------------------------
# Univariate forecasting methods.  Each receives an ordered list of
# training values and returns a single 1-step-ahead forecast.
# ----------------------------------------------------------------------
def m_naive(hist):
    return float(hist[-1])


def m_mean3(hist):
    return float(np.mean(hist[-3:]))


def m_median(hist):
    return float(np.median(hist))


def m_trend(hist):
    if len(hist) < 2:
        return float(hist[-1])
    x = np.arange(len(hist))
    a, b = np.polyfit(x, hist, 1)
    return float(a * len(hist) + b)


def m_log_trend(hist):
    """Linear regression in log space.  Falls back to plain Trend on non-positives."""
    arr = np.asarray(hist, dtype=float)
    if len(arr) < 2 or np.any(arr <= 0):
        return m_trend(hist)
    x = np.arange(len(arr))
    a, b = np.polyfit(x, np.log(arr), 1)
    return float(np.exp(a * len(arr) + b))


def m_weighted(hist, lam: float = 0.7):
    """Exponentially weighted average, most recent points weighted highest."""
    arr = np.asarray(hist, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    w = np.array([lam ** (n - 1 - i) for i in range(n)])
    return float(np.sum(arr * w) / np.sum(w))


def m_ses(hist):
    """Simple exponential smoothing with alpha picked by leave-one-out MSE on the
    last point of the training series."""
    arr = np.asarray(hist, dtype=float)
    if len(arr) < 2:
        return float(arr[-1])

    def fit_pred(alpha, x):
        level = x[0]
        for v in x[1:]:
            level = alpha * v + (1 - alpha) * level
        return level

    if len(arr) >= 3:
        best_alpha, best_err = 0.5, float("inf")
        for a in np.linspace(0.1, 0.9, 9):
            pred_last = fit_pred(a, arr[:-1])
            err = (pred_last - arr[-1]) ** 2
            if err < best_err:
                best_err, best_alpha = err, a
        alpha = best_alpha
    else:
        alpha = 0.5
    return float(fit_pred(alpha, arr))


def m_damped(hist):
    """Holt damped-trend with grid-searched alpha/beta/phi."""
    arr = np.asarray(hist, dtype=float)
    if len(arr) < 3:
        return m_trend(hist)

    def forecast(alpha, beta, phi, x):
        L = x[0]
        T = x[1] - x[0]
        for v in x[1:]:
            L_new = alpha * v + (1 - alpha) * (L + phi * T)
            T = beta * (L_new - L) + (1 - beta) * phi * T
            L = L_new
        return L + phi * T

    best, best_err = m_trend(hist), float("inf")
    if len(arr) >= 4:
        for a, b, p in product([0.3, 0.6, 0.9], [0.1, 0.3, 0.6], [0.7, 0.9, 0.98]):
            pred_last = forecast(a, b, p, arr[:-1])
            err = (pred_last - arr[-1]) ** 2
            if err < best_err:
                best_err = err
                best = forecast(a, b, p, arr)
    else:
        best = forecast(0.6, 0.3, 0.9, arr)
    return float(best)


def m_theta(hist):
    """Theta method (theta=2): average of SES (theta=0 line) and linear trend
    extrapolation (theta=2 line). A strong, robust univariate baseline."""
    if len(hist) < 2:
        return float(hist[-1])
    return 0.5 * (m_ses(hist) + m_trend(hist))


def m_robust_mean(hist):
    """Mean of last 3 observations after dropping MAD-outliers (>3 MAD)."""
    arr = np.asarray(hist[-3:], dtype=float)
    if len(arr) <= 1:
        return float(arr[-1])
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) or 1.0
    keep = arr[np.abs(arr - med) <= 3 * mad]
    return float(np.mean(keep)) if len(keep) else float(med)


UNIVARIATE = {
    "Naive": m_naive,
    "Mean3": m_mean3,
    "Median": m_median,
    "Trend": m_trend,
    "LogTrend": m_log_trend,
    "Weighted": m_weighted,
    "SES": m_ses,
    "DampedTrend": m_damped,
    "Theta": m_theta,
    "RobustMean": m_robust_mean,
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def line_history(df, line_key, col, max_year):
    g = (df[(df["Line_Key"] == line_key) & (df["Year"] < max_year)]
         .sort_values("Year"))
    return g[col].astype(float).tolist(), g["Year"].astype(int).tolist()


def fit_predict_uni(name, hist):
    fn = UNIVARIATE[name]
    try:
        p = fn(hist)
        if not np.isfinite(p):
            return None
        return float(p)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Multivariate / cross-series methods
# ----------------------------------------------------------------------
def m_drift_naive(df, line_key, target_year):
    """Last value scaled by aggregate YoY growth."""
    hist, years = line_history(df, line_key, "Total_Engage_Vises", target_year)
    if not hist:
        return None
    agg = (df[df["Year"] < target_year]
           .groupby("Year")["Total_Engage_Vises"].sum())
    if len(agg) < 2:
        return float(hist[-1])
    growth = agg.iloc[-1] / agg.iloc[-2] if agg.iloc[-2] > 0 else 1.0
    growth = float(np.clip(growth, 0.5, 2.0))
    return float(hist[-1] * growth)


def m_exec_rate(df, line_key, target_year):
    """Predict execution rate (engage / credits), then multiply by the target
    year's known credits (or last year credits if missing)."""
    g = df[(df["Line_Key"] == line_key)].sort_values("Year")
    train = g[g["Year"] < target_year]
    rates = []
    for _, row in train.iterrows():
        c = row.get("Total_Credits_Vises") or row.get("Credits_Ouverts_Vises")
        e = row.get("Total_Engage_Vises")
        if c and e is not None and c > 0:
            rates.append(e / c)
    if not rates:
        return None
    rate = float(np.median(rates[-3:])) if len(rates) >= 3 else float(np.mean(rates))
    target_row = g[g["Year"] == target_year]
    target_credits = None
    if not target_row.empty:
        c = target_row.iloc[0].get("Total_Credits_Vises") or target_row.iloc[0].get("Credits_Ouverts_Vises")
        if c and c > 0:
            target_credits = float(c)
    if target_credits is None:
        last = train.iloc[-1]
        target_credits = (last.get("Total_Credits_Vises")
                          or last.get("Credits_Ouverts_Vises") or 0)
    return float(rate * target_credits) if target_credits else None


def hierarchical_reconcile(df, target_year, base_preds_naive):
    """Bottom-up forecast reconciled with chapter-level trend.

    base_preds_naive : dict Line_Key -> bottom-up prediction (any method).
    Adjusts each line's prediction so that the sum across a chapter matches
    a chapter-level trend forecast.  Reduces global bias.
    """
    g = df[df["Year"] < target_year].copy()
    chap_totals = g.groupby(["Year", "Chap"])["Total_Engage_Vises"].sum().reset_index()
    out = dict(base_preds_naive)
    for chap, grp in chap_totals.groupby("Chap"):
        ser = grp.sort_values("Year")["Total_Engage_Vises"].astype(float).tolist()
        chap_pred = m_theta(ser) if len(ser) >= 2 else (ser[-1] if ser else 0)
        # lines belonging to this chapter
        chap_lines = df[(df["Chap"] == chap)]["Line_Key"].dropna().unique()
        bu_sum = sum(out.get(k, 0) or 0 for k in chap_lines)
        if bu_sum <= 0 or chap_pred <= 0:
            continue
        scale = chap_pred / bu_sum
        # bound the scaling factor so we don't over-correct
        scale = float(np.clip(scale, 0.7, 1.4))
        for k in chap_lines:
            if k in out and out[k] is not None:
                out[k] = out[k] * scale
    return out


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def main():
    print(f"Loading panel from {PANEL.name} ...")
    df = pd.read_excel(PANEL)
    df = df[~df["Year"].isin(PARTIAL_YEARS)]
    df = df[df["Level"] == "Ligne"].copy()
    print(f"  {len(df):,} ligne-year rows, {df['Line_Key'].nunique()} unique lignes")
    print(f"  years: {sorted(df['Year'].unique())}")

    # Anomaly filtering: drop Line_Keys flagged by extraction rules
    suspect_keys = set()
    if ANOM_EXTRACTION.exists():
        anom = pd.read_csv(ANOM_EXTRACTION)
        suspect_keys = set(anom["Line_Key"].dropna().astype(str))
        print(f"  filtering out {len(suspect_keys)} Line_Keys flagged by extraction anomalies")
        df = df[~df["Line_Key"].astype(str).isin(suspect_keys)]
        print(f"  after filter: {df['Line_Key'].nunique()} lignes")

    # Eligible lignes = have an actual 2025 + at least 3 training years
    eligible = []
    for key, grp in df.groupby("Line_Key"):
        hist = grp[grp["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].dropna()
        actual = grp[grp["Year"] == BACKTEST_YEAR]["Total_Engage_Vises"].dropna()
        if len(hist) >= 3 and len(actual) == 1:
            eligible.append(key)
    print(f"  eligible lignes for backtest: {len(eligible)}")

    # --------------------------------------------------------------
    # 1. Univariate predictions for every (line, method)
    # --------------------------------------------------------------
    detail_rows = []
    preds_by_method: dict[str, dict] = {m: {} for m in UNIVARIATE}
    preds_by_method["DriftNaive"] = {}
    preds_by_method["ExecRate"] = {}

    actuals = {}
    for key in eligible:
        g = df[df["Line_Key"] == key].sort_values("Year")
        intit = g["Intitule"].iloc[0]
        hist_vals = g[g["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].astype(float).tolist()
        actual = float(g[g["Year"] == BACKTEST_YEAR]["Total_Engage_Vises"].iloc[0])
        actuals[key] = actual

        for name in UNIVARIATE:
            p = fit_predict_uni(name, hist_vals)
            if p is None:
                continue
            preds_by_method[name][key] = p
            detail_rows.append({
                "Line_Key": key, "Intitule": intit, "Method": name,
                "Actual_2025": round(actual), "Predicted_2025": round(p),
                "sMAPE_pct": round(smape_one(actual, p), 2),
            })

        p_dn = m_drift_naive(df, key, BACKTEST_YEAR)
        if p_dn is not None and np.isfinite(p_dn):
            preds_by_method["DriftNaive"][key] = p_dn
            detail_rows.append({
                "Line_Key": key, "Intitule": intit, "Method": "DriftNaive",
                "Actual_2025": round(actual), "Predicted_2025": round(p_dn),
                "sMAPE_pct": round(smape_one(actual, p_dn), 2),
            })

        p_er = m_exec_rate(df, key, BACKTEST_YEAR)
        if p_er is not None and np.isfinite(p_er):
            preds_by_method["ExecRate"][key] = p_er
            detail_rows.append({
                "Line_Key": key, "Intitule": intit, "Method": "ExecRate",
                "Actual_2025": round(actual), "Predicted_2025": round(p_er),
                "sMAPE_pct": round(smape_one(actual, p_er), 2),
            })

    # --------------------------------------------------------------
    # 2. Hierarchical reconciliation (over Theta base predictions)
    # --------------------------------------------------------------
    base_theta = {k: preds_by_method["Theta"].get(k) for k in eligible}
    reconciled = hierarchical_reconcile(df, BACKTEST_YEAR, base_theta)
    preds_by_method["HierTheta"] = reconciled
    for key, p in reconciled.items():
        if p is None or not np.isfinite(p):
            continue
        detail_rows.append({
            "Line_Key": key,
            "Intitule": df[df["Line_Key"] == key]["Intitule"].iloc[0],
            "Method": "HierTheta",
            "Actual_2025": round(actuals[key]),
            "Predicted_2025": round(p),
            "sMAPE_pct": round(smape_one(actuals[key], p), 2),
        })

    detail = pd.DataFrame(detail_rows)
    print(f"\nGenerated {len(detail):,} (line, method) predictions")

    # --------------------------------------------------------------
    # 3. Leaderboard for plain methods
    # --------------------------------------------------------------
    lb = (detail.groupby("Method")
                .agg(n=("sMAPE_pct", "size"),
                     mean_sMAPE=("sMAPE_pct", "mean"),
                     median_sMAPE=("sMAPE_pct", "median"),
                     p75=("sMAPE_pct", lambda s: np.percentile(s, 75)),
                     pct_le_10=("sMAPE_pct", lambda s: 100 * np.mean(s <= 10)),
                     pct_le_25=("sMAPE_pct", lambda s: 100 * np.mean(s <= 25)))
                .round(2).reset_index().sort_values("median_sMAPE"))
    best_idx = detail.groupby("Line_Key")["sMAPE_pct"].idxmin()
    best_per_line = detail.loc[best_idx]
    wins = best_per_line["Method"].value_counts().to_dict()
    lb["wins"] = lb["Method"].map(lambda m: wins.get(m, 0))

    print("\n=== METHOD LEADERBOARD (lower median_sMAPE = better) ===")
    print(lb.to_string(index=False))

    # --------------------------------------------------------------
    # 4. Selection / ensemble strategies
    # --------------------------------------------------------------
    strategies = {}

    # Oracle (upper bound)
    strategies["Oracle"] = {row["Line_Key"]: row["Predicted_2025"]
                            for _, row in best_per_line.iterrows()}

    # Adaptive: per-line method chosen by walk-forward CV on 2023 and 2024
    adaptive_picks = {}
    adaptive_preds = {}
    for key in eligible:
        g = df[df["Line_Key"] == key].sort_values("Year")
        vals_full = g[g["Year"] < BACKTEST_YEAR][["Year", "Total_Engage_Vises"]]
        years_train = vals_full["Year"].astype(int).tolist()
        hist_train = vals_full["Total_Engage_Vises"].astype(float).tolist()
        cv_years = [y for y in years_train if y >= 2023]
        if len(cv_years) < 1:
            adaptive_picks[key] = "Theta"
        else:
            method_errs = {}
            for name in UNIVARIATE:
                errs = []
                for cy in cv_years:
                    idx = years_train.index(cy)
                    sub_train = hist_train[:idx]
                    if len(sub_train) < 2:
                        continue
                    p = fit_predict_uni(name, sub_train)
                    if p is None:
                        continue
                    errs.append(smape_one(hist_train[idx], p))
                if errs:
                    method_errs[name] = np.mean(errs)
            adaptive_picks[key] = min(method_errs, key=method_errs.get) if method_errs else "Theta"
        adaptive_preds[key] = preds_by_method[adaptive_picks[key]].get(key) \
                              or preds_by_method["Naive"].get(key)
    strategies["Adaptive"] = adaptive_preds

    # Ensemble strategies based on top methods of the leaderboard
    top3 = lb.sort_values("median_sMAPE")["Method"].tolist()[:3]
    print(f"\nTop 3 methods (by median_sMAPE) for ensembles: {top3}")
    ens_median = {}
    ens_mean = {}
    for key in eligible:
        preds = [preds_by_method[m].get(key) for m in top3]
        preds = [p for p in preds if p is not None and np.isfinite(p)]
        if not preds:
            continue
        ens_median[key] = float(np.median(preds))
        ens_mean[key] = float(np.mean(preds))
    strategies["EnsembleMedianTop3"] = ens_median
    strategies["EnsembleMeanTop3"] = ens_mean

    # Trimmed mean over all UNIVARIATE methods
    ens_trim = {}
    for key in eligible:
        preds = [preds_by_method[m].get(key) for m in UNIVARIATE]
        preds = sorted([p for p in preds if p is not None and np.isfinite(p)])
        if len(preds) >= 4:
            preds = preds[1:-1]
        if preds:
            ens_trim[key] = float(np.mean(preds))
    strategies["EnsembleTrimmed"] = ens_trim

    # StackingLite: inverse-MAE weights from walk-forward CV
    stacking = {}
    for key in eligible:
        g = df[df["Line_Key"] == key].sort_values("Year")
        years_train = g[g["Year"] < BACKTEST_YEAR]["Year"].astype(int).tolist()
        hist_train = g[g["Year"] < BACKTEST_YEAR]["Total_Engage_Vises"].astype(float).tolist()
        cv_years = [y for y in years_train if y >= 2023]
        weights = {}
        for name in UNIVARIATE:
            errs = []
            for cy in cv_years:
                idx = years_train.index(cy)
                sub_train = hist_train[:idx]
                if len(sub_train) < 2:
                    continue
                p = fit_predict_uni(name, sub_train)
                if p is None:
                    continue
                errs.append(abs(hist_train[idx] - p))
            if errs:
                mae = np.mean(errs)
                weights[name] = 1.0 / (mae + 1.0)  # +1 avoids div-by-zero
        if not weights:
            stacking[key] = preds_by_method["Theta"].get(key)
            continue
        w_sum = sum(weights.values())
        blend = 0.0
        for name, w in weights.items():
            p = preds_by_method[name].get(key)
            if p is None:
                continue
            blend += (w / w_sum) * p
        stacking[key] = blend
    strategies["StackingLite"] = stacking

    # --------------------------------------------------------------
    # 5. Score strategies
    # --------------------------------------------------------------
    strat_rows = []
    for name, preds in strategies.items():
        errs = []
        signed = []
        total_a, total_p = 0.0, 0.0
        for key, p in preds.items():
            if p is None or not np.isfinite(p):
                continue
            a = actuals[key]
            errs.append(smape_one(a, p))
            signed.append(p - a)
            total_a += a
            total_p += p
        if not errs:
            continue
        errs = np.array(errs)
        strat_rows.append({
            "Strategy": name,
            "n": len(errs),
            "mean_sMAPE": round(float(np.mean(errs)), 2),
            "median_sMAPE": round(float(np.median(errs)), 2),
            "p75_sMAPE": round(float(np.percentile(errs, 75)), 2),
            "pct_le_10": round(100 * float(np.mean(errs <= 10)), 1),
            "pct_le_25": round(100 * float(np.mean(errs <= 25)), 1),
            "global_bias_pct": round(100 * (total_p - total_a) / total_a, 2) if total_a else 0,
        })
    strat = pd.DataFrame(strat_rows).sort_values("median_sMAPE")
    print("\n=== STRATEGY LEADERBOARD (lower median_sMAPE = better) ===")
    print(strat.to_string(index=False))

    # --------------------------------------------------------------
    # 6. Write outputs
    # --------------------------------------------------------------
    detail.to_csv(OUT / "advanced_per_line.csv", index=False)
    lb.to_csv(OUT / "advanced_leaderboard.csv", index=False)
    strat.to_csv(OUT / "advanced_summary.csv", index=False)
    pd.DataFrame(
        [{"Line_Key": k, "Method": v} for k, v in adaptive_picks.items()]
    ).to_csv(OUT / "advanced_best_strategy.csv", index=False)

    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
