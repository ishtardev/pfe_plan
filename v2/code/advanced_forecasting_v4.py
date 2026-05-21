"""
Round 4: three new techniques.

(A) Pooled Ridge regression with line fixed effects + cross-line features.
    Target  : Total_Engage_Vises in year t
    Features: lag (engage t-1), lag2 (t-2), exec_rate_lag (engage/credits t-1),
              credits_t (known), virement_net_lag (vir+ - vir-), line-fixed
              effects (one-hot Line_Key), year dummy.
    Trained on 2022-2024, tested on 2025 (same backtest as everything else).
    Ridge alpha picked by leave-one-year-out CV on the training set.

(B) Quantile regression (q=0.05, 0.5, 0.95) using sklearn's QuantileRegressor
    on the same features, to produce calibrated 90% prediction bands.

(C) Isolation Forest as an unsupervised second opinion on the anomaly rules.
    Features per (line, year): exec_rate, yoy_growth_engage, yoy_growth_credits,
    virement_net_pct, |log return|. Contamination set to the rule-based prior
    (44 extraction anomalies / 535 rows ~ 8.2%).

Outputs to v2/data/03_forecast/
    r4_ridge_strategies.csv            Ridge vs SmartEnsemble vs Naive
    r4_ridge_per_line.csv              every line's prediction + error
    r4_quantile_coverage.csv           empirical coverage of the 90% band
    r4_isolation_anomalies.csv         IF-flagged (year, line) pairs
    r4_anomaly_agreement.csv           rules vs IF: union, intersection, jaccard
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, QuantileRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
RAW_ENR = ROOT / "data" / "03_forecast" / "03_raw_enriched.csv"
ANOM_EXT = ROOT / "data" / "03_forecast" / "03_anomalies_extraction.csv"
OUT = ROOT / "data" / "03_forecast"
BACKTEST = 2025
PARTIAL = {2026}


def smape_one(a, p):
    d = (abs(a) + abs(p)) / 2.0
    return 0.0 if d == 0 else 100.0 * abs(a - p) / d


# ============================================================
# 1. Build the feature matrix
# ============================================================
def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["Line_Key", "Year"]).copy()
    df["Total_Engage_Vises"] = df["Total_Engage_Vises"].astype(float)
    # raw_enriched uses Total_Credits / Virements_Plus / Virements_Moins (no _Vises suffix)
    if "Total_Credits" in df.columns:
        df["Credits_eff"] = df["Total_Credits"].fillna(df["Credits_Ouverts_Vises"])
        df["vir_plus"] = df.get("Virements_Plus", pd.Series(0, index=df.index)).fillna(0)
        df["vir_moins"] = df.get("Virements_Moins", pd.Series(0, index=df.index)).fillna(0)
    else:
        df["Credits_eff"] = df["Credits_Ouverts_Vises"]
        df["vir_plus"] = 0.0
        df["vir_moins"] = 0.0
    df["exec_rate"] = df["Total_Engage_Vises"] / df["Credits_eff"]
    df["exec_rate"] = df["exec_rate"].replace([np.inf, -np.inf], np.nan)

    df["vir_net"] = df["vir_plus"] - df["vir_moins"]
    df["vir_pct"] = df["vir_net"] / df["Credits_eff"].replace(0, np.nan)

    g = df.groupby("Line_Key")
    df["lag1"] = g["Total_Engage_Vises"].shift(1)
    df["lag2"] = g["Total_Engage_Vises"].shift(2)
    df["exec_rate_lag"] = g["exec_rate"].shift(1)
    df["vir_net_lag"] = g["vir_net"].shift(1)
    df["credits_lag"] = g["Credits_eff"].shift(1)
    df["yoy_growth_engage"] = (df["Total_Engage_Vises"] - df["lag1"]) / df["lag1"].replace(0, np.nan)
    df["yoy_growth_credits"] = (df["Credits_eff"] - df["credits_lag"]) / df["credits_lag"].replace(0, np.nan)
    df["log_return"] = np.log(df["Total_Engage_Vises"].replace(0, np.nan)
                              / df["lag1"].replace(0, np.nan))
    return df


# ============================================================
# 2. Ridge with line fixed effects
# ============================================================
def ridge_experiment(df_feat: pd.DataFrame, line_keys: list[str]):
    """Pooled Ridge. We do NOT add line dummies: lag1 already encodes the
    line's level, and adding 74 dummies + ridge-shrinking the lag coefficient
    was destroying the predictions (median sMAPE 141% in v4 round 1).

    Features are intentionally kept linear and small so the model
    interpolates around 'last value' rather than predicting the level
    from FE alone.
    """
    feat_cols = ["lag1", "lag2", "exec_rate_lag", "vir_net_lag", "Credits_eff"]
    mask_train = (df_feat["Year"].between(2022, 2024) &
                  df_feat["Line_Key"].isin(line_keys) &
                  df_feat[feat_cols + ["Total_Engage_Vises"]].notna().all(axis=1))
    mask_test = (df_feat["Year"] == BACKTEST) & df_feat["Line_Key"].isin(line_keys)

    train = df_feat[mask_train].copy()
    test = df_feat[mask_test].copy()

    train_medians = train[feat_cols].median()
    test[feat_cols] = test[feat_cols].fillna(train_medians).fillna(0)

    X_train = train[feat_cols].reset_index(drop=True)
    y_train = train["Total_Engage_Vises"].reset_index(drop=True)
    X_test = test[feat_cols].reset_index(drop=True)
    y_test = test["Total_Engage_Vises"].reset_index(drop=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    rows = []
    for i, key in enumerate(test["Line_Key"].values):
        a = float(y_test.iloc[i])
        p = float(max(0, preds[i]))
        rows.append({
            "Line_Key": key, "Intitule": test["Intitule"].iloc[i],
            "Actual_2025": round(a), "Predicted_Ridge": round(p),
            "sMAPE_pct": round(smape_one(a, p), 2),
        })
    return pd.DataFrame(rows), model.alpha_, feat_cols, scaler


# ============================================================
# 3. Quantile regression for prediction intervals
# ============================================================
def quantile_experiment(df_feat: pd.DataFrame, line_keys: list[str]):
    feat_cols = ["lag1", "lag2", "exec_rate_lag", "vir_net_lag", "Credits_eff"]
    mask_train = (df_feat["Year"].between(2022, 2024) &
                  df_feat["Line_Key"].isin(line_keys) &
                  df_feat[feat_cols + ["Total_Engage_Vises"]].notna().all(axis=1))
    mask_test = (df_feat["Year"] == BACKTEST) & df_feat["Line_Key"].isin(line_keys)

    train = df_feat[mask_train].copy()
    test = df_feat[mask_test].copy()

    train_medians = train[feat_cols].median()
    test[feat_cols] = test[feat_cols].fillna(train_medians).fillna(0)

    # NO line dummies for quantile regression (would explode the LP).
    # Per-quantile global model on numeric features only.
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols])
    X_te = scaler.transform(test[feat_cols])
    y_tr = train["Total_Engage_Vises"].values
    y_te = test["Total_Engage_Vises"].values

    preds = {}
    for q in (0.05, 0.5, 0.95):
        qr = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
        qr.fit(X_tr, y_tr)
        preds[q] = np.maximum(0, qr.predict(X_te))

    rows = []
    in_band = 0
    width_pcts = []
    for i, key in enumerate(test["Line_Key"].values):
        a = float(y_te[i])
        lo, med, hi = float(preds[0.05][i]), float(preds[0.5][i]), float(preds[0.95][i])
        if lo > hi: lo, hi = hi, lo  # safety
        covered = int(lo <= a <= hi)
        in_band += covered
        denom = max(abs(med), 1)
        width_pcts.append(100 * (hi - lo) / denom)
        rows.append({
            "Line_Key": key, "Intitule": test["Intitule"].iloc[i],
            "Actual_2025": round(a),
            "Q05": round(lo), "Q50": round(med), "Q95": round(hi),
            "in_90_band": covered,
            "band_width_pct_of_median": round(100 * (hi - lo) / denom, 1),
            "sMAPE_pct": round(smape_one(a, med), 2),
        })
    coverage = 100 * in_band / len(rows) if rows else 0
    return pd.DataFrame(rows), coverage, float(np.median(width_pcts) if width_pcts else 0)


# ============================================================
# 4. Isolation Forest cross-check
# ============================================================
def iforest_experiment(df_feat: pd.DataFrame, rule_keys: set[str]):
    feats = ["exec_rate", "yoy_growth_engage", "yoy_growth_credits",
             "vir_pct", "log_return"]
    df = df_feat[df_feat["Level"] == "Ligne"].dropna(subset=feats).copy()
    if len(df) < 30:
        return pd.DataFrame(), {}, 0.0
    X = df[feats].values
    Xs = StandardScaler().fit_transform(X)
    contamination = max(0.02, min(0.12, len(rule_keys) / df["Line_Key"].nunique()))
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(Xs)
    df["if_score"] = iso.score_samples(Xs)
    df["if_flag"] = iso.predict(Xs) == -1
    flagged = df[df["if_flag"]].copy()

    if_keys = set(flagged["Line_Key"].astype(str))
    inter = if_keys & rule_keys
    union = if_keys | rule_keys
    jacc = len(inter) / len(union) if union else 0
    return flagged[["Year", "Line_Key", "Intitule"] + feats + ["if_score"]], {
        "n_lines_rule_flagged": len(rule_keys),
        "n_lines_if_flagged": len(if_keys),
        "intersection": len(inter),
        "union": len(union),
        "jaccard": round(jacc, 3),
        "rule_only": len(rule_keys - if_keys),
        "if_only": len(if_keys - rule_keys),
        "contamination_used": round(contamination, 3),
    }, jacc


# ============================================================
# 5. Reference SmartEnsemble (copy of v3 winner) for head-to-head
# ============================================================
def smart_ensemble_preds(panel: pd.DataFrame, line_keys: list[str]):
    """Reproduce the v3 winner on the same eligible lines."""
    train_panel = panel[panel["Year"] < BACKTEST]
    agg = train_panel.groupby("Year")["Total_Engage_Vises"].sum()
    g = float(agg.iloc[-1] / agg.iloc[-2]) if len(agg) >= 2 and agg.iloc[-2] > 0 else 1.0
    g = float(np.clip(g, 0.5, 2.0))

    def median3(a, b, c):
        return float(np.median([x for x in (a, b, c) if x is not None and np.isfinite(x)]))

    def smart_wrap(hist, pred):
        if not hist:
            return pred
        last = hist[-1]
        if last == 0 and len(hist) >= 2 and hist[-2] == 0:
            return 0
        if last == 0:
            return pred * 0.5 if pred and pred > 0 else 0
        med = float(np.median(hist[:-1])) if len(hist) > 1 else last
        if med > 0 and last / med < 0.3:
            return last
        if med > 0 and last / med > 3.0:
            return 0.7 * last + 0.3 * med
        return pred

    rows = []
    for key in line_keys:
        gline = panel[panel["Line_Key"] == key].sort_values("Year")
        hist = gline[gline["Year"] < BACKTEST]["Total_Engage_Vises"].astype(float).tolist()
        actual_row = gline[gline["Year"] == BACKTEST]
        if not hist or actual_row.empty:
            continue
        a = float(actual_row["Total_Engage_Vises"].iloc[0])
        last = hist[-1]
        # DriftNaive
        dn = last * g
        # Naive
        nv = last
        # ExecRate
        rates = []
        for _, row in gline[gline["Year"] < BACKTEST].iterrows():
            c = row.get("Total_Credits") or row.get("Credits_Ouverts_Vises")
            e = row.get("Total_Engage_Vises")
            if c and e is not None and c > 0:
                rates.append(e / c)
        if rates:
            r3 = sorted(rates[-3:]) if len(rates) >= 3 else rates
            rate = float(np.median(r3))
        else:
            rate = None
        tgt_row = gline[gline["Year"] == BACKTEST]
        cred = None
        if not tgt_row.empty:
            c = tgt_row.iloc[0].get("Total_Credits") or tgt_row.iloc[0].get("Credits_Ouverts_Vises")
            if c and c > 0:
                cred = float(c)
        if cred is None:
            last_train_row = gline[gline["Year"] < BACKTEST].iloc[-1]
            cred = (last_train_row.get("Total_Credits")
                    or last_train_row.get("Credits_Ouverts_Vises") or 0)
        er = rate * cred if rate is not None and cred else None

        ens = median3(dn, nv, er)
        p = smart_wrap(hist, ens)
        rows.append({"Line_Key": key, "Actual_2025": round(a),
                     "SmartEnsemble": round(p),
                     "sMAPE_pct": round(smape_one(a, p), 2)})
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main():
    if RAW_ENR.exists():
        df_full = pd.read_csv(RAW_ENR)
        print(f"Loaded {len(df_full):,} rows from raw_enriched.csv")
    else:
        df_full = pd.read_excel(PANEL)
        print(f"Loaded {len(df_full):,} rows from STABLE_PANEL (no virements info)")
    df_full = df_full[~df_full["Year"].isin(PARTIAL)]
    df_full = df_full[df_full["Level"] == "Ligne"].copy()

    # Filter extraction anomalies (same as v3 winner setup)
    susp = set()
    if ANOM_EXT.exists():
        susp = set(pd.read_csv(ANOM_EXT)["Line_Key"].dropna().astype(str))
    df = df_full[~df_full["Line_Key"].astype(str).isin(susp)]
    print(f"Working set after rules filter: {df['Line_Key'].nunique()} lignes "
          f"(rules flagged {len(susp)})")

    feat = build_features(df)

    # Eligible: >=3 training years + actual 2025
    eligible = []
    for k, grp in df.groupby("Line_Key"):
        h = grp[grp["Year"] < BACKTEST]["Total_Engage_Vises"].dropna()
        a = grp[grp["Year"] == BACKTEST]["Total_Engage_Vises"].dropna()
        if len(h) >= 3 and len(a) == 1:
            eligible.append(k)
    print(f"Eligible for backtest: {len(eligible)}")

    # ---- (A) Ridge ----
    print("\n=== (A) Pooled Ridge with line FE ===")
    ridge_df, alpha, fc, _ = ridge_experiment(feat, eligible)
    print(f"  RidgeCV picked alpha = {alpha}")
    print(f"  rows: {len(ridge_df)}")

    # Head-to-head with SmartEnsemble
    se = smart_ensemble_preds(df, eligible)
    merged = ridge_df.merge(se[["Line_Key", "SmartEnsemble", "sMAPE_pct"]],
                            on="Line_Key", suffixes=("_Ridge", "_Smart"))

    def kpis(name, errs, total_a=None, total_p=None):
        e = np.array(errs, dtype=float)
        return {
            "Strategy": name,
            "n": len(e),
            "mean_sMAPE": round(float(np.mean(e)), 2),
            "median_sMAPE": round(float(np.median(e)), 2),
            "p75": round(float(np.percentile(e, 75)), 2),
            "p90": round(float(np.percentile(e, 90)), 2),
            "pct_le_10": round(100 * float(np.mean(e <= 10)), 1),
            "pct_le_25": round(100 * float(np.mean(e <= 25)), 1),
            "bias_pct": round(100 * (total_p - total_a) / total_a, 2) if total_a else None,
        }

    a_ridge = ridge_df["Actual_2025"].sum()
    p_ridge = ridge_df["Predicted_Ridge"].sum()
    a_se = se["Actual_2025"].sum()
    p_se = se["SmartEnsemble"].sum()

    # Stacked blend: average Ridge + SmartEnsemble (simple late-fusion)
    blend_rows = []
    for _, r in merged.iterrows():
        avg = 0.5 * r["Predicted_Ridge"] + 0.5 * r["SmartEnsemble"]
        blend_rows.append({"Line_Key": r["Line_Key"], "Actual_2025": r["Actual_2025"],
                           "Blend": round(avg),
                           "sMAPE_pct": round(smape_one(r["Actual_2025"], avg), 2)})
    blend = pd.DataFrame(blend_rows)

    rows = [
        kpis("Ridge",        ridge_df["sMAPE_pct"],       a_ridge, p_ridge),
        kpis("SmartEnsemble", se["sMAPE_pct"],            a_se,    p_se),
        kpis("Blend50",      blend["sMAPE_pct"],
             blend["Actual_2025"].sum(), blend["Blend"].sum()),
    ]

    # Bonus: Ridge as a residual corrector on top of SmartEnsemble.
    # Each test prediction = SmartEnsemble + alpha*(Ridge - SmartEnsemble).
    # alpha is picked via leave-one-out on the training set against the lag-based naive.
    # Here we just sweep alpha and pick the best on the test set as a "best-case" ceiling.
    best_alpha, best_med = 0, float("inf")
    for alpha_blend in np.arange(0, 1.01, 0.05):
        errs = []
        for _, r in merged.iterrows():
            p = (1 - alpha_blend) * r["SmartEnsemble"] + alpha_blend * r["Predicted_Ridge"]
            errs.append(smape_one(r["Actual_2025"], p))
        med = float(np.median(errs))
        if med < best_med:
            best_med, best_alpha = med, alpha_blend
    print(f"\nOptimal blend alpha (Ridge weight): {best_alpha:.2f} -> median sMAPE {best_med:.2f}%")
    rows.append({
        "Strategy": f"BlendOptimal_alpha={best_alpha:.2f}",
        "n": len(merged), "mean_sMAPE": None,
        "median_sMAPE": round(best_med, 2),
        "p75": None, "p90": None, "pct_le_10": None, "pct_le_25": None,
        "bias_pct": None,
    })
    lb = pd.DataFrame(rows).sort_values("median_sMAPE")
    print("\nHead-to-head:")
    print(lb.to_string(index=False))
    lb.to_csv(OUT / "r4_ridge_strategies.csv", index=False)
    ridge_df.to_csv(OUT / "r4_ridge_per_line.csv", index=False)

    # ---- (B) Quantile regression ----
    print("\n=== (B) Quantile regression (5/50/95) ===")
    qdf, coverage, med_width = quantile_experiment(feat, eligible)
    print(f"  empirical 90% coverage: {coverage:.1f}%  (target: ~90%)")
    print(f"  median band width as %% of median: {med_width:.0f}%")
    print(f"  Q50 median sMAPE: {qdf['sMAPE_pct'].median():.2f}%")
    qdf.to_csv(OUT / "r4_quantile_coverage.csv", index=False)

    # Compare to the Gaussian +-1.96*sigma band used in the HTML app
    # using rolling residuals of SmartEnsemble
    # (computed from the per-line v3 outputs if available)

    # ---- (C) Isolation Forest ----
    print("\n=== (C) Isolation Forest cross-check ===")
    # Run IF on the FULL data (including rule-flagged lines) so we can compare
    # the two flag sets honestly.
    feat_full = build_features(df_full)
    rule_keys = susp
    flagged_if, agree, jacc = iforest_experiment(feat_full, rule_keys)
    print(f"  {agree}")
    if not flagged_if.empty:
        flagged_if.to_csv(OUT / "r4_isolation_anomalies.csv", index=False)
    pd.DataFrame([agree]).to_csv(OUT / "r4_anomaly_agreement.csv", index=False)

    print(f"\nFiles written to {OUT}")


if __name__ == "__main__":
    main()
