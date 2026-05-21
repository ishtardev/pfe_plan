"""
03c_ml_anomalies_extended.py
============================
Extends 03b with four additional signals and produces a 7-signal ensemble
score per (Line_Key, Year). Each new signal looks at the data through a
DIFFERENT lens than IsolationForest:

  4. LOF                 : LocalOutlierFactor — *local* density (a line that
                           is weird relative to its neighbours, not the world).
  5. MAHALANOBIS         : EllipticEnvelope — statistical outlier under a
                           multivariate Gaussian fit (interpretable "z-distance").
  6. BENFORD             : Benford's law fit on the leading digit of Total_Engage
                           per Programme (classic audit test). Lines inherit
                           the chi2 score of their programme.
  7. CHANGE_POINT        : per Line_Key, magnitude of the biggest year-to-year
                           regime shift in (taux_engagement, virement_net_pct).
                           Captures "this line was stable then broke".

Inputs:
    - v2/data/03_forecast/03b_ml_features.csv      (built by 03b)
    - v2/data/03_forecast/03b_ml_anomaly_scores.csv (rules + iso + forecast)
    - v2/data/03_forecast/03_raw_enriched.csv

Outputs:
    - v2/data/03_forecast/03c_ml_anomaly_scores_ext.csv
    - v2/data/03_forecast/03c_ml_top_suspects_ext.csv
    - v2/data/03_forecast/03c_signal_agreement.csv (how often signals agree)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "03_forecast"
FEAT = DATA / "03b_ml_features.csv"
PREV = DATA / "03b_ml_anomaly_scores.csv"
RAW  = DATA / "03_raw_enriched.csv"

RANDOM_STATE = 42
CONTAMINATION = 0.05
FEAT_COLS = [
    "taux_engagement", "taux_vs_ouvert", "virement_net_pct",
    "virement_gross_pct", "credits_topup_pct",
    "delta_credits_yoy", "delta_engage_yoy", "taux_vs_prog_avg",
]


def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    return pd.Series(0.0, index=s.index) if hi - lo < 1e-12 else (s - lo) / (hi - lo)


# ---------- Signal 4: LOF ----------------------------------------------------
def signal_lof(features: pd.DataFrame) -> pd.DataFrame:
    X = StandardScaler().fit_transform(features[FEAT_COLS].values)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
    pred = lof.fit_predict(X)
    raw_score = -lof.negative_outlier_factor_   # higher = more anomalous
    return features[["Year", "Line_Key"]].assign(
        lof_score_raw=raw_score, lof_flag=(pred == -1).astype(int)
    )


# ---------- Signal 5: Mahalanobis (EllipticEnvelope) -------------------------
def signal_mahalanobis(features: pd.DataFrame) -> pd.DataFrame:
    X = StandardScaler().fit_transform(features[FEAT_COLS].values)
    try:
        env = EllipticEnvelope(
            contamination=CONTAMINATION, random_state=RANDOM_STATE, support_fraction=0.9
        ).fit(X)
        dist = -env.score_samples(X)          # higher = more anomalous
        flag = (env.predict(X) == -1).astype(int)
    except Exception as e:
        print(f"[warn] EllipticEnvelope failed ({e}) — filling zeros")
        dist = np.zeros(len(X)); flag = np.zeros(len(X), dtype=int)
    return features[["Year", "Line_Key"]].assign(maha_score_raw=dist, maha_flag=flag)


# ---------- Signal 6: Benford's law per Programme ---------------------------
def first_digit(x: float) -> int | None:
    if x is None or x <= 0 or np.isnan(x):
        return None
    s = f"{x:.20g}".lstrip("-0.")
    return int(s[0]) if s and s[0].isdigit() and s[0] != "0" else None


def benford_chi2(values: pd.Series) -> float:
    digits = values.dropna().apply(first_digit).dropna().astype(int)
    digits = digits[(digits >= 1) & (digits <= 9)]
    if len(digits) < 10:
        return np.nan
    expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)]) * len(digits)
    observed = np.array([(digits == d).sum() for d in range(1, 10)])
    return float(((observed - expected) ** 2 / expected).sum())


def signal_benford(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    prog_chi2 = (raw.groupby("Prog")["Total_Engage"]
                    .apply(benford_chi2)
                    .rename("benford_chi2").reset_index())
    prog_chi2["benford_chi2"] = prog_chi2["benford_chi2"].fillna(prog_chi2["benford_chi2"].median())
    # Attach Prog to each line
    line_to_prog = raw.drop_duplicates("Line_Key").set_index("Line_Key")["Prog"]
    out = features[["Year", "Line_Key"]].copy()
    out["Prog"] = out["Line_Key"].map(line_to_prog)
    out = out.merge(prog_chi2, on="Prog", how="left")
    return out[["Year", "Line_Key", "benford_chi2"]]


# ---------- Signal 7: Change-point magnitude per Line_Key -------------------
def signal_changepoint(features: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute jump in (taux_engagement, virement_net_pct) for each line.
    A stable line has small jumps; a 'broken' line has at least one huge jump."""
    df = features.sort_values(["Line_Key", "Year"]).copy()
    df["jump_taux"]    = df.groupby("Line_Key")["taux_engagement"].diff().abs()
    df["jump_virnet"]  = df.groupby("Line_Key")["virement_net_pct"].diff().abs()
    df["jump_combined"] = df[["jump_taux", "jump_virnet"]].max(axis=1)
    # Take max jump per Line_Key, broadcast to all its years
    max_jump = df.groupby("Line_Key")["jump_combined"].max().rename("changepoint_score")
    out = df[["Year", "Line_Key"]].merge(max_jump, on="Line_Key", how="left")
    out["changepoint_score"] = out["changepoint_score"].fillna(0.0)
    return out


# ---------- Main ------------------------------------------------------------
def main() -> None:
    features = pd.read_csv(FEAT)
    prev     = pd.read_csv(PREV)
    raw      = pd.read_csv(RAW)

    lof   = signal_lof(features)
    maha  = signal_mahalanobis(features)
    benf  = signal_benford(raw, features)
    chgpt = signal_changepoint(features)

    df = prev.merge(lof,   on=["Line_Key", "Year"], how="left") \
             .merge(maha,  on=["Line_Key", "Year"], how="left") \
             .merge(benf,  on=["Line_Key", "Year"], how="left") \
             .merge(chgpt, on=["Line_Key", "Year"], how="left")

    df["score_lof"]         = minmax(df["lof_score_raw"])
    df["score_maha"]        = minmax(df["maha_score_raw"])
    df["score_benford"]     = minmax(df["benford_chi2"])
    df["score_changepoint"] = minmax(df["changepoint_score"])

    # 7-signal average: rules, iso, forecast (from prev) + lof, maha, benford, chgpt
    comps = ["score_rules", "score_iso", "score_forecast",
             "score_lof", "score_maha", "score_benford", "score_changepoint"]
    df["FINAL_SCORE_EXT"] = df[comps].mean(axis=1)
    df["FINAL_FLAG_EXT"]  = (df["FINAL_SCORE_EXT"] >= df["FINAL_SCORE_EXT"].quantile(0.90)).astype(int)

    # Signal agreement: per row, how many signals individually flag it
    # Forecast residual is only populated for 2025 backtest rows; threshold
    # against the 90th percentile of NON-ZERO values, else all-zero rows pass.
    fcst = df["forecast_smape"]
    fcst_thr = fcst[fcst > 0].quantile(0.90) if (fcst > 0).any() else np.inf
    benf_thr = df["benford_chi2"].quantile(0.90)
    chgpt_thr = df["changepoint_score"].quantile(0.90)

    df["n_signals_flag"] = (
        (df["rules_count"] >= 2).astype(int)
        + df["iso_flag"]
        + (fcst >= fcst_thr).astype(int)
        + df["lof_flag"]
        + df["maha_flag"]
        + (df["benford_chi2"] >= benf_thr).astype(int)
        + (df["changepoint_score"] >= chgpt_thr).astype(int)
    )

    cols = ["Year", "Line_Key", "Intitule",
            "rules_count", "iso_flag", "forecast_smape",
            "lof_flag", "maha_flag", "benford_chi2", "changepoint_score",
            "score_rules", "score_iso", "score_forecast",
            "score_lof", "score_maha", "score_benford", "score_changepoint",
            "FINAL_SCORE_EXT", "FINAL_FLAG_EXT", "n_signals_flag"]
    df[cols].to_csv(DATA / "03c_ml_anomaly_scores_ext.csv", index=False)

    top = (df.sort_values("FINAL_SCORE_EXT", ascending=False)
             .groupby("Line_Key", as_index=False).first()
             .sort_values("FINAL_SCORE_EXT", ascending=False).head(30))
    top[cols].to_csv(DATA / "03c_ml_top_suspects_ext.csv", index=False)

    # Pairwise agreement matrix: do these signals fire on the same rows?
    flag_cols = {
        "rules≥2":   (df["rules_count"] >= 2).astype(int),
        "iso":       df["iso_flag"],
        "forecast":  (fcst >= fcst_thr).astype(int),
        "lof":       df["lof_flag"],
        "maha":      df["maha_flag"],
        "benford":   (df["benford_chi2"] >= benf_thr).astype(int),
        "chgpt":     (df["changepoint_score"] >= chgpt_thr).astype(int),
    }
    flag_df = pd.DataFrame(flag_cols)
    n = len(flag_df)
    agree = pd.DataFrame(index=flag_df.columns, columns=flag_df.columns, dtype=float)
    for a in flag_df.columns:
        for b in flag_df.columns:
            both = ((flag_df[a] == 1) & (flag_df[b] == 1)).sum()
            either = ((flag_df[a] == 1) | (flag_df[b] == 1)).sum()
            agree.loc[a, b] = round(both / either, 3) if either else 0.0
    agree.to_csv(DATA / "03c_signal_agreement.csv")

    print("=" * 72)
    print("EXTENDED ML ANOMALY DETECTION — 7-signal ensemble")
    print("=" * 72)
    print(f"Rows scored                : {len(df)}")
    print(f"Individual flag counts:")
    for k, v in flag_cols.items():
        print(f"  {k:<10s} : {int(v.sum())}")
    print(f"FINAL_FLAG_EXT (top 10%)   : {int(df['FINAL_FLAG_EXT'].sum())}")
    print(f"Rows flagged by ≥3 signals : {int((df['n_signals_flag'] >= 3).sum())}")
    print(f"Rows flagged by ≥4 signals : {int((df['n_signals_flag'] >= 4).sum())}")
    print()
    print("Top 10 suspects (ext):")
    print(top[["Line_Key", "Intitule", "FINAL_SCORE_EXT", "n_signals_flag"]].head(10).to_string(index=False))
    print()
    print("Signal agreement (Jaccard, top-right corner):")
    print(agree.to_string())


if __name__ == "__main__":
    main()
