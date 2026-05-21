"""
03b_ml_anomalies.py
===================
ML-based anomaly detector that *extends* the rule-based detector
(`03_anomaly_detection.py`). It produces a continuous suspicion score
per (Line_Key, Year) by combining three independent signals:

  1. RULES        : count of rule flags raised by 03_anomaly_detection.py
                    (extraction + business). Already in 03_anomalies_*.csv.
  2. ISO_FOREST   : multivariate outlier score (IsolationForest) over
                    behavioural features (rates, virements, deltas).
  3. FORECAST_RES : how badly the SmartEnsemble forecast missed in the
                    2025 backtest (sMAPE_pct of best method per line).

Each of the three is normalised to [0, 1] and averaged into FINAL_SCORE.

Inputs (v2/data/03_forecast/):
    - 03_raw_enriched.csv
    - 03_anomalies_extraction.csv
    - 03_anomalies_business.csv
    - 04_backtest_2025_best_per_line.csv

Outputs (v2/data/03_forecast/):
    - 03b_ml_features.csv         (engineered feature panel)
    - 03b_ml_anomaly_scores.csv   (per (Line_Key, Year) scores)
    - 03b_ml_top_suspects.csv     (top 30 lines ranked by FINAL_SCORE)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "03_forecast"
RAW      = DATA_DIR / "03_raw_enriched.csv"
ANOM_EXT = DATA_DIR / "03_anomalies_extraction.csv"
ANOM_BUS = DATA_DIR / "03_anomalies_business.csv"
BACKTEST = DATA_DIR / "04_backtest_2025_best_per_line.csv"

RANDOM_STATE = 42
CONTAMINATION = 0.05   # expected fraction of outliers (tune later)


# ---------- 1. Feature engineering ------------------------------------------
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    eps = 1e-9
    df["taux_engagement"]      = df["Total_Engage"] / (df["Total_Credits"] + eps)
    df["taux_vs_ouvert"]       = df["Total_Engage"] / (df["Credits_Ouverts"] + eps)
    df["virement_net_pct"]     = (df["Virements_Plus"] - df["Virements_Moins"]) / (df["Credits_Ouverts"] + eps)
    df["virement_gross_pct"]   = (df["Virements_Plus"] + df["Virements_Moins"]) / (df["Credits_Ouverts"] + eps)
    df["credits_topup_pct"]    = (df["Total_Credits"] - df["Credits_Ouverts"]) / (df["Credits_Ouverts"] + eps)

    df = df.sort_values(["Line_Key", "Year"])
    df["delta_credits_yoy"] = df.groupby("Line_Key")["Total_Credits"].pct_change()
    df["delta_engage_yoy"]  = df.groupby("Line_Key")["Total_Engage"].pct_change()

    # Programme-relative deviation: how far from peer average
    prog_avg = df.groupby(["Prog", "Year"])["taux_engagement"].transform("mean")
    df["taux_vs_prog_avg"] = df["taux_engagement"] - prog_avg

    feats = [
        "taux_engagement", "taux_vs_ouvert", "virement_net_pct",
        "virement_gross_pct", "credits_topup_pct",
        "delta_credits_yoy", "delta_engage_yoy", "taux_vs_prog_avg",
    ]
    out = df[["Year", "Line_Key", "Intitule"] + feats].copy()
    out[feats] = out[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, feats


# ---------- 2. Signal 1: rule-flag count ------------------------------------
def signal_rules(raw: pd.DataFrame) -> pd.DataFrame:
    """Count rule flags per (Line_Key, Year). Extraction flags have no Year
    -> we broadcast them to every Year of that Line_Key (identity issues
    affect the whole series)."""
    ext = pd.read_csv(ANOM_EXT)
    bus = pd.read_csv(ANOM_BUS)

    years = raw[["Line_Key", "Year"]].drop_duplicates()

    # Extraction: one flag per Line_Key -> replicate across years
    ext_count = ext.groupby("Line_Key").size().rename("rules_ext").reset_index()
    ext_by_year = years.merge(ext_count, on="Line_Key", how="left")

    # Business: per (Line_Key, Year)
    bus["Year"] = pd.to_numeric(bus["Year"], errors="coerce")
    bus_count = (bus.dropna(subset=["Year"])
                    .groupby(["Line_Key", "Year"]).size()
                    .rename("rules_bus").reset_index())
    bus_count["Year"] = bus_count["Year"].astype(int)

    out = years.merge(ext_by_year, on=["Line_Key", "Year"], how="left") \
               .merge(bus_count,  on=["Line_Key", "Year"], how="left")
    out[["rules_ext", "rules_bus"]] = out[["rules_ext", "rules_bus"]].fillna(0).astype(int)
    out["rules_count"] = out["rules_ext"] + out["rules_bus"]
    return out


# ---------- 3. Signal 2: IsolationForest ------------------------------------
def signal_isoforest(features: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = StandardScaler().fit_transform(features[feat_cols].values)
    iso = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    ).fit(X)
    # score_samples: higher = more normal. Flip + min-max to [0,1] where 1 = most anomalous.
    raw_score = -iso.score_samples(X)
    out = features[["Year", "Line_Key"]].copy()
    out["iso_score_raw"] = raw_score
    out["iso_flag"]      = (iso.predict(X) == -1).astype(int)
    return out


# ---------- 4. Signal 3: forecast residual ----------------------------------
def signal_forecast() -> pd.DataFrame:
    if not BACKTEST.exists():
        print("[warn] backtest file missing — skipping forecast-residual signal")
        return pd.DataFrame(columns=["Line_Key", "Year", "forecast_smape"])
    bt = pd.read_csv(BACKTEST)
    # Best method per line in 2025 backtest. The column may be "Yes"/"No" or 1/0.
    if "Is_Best" in bt.columns:
        bt = bt[bt["Is_Best"].astype(str).str.lower().isin(["yes", "1", "true"])]
    out = bt[["Line_Key", "sMAPE_pct"]].rename(columns={"sMAPE_pct": "forecast_smape"})
    out["Year"] = 2025
    return out


# ---------- 5. Combine ------------------------------------------------------
def minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def main() -> None:
    raw = pd.read_csv(RAW)
    features, feat_cols = build_features(raw)
    features.to_csv(DATA_DIR / "03b_ml_features.csv", index=False)

    rules    = signal_rules(raw)
    iso      = signal_isoforest(features, feat_cols)
    fcast    = signal_forecast()

    df = features.merge(rules, on=["Line_Key", "Year"], how="left") \
                 .merge(iso,   on=["Line_Key", "Year"], how="left") \
                 .merge(fcast, on=["Line_Key", "Year"], how="left")

    df["rules_count"]    = df["rules_count"].fillna(0)
    df["iso_score_raw"]  = df["iso_score_raw"].fillna(df["iso_score_raw"].median())
    # Forecast residual is only available for the 2025 backtest year.
    # Treat missing as 0 (no evidence) — NOT the median, otherwise every
    # row gets a non-zero forecast score and the signal becomes useless.
    df["forecast_smape"] = df["forecast_smape"].fillna(0.0)

    df["score_rules"]    = minmax(df["rules_count"])
    df["score_iso"]      = minmax(df["iso_score_raw"])
    df["score_forecast"] = minmax(df["forecast_smape"])

    df["FINAL_SCORE"] = (df["score_rules"] + df["score_iso"] + df["score_forecast"]) / 3.0
    df["FINAL_FLAG"]  = (df["FINAL_SCORE"] >= df["FINAL_SCORE"].quantile(0.90)).astype(int)

    cols = ["Year", "Line_Key", "Intitule",
            "rules_count", "iso_score_raw", "iso_flag", "forecast_smape",
            "score_rules", "score_iso", "score_forecast",
            "FINAL_SCORE", "FINAL_FLAG"]
    df[cols].to_csv(DATA_DIR / "03b_ml_anomaly_scores.csv", index=False)

    # Top suspects: aggregate to Line_Key (max year score)
    top = (df.sort_values("FINAL_SCORE", ascending=False)
             .groupby("Line_Key", as_index=False)
             .first()
             .sort_values("FINAL_SCORE", ascending=False)
             .head(30))
    top[cols].to_csv(DATA_DIR / "03b_ml_top_suspects.csv", index=False)

    print("=" * 72)
    print("ML ANOMALY DETECTION — summary")
    print("=" * 72)
    print(f"Rows scored                 : {len(df)}")
    print(f"IsolationForest flagged     : {int(df['iso_flag'].sum())}")
    print(f"Rule flags raised           : {int(df['rules_count'].sum())}")
    print(f"FINAL_FLAG (top 10% score)  : {int(df['FINAL_FLAG'].sum())}")
    print()
    print("Top 10 suspects (Line_Key, FINAL_SCORE):")
    print(top[["Line_Key", "Intitule", "FINAL_SCORE"]].head(10).to_string(index=False))
    print()
    print(f"Files written to {DATA_DIR}")


if __name__ == "__main__":
    main()
