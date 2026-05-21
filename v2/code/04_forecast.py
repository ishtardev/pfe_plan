"""Forecast Total_Engage_Vises for next year (2027) at the Ligne level.

Strategy:
  - 5 complete training years: 2021-2025 (2026 is partial -> excluded).
  - Rolling backtest on 2024 & 2025 to pick the best method per line.
  - 5 methods:
        1. Naive       : previous year's value (baseline)
        2. Mean3       : mean of last 3 years (baseline)
        3. Trend       : linear regression on (year, value)
        4. Ridge       : ridge regression on engineered lag features (per line)
        5. XGB         : XGBoost trained GLOBALLY across all 107 lines pooled
                         (one model, line identity passed as a categorical feature)

Outputs (in v2/data/03_forecast/):
  - forecast_2027.xlsx        : per-line forecast for 2027 with chosen method
  - validation_report.xlsx    : per-line/per-method backtest errors
  - method_leaderboard.xlsx   : global ranking of methods
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL = DATA_DIR / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT_DIR = DATA_DIR / "03_forecast"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_YEARS = list(range(2021, 2026))   # 2021..2025 complete
VALIDATION_YEARS = [2024, 2025]         # rolling backtest
FORECAST_YEAR = 2027                    # final forecast horizon


# ---------- forecasting methods ----------
# Each method takes a history (pd.Series indexed by Year) and a target Year,
# and returns a float forecast.

def m_naive(hist: pd.Series, target_year: int) -> float:
    return float(hist.iloc[-1])


def m_mean3(hist: pd.Series, target_year: int) -> float:
    return float(hist.tail(3).mean())


def m_trend(hist: pd.Series, target_year: int) -> float:
    x = hist.index.values.astype(float).reshape(-1, 1)
    y = hist.values.astype(float)
    if len(x) < 2:
        return float(y[-1])
    model = LinearRegression().fit(x, y)
    pred = float(model.predict(np.array([[target_year]]))[0])
    return max(pred, 0.0)  # no negative spending


def m_ridge(hist: pd.Series, target_year: int) -> float:
    """Ridge on [lag1, lag2, mean_past, slope] features.

    With only ~5 points we fit on 'within-series' synthetic samples:
    for each year t in hist where lag2 is available, build a sample.
    Falls back to trend if too few rows.
    """
    years = hist.index.values.astype(int)
    vals = hist.values.astype(float)
    samples_X, samples_y = [], []
    for i in range(2, len(vals)):
        lag1 = vals[i - 1]
        lag2 = vals[i - 2]
        mean_past = vals[:i].mean()
        slope = np.polyfit(years[:i], vals[:i], 1)[0] if i >= 2 else 0.0
        samples_X.append([lag1, lag2, mean_past, slope])
        samples_y.append(vals[i])
    if len(samples_X) < 2:
        return m_trend(hist, target_year)
    X = np.array(samples_X)
    y = np.array(samples_y)
    model = Ridge(alpha=1.0).fit(X, y)
    # Build feature row for target_year
    lag1 = vals[-1]
    lag2 = vals[-2]
    mean_past = vals.mean()
    slope = np.polyfit(years, vals, 1)[0]
    pred = float(model.predict(np.array([[lag1, lag2, mean_past, slope]]))[0])
    return max(pred, 0.0)


METHODS = {
    "Naive": m_naive,
    "Mean3": m_mean3,
    "Trend": m_trend,
    "Ridge": m_ridge,
}


# ---------- global XGBoost (cross-line pooling) ----------
# Per-line history has only ~5 points (too few for trees). We pool ALL lines into
# one training set: each row = (line_id, year, lag1, lag2, mean_past, slope) -> y.
# The model learns generic temporal patterns + line-specific intercepts via the
# line_id categorical.

XGB_PREDICTIONS: dict[tuple[str, int], float] = {}


def _build_xgb_dataset(lignes: pd.DataFrame, max_year: int):
    """Build (X, y, predict_rows) where predict_rows lets us forecast max_year+1.

    Trains on rows where year <= max_year and at least 2 prior years exist.
    """
    rows_train, rows_predict = [], []
    line_ids = {k: i for i, k in enumerate(sorted(lignes["Line_Key"].unique()))}
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub[sub.Year <= max_year].sort_values("Year")
        vals = sub["Total_Engage_Vises"].values.astype(float)
        years = sub["Year"].values.astype(int)
        lid = line_ids[key]
        # training samples: predict year[i] from years[:i]
        for i in range(2, len(vals)):
            lag1, lag2 = vals[i - 1], vals[i - 2]
            mean_past = vals[:i].mean()
            slope = np.polyfit(years[:i], vals[:i], 1)[0]
            rows_train.append([lid, years[i], lag1, lag2, mean_past, slope, vals[i]])
        # forecast row for max_year + 1
        if len(vals) >= 2:
            lag1, lag2 = vals[-1], vals[-2]
            mean_past = vals.mean()
            slope = np.polyfit(years, vals, 1)[0]
            rows_predict.append([key, lid, max_year + 1, lag1, lag2, mean_past, slope])
    cols_train = ["line_id", "year", "lag1", "lag2", "mean_past", "slope", "y"]
    cols_pred = ["Line_Key", "line_id", "year", "lag1", "lag2", "mean_past", "slope"]
    return (
        pd.DataFrame(rows_train, columns=cols_train),
        pd.DataFrame(rows_predict, columns=cols_pred),
    )


def train_xgb_for_year(lignes: pd.DataFrame, target_year: int):
    """Train XGB on all data with year < target_year, predict target_year.
    Cache results in XGB_PREDICTIONS keyed by (Line_Key, target_year)."""
    train_df, pred_df = _build_xgb_dataset(lignes, max_year=target_year - 1)
    if train_df.empty or pred_df.empty:
        return
    feat = ["line_id", "year", "lag1", "lag2", "mean_past", "slope"]
    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_df[feat], train_df["y"])
    preds = model.predict(pred_df[feat])
    for key, p in zip(pred_df["Line_Key"], preds):
        XGB_PREDICTIONS[(key, target_year)] = max(float(p), 0.0)


def m_xgb(hist: pd.Series, target_year: int) -> float:
    """Look up the pre-trained global XGB prediction; fallback to naive."""
    # We can't know the line key from `hist` alone — caller must set _current_key.
    key = getattr(m_xgb, "_current_key", None)
    if key is not None and (key, target_year) in XGB_PREDICTIONS:
        return XGB_PREDICTIONS[(key, target_year)]
    return float(hist.iloc[-1])


METHODS["XGB"] = m_xgb


# ---------- metrics ----------
def smape(actual: float, pred: float) -> float:
    """Symmetric MAPE in %. Robust to zeros."""
    denom = (abs(actual) + abs(pred)) / 2
    if denom == 0:
        return 0.0
    return 100 * abs(actual - pred) / denom


# ---------- main pipeline ----------
def load_lines() -> pd.DataFrame:
    df = pd.read_excel(
        PANEL,
        dtype={"Chap": str, "Prog": str, "Reg": str, "Proj": str, "Lb": str},
    )
    lignes = df[df.Level == "Ligne"].copy()
    lignes = lignes[lignes.Year.isin(TRAIN_YEARS)]
    return lignes


def rolling_validation(lignes: pd.DataFrame) -> pd.DataFrame:
    """For each (line, validation_year, method), compute the forecast & error."""
    records = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        intitule = sub["Intitule"].iloc[-1]
        for val_year in VALIDATION_YEARS:
            hist = sub.loc[sub.index < val_year, "Total_Engage_Vises"]
            if len(hist) < 2 or val_year not in sub.index:
                continue
            actual = float(sub.loc[val_year, "Total_Engage_Vises"])
            for name, fn in METHODS.items():
                try:
                    m_xgb._current_key = key  # used by m_xgb lookup
                    pred = fn(hist, val_year)
                except Exception:
                    continue
                records.append({
                    "Line_Key": key,
                    "Intitule": intitule,
                    "Year": val_year,
                    "Method": name,
                    "Actual": actual,
                    "Predicted": pred,
                    "Abs_Error": abs(actual - pred),
                    "sMAPE_%": smape(actual, pred),
                })
    return pd.DataFrame(records)


def pick_best_method(val_df: pd.DataFrame) -> pd.DataFrame:
    """Best method per line = lowest mean sMAPE across validation years."""
    agg = (val_df.groupby(["Line_Key", "Method"])
                 .agg(mean_sMAPE=("sMAPE_%", "mean"),
                      mean_AbsErr=("Abs_Error", "mean"))
                 .reset_index())
    idx = agg.groupby("Line_Key")["mean_sMAPE"].idxmin()
    return agg.loc[idx].rename(columns={"Method": "Best_Method"}).reset_index(drop=True)


def forecast_next(lignes: pd.DataFrame, best: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Train on full history, forecast target_year using each line's best method."""
    best_map = dict(zip(best["Line_Key"], best["Best_Method"]))
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        hist = sub["Total_Engage_Vises"]
        method_name = best_map.get(key, "Naive")
        fn = METHODS[method_name]
        try:
            m_xgb._current_key = key
            pred = fn(hist, target_year)
        except Exception:
            pred = float(hist.iloc[-1])
        rows.append({
            "Line_Key": key,
            "Intitule": sub["Intitule"].iloc[-1],
            "Last_Year_Engage": float(hist.iloc[-1]),
            "Mean_Engage_5y": float(hist.mean()),
            "Best_Method": method_name,
            f"Forecast_{target_year}": pred,
            f"Change_vs_{hist.index.max()}_%": 100 * (pred - hist.iloc[-1]) / hist.iloc[-1]
                                                 if hist.iloc[-1] else np.nan,
        })
    return pd.DataFrame(rows).sort_values(f"Forecast_{target_year}", ascending=False)


def main():
    lignes = load_lines()
    print(f"Loaded {lignes.Line_Key.nunique()} lines x {lignes.Year.nunique()} years "
          f"({len(lignes)} rows)\n")

    print("=== Pre-training global XGBoost models (one per target year) ===")
    for yr in VALIDATION_YEARS + [FORECAST_YEAR]:
        train_xgb_for_year(lignes, yr)
        print(f"  trained XGB -> {yr} ({sum(1 for k in XGB_PREDICTIONS if k[1]==yr)} lines)")
    print()

    print("=== Rolling backtest (predicting 2024 & 2025) ===")
    val_df = rolling_validation(lignes)
    leaderboard = (val_df.groupby("Method")
                         .agg(mean_sMAPE=("sMAPE_%", "mean"),
                              median_sMAPE=("sMAPE_%", "median"),
                              mean_AbsErr=("Abs_Error", "mean"),
                              n=("Method", "size"))
                         .round(2)
                         .sort_values("mean_sMAPE"))
    print(leaderboard.to_string())
    print()

    best = pick_best_method(val_df)
    print("Best-method distribution across lines:")
    print(best["Best_Method"].value_counts().to_string())
    print()

    print(f"=== Forecasting {FORECAST_YEAR} ===")
    fc = forecast_next(lignes, best, FORECAST_YEAR)
    total = fc[f"Forecast_{FORECAST_YEAR}"].sum()
    last = fc["Last_Year_Engage"].sum()
    print(f"Sum of forecasts {FORECAST_YEAR}: {total:,.0f}")
    print(f"Sum last year (2025) : {last:,.0f}")
    print(f"Implied change       : {(total - last) / last * 100:+.2f}%")

    # Save outputs
    val_df.to_excel(OUT_DIR / "04_validation_report.xlsx", index=False)
    leaderboard.to_excel(OUT_DIR / "04_method_leaderboard.xlsx")
    best.to_excel(OUT_DIR / "04_best_method_per_line.xlsx", index=False)
    fc.to_excel(OUT_DIR / f"04_forecast_{FORECAST_YEAR}.xlsx", index=False)
    print(f"\nOutputs written to {OUT_DIR.relative_to(DATA_DIR.parent.parent)}/")


if __name__ == "__main__":
    main()
