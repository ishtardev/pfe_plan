"""Execution-rate model: predict Total_Engage_Vises FROM Credits_Ouverts_Vises.

Operational use case:
    The manager decides next year's Credits_Ouverts per line.
    This script predicts the resulting Total_Engage for that scenario.

Method:
    For each line, estimate execution_rate = Engage / Credits.
    Then forecast: Engage_hat = predicted_rate * Credits_input.

    We compare 4 ways of estimating the rate:
        1. RateLast   : last year's rate
        2. RateMean   : average of all years
        3. RateMed    : median of all years (robust to outliers)
        4. RateTrend  : linear trend on rate over years

Inputs:
    - v2/data/02_cleaned/SituationChap_STABLE_PANEL.xlsx
    - v2/data/03_forecast/05_credits_scenario_2027.xlsx  (optional manager input)
        If absent, we assume Credits_2027 = Credits_2026 (status-quo scenario).

Outputs (in v2/data/03_forecast/):
    - 05_rate_validation.xlsx                  : backtest of each rate method
    - 05_rate_leaderboard.xlsx                 : which rate method wins on average
    - 05_engage_forecast_2027.xlsx             : per-line predicted Engage for 2027
    - 05_credits_scenario_2027_TEMPLATE.xlsx   : empty template for manager input
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL = DATA_DIR / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT_DIR = DATA_DIR / "03_forecast"
OUT_DIR.mkdir(exist_ok=True)
SCENARIO_FILE = OUT_DIR / "05_credits_scenario_2027.xlsx"
TEMPLATE_FILE = OUT_DIR / "05_credits_scenario_2027_TEMPLATE.xlsx"

COMPLETE_YEARS = list(range(2021, 2026))  # 2026 partial -> excluded from rate estimation
VALIDATION_YEARS = [2024, 2025]
FORECAST_YEAR = 2027


# ---------- rate-estimation methods ----------
# Each takes a Series of historical rates indexed by Year, returns a forecast rate.

def r_last(rates: pd.Series, target_year: int) -> float:
    return float(rates.iloc[-1])


def r_mean(rates: pd.Series, target_year: int) -> float:
    return float(rates.mean())


def r_median(rates: pd.Series, target_year: int) -> float:
    return float(rates.median())


def r_trend(rates: pd.Series, target_year: int) -> float:
    if len(rates) < 2:
        return float(rates.iloc[-1])
    slope, intercept = np.polyfit(rates.index.astype(float), rates.values.astype(float), 1)
    pred = slope * target_year + intercept
    return float(np.clip(pred, 0.0, 2.0))  # cap at 200% to avoid extrapolation explosions


RATE_METHODS = {
    "RateLast": r_last,
    "RateMean": r_mean,
    "RateMed": r_median,
    "RateTrend": r_trend,
}


def smape(actual: float, pred: float) -> float:
    denom = (abs(actual) + abs(pred)) / 2
    return 0.0 if denom == 0 else 100 * abs(actual - pred) / denom


# ---------- pipeline ----------
def load_lines() -> pd.DataFrame:
    df = pd.read_excel(
        PANEL,
        dtype={"Chap": str, "Prog": str, "Reg": str, "Proj": str, "Lb": str},
    )
    lignes = df[(df.Level == "Ligne") & df.Year.isin(COMPLETE_YEARS)].copy()
    # Execution rate; guard against zero credits
    lignes["Rate"] = np.where(
        lignes["Credits_Ouverts_Vises"] > 0,
        lignes["Total_Engage_Vises"] / lignes["Credits_Ouverts_Vises"],
        np.nan,
    )
    return lignes


def rolling_validation(lignes: pd.DataFrame) -> pd.DataFrame:
    """For each (line, val_year, method): predict Engage_val from Credits_val * rate_hat."""
    records = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        for val_year in VALIDATION_YEARS:
            if val_year not in sub.index:
                continue
            hist_rates = sub.loc[sub.index < val_year, "Rate"].dropna()
            if len(hist_rates) < 2:
                continue
            actual_engage = float(sub.loc[val_year, "Total_Engage_Vises"])
            credits_val = float(sub.loc[val_year, "Credits_Ouverts_Vises"])
            for name, fn in RATE_METHODS.items():
                rate_hat = fn(hist_rates, val_year)
                pred_engage = rate_hat * credits_val
                records.append({
                    "Line_Key": key,
                    "Intitule": sub["Intitule"].iloc[-1],
                    "Year": val_year,
                    "Method": name,
                    "Rate_hat": rate_hat,
                    "Credits": credits_val,
                    "Actual_Engage": actual_engage,
                    "Pred_Engage": pred_engage,
                    "Abs_Error": abs(actual_engage - pred_engage),
                    "sMAPE_%": smape(actual_engage, pred_engage),
                })
    return pd.DataFrame(records)


def pick_best_rate_method(val_df: pd.DataFrame) -> pd.DataFrame:
    agg = (val_df.groupby(["Line_Key", "Method"])
                 .agg(mean_sMAPE=("sMAPE_%", "mean"),
                      mean_AbsErr=("Abs_Error", "mean"))
                 .reset_index())
    idx = agg.groupby("Line_Key")["mean_sMAPE"].idxmin()
    return (agg.loc[idx]
               .rename(columns={"Method": "Best_Rate_Method"})
               .reset_index(drop=True))


def load_or_build_scenario(lignes: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame [Line_Key, Intitule, Credits_2027].

    If the manager file exists -> read it. Otherwise build a status-quo scenario
    using the most recent observed Credits (could be 2026 partial; that's the user
    intent: 'consider 2026 as current').
    """
    # Always rebuild template so manager can update it
    panel = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str, "Proj": str, "Lb": str})
    latest = (panel[panel.Level == "Ligne"]
              .sort_values("Year")
              .groupby("Line_Key")
              .tail(1)[["Line_Key", "Intitule", "Year", "Credits_Ouverts_Vises"]]
              .rename(columns={"Credits_Ouverts_Vises": "Credits_Latest",
                               "Year": "Latest_Year"}))
    template = latest.copy()
    template[f"Credits_{FORECAST_YEAR}"] = template["Credits_Latest"]
    template.to_excel(TEMPLATE_FILE, index=False)

    if SCENARIO_FILE.exists():
        print(f"  Using manager scenario: {SCENARIO_FILE.name}")
        sc = pd.read_excel(SCENARIO_FILE)
    else:
        print(f"  No scenario file found -> status-quo (Credits_{FORECAST_YEAR} = latest)")
        sc = template.copy()
    return sc[["Line_Key", "Intitule", f"Credits_{FORECAST_YEAR}"]]


def forecast_engage(lignes: pd.DataFrame,
                    best: pd.DataFrame,
                    scenario: pd.DataFrame) -> pd.DataFrame:
    best_map = dict(zip(best["Line_Key"], best["Best_Rate_Method"]))
    scen_map = dict(zip(scenario["Line_Key"], scenario[f"Credits_{FORECAST_YEAR}"]))
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        hist_rates = sub["Rate"].dropna()
        if hist_rates.empty:
            continue
        method = best_map.get(key, "RateMed")
        rate_hat = RATE_METHODS[method](hist_rates, FORECAST_YEAR)
        credits_in = scen_map.get(key, float(sub["Credits_Ouverts_Vises"].iloc[-1]))
        pred_engage = rate_hat * credits_in
        rows.append({
            "Line_Key": key,
            "Intitule": sub["Intitule"].iloc[-1],
            f"Credits_{FORECAST_YEAR}_input": credits_in,
            "Best_Rate_Method": method,
            "Predicted_Rate": rate_hat,
            f"Predicted_Engage_{FORECAST_YEAR}": pred_engage,
            "Last_Engage_2025": float(sub.loc[2025, "Total_Engage_Vises"]) if 2025 in sub.index else np.nan,
        })
    fc = pd.DataFrame(rows)
    fc[f"Change_vs_2025_%"] = 100 * (fc[f"Predicted_Engage_{FORECAST_YEAR}"] - fc["Last_Engage_2025"]) / fc["Last_Engage_2025"].replace(0, np.nan)
    return fc.sort_values(f"Predicted_Engage_{FORECAST_YEAR}", ascending=False)


def main():
    lignes = load_lines()
    n_lines = lignes.Line_Key.nunique()
    print(f"Loaded {n_lines} stable Ligne items x {lignes.Year.nunique()} complete years\n")

    # Sanity: distribution of execution rates
    rate_summary = lignes.groupby("Year")["Rate"].agg(["median", "mean", "std"]).round(3)
    print("Execution rate per year (Engage / Credits):")
    print(rate_summary.to_string())
    print()

    print("=== Rolling backtest (predict Engage 2024 & 2025 from Credits * rate_hat) ===")
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

    best = pick_best_rate_method(val_df)
    print("Best rate-method per line:")
    print(best["Best_Rate_Method"].value_counts().to_string())
    print()

    print(f"=== Building {FORECAST_YEAR} scenario ===")
    scenario = load_or_build_scenario(lignes)

    print(f"=== Forecasting Engage {FORECAST_YEAR} ===")
    fc = forecast_engage(lignes, best, scenario)
    total_pred = fc[f"Predicted_Engage_{FORECAST_YEAR}"].sum()
    total_credits = fc[f"Credits_{FORECAST_YEAR}_input"].sum()
    total_last = fc["Last_Engage_2025"].sum()
    print(f"Total Credits  input {FORECAST_YEAR} : {total_credits:,.0f}")
    print(f"Total Engage forecast {FORECAST_YEAR} : {total_pred:,.0f}")
    print(f"Total Engage 2025            : {total_last:,.0f}")
    print(f"Implied global exec rate     : {total_pred / total_credits:.1%}")
    print(f"Change vs 2025               : {(total_pred - total_last) / total_last * 100:+.2f}%")

    val_df.to_excel(OUT_DIR / "05_rate_validation.xlsx", index=False)
    leaderboard.to_excel(OUT_DIR / "05_rate_leaderboard.xlsx")
    best.to_excel(OUT_DIR / "05_best_rate_method_per_line.xlsx", index=False)
    fc.to_excel(OUT_DIR / f"05_engage_forecast_{FORECAST_YEAR}.xlsx", index=False)
    print(f"\nOutputs in {OUT_DIR.relative_to(DATA_DIR.parent.parent)}/")
    print(f"Manager template: {TEMPLATE_FILE.name}  <-- fill Credits_{FORECAST_YEAR} column "
          f"and rename to credits_scenario_{FORECAST_YEAR}.xlsx")


if __name__ == "__main__":
    main()
