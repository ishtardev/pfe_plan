"""Multi-scenario comparison: how does predicted Engage change under different
Credits envelopes set by the manager?

Scenarios (all built from latest observed Credits per line):
    - StatusQuo   : Credits_2027 = latest                (multiplier 1.00)
    - Austerity   : Credits_2027 = latest * 0.90         (-10%)
    - Expansion   : Credits_2027 = latest * 1.10         (+10%)
    - Manager     : whatever is in credits_scenario_2027.xlsx (if present)

For each scenario, predict Engage per line using the per-line best rate method
(reuses logic from 04_rate_forecast.py).

Output: v2/data/03_forecast/06_scenario_comparison_2027.xlsx
    - Sheet "by_line"      : one row per line, columns = scenarios
    - Sheet "by_programme" : aggregated to Programme level
    - Sheet "totals"       : grand totals per scenario
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL = DATA_DIR / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT_DIR = DATA_DIR / "03_forecast"
MANAGER_FILE = OUT_DIR / "05_credits_scenario_2027.xlsx"

COMPLETE_YEARS = list(range(2021, 2026))
VALIDATION_YEARS = [2024, 2025]
FORECAST_YEAR = 2027

SCENARIOS = {
    "Austerity_-10%": 0.90,
    "StatusQuo":      1.00,
    "Expansion_+10%": 1.10,
}


# rate methods (same as 04_rate_forecast)
def r_last(rates, _): return float(rates.iloc[-1])
def r_mean(rates, _): return float(rates.mean())
def r_median(rates, _): return float(rates.median())
def r_trend(rates, target):
    if len(rates) < 2: return float(rates.iloc[-1])
    slope, intercept = np.polyfit(rates.index.astype(float), rates.values.astype(float), 1)
    return float(np.clip(slope * target + intercept, 0.0, 2.0))

RATE_METHODS = {"RateLast": r_last, "RateMean": r_mean, "RateMed": r_median, "RateTrend": r_trend}


def smape(a, p):
    d = (abs(a) + abs(p)) / 2
    return 0.0 if d == 0 else 100 * abs(a - p) / d


def load_lines():
    df = pd.read_excel(
        PANEL,
        dtype={"Chap": str, "Prog": str, "Reg": str, "Proj": str, "Lb": str},
    )
    lignes = df[(df.Level == "Ligne") & df.Year.isin(COMPLETE_YEARS)].copy()
    lignes["Rate"] = np.where(
        lignes["Credits_Ouverts_Vises"] > 0,
        lignes["Total_Engage_Vises"] / lignes["Credits_Ouverts_Vises"],
        np.nan,
    )
    return lignes


def best_rate_method_per_line(lignes):
    records = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        for val_year in VALIDATION_YEARS:
            if val_year not in sub.index: continue
            hist = sub.loc[sub.index < val_year, "Rate"].dropna()
            if len(hist) < 2: continue
            actual = float(sub.loc[val_year, "Total_Engage_Vises"])
            credits = float(sub.loc[val_year, "Credits_Ouverts_Vises"])
            for name, fn in RATE_METHODS.items():
                pred = fn(hist, val_year) * credits
                records.append((key, name, smape(actual, pred)))
    val = pd.DataFrame(records, columns=["Line_Key", "Method", "sMAPE"])
    agg = val.groupby(["Line_Key", "Method"])["sMAPE"].mean().reset_index()
    idx = agg.groupby("Line_Key")["sMAPE"].idxmin()
    return dict(zip(agg.loc[idx, "Line_Key"], agg.loc[idx, "Method"]))


def latest_credits_per_line(panel_full):
    return (panel_full[panel_full.Level == "Ligne"]
            .sort_values("Year")
            .groupby("Line_Key")
            .tail(1)
            .set_index("Line_Key")["Credits_Ouverts_Vises"])


def forecast_scenario(lignes, best_map, credits_per_line):
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        hist = sub["Rate"].dropna()
        if hist.empty: continue
        method = best_map.get(key, "RateMed")
        rate_hat = RATE_METHODS[method](hist, FORECAST_YEAR)
        credits = float(credits_per_line.get(key, sub["Credits_Ouverts_Vises"].iloc[-1]))
        rows.append({
            "Line_Key": key,
            "Intitule": sub["Intitule"].iloc[-1],
            "Credits": credits,
            "Predicted_Engage": rate_hat * credits,
        })
    return pd.DataFrame(rows)


def main():
    lignes = load_lines()
    panel_full = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str, "Proj": str, "Lb": str})

    print(f"Lines: {lignes.Line_Key.nunique()} | training years: {COMPLETE_YEARS}\n")
    best_map = best_rate_method_per_line(lignes)
    latest_credits = latest_credits_per_line(panel_full)

    # Build per-line table; one column per scenario
    by_line = None
    totals = {}
    for name, mult in SCENARIOS.items():
        credits_scen = latest_credits * mult
        fc = forecast_scenario(lignes, best_map, credits_scen)
        fc = fc.rename(columns={
            "Credits": f"Credits_{name}",
            "Predicted_Engage": f"Engage_{name}",
        })
        if by_line is None:
            by_line = fc
        else:
            by_line = by_line.merge(
                fc[["Line_Key", f"Credits_{name}", f"Engage_{name}"]],
                on="Line_Key",
            )
        totals[name] = fc[f"Engage_{name}"].sum()

    # Optional manager scenario
    if MANAGER_FILE.exists():
        mgr = pd.read_excel(MANAGER_FILE)
        col = next((c for c in mgr.columns if str(FORECAST_YEAR) in str(c) and "Credit" in str(c)), None)
        if col:
            mgr_credits = mgr.set_index("Line_Key")[col]
            fc = forecast_scenario(lignes, best_map, mgr_credits)
            fc = fc.rename(columns={"Credits": "Credits_Manager", "Predicted_Engage": "Engage_Manager"})
            by_line = by_line.merge(fc[["Line_Key", "Credits_Manager", "Engage_Manager"]], on="Line_Key")
            totals["Manager"] = fc["Engage_Manager"].sum()
            print(f"Manager scenario loaded from {MANAGER_FILE.name}\n")

    # Add Programme code for aggregation
    prog_map = (panel_full[panel_full.Level == "Ligne"]
                .drop_duplicates("Line_Key")
                .set_index("Line_Key")[["Chap", "Prog"]])
    by_line = by_line.merge(prog_map, left_on="Line_Key", right_index=True, how="left")

    # by_programme aggregation
    engage_cols = [c for c in by_line.columns if c.startswith("Engage_")]
    credits_cols = [c for c in by_line.columns if c.startswith("Credits_")]
    by_prog = (by_line.groupby(["Chap", "Prog"])[engage_cols + credits_cols]
                       .sum()
                       .reset_index())

    # totals summary
    totals_df = pd.DataFrame({
        "Scenario": list(totals.keys()),
        "Total_Engage_Forecast_2027": list(totals.values()),
    })
    base = totals.get("StatusQuo", list(totals.values())[0])
    totals_df["Delta_vs_StatusQuo_%"] = (totals_df["Total_Engage_Forecast_2027"] / base - 1) * 100

    print("=== Grand totals per scenario ===")
    print(totals_df.round(2).to_string(index=False))

    out = OUT_DIR / "06_scenario_comparison_2027.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        totals_df.to_excel(w, sheet_name="totals", index=False)
        by_prog.to_excel(w, sheet_name="by_programme", index=False)
        by_line.to_excel(w, sheet_name="by_line", index=False)
    print(f"\nSaved {out.relative_to(DATA_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
