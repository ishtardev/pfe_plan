"""Build a fully offline HTML dashboard for the budget manager.

How it works:
    1. Loads the stable panel.
    2. Computes per-line: best rate method, predicted rate for 2027,
       last observed Credits & Engage, programme/chapter labels.
    3. Embeds this model as a JSON blob inside dashboard.html.
    4. Manager opens dashboard.html in any browser (Chrome/Edge/Firefox).
       He uploads his Credits-scenario xlsx; the page predicts Engage line-by-line
       and lets him download an xlsx with the results.

No internet required: SheetJS is vendored locally next to dashboard.html.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DASH_DIR = ROOT / "dashboard"
DASH_DIR.mkdir(exist_ok=True)
PANEL = DATA_DIR / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"

COMPLETE_YEARS = list(range(2021, 2026))
VALIDATION_YEARS = [2024, 2025]
FORECAST_YEAR = 2027


def r_last(rates, _): return float(rates.iloc[-1])
def r_mean(rates, _): return float(rates.mean())
def r_median(rates, _): return float(rates.median())
def r_trend(rates, target):
    if len(rates) < 2: return float(rates.iloc[-1])
    s, i = np.polyfit(rates.index.astype(float), rates.values.astype(float), 1)
    return float(np.clip(s * target + i, 0.0, 2.0))

METHODS = {"RateLast": r_last, "RateMean": r_mean, "RateMed": r_median, "RateTrend": r_trend}


def smape(a, p):
    d = (abs(a) + abs(p)) / 2
    return 0.0 if d == 0 else 100 * abs(a - p) / d


def main():
    panel = pd.read_excel(
        PANEL,
        dtype={"Chap": str, "Prog": str, "Reg": str, "Proj": str, "Lb": str},
    )
    lignes_all = panel[panel.Level == "Ligne"].copy()
    lignes = lignes_all[lignes_all.Year.isin(COMPLETE_YEARS)].copy()
    lignes["Rate"] = np.where(
        lignes["Credits_Ouverts_Vises"] > 0,
        lignes["Total_Engage_Vises"] / lignes["Credits_Ouverts_Vises"],
        np.nan,
    )

    # --- backtest -> best method per line ---
    val_rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        for vy in VALIDATION_YEARS:
            if vy not in sub.index: continue
            hist = sub.loc[sub.index < vy, "Rate"].dropna()
            if len(hist) < 2: continue
            actual = float(sub.loc[vy, "Total_Engage_Vises"])
            credits = float(sub.loc[vy, "Credits_Ouverts_Vises"])
            for name, fn in METHODS.items():
                pred = fn(hist, vy) * credits
                val_rows.append((key, name, smape(actual, pred)))
    val = pd.DataFrame(val_rows, columns=["Line_Key", "Method", "sMAPE"])
    agg = val.groupby(["Line_Key", "Method"])["sMAPE"].mean().reset_index()
    idx = agg.groupby("Line_Key")["sMAPE"].idxmin()
    best_map = dict(zip(agg.loc[idx, "Line_Key"], agg.loc[idx, "Method"]))
    best_score = dict(zip(agg.loc[idx, "Line_Key"], agg.loc[idx, "sMAPE"]))

    # --- ML anomaly score (max FINAL_SCORE per Line_Key) ---
    ml_path = DATA_DIR / "03_forecast" / "03b_ml_anomaly_scores.csv"
    if ml_path.exists():
        ml = pd.read_csv(ml_path)
        ml_score = ml.groupby("Line_Key")["FINAL_SCORE"].max().to_dict()
        ml_year = ml.sort_values("FINAL_SCORE", ascending=False) \
                    .drop_duplicates("Line_Key").set_index("Line_Key")["Year"].to_dict()
    else:
        ml_score, ml_year = {}, {}

    # --- final rate per line (trained on full history) ---
    latest = (lignes_all.sort_values("Year")
                        .groupby("Line_Key")
                        .tail(1)
                        .set_index("Line_Key"))
    model = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        hist = sub["Rate"].dropna()
        if hist.empty: continue
        method = best_map.get(key, "RateMed")
        rate_hat = METHODS[method](hist, FORECAST_YEAR)
        latest_row = latest.loc[key] if key in latest.index else None
        last_credits = float(latest_row["Credits_Ouverts_Vises"]) if latest_row is not None else float(sub["Credits_Ouverts_Vises"].iloc[-1])
        last_engage_2025 = float(sub.loc[2025, "Total_Engage_Vises"]) if 2025 in sub.index else None
        model.append({
            "Line_Key": key,
            "Intitule": str(sub["Intitule"].iloc[-1]),
            "Chap": str(sub["Chap"].iloc[-1]) if "Chap" in sub.columns else "",
            "Prog": str(sub["Prog"].iloc[-1]) if "Prog" in sub.columns else "",
            "Best_Method": method,
            "Backtest_sMAPE": round(best_score.get(key, float("nan")), 2),
            "Predicted_Rate_2027": round(rate_hat, 4),
            "Last_Credits": last_credits,
            "Last_Engage_2025": last_engage_2025,
            "Last_Year_Observed": int(latest_row["Year"]) if latest_row is not None else max(COMPLETE_YEARS),
            "ML_Risk_Score": round(float(ml_score.get(key, 0.0)), 3),
            "ML_Risk_Year": int(ml_year[key]) if key in ml_year and not pd.isna(ml_year[key]) else None,
        })
    model_df = pd.DataFrame(model).sort_values("Last_Engage_2025", ascending=False, na_position="last")

    # Programme labels: pull the Intitule from Level=='Programme' rows
    prog_lbl = (panel[panel.Level == "Programme"]
                .drop_duplicates(["Chap", "Prog"])
                .set_index(["Chap", "Prog"])["Intitule"]
                .to_dict())
    prog_labels = {f"{c}|{p}": v for (c, p), v in prog_lbl.items()}
    chap_lbl = (panel[panel.Level == "Chapitre"]
                .drop_duplicates(["Chap"])
                .set_index("Chap")["Intitule"]
                .to_dict())

    payload = {
        "forecast_year": FORECAST_YEAR,
        "training_years": COMPLETE_YEARS,
        "n_lines": int(len(model_df)),
        "lines": model_df.to_dict(orient="records"),
        "programme_labels": prog_labels,
        "chapter_labels": chap_lbl,
        "backtest_summary": {
            "median_sMAPE": float(np.median([s for s in best_score.values() if s == s])),
            "mean_sMAPE":   float(np.mean([s for s in best_score.values() if s == s])),
        },
    }

    # Build HTML
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    html = template.replace("/*__MODEL_JSON__*/", json.dumps(payload, ensure_ascii=False))
    out_html = DASH_DIR / "dashboard.html"
    out_html.write_text(html, encoding="utf-8")

    # Build the template CSV the manager will fill (so he doesn't have to know columns)
    template_csv = pd.DataFrame({
        "Line_Key": model_df["Line_Key"],
        "Intitule": model_df["Intitule"],
        "Credits_2027": model_df["Last_Credits"].round(0),
    })
    template_csv.to_excel(DASH_DIR / "scenario_template_2027.xlsx", index=False)
    template_csv.to_csv(DASH_DIR / "scenario_template_2027.csv", index=False, encoding="utf-8-sig")

    print(f"Built {out_html} ({out_html.stat().st_size / 1024:.0f} KB)")
    print(f"Lines embedded: {payload['n_lines']}")
    print(f"Backtest sMAPE median: {payload['backtest_summary']['median_sMAPE']:.2f}%")
    print(f"\nManager workflow:")
    print(f"  1. Open dashboard/dashboard.html in any browser")
    print(f"  2. (Optional) Edit scenario_template_2027.xlsx then upload it")
    print(f"  3. View predictions & download results")


if __name__ == "__main__":
    main()
