"""Compare backtest 2025 with different training windows."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"

def smape(a, p):
    d = (abs(a) + abs(p)) / 2
    return 0.0 if d == 0 else 100 * abs(a - p) / d

def naive(h): return h[-1]
def mean3(h): return float(np.mean(h[-3:]))
def median_(h): return float(np.median(h))
def trend(h):
    if len(h) < 2: return float(h[-1])
    a, b = np.polyfit(np.arange(len(h)), h, 1)
    return float(a * len(h) + b)
METHODS = {"Naive": naive, "Mean3": mean3, "Median": median_, "Trend": trend}

def run(df, train_years, target=2025):
    rows = []
    for key, grp in df.groupby("Line_Key"):
        g = grp.sort_values("Year")
        train = g[g["Year"].isin(train_years)].dropna(subset=["Total_Engage_Vises"])
        actual_row = g[g["Year"] == target].dropna(subset=["Total_Engage_Vises"])
        if len(train) < 2 or actual_row.empty:
            continue
        actual = float(actual_row["Total_Engage_Vises"].iloc[0])
        tv = train["Total_Engage_Vises"].astype(float).tolist()
        for name, fn in METHODS.items():
            try: pred = fn(tv)
            except: continue
            if not np.isfinite(pred): continue
            rows.append({"Line_Key": key, "Method": name,
                         "Actual": actual, "Predicted": pred,
                         "sMAPE": smape(actual, pred),
                         "AbsErr": abs(actual - pred)})
    res = pd.DataFrame(rows)
    best = res.loc[res.groupby("Line_Key")["sMAPE"].idxmin()]
    lb = (res.groupby("Method")
              .agg(n=("sMAPE", "size"),
                   mean_sMAPE=("sMAPE", "mean"),
                   median_sMAPE=("sMAPE", "median"),
                   mean_AbsErr=("AbsErr", "mean"))
              .round(2).reset_index())
    wins = best["Method"].value_counts().to_dict()
    lb["wins"] = lb["Method"].map(lambda m: wins.get(m, 0))
    lb = lb.sort_values("mean_sMAPE")
    sm = best["sMAPE"].values
    summary = {
        "n_lignes": len(best),
        "median_sMAPE_best": round(float(np.median(sm)), 2),
        "mean_sMAPE_best": round(float(np.mean(sm)), 2),
        "pct_under_10pct": round(100 * float(np.mean(sm <= 10)), 1),
        "pct_under_25pct": round(100 * float(np.mean(sm <= 25)), 1),
        "global_error_pct": round(100 * (best["Predicted"].sum() - best["Actual"].sum()) / best["Actual"].sum(), 2),
    }
    return lb, summary

df = pd.read_excel(PANEL)
df = df[df["Level"] == "Ligne"]
df = df[~df["Year"].isin([2026])]

print("="*72)
print("CONFIG A: train 2021-2024 -> predict 2025 (4 training years)")
print("="*72)
lb_a, sum_a = run(df, train_years=[2021,2022,2023,2024])
print(lb_a.to_string(index=False))
print("Summary:", sum_a)

print()
print("="*72)
print("CONFIG B: train 2022-2024 -> predict 2025 (3 training years, drop 2021)")
print("="*72)
lb_b, sum_b = run(df, train_years=[2022,2023,2024])
print(lb_b.to_string(index=False))
print("Summary:", sum_b)

print()
print("="*72)
print("DELTA  (B - A)")
print("="*72)
for k in sum_a:
    a, b = sum_a[k], sum_b[k]
    if isinstance(a, (int, float)):
        diff = b - a
        arrow = "↓ better" if (k.startswith("median") or k.startswith("mean_sMAPE") or k.startswith("global")) and diff < 0 else \
                "↑ better" if k.startswith("pct_") and diff > 0 else \
                "↑ worse" if (k.startswith("median") or k.startswith("mean_sMAPE")) and diff > 0 else \
                "↓ worse" if k.startswith("pct_") and diff < 0 else ""
        print(f"  {k:25s}  A={a:>8}  B={b:>8}  Δ={diff:+.2f}  {arrow}")
