"""
Per-fold stability check: Lasso vs Hist_mean vs XGBoost on Matériel
Uses data_long.csv with the same walk-forward setup as model.py
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("data_long.csv")
cat_map = {"INVESTISSEMENT": 0, "MATERIEL": 1, "PERSONNEL": 2}
df["cat_enc"] = df["category"].map(cat_map)

FEATURES   = ["quarter_num", "cat_enc", "taux_lag1", "taux_lag4", "lf_ratio"]
TARGET     = "taux"
df_model   = df.dropna(subset=FEATURES).copy()
test_years = [y for y in sorted(df_model["year"].unique())
              if y < 2025 and (df_model["year"] < y).any()]

xgb_p   = dict(n_estimators=200, max_depth=2, learning_rate=0.05,
               subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
               random_state=42, verbosity=0)

fold_results = []

for fold, test_year in enumerate(test_years):
    train = df_model[df_model["year"] < test_year]   # strictly past only
    test  = df_model[df_model["year"] == test_year]

    # Filter to MATERIEL only
    mat_enc   = cat_map["MATERIEL"]
    tr_m = train[train["cat_enc"] == mat_enc]
    te_m = test[test["cat_enc"]   == mat_enc]
    if len(te_m) == 0:
        continue

    y_true = te_m[TARGET].values

    # ── XGBoost (trained on all categories, same as main model)
    xgb = XGBRegressor(**xgb_p)
    xgb.fit(train[FEATURES], train[TARGET])
    pred_xgb = xgb.predict(te_m[FEATURES])

    # ── Lasso (trained on all categories)
    lasso = Pipeline([("sc", StandardScaler()), ("reg", Lasso(alpha=0.01, max_iter=5000))])
    lasso.fit(train[FEATURES], train[TARGET])
    pred_lasso = lasso.predict(te_m[FEATURES])

    # ── Hist_mean per (cat_enc, quarter_num) — same approach as model.py
    hist_mean = (tr_m.groupby("quarter_num")[TARGET]
                     .mean()
                     .reset_index()
                     .rename(columns={TARGET: "hist_mean"}))
    te_m2 = te_m.merge(hist_mean, on="quarter_num", how="left")
    pred_hist = te_m2["hist_mean"].values

    for name, preds in [("XGBoost", pred_xgb),
                        ("Lasso",   pred_lasso),
                        ("Hist_mean", pred_hist)]:
        valid = ~np.isnan(preds) & ~np.isnan(y_true)
        rmse  = np.sqrt(mean_squared_error(y_true[valid], preds[valid]))
        fold_results.append({"fold": test_year, "model": name, "rmse": round(rmse, 4)})

df_folds = pd.DataFrame(fold_results)
pivot = df_folds.pivot_table(index="fold", columns="model", values="rmse")
pivot.columns.name = None
pivot.index.name   = "Test year"
# Add winner column
pivot["🏆 Winner"] = pivot[["XGBoost", "Lasso", "Hist_mean"]].idxmin(axis=1)

print("=== MATÉRIEL — RMSE per fold ===")
print(pivot.to_string())
print()

means = df_folds.groupby("model")["rmse"].mean().round(4).sort_values()
print("=== Mean RMSE across folds ===")
for m, v in means.items():
    print(f"  {m:12s}  {v:.4f}")

print()
lasso_wins = (pivot["🏆 Winner"] == "Lasso").sum()
hist_wins  = (pivot["🏆 Winner"] == "Hist_mean").sum()
xgb_wins   = (pivot["🏆 Winner"] == "XGBoost").sum()
print(f"Lasso wins:     {lasso_wins}/{len(pivot)} folds")
print(f"Hist_mean wins: {hist_wins}/{len(pivot)} folds")
print(f"XGBoost wins:   {xgb_wins}/{len(pivot)} folds")
