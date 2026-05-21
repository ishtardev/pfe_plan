"""
ML Modeling: Prediction of taux d'execution budgetaire
Ministere de la Justice - 2020-2025 (quarterly)

Models   : Random Forest, XGBoost
Baselines: Seasonal naive (taux_lag4), Historical mean per quarter/category
CV       : Leave-One-Year-Out (LOYO) - 6 folds
Metrics  : RMSE, MAE, MAPE
Output   : results_cv.csv, figures/
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "DejaVu Sans"

os.makedirs("figures", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data_long.csv")

# Encode category as integer
cat_map = {"INVESTISSEMENT": 0, "MATERIEL": 1, "PERSONNEL": 2}
df["cat_enc"] = df["category"].map(cat_map)

# Features and target
FEATURES = ["quarter_num", "cat_enc", "taux_lag1", "taux_lag4", "lf_ratio"]
TARGET   = "taux"

# Drop rows with any NaN in features (mostly 2020 T1 which has no lags)
df_model = df.dropna(subset=FEATURES).copy()
print(f"Usable rows after dropping NaN lags: {len(df_model)}")

years = sorted(df_model["year"].unique())

# ── 2. Walk-Forward (Expanding Window) Cross-Validation ──────────────────────
# Train strictly on past years; never use future data.
# 2021 has no prior training data → skip. Test folds: 2022, 2023, 2024, 2025.
results = []

rf_params  = dict(n_estimators=200, max_depth=3, min_samples_leaf=2,
                  random_state=42, n_jobs=-1)
xgb_params = dict(n_estimators=200, max_depth=2, learning_rate=0.05,
                  subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
                  random_state=42, verbosity=0)
lgb_params = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  num_leaves=15, min_child_samples=2,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbose=-1)

test_years_wf = [y for y in years if (df_model["year"] < y).any()]

for test_year in test_years_wf:
    train = df_model[df_model["year"] < test_year]
    test  = df_model[df_model["year"] == test_year]

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    # ── Random Forest
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # ── XGBoost
    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    # ── LightGBM
    lgbm = lgb.LGBMRegressor(**lgb_params)
    lgbm.fit(X_train, y_train)
    pred_lgbm = lgbm.predict(X_test)

    # ── Linear Regression (scaled)
    lr = Pipeline([("scaler", StandardScaler()),
                   ("model", LinearRegression())])
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # ── Ridge (alpha=1.0, scaled)
    ridge = Pipeline([("scaler", StandardScaler()),
                      ("model", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    # ── Lasso (alpha=0.01, scaled)
    lasso = Pipeline([("scaler", StandardScaler()),
                      ("model", Lasso(alpha=0.01, max_iter=5000))])
    lasso.fit(X_train, y_train)
    pred_lasso = lasso.predict(X_test)

    # ── Baseline 1: Seasonal naive (taux_lag4)
    pred_naive = test["taux_lag4"].values

    # ── Baseline 2: Historical mean per (category, quarter)
    hist_mean = (train.groupby(["cat_enc", "quarter_num"])[TARGET]
                      .mean()
                      .reset_index()
                      .rename(columns={TARGET: "hist_mean"}))
    test2 = test.merge(hist_mean, on=["cat_enc", "quarter_num"], how="left")
    pred_hist = test2["hist_mean"].values

    for cat_name, cat_code in cat_map.items():
        mask = test["cat_enc"].values == cat_code
        for model_name, preds in [("RF", pred_rf), ("XGBoost", pred_xgb),
                                   ("LightGBM", pred_lgbm), ("LinReg", pred_lr),
                                   ("Ridge", pred_ridge), ("Lasso", pred_lasso),
                                   ("Naive_lag4", pred_naive), ("Hist_mean", pred_hist)]:
            y_true_cat = y_test.values[mask]
            y_pred_cat = preds[mask]

            # Skip if NaN in naive baseline
            valid = ~np.isnan(y_pred_cat) & ~np.isnan(y_true_cat)
            if valid.sum() == 0:
                continue

            rmse = np.sqrt(mean_squared_error(y_true_cat[valid], y_pred_cat[valid]))
            mae  = mean_absolute_error(y_true_cat[valid], y_pred_cat[valid])
            # MAPE: skip zeros in denominator
            nonzero = y_true_cat[valid] != 0
            mape = (np.abs((y_true_cat[valid][nonzero] - y_pred_cat[valid][nonzero])
                           / y_true_cat[valid][nonzero])).mean() * 100 if nonzero.sum() > 0 else np.nan

            results.append(dict(test_year=test_year, category=cat_name,
                                model=model_name, rmse=rmse, mae=mae, mape=mape))

results_df = pd.DataFrame(results)
results_df.to_csv("results_cv.csv", index=False)

# ── 3. Summary table ──────────────────────────────────────────────────────────
summary = (results_df
           .groupby(["model", "category"])[["rmse", "mae", "mape"]]
           .mean()
           .round(4))
print("\n=== Walk-Forward CV Results (mean across test folds 2022–2025) ===")
print(summary.to_string())

# ── 4. Feature importance (train on full data) ────────────────────────────────
X_all = df_model[FEATURES]
y_all = df_model[TARGET]

rf_full = RandomForestRegressor(**rf_params)
rf_full.fit(X_all, y_all)

xgb_full = XGBRegressor(**xgb_params)
xgb_full.fit(X_all, y_all)

feat_labels = ["Trimestre", "Catégorie", "Taux lag-1", "Taux lag-4 (saisonnier)", "Ratio LF"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, model, name in [(axes[0], rf_full, "Random Forest"),
                         (axes[1], xgb_full, "XGBoost")]:
    imp = model.feature_importances_
    order = np.argsort(imp)
    ax.barh([feat_labels[i] for i in order], imp[order], color="#2c7bb6")
    ax.set_title(f"Importance des variables — {name}", fontsize=11)
    ax.set_xlabel("Importance (Gini / Gain)")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved figures/feature_importance.png")

# ── 5. Actual vs Predicted plot (Investissement only, all CV preds) ───────────
inv_preds = {"RF": [], "XGBoost": [], "Naive_lag4": [], "Hist_mean": []}
inv_actual = []

for test_year in test_years_wf:
    train = df_model[df_model["year"] < test_year]
    test  = df_model[df_model["year"] == test_year]
    mask  = test["cat_enc"].values == cat_map["INVESTISSEMENT"]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test = test[FEATURES]

    rf  = RandomForestRegressor(**rf_params).fit(X_train, y_train)
    xgb = XGBRegressor(**xgb_params).fit(X_train, y_train)

    hist_mean = (train.groupby(["cat_enc", "quarter_num"])[TARGET]
                      .mean().reset_index()
                      .rename(columns={TARGET: "hist_mean"}))
    test_hm = test.merge(hist_mean, on=["cat_enc", "quarter_num"], how="left")

    inv_actual.extend(test[TARGET].values[mask].tolist())
    inv_preds["RF"].extend(rf.predict(X_test)[mask].tolist())
    inv_preds["XGBoost"].extend(xgb.predict(X_test)[mask].tolist())
    inv_preds["Naive_lag4"].extend(test["taux_lag4"].values[mask].tolist())
    inv_preds["Hist_mean"].extend(test_hm["hist_mean"].values[mask].tolist())

inv_df = df_model[df_model["cat_enc"] == cat_map["INVESTISSEMENT"]].copy()
inv_df = inv_df[inv_df["year"].isin(test_years_wf)]
x_labels = [f"{r.year} T{int(r.quarter_num)}" for _, r in inv_df.iterrows()]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(len(inv_actual)), inv_actual, "ko-", label="Réalisé", linewidth=2)
colors  = {"RF": "#2c7bb6", "XGBoost": "#d7191c", "Naive_lag4": "#fdae61", "Hist_mean": "#7b2d8b"}
markers = {"RF": "s",       "XGBoost": "^",        "Naive_lag4": "x",        "Hist_mean": "D"}
for mname, preds in inv_preds.items():
    clean = [p if not np.isnan(p) else np.nan for p in preds]
    ax.plot(range(len(clean)), clean, f"{markers[mname]}--",
            color=colors[mname], label=mname, linewidth=1.5, markersize=6)

ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Taux d'exécution")
ax.set_title("Taux d'exécution — Dépenses d'Investissement\n(validation walk-forward, fenêtre expansive 2022–2025)", fontsize=11)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted_investissement.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/actual_vs_predicted_investissement.png")

# ── 6. Seasonality plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
cat_colors = {"INVESTISSEMENT": "#d7191c", "MATERIEL": "#2c7bb6", "PERSONNEL": "#1a9641"}

for ax, (cat, color) in zip(axes, cat_colors.items()):
    sub = df[df["category"] == cat]
    for yr in years:
        yr_data = sub[sub["year"] == yr].sort_values("quarter_num")
        ax.plot([1,2,3,4], yr_data["taux"].values, "o-", alpha=0.6, label=str(yr))
    ax.set_title(cat.capitalize(), fontsize=10)
    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(["T1","T2","T3","T4"])
    ax.set_ylabel("Taux d'exécution")
    ax.set_ylim(0, 1.3)
    ax.spines[["top", "right"]].set_visible(False)
    if cat == "INVESTISSEMENT":
        ax.legend(fontsize=7)

plt.suptitle("Profil saisonnier du taux d'exécution par catégorie (2020–2025)", fontsize=11)
plt.tight_layout()
plt.savefig("figures/seasonality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/seasonality.png")

# ── 7. Rolling vs Direct Forecast Comparison ─────────────────────────────────
# Direct  : at T2 snapshot, predict T4 in one step (no T3 information)
# Rolling : at T2 snapshot, first predict T3, then use predicted T3 to predict T4
# Both use walk-forward expanding-window CV (same test folds: 2022–2025)

# Build wide format: one row per (year, category), columns T1..T4
meta_wide = df.groupby(["year", "category"]).agg(
    cat_enc=("cat_enc", "first"),
    lf_ratio=("lf_ratio", "first")
).reset_index()

taux_wide = (df.pivot_table(index=["year", "category"], columns="quarter_num", values="taux")
               .reset_index())
taux_wide.columns = ["year", "category", "T1", "T2", "T3", "T4"]

wide = meta_wide.merge(taux_wide, on=["year", "category"])
wide = wide.dropna(subset=["T1", "T2", "T3", "T4"])
wide = wide.sort_values(["category", "year"]).reset_index(drop=True)

# Lagged annual targets (T4 and T3 of previous year as features)
wide["T4_lag"] = wide.groupby("category")["T4"].shift(1)
wide["T3_lag"] = wide.groupby("category")["T3"].shift(1)
wide_model = wide.dropna(subset=["T4_lag", "T3_lag"]).copy()

# Feature sets
# Direct: T2 snapshot only → predict T4
FEAT_DIR   = ["cat_enc", "T1", "T2", "T4_lag", "lf_ratio"]
# Rolling step 1: T2 snapshot → predict T3
FEAT_S1    = ["cat_enc", "T1", "T2", "T3_lag", "lf_ratio"]
# Rolling step 2: T2 + predicted T3 → predict T4
FEAT_S2    = ["cat_enc", "T1", "T2", "T3_pred", "T4_lag", "lf_ratio"]

test_years_wide = [y for y in sorted(wide_model["year"].unique())
                   if (wide_model["year"] < y).any()]

rolling_results = []

for test_year in test_years_wide:
    train_w = wide_model[wide_model["year"] < test_year]
    test_w  = wide_model[wide_model["year"] == test_year].copy()
    if len(train_w) < 3:
        continue

    # ── Direct: T2 → T4
    rf_dir  = RandomForestRegressor(**rf_params).fit(train_w[FEAT_DIR], train_w["T4"])
    xgb_dir = XGBRegressor(**xgb_params).fit(train_w[FEAT_DIR], train_w["T4"])
    pred_dir_rf  = rf_dir.predict(test_w[FEAT_DIR])
    pred_dir_xgb = xgb_dir.predict(test_w[FEAT_DIR])

    # ── Rolling step 1: T2 → T3 (predict intermediate T3)
    rf_s1  = RandomForestRegressor(**rf_params).fit(train_w[FEAT_S1], train_w["T3"])
    xgb_s1 = XGBRegressor(**xgb_params).fit(train_w[FEAT_S1], train_w["T3"])
    test_w["T3_pred_rf"]  = rf_s1.predict(test_w[FEAT_S1])
    test_w["T3_pred_xgb"] = xgb_s1.predict(test_w[FEAT_S1])

    # ── Rolling step 2: use predicted T3 → T4
    # Train step-2 models using ACTUAL T3 in training (oracle), pred T3 at test
    train_s2 = train_w.assign(T3_pred=train_w["T3"])

    # RF rolling
    rf_s2 = RandomForestRegressor(**rf_params).fit(train_s2[FEAT_S2], train_s2["T4"])
    test_w["T3_pred"] = test_w["T3_pred_rf"]
    pred_roll_rf = rf_s2.predict(test_w[FEAT_S2])

    # XGBoost rolling
    xgb_s2 = XGBRegressor(**xgb_params).fit(train_s2[FEAT_S2], train_s2["T4"])
    test_w["T3_pred"] = test_w["T3_pred_xgb"]
    pred_roll_xgb = xgb_s2.predict(test_w[FEAT_S2])

    y_true = test_w["T4"].values

    for cat_name, cat_code in cat_map.items():
        mask = test_w["cat_enc"].values == cat_code
        if mask.sum() == 0:
            continue
        for model_name, preds in [("RF_Direct", pred_dir_rf),
                                   ("XGB_Direct", pred_dir_xgb),
                                   ("RF_Rolling", pred_roll_rf),
                                   ("XGB_Rolling", pred_roll_xgb)]:
            y_t = y_true[mask]; y_p = preds[mask]
            valid = ~np.isnan(y_t) & ~np.isnan(y_p)
            if valid.sum() == 0:
                continue
            rmse = np.sqrt(mean_squared_error(y_t[valid], y_p[valid]))
            mae  = mean_absolute_error(y_t[valid], y_p[valid])
            nonzero = y_t[valid] != 0
            mape = (np.abs((y_t[valid][nonzero] - y_p[valid][nonzero])
                           / y_t[valid][nonzero])).mean() * 100 if nonzero.sum() > 0 else np.nan
            rolling_results.append(dict(test_year=test_year, category=cat_name,
                                        model=model_name, rmse=rmse, mae=mae, mape=mape))

rolling_df = pd.DataFrame(rolling_results)
rolling_df.to_csv("results_rolling.csv", index=False)

rolling_summary = (rolling_df
                   .groupby(["model", "category"])[["rmse", "mae", "mape"]]
                   .mean()
                   .round(4))
print("\n=== Direct vs Rolling Walk-Forward CV Results (mean 2022–2025) ===")
print(rolling_summary.to_string())

# ── 8. Figure: Direct vs Rolling RMSE bar chart ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
cat_list  = ["INVESTISSEMENT", "MATERIEL", "PERSONNEL"]
model_order  = ["RF_Direct", "RF_Rolling", "XGB_Direct", "XGB_Rolling"]
bar_labels   = ["RF\nDirect", "RF\nRoulant", "XGB\nDirect", "XGB\nRoulant"]
bar_colors   = ["#2c7bb6", "#74add1", "#d7191c", "#fdae61"]

for ax, cat in zip(axes, cat_list):
    sub = rolling_df[rolling_df["category"] == cat]
    means = sub.groupby("model")["rmse"].mean().reindex(model_order)
    bars = ax.bar(range(4), means.values, color=bar_colors, width=0.6)
    ax.set_title(cat.capitalize(), fontsize=10)
    ax.set_ylabel("RMSE moyen (4 folds)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, means.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7)

plt.suptitle("Prévision directe (T2→T4) vs roulante (T2→T3→T4)\nRMSE moyen par catégorie, walk-forward 2022–2025",
             fontsize=11)
plt.tight_layout()
plt.savefig("figures/direct_vs_rolling.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/direct_vs_rolling.png")

print("\nDone. Check results_cv.csv, results_rolling.csv and figures/")
