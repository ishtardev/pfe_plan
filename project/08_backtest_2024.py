"""
Backtesting 2024 — Valider la précision du modèle
Entraîner sur 2020-2023 uniquement, prédire 2024, comparer à réel
"""

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Charger données ──────────────────────────────────────────────────────────
df = pd.read_csv("data_lines.csv")

# Encodages
type_map = {"acquisitions": 0, "equipements": 1, "etudes": 2,
            "fournitures": 3, "travaux": 4, "personnel": 5}
prog_map = {300: 0, 301: 1, 302: 2, 303: 3}
budget_map = {"INVESTISSEMENT": 0, "MATERIEL": 1, "PERSONNEL": 2}
df["type_enc"] = df["type_ligne"].map(type_map)
df["prog_enc"] = df["programme"].map(prog_map)
df["budget_enc"] = df["budget_type"].map(budget_map)
df["reg_enc"] = df["region"].astype(int)

YEARS = sorted(df["year"].unique())

# ── Construire dataset historique 2020-2025 ──────────────────────────────────
_t3 = (df[df["quarter"] == 3][["line_id", "year", "taux"]].sort_values(["line_id", "year"]))
_t3_lookup = {}
for _lid, _grp in _t3.groupby("line_id", sort=False):
    _yrs = _grp["year"].tolist()
    _vals = _grp["taux"].tolist()
    for _i, _yr in enumerate(_yrs):
        _t3_lookup[(_lid, _yr)] = float(np.mean(_vals[:_i])) if _i > 0 else np.nan

rows = []
for year in YEARS:
    prev_year = year - 1
    if prev_year not in YEARS:
        continue
    for lid in df["line_id"].unique():
        cur = df[(df["line_id"] == lid) & (df["year"] == year)].set_index("quarter")
        prev = df[(df["line_id"] == lid) & (df["year"] == prev_year)].set_index("quarter")
        if len(cur) < 4 or len(prev) < 4:
            continue
        rows.append({
            "line_id": lid,
            "year": year,
            "taux_T2": cur.loc[2, "taux"],
            "taux_T1": cur.loc[1, "taux"],
            "taux_T4_lag": prev.loc[4, "taux"],
            "taux_T2_lag": prev.loc[2, "taux"],
            "hist_avg_T3": _t3_lookup.get((lid, year), np.nan),
            "taux_T3": cur.loc[3, "taux"],
            "taux_T3_lag": prev.loc[3, "taux"],
            "lf_ratio": cur.loc[2, "lf_ratio"],
            "lf_share": cur.loc[2, "lf_share"],
            "type_enc": cur.loc[2, "type_enc"],
            "prog_enc": cur.loc[2, "prog_enc"],
            "reg_enc": cur.loc[2, "reg_enc"],
            "lf_mdh": cur.loc[2, "lf_mdh"],
            "ligne_label": cur.loc[2, "ligne_label"],
            "programme": cur.loc[2, "programme"],
            "region_label": cur.loc[2, "region_label"],
            "budget_type": cur.loc[2, "budget_type"],
            "budget_enc": cur.loc[2, "budget_enc"],
            "target_T4": cur.loc[4, "taux"],
        })

pred_df = pd.DataFrame(rows)
_global_t3 = pred_df["hist_avg_T3"].mean()
pred_df["hist_avg_T3"] = pred_df["hist_avg_T3"].fillna(_global_t3)

FEATURES = ["taux_T2", "taux_T1", "taux_T4_lag", "taux_T2_lag", "hist_avg_T3",
            "lf_ratio", "lf_share", "type_enc", "prog_enc", "reg_enc", "budget_enc"]
FEATURES_T3 = FEATURES + ["taux_T3", "taux_T3_lag"]

xgb_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, reg_alpha=0.1, random_state=42, verbosity=0)
lasso_params = dict(alpha=0.01, max_iter=5000)

# ── BACKTEST 2024: Entraîner 2020-2023, Prédire 2024 ────────────────────────
print("\n" + "="*100)
print("BACKTESTING 2024 — Validation du modèle sur données historiques")
print("="*100)

# Données d'entraînement (uniquement 2020-2023)
train_data = pred_df[pred_df["year"].isin([2020, 2021, 2022, 2023])]

# Données de test (2024 — année qu'on va prédire)
test_2024 = pred_df[pred_df["year"] == 2024].copy()
test_2024["pred_T4"] = np.nan

print(f"\nEntraînement : {len(train_data)} observations (2020-2023)")
print(f"Test 2024 : {len(test_2024)} observations")

# INVESTISSEMENT: XGBoost
tr_i = train_data[train_data["budget_type"] == "INVESTISSEMENT"]
te_i = test_2024[test_2024["budget_type"] == "INVESTISSEMENT"]
if len(tr_i) >= 2 and len(te_i) >= 1:
    xgb_inv = XGBRegressor(**xgb_params)
    xgb_inv.fit(tr_i[FEATURES], tr_i["target_T4"])
    mask_i = test_2024["budget_type"] == "INVESTISSEMENT"
    test_2024.loc[mask_i, "pred_T4"] = xgb_inv.predict(
        test_2024.loc[mask_i, FEATURES]).clip(0, 1.30)
    mae_i = mean_absolute_error(te_i["target_T4"], test_2024.loc[mask_i, "pred_T4"])
    rmse_i = np.sqrt(mean_squared_error(te_i["target_T4"], test_2024.loc[mask_i, "pred_T4"]))
    print(f"\nINVESTISSEMENT (XGBoost):")
    print(f"  MAE: {mae_i:.4f}  |  RMSE: {rmse_i:.4f}  |  {len(te_i)} lignes")

# MATERIEL: Lasso
tr_m = train_data[train_data["budget_type"] == "MATERIEL"]
te_m = test_2024[test_2024["budget_type"] == "MATERIEL"]
if len(tr_m) >= 2 and len(te_m) >= 1:
    lasso_mat = Pipeline([("sc", StandardScaler()), ("reg", Lasso(**lasso_params))])
    lasso_mat.fit(tr_m[FEATURES], tr_m["target_T4"])
    mask_m = test_2024["budget_type"] == "MATERIEL"
    test_2024.loc[mask_m, "pred_T4"] = np.clip(
        lasso_mat.predict(test_2024.loc[mask_m, FEATURES]), 0, 1.30)
    mae_m = mean_absolute_error(te_m["target_T4"], test_2024.loc[mask_m, "pred_T4"])
    rmse_m = np.sqrt(mean_squared_error(te_m["target_T4"], test_2024.loc[mask_m, "pred_T4"]))
    print(f"\nMATERIEL (Lasso):")
    print(f"  MAE: {mae_m:.4f}  |  RMSE: {rmse_m:.4f}  |  {len(te_m)} lignes")

# PERSONNEL: Moyenne historique
tr_p = train_data[train_data["budget_type"] == "PERSONNEL"]
te_p = test_2024[test_2024["budget_type"] == "PERSONNEL"]
if len(tr_p) >= 1 and len(te_p) >= 1:
    hist_p = tr_p.groupby("line_id")["target_T4"].mean()
    overall_p = tr_p["target_T4"].mean()
    mask_p = test_2024["budget_type"] == "PERSONNEL"
    test_2024.loc[mask_p, "pred_T4"] = (
        test_2024.loc[mask_p, "line_id"].map(hist_p).fillna(overall_p).clip(0, 1.30).values
    )
    mae_p = mean_absolute_error(te_p["target_T4"], test_2024.loc[mask_p, "pred_T4"])
    rmse_p = np.sqrt(mean_squared_error(te_p["target_T4"], test_2024.loc[mask_p, "pred_T4"]))
    print(f"\nPERSONNEL (Hist. moyenne):")
    print(f"  MAE: {mae_p:.4f}  |  RMSE: {rmse_p:.4f}  |  {len(te_p)} lignes")

# Global metrics
mae_global = mean_absolute_error(test_2024["target_T4"], test_2024["pred_T4"])
rmse_global = np.sqrt(mean_squared_error(test_2024["target_T4"], test_2024["pred_T4"]))
mape_global = mean_absolute_percentage_error(test_2024["target_T4"], test_2024["pred_T4"]) * 100

print(f"\n" + "-"*100)
print(f"RÉSULTATS GLOBAUX 2024:")
print(f"  MAE: {mae_global:.4f}  (écart moyen en points)")
print(f"  RMSE: {rmse_global:.4f}")
print(f"  MAPE: {mape_global:.1f}%  (erreur en %)")
print(f"  Accuracy: {100 - mape_global:.1f}%")
print(f"-"*100)

# ── Calcul écarts et classification ──────────────────────────────────────────
test_2024["ecart"] = test_2024["pred_T4"] - test_2024["target_T4"]
test_2024["ecart_pp"] = test_2024["ecart"] * 100
test_2024["ecart_abs"] = test_2024["ecart"].abs()

# Classification: bon/acceptable/mauvais
def classify_accuracy(ecart):
    ae = abs(ecart)
    if ae < 0.05:
        return "Excellent (< 5pp)"
    elif ae < 0.10:
        return "Bon (5-10pp)"
    elif ae < 0.15:
        return "Acceptable (10-15pp)"
    else:
        return "Mauvais (> 15pp)"

test_2024["accuracy_class"] = test_2024["ecart"].apply(classify_accuracy)

# Statistiques par classe
print(f"\nDISTRIBUTION DES ERREURS:")
dist = test_2024["accuracy_class"].value_counts().sort_index()
for cls, cnt in dist.items():
    pct = 100 * cnt / len(test_2024)
    print(f"  {cls}: {cnt:3d} lignes ({pct:5.1f}%)")

# ── Exporter résultats ──────────────────────────────────────────────────────
export_df = test_2024[[
    "line_id", "programme", "ligne_label", "region_label", "budget_type",
    "lf_mdh", "taux_T2", "target_T4", "pred_T4", "ecart_pp", "accuracy_class"
]].copy()

export_df.columns = [
    "line_id", "programme", "ligne", "region", "budget_type",
    "lf_mdh", "T2", "T4_reel", "T4_predit", "ecart_pp", "precision"
]

# Conversions %
for col in ["T2", "T4_reel", "T4_predit"]:
    export_df[col] = (export_df[col] * 100).round(1)
export_df["ecart_pp"] = export_df["ecart_pp"].round(1)

# Trier par erreur décroissante
export_df = export_df.sort_values("ecart_pp", ascending=False, key=abs)

# Export Excel
output_path = "../projectv2/backtest_2024.xlsx"
export_df.to_excel(output_path, index=False, sheet_name="Backtest 2024")

print(f"\n✓ Backtest 2024 exporté: {output_path}")
print(f"  {len(export_df)} lignes (lignes avec prédictions 2024)")
print(f"  Mesure: Prédire T4 2024 à partir de données 2020-2023")
print(f"  Verdict: Modèle {100 - mape_global:.0f}% précis sur données historiques")
