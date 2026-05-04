"""
Système d'alerte précoce — Prévision T4 par ligne budgétaire
Modèle : XGBoost entraîné sur 2021-2024, prévision 2025

Principe : à fin T2 (30 juin), prédire le taux d'exécution au 31 décembre
pour chaque ligne budgétaire d'investissement.

Output : figures/alert_dashboard.png, alert_results_2025.csv
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

os.makedirs("figures", exist_ok=True)

# ── 1. Chargement et préparation ─────────────────────────────────────────────
df = pd.read_csv("data_lines.csv")

type_map = {"acquisitions": 0, "equipements": 1, "etudes": 2,
            "fournitures": 3, "travaux": 4}
prog_map = {300: 0, 301: 1, 302: 2, 303: 3}
reg_map  = {
    "00": 0, "01": 1, "02": 2, "03": 3, "04": 4,
    "05": 5, "06": 6, "07": 7, "08": 8, "09": 9,
    "10": 10, "11": 11, "12": 12,
}

df["type_enc"] = df["type_ligne"].map(type_map)
df["prog_enc"] = df["programme"].map(prog_map)
df["reg_enc"]  = df["region"].map(reg_map)

YEARS = sorted(df["year"].unique())

# ── 2. Construction du dataset d'entraînement ─────────────────────────────────
# Une observation = (ligne, année) avec snapshot à fin T2 → cible = taux T4
rows = []
for year in YEARS:
    prev_year = year - 1
    if prev_year not in YEARS:
        continue
    for lid in df["line_id"].unique():
        cur  = df[(df["line_id"] == lid) & (df["year"] == year)].set_index("quarter")
        prev = df[(df["line_id"] == lid) & (df["year"] == prev_year)].set_index("quarter")
        if len(cur) < 4 or len(prev) < 4:
            continue
        rows.append({
            "line_id":       lid,
            "year":          year,
            # Features observées fin T2
            "taux_T2":       cur.loc[2,  "taux"],
            "taux_T1":       cur.loc[1,  "taux"],
            "taux_T4_lag":   prev.loc[4, "taux"],   # T4 année précédente
            "taux_T2_lag":   prev.loc[2, "taux"],   # T2 année précédente
            "lf_ratio":      cur.loc[2,  "lf_ratio"],
            "lf_share":      cur.loc[2,  "lf_share"],
            "type_enc":      cur.loc[2,  "type_enc"],
            "prog_enc":      cur.loc[2,  "prog_enc"],
            "reg_enc":       cur.loc[2,  "reg_enc"],
            # Métadonnées
            "lf_mdh":        cur.loc[2,  "lf_mdh"],
            "ligne_label":   cur.loc[2,  "ligne_label"],
            "type_ligne":    cur.loc[2,  "type_ligne"],
            "programme":     cur.loc[2,  "programme"],
            "region_label":  cur.loc[2,  "region_label"],
            # Cible
            "target_T4":     cur.loc[4,  "taux"],
        })

pred_df = pd.DataFrame(rows)

FEATURES = ["taux_T2", "taux_T1", "taux_T4_lag", "taux_T2_lag",
            "lf_ratio", "lf_share", "type_enc", "prog_enc", "reg_enc"]

# ── 3. Validation croisée LOYO (2021-2024) ────────────────────────────────────
xgb_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                  subsample=0.8, reg_alpha=0.1, random_state=42, verbosity=0)

train_years = [y for y in YEARS if y >= 2021 and y < 2025]
loyo_rmse = []

for y in train_years:
    tr = pred_df[(pred_df["year"].isin(train_years)) & (pred_df["year"] != y)]
    te = pred_df[pred_df["year"] == y]
    m = XGBRegressor(**xgb_params)
    m.fit(tr[FEATURES], tr["target_T4"])
    rmse = np.sqrt(mean_squared_error(te["target_T4"],
                                       m.predict(te[FEATURES]).clip(0, 1.30)))
    loyo_rmse.append(rmse)
    print(f"  LOYO fold {y}: RMSE = {rmse:.4f}")

print(f"\nLOYO RMSE moyen (ligne-niveau) : {np.mean(loyo_rmse):.4f} ± {np.std(loyo_rmse):.4f}")

# ── 4. Entraînement final sur 2021-2024, prévision 2025 ──────────────────────
train_full = pred_df[pred_df["year"].isin(train_years)]
test_2025  = pred_df[pred_df["year"] == 2025].copy()

# Modèle central (moyenne)
xgb_final = XGBRegressor(**xgb_params)
xgb_final.fit(train_full[FEATURES], train_full["target_T4"])
test_2025["pred_T4"] = xgb_final.predict(test_2025[FEATURES]).clip(0, 1.30)

# VaR budgétaire — régression par quantile
# Q05 : borne pessimiste (pire cas à 95 % de confiance)
# Q95 : borne optimiste
q_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, reg_alpha=0.1, random_state=42, verbosity=0,
                objective="reg:quantileerror")

xgb_q05 = XGBRegressor(**q_params, quantile_alpha=0.05)
xgb_q05.fit(train_full[FEATURES], train_full["target_T4"])
test_2025["var_q05"] = xgb_q05.predict(test_2025[FEATURES]).clip(0, 1.30)

xgb_q95 = XGBRegressor(**q_params, quantile_alpha=0.95)
xgb_q95.fit(train_full[FEATURES], train_full["target_T4"])
test_2025["var_q95"] = xgb_q95.predict(test_2025[FEATURES]).clip(0, 1.30)

# VaR MDH : crédits à risque dans le scénario pessimiste
test_2025["var_mdh"] = (
    test_2025["lf_mdh"] * (1 - test_2025["var_q05"])
).clip(0).round(2)

print(f"\nVaR budgétaire (Q05) — scénario pessimiste à 95 % de confiance :")
print(f"  Crédits à risque VaR total : "
      f"{test_2025['var_mdh'].sum():.1f} MDH")

# ── 5. Classification du risque (basée sur Q05 — scénario pessimiste) ────────
def risk(taux):
    if taux < 0.60: return ("Critique",  "#d73027")
    if taux < 0.80: return ("Attention", "#fc8d59")
    return           ("OK",              "#1a9850")

# Risque basé sur la prévision centrale
test_2025[["risk_label", "risk_color"]] = pd.DataFrame(
    test_2025["pred_T4"].apply(risk).tolist(), index=test_2025.index
)
# Risque VaR — plus conservateur (basé sur Q05)
test_2025[["var_risk_label", "var_risk_color"]] = pd.DataFrame(
    test_2025["var_q05"].apply(risk).tolist(), index=test_2025.index
)
test_2025["credit_risque_mdh"] = (
    test_2025["lf_mdh"] * (1 - test_2025["pred_T4"])
).clip(0).round(3)

test_2025 = test_2025.sort_values("pred_T4").reset_index(drop=True)

# ── 6. Impression du tableau de bord ─────────────────────────────────────────
print("\n" + "=" * 95)
print("TABLEAU DE BORD — PRÉVISION T4 AU 30 JUIN 2025")
print("=" * 95)
header = f"{'Prog':<5} {'Ligne budgétaire':<42} {'Rég.':<10} {'LF(MDH)':>8} "
header += f"{'T2 réel':>8} {'T4 prévu':>9} {'VaR Q05':>9} {'VaR Q95':>9} {'Risque':>10} {'VaR MDH':>9}"
print(header)
print("-" * 110)
for _, r in test_2025.iterrows():
    lbl = r.ligne_label[:41]
    print(f"P{int(r.programme):<4} {lbl:<42} {r.region_label:<25} "
          f"{r.lf_mdh:>8.2f} {r.taux_T2:>8.1%} {r.pred_T4:>9.1%} "
          f"{r.var_q05:>9.1%} {r.var_q95:>9.1%} "
          f"{r.risk_label:>10} {r.var_mdh:>9.2f}")

n_crit = (test_2025["risk_label"] == "Critique").sum()
n_att  = (test_2025["risk_label"] == "Attention").sum()
n_ok   = (test_2025["risk_label"] == "OK").sum()
mdh_risque = test_2025[test_2025["risk_label"] != "OK"]["credit_risque_mdh"].sum()
print("-" * 110)
print(f"  {n_crit} lignes critiques | {n_att} en attention | {n_ok} OK")
print(f"  Crédits à risque (prévision centrale) : {mdh_risque:.1f} MDH")
print(f"  Crédits à risque (VaR Q05 pessimiste)  : {test_2025['var_mdh'].sum():.1f} MDH")

test_2025.to_csv("alert_results_2025.csv", index=False)

# ── 7. Figure dashboard ───────────────────────────────────────────────────────
n = len(test_2025)
fig, ax = plt.subplots(figsize=(13, 0.45 * n + 2.5))

BAR_H = 0.32

for idx, (_, row) in enumerate(test_2025.iterrows()):
    y = idx
    # Barre T4 prévu (couleur risque centrale)
    ax.barh(y + BAR_H / 2, row.pred_T4, height=BAR_H,
            color=row.risk_color, alpha=0.88, zorder=3)
    # Intervalle de confiance [Q05, Q95]
    ax.plot([row.var_q05, row.var_q95], [y + BAR_H / 2, y + BAR_H / 2],
            color="#333333", linewidth=2.0, zorder=5, solid_capstyle="round")
    ax.plot([row.var_q05, row.var_q05], [y + BAR_H/2 - 0.12, y + BAR_H/2 + 0.12],
            color="#333333", linewidth=1.5, zorder=5)
    ax.plot([row.var_q95, row.var_q95], [y + BAR_H/2 - 0.12, y + BAR_H/2 + 0.12],
            color="#333333", linewidth=1.5, zorder=5)
    # Barre T2 observé (bleu)
    ax.barh(y - BAR_H / 2, row.taux_T2, height=BAR_H,
            color="#4575b4", alpha=0.75, zorder=3)
    # Étiquettes texte
    ax.text(min(row.pred_T4 + 0.012, 1.22), y + BAR_H / 2,
            f"{row.pred_T4:.0%}", va="center", fontsize=7.5,
            color=row.risk_color, fontweight="bold")
    ax.text(max(row.taux_T2 + 0.012, 0.015), y - BAR_H / 2,
            f"{row.taux_T2:.0%}", va="center", fontsize=7, color="#2166ac")

# Étiquettes Y : Prog | Ligne (Région)
ylabels = [
    f"P{int(r.programme)} │ {r.ligne_label[:40]} ({r.region_label})"
    for _, r in test_2025.iterrows()
]
ax.set_yticks(range(n))
ax.set_yticklabels(ylabels, fontsize=7.5)
ax.invert_yaxis()

# Lignes de seuil
ax.axvline(0.80, color="black",   linestyle="--", linewidth=1.3, zorder=5)
ax.axvline(0.60, color="#d73027", linestyle=":",  linewidth=1.1, zorder=5)
ax.set_xlim(0, 1.30)

ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xlabel("Taux d'exécution budgétaire")
ax.set_title(
    "Système d'alerte précoce — Prévision T4 par ligne budgétaire\n"
    "Dépenses d'investissement — Ministère de la Justice (30 juin 2025)\n"
    r"$\it{Données\ simulées\ —\ calibrées\ sur\ agrégats\ TGR\ réels\ 2020{-}2025}$",
    fontsize=10.5, pad=10
)

# Légende
patches = [
    mpatches.Patch(color="#4575b4", alpha=0.75,
                   label="Taux T2 réalisé (observé à fin juin)"),
    mpatches.Patch(color="#1a9850", alpha=0.88,
                   label="T4 prévu — OK (≥ 80 %)"),
    mpatches.Patch(color="#fc8d59", alpha=0.88,
                   label="T4 prévu — Attention (60–80 %)"),
    mpatches.Patch(color="#d73027", alpha=0.88,
                   label="T4 prévu — Critique (< 60 %)"),
    plt.Line2D([0],[0], color="#333333", linewidth=2,
                   label="Intervalle de confiance 90 % [Q05–Q95]"),
    plt.Line2D([0],[0], color="black",   linestyle="--", label="Seuil cible : 80 %"),
    plt.Line2D([0],[0], color="#d73027", linestyle=":",  label="Seuil critique : 60 %"),
]
ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)
ax.spines[["top","right"]].set_visible(False)
ax.set_axisbelow(True)
ax.xaxis.grid(True, alpha=0.3, linestyle=":")

plt.tight_layout()
plt.savefig("figures/alert_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSauvegardé figures/alert_dashboard.png")
print("Sauvegardé alert_results_2025.csv")
