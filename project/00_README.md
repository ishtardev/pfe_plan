# Pipeline du projet — Prévision et alerte budgétaire

## Vue d'ensemble

Le projet suit une chaîne en **6 étapes numérotées**. Pour reproduire les
résultats du mémoire, exécuter les scripts dans l'ordre :

```
raw_data.xlsx
      │
      ▼  01_reshape_raw.py
data_long.csv  ──────────────────────────────► 03_train_models.py ──► results_cv.csv
   (72 obs, agrégés TGR)                    │  04_fold_stability.py    results_rolling.csv
                                            │                          figures/
      02_simulate_lines.py                  │
      (TGR réels + morasse LOF)             │
      ▼                                     │
data_lines.csv ────────────────────────────►   05_alert_system.py ──► alert_results_2025.csv
   (3 120 obs, niveau ligne)                   06_dashboard.py        audit_log.txt
                                                                       predictions_*
```

## Détail des scripts

| # | Script | Entrée | Sortie | Rôle |
|---|---|---|---|---|
| 01 | `01_reshape_raw.py` | `raw_data.xlsx` | `data_long.csv` | Mise en forme des taux TGR (wide → long). |
| 02 | `02_simulate_lines.py` | (TGR + morasse en dur) | `data_lines.csv` | Désagrégation contrainte au niveau ligne budgétaire. |
| 03 | `03_train_models.py` | `data_long.csv` | `results_cv.csv`, `results_rolling.csv`, `figures/` | Évaluation walk-forward de 8 modèles (RF, XGBoost, LightGBM, LinReg, Ridge, Lasso, Naïve, Hist_mean). |
| 04 | `04_fold_stability.py` | `data_long.csv` | console | Analyse de stabilité des folds. |
| 05 | `05_alert_system.py` | `data_lines.csv` | `alert_results_2025.csv` | Détection d'anomalies (z-score) sur prévisions T4 par ligne. |
| 06 | `06_dashboard.py` | `data_lines.csv` | dashboard Tkinter + `audit_log.txt`, `predictions_cache.json` | Interface utilisateur (KPI, vue détaillée, export). |

## Distinction importante

- **Évaluation des modèles (étape 03)** : conduite sur `data_long.csv` → 72 observations **réelles** (taux TGR agrégés 2020–2025). C'est ce qui produit les RMSE/MAE/MAPE rapportés au chapitre 4.
- **Système opérationnel (étapes 05–06)** : utilise `data_lines.csv` → 3 120 observations au niveau ligne, obtenues par désagrégation contrainte (cf. chapitre 4, §4.1.2). Permet la détection d'anomalies fine et le tableau de bord.

## Fichiers connexes

- `variables.md` — dictionnaire des variables.
- `previsions_2025_20260429.xlsx` — export horodaté des prévisions.
- `Système_de_Pilotage_Budgétaire.pbix` — version Power BI (annexe).
- `_archive/` — anciens scripts / utilitaires.
