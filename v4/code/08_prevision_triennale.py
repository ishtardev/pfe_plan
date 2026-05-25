"""Prévision triennale 2026–2028 par ligne budgétaire.

Approche :
- Meilleur modèle par ligne sélectionné sur la validation 2025
- Credits_Ouverts extrapolé par tendance linéaire
- Taux d'engagement prédit par le meilleur modèle
- Intervalles de confiance s'élargissant avec l'horizon :
    margin_h = 2 × σ(taux) × CO_pred × √h   (h = 1, 2, 3)

Output : 03_extracted/SituationChap_PREVISION_TRIENNALE.xlsx
  Sheet "Prévision_Triennale" -- une ligne par budget_line, 3 ans côte à côte
  Sheet "Format_Long"         -- une ligne par (budget_line × année), pour Power BI
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent
SRC      = DATA_DIR / "03_extracted" / "SituationChap_LIGNES_BUDGETAIRES_COMMUNS.xlsx"
VAL_2025 = DATA_DIR / "03_extracted" / "SituationChap_VALIDATION_2025.xlsx"
OUT_DIR  = DATA_DIR / "03_extracted"

ID_COLS = ["Chap", "Chap_Intitule", "Prog", "Prog_Intitule",
           "Reg",  "Reg_Intitule",  "Proj", "Proj_Intitule",
           "Lb",   "Intitule"]

MODELS        = ["linear_trend", "average", "weighted_avg", "naive", "exp_smoothing"]
FORECAST_YEARS = [2026, 2027, 2028]


# ------------------------------------------------------------------ #
#  Taux prediction methods (identiques à 05_model_comparison)         #
# ------------------------------------------------------------------ #

def _linear(years, taux, target):
    mask = ~np.isnan(taux)
    if mask.sum() < 2: return np.nan
    a, b = np.polyfit(years[mask].astype(float), taux[mask].astype(float), 1)
    return float(np.clip(a * target + b, 0, 1))

def _average(taux):
    vals = taux[~np.isnan(taux)]
    return float(np.clip(vals.mean(), 0, 1)) if len(vals) > 0 else np.nan

def _weighted(years, taux):
    mask = ~np.isnan(taux)
    if mask.sum() == 0: return np.nan
    y = taux[mask]
    w = np.arange(1, len(y) + 1, dtype=float); w /= w.sum()
    return float(np.clip(np.dot(w, y), 0, 1))

def _naive(taux):
    vals = taux[~np.isnan(taux)]
    return float(np.clip(vals[-1], 0, 1)) if len(vals) > 0 else np.nan

def _exp_smooth(years, taux, target):
    mask = ~np.isnan(taux)
    if mask.sum() < 2: return np.nan
    y = taux[mask].astype(float)
    try:
        fit   = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        steps = target - int(years[mask][-1])
        return float(np.clip(fit.forecast(steps)[-1], 0, 1))
    except Exception:
        return np.nan

def predict_taux(model_name, years, taux, target):
    if model_name == "linear_trend":  return _linear(years, taux, target)
    if model_name == "average":       return _average(taux)
    if model_name == "weighted_avg":  return _weighted(years, taux)
    if model_name == "naive":         return _naive(taux)
    if model_name == "exp_smoothing": return _exp_smooth(years, taux, target)
    return np.nan

def predict_co(years, co_vals, target):
    mask = ~np.isnan(co_vals)
    if mask.sum() < 2: return np.nan
    a, b = np.polyfit(years[mask].astype(float), co_vals[mask].astype(float), 1)
    return float(max(0.0, a * target + b))

def fiabilite_label(mape):
    if np.isnan(mape):   return "À valider manuellement"
    if mape < 20:        return "Fiable"
    if mape < 50:        return "Acceptable"
    if mape < 100:       return "Incertain"
    return "À valider manuellement"

def taux_cv_label(taux_vals):
    valid = taux_vals[~np.isnan(taux_vals)]
    if len(valid) < 2: return "Insuffisant"
    mean = np.mean(valid)
    if mean == 0: return "Insuffisant"
    cv = np.std(valid, ddof=1) / abs(mean)
    if cv < 0.10: return "Stable"
    if cv < 0.25: return "Modéré"
    return "Volatile"

_CONFIANCE_MATRIX = {
    ("Fiable",               "Stable"):      "Très fiable",
    ("Fiable",               "Modéré"):      "Fiable",
    ("Fiable",               "Volatile"):    "Acceptable",
    ("Acceptable",           "Stable"):      "Fiable",
    ("Acceptable",           "Modéré"):      "Acceptable",
    ("Acceptable",           "Volatile"):    "Incertain",
    ("Incertain",            "Stable"):      "Acceptable",
    ("Incertain",            "Modéré"):      "Incertain",
    ("Incertain",            "Volatile"):    "À vérifier",
    ("À valider manuellement","Stable"):     "Incertain",
    ("À valider manuellement","Modéré"):     "À vérifier",
    ("À valider manuellement","Volatile"):   "À vérifier",
    ("À valider manuellement","Insuffisant"):"À vérifier",
}

def confiance_globale(fiabilite, stabilite):
    return _CONFIANCE_MATRIX.get((fiabilite, stabilite), "À vérifier")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    df = pd.read_excel(SRC)
    df["Line_Key"] = (df["Chap"].astype(str) + "-" + df["Prog"].astype(str) + "-" +
                      df["Reg"].astype(str)  + "-" + df["Proj"].astype(str) + "-" +
                      df["Lb"].astype(str))

    meta       = df.drop_duplicates("Line_Key").set_index("Line_Key")[ID_COLS]
    keys       = df["Line_Key"].unique()
    all_years  = np.array(sorted([y for y in df["Year"].unique() if y != 2026]))

    # -- Récupérer le meilleur modèle par ligne depuis la validation 2025 --
    val_df     = pd.read_excel(VAL_2025, sheet_name="Meilleur_Modele")
    val_df["Line_Key"] = (val_df["Chap"].astype(str) + "-" + val_df["Prog"].astype(str) + "-" +
                          val_df["Reg"].astype(str)  + "-" + val_df["Proj"].astype(str) + "-" +
                          val_df["Lb"].astype(str))
    best_model_map = val_df.set_index("Line_Key")["Modele_Utilise"].to_dict()
    best_mape_map  = val_df.set_index("Line_Key")["Erreur_Pct"].to_dict()

    wide_rows = []   # format large (une ligne par budget_line)
    long_rows = []   # format long (une ligne par budget_line × année)

    for key in keys:
        grp  = df[df["Line_Key"] == key].set_index("Year")
        base = {c: meta.loc[key, c] for c in ID_COLS}
        base["Line_Key"] = key

        co_all   = np.array([grp.loc[y, "Credits_Ouverts_Vises"]
                              if y in grp.index else np.nan for y in all_years])
        taux_all = np.array([
            (grp.loc[y, "Total_Engage_Vises"] / grp.loc[y, "Credits_Ouverts_Vises"]
             if y in grp.index and grp.loc[y, "Credits_Ouverts_Vises"] not in (0, np.nan)
             else np.nan)
            for y in all_years
        ])

        best_model = best_model_map.get(key, "average")
        mape_val   = best_mape_map.get(key, np.nan)
        if isinstance(mape_val, str):
            try: mape_val = float(mape_val)
            except: mape_val = np.nan

        fiabilite  = fiabilite_label(mape_val)
        stabilite  = taux_cv_label(taux_all)
        confiance  = confiance_globale(fiabilite, stabilite)

        # Sigma du taux pour les intervalles
        valid_taux = taux_all[~np.isnan(taux_all)]
        sigma_taux = np.std(valid_taux, ddof=1) if len(valid_taux) >= 2 else np.nan

        wide_row = {**base,
                    "Modele_Utilise":   best_model,
                    "Validation_MAPE":  round(mape_val, 1) if not np.isnan(mape_val) else np.nan,
                    "Fiabilite":        fiabilite,
                    "Stabilite_Taux":   stabilite,
                    "Confiance_Globale":confiance}

        for h, year in enumerate(FORECAST_YEARS, start=1):
            co_pred     = predict_co(all_years, co_all, year)
            t_pred      = max(0.0, predict_taux(best_model, all_years, taux_all, year)) if not np.isnan(predict_taux(best_model, all_years, taux_all, year)) else np.nan
            engage_pred = co_pred * t_pred if not (np.isnan(co_pred) or np.isnan(t_pred)) else np.nan

            # Interval s'élargit avec √h
            if not (np.isnan(sigma_taux) or np.isnan(co_pred)):
                margin = 2.0 * sigma_taux * abs(co_pred) * np.sqrt(h)
            else:
                margin = np.nan

            wide_row[f"CO_Pred_{year}"]     = round(co_pred, 2)     if not np.isnan(co_pred)     else np.nan
            wide_row[f"Engage_Pred_{year}"] = round(engage_pred, 2) if not np.isnan(engage_pred) else np.nan
            wide_row[f"Int_Bas_{year}"]     = round(max(0, engage_pred - margin), 2) if not (np.isnan(engage_pred) or np.isnan(margin)) else np.nan
            wide_row[f"Int_Haut_{year}"]    = round(engage_pred + margin, 2) if not (np.isnan(engage_pred) or np.isnan(margin)) else np.nan

            long_rows.append({
                **base,
                "Modele_Utilise":    best_model,
                "Validation_MAPE":   round(mape_val, 1) if not np.isnan(mape_val) else np.nan,
                "Fiabilite":         fiabilite,
                "Stabilite_Taux":    stabilite,
                "Confiance_Globale": confiance,
                "Annee":             year,
                "Horizon_Annees":    h,
                "Credits_Ouverts_Pred": round(co_pred, 2)     if not np.isnan(co_pred)     else np.nan,
                "Taux_Engagement_Pred": round(t_pred, 4)      if not np.isnan(t_pred)      else np.nan,
                "Total_Engage_Pred":    round(engage_pred, 2) if not np.isnan(engage_pred) else np.nan,
                "Intervalle_Bas":       round(max(0, engage_pred - margin), 2) if not (np.isnan(engage_pred) or np.isnan(margin)) else np.nan,
                "Intervalle_Haut":      round(engage_pred + margin, 2) if not (np.isnan(engage_pred) or np.isnan(margin)) else np.nan,
            })

        wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)

    # Résumé
    print("=== PRÉVISION TRIENNALE 2026–2028 ===")
    for year in FORECAST_YEARS:
        total = wide_df[f"Engage_Pred_{year}"].sum() / 1e6
        print(f"  {year} : Total prévu = {total:,.1f} M MAD")
    print(f"\n  Lignes 'Très fiable'  : {(wide_df['Confiance_Globale'] == 'Très fiable').sum()}")
    print(f"  Lignes 'À vérifier'   : {(wide_df['Confiance_Globale'] == 'À vérifier').sum()}")

    out_path = OUT_DIR / "SituationChap_PREVISION_TRIENNALE.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        wide_df.to_excel(writer, sheet_name="Prévision_Triennale", index=False)
        long_df.to_excel(writer, sheet_name="Format_Long",         index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
