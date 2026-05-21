"""Compare 5 forecasting models per budget line on each validation year.

Models compared (all applied to the taux d'engagement):
  1. linear_trend    -- OLS line through all train years
  2. average         -- simple mean of historical taux
  3. weighted_avg    -- recent years weighted more
  4. naive           -- last known year's taux
  5. exp_smoothing   -- exponential smoothing (auto-optimised alpha)

For each validation year a dedicated Excel file is produced with 2 sheets:
  Sheet "Comparaison_Modeles" -- all 5 models' predictions vs actual
  Sheet "Meilleur_Modele"     -- best model per line, actual, error, interval, fiabilite

A final prediction file is also produced for the next year:
  SituationChap_PREVISION_BEST_{next_year}.xlsx

Confidence interval: ±2 × (std of historical taux × predicted Credits_Ouverts)
  -- consistent across all model types, reflects historical volatility of execution.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent
SRC      = DATA_DIR / "03_extracted" / "SituationChap_LIGNES_BUDGETAIRES_COMMUNS.xlsx"
OUT_DIR  = DATA_DIR / "03_extracted"

ID_COLS = ["Chap", "Chap_Intitule", "Prog", "Prog_Intitule",
           "Reg",  "Reg_Intitule",  "Proj", "Proj_Intitule",
           "Lb",   "Intitule"]

MODELS = ["linear_trend", "average", "weighted_avg", "naive", "exp_smoothing"]


# ------------------------------------------------------------------ #
#  Taux prediction methods (input: array of historical taux values)   #
# ------------------------------------------------------------------ #

def _linear(years: np.ndarray, taux: np.ndarray, target: int) -> float:
    mask = ~np.isnan(taux)
    if mask.sum() < 2:
        return np.nan
    a, b = np.polyfit(years[mask].astype(float), taux[mask].astype(float), 1)
    return float(np.clip(a * target + b, 0, 1))


def _average(taux: np.ndarray) -> float:
    vals = taux[~np.isnan(taux)]
    return float(np.clip(vals.mean(), 0, 1)) if len(vals) > 0 else np.nan


def _weighted(years: np.ndarray, taux: np.ndarray) -> float:
    """Exponentially increasing weights: most recent year gets highest weight."""
    mask = ~np.isnan(taux)
    if mask.sum() == 0:
        return np.nan
    y = taux[mask]
    # Weights: 1, 2, 3, ... proportional to position
    w = np.arange(1, len(y) + 1, dtype=float)
    w /= w.sum()
    return float(np.clip(np.dot(w, y), 0, 1))


def _naive(taux: np.ndarray) -> float:
    """Last non-NaN value."""
    vals = taux[~np.isnan(taux)]
    return float(np.clip(vals[-1], 0, 1)) if len(vals) > 0 else np.nan


def _exp_smooth(years: np.ndarray, taux: np.ndarray, target: int) -> float:
    mask = ~np.isnan(taux)
    if mask.sum() < 2:
        return np.nan
    y = taux[mask].astype(float)
    try:
        model = SimpleExpSmoothing(y, initialization_method="estimated")
        fit   = model.fit(optimized=True)
        steps = target - int(years[mask][-1])
        pred  = fit.forecast(steps)[-1]
        return float(np.clip(pred, 0, 1))
    except Exception:
        return np.nan


def predict_taux(model_name: str, years: np.ndarray,
                 taux: np.ndarray, target: int) -> float:
    if model_name == "linear_trend":
        return _linear(years, taux, target)
    if model_name == "average":
        return _average(taux)
    if model_name == "weighted_avg":
        return _weighted(years, taux)
    if model_name == "naive":
        return _naive(taux)
    if model_name == "exp_smoothing":
        return _exp_smooth(years, taux, target)
    return np.nan


# ------------------------------------------------------------------ #
#  Credits Ouverts prediction (always linear trend, most stable)      #
# ------------------------------------------------------------------ #

def predict_co(years: np.ndarray, co_vals: np.ndarray, target: int) -> float:
    mask = ~np.isnan(co_vals)
    if mask.sum() < 2:
        return np.nan
    a, b = np.polyfit(years[mask].astype(float), co_vals[mask].astype(float), 1)
    return float(a * target + b)


def fiabilite_label(mape: float) -> str:
    if np.isnan(mape):
        return "À valider manuellement"
    if mape < 20:
        return "Fiable"
    if mape < 50:
        return "Acceptable"
    if mape < 100:
        return "Incertain"
    return "À valider manuellement"


_CONFIANCE_MATRIX = {
    # (fiabilite, stabilite) -> confiance_globale
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
    ("À valider manuellement","Insuffisant"): "À vérifier",
}


def confiance_globale(fiabilite: str, stabilite: str) -> str:
    return _CONFIANCE_MATRIX.get((fiabilite, stabilite), "À vérifier")


def taux_cv_label(taux_vals: np.ndarray) -> str:
    """Stabilité intrinsèque du taux basée sur le coefficient de variation."""
    valid = taux_vals[~np.isnan(taux_vals)]
    if len(valid) < 2:
        return "Insuffisant"
    mean = np.mean(valid)
    if mean == 0:
        return "Insuffisant"
    cv = np.std(valid, ddof=1) / abs(mean)
    if cv < 0.10:
        return "Stable"
    if cv < 0.25:
        return "Modéré"
    return "Volatile"


def taux_interval(taux_vals: np.ndarray, co_pred: float) -> tuple[float, float]:
    """±2σ of historical taux × predicted CO — consistent across all model types."""
    valid = taux_vals[~np.isnan(taux_vals)]
    if len(valid) < 2 or np.isnan(co_pred):
        return np.nan, np.nan
    sigma = np.std(valid, ddof=1)
    margin = 2.0 * sigma * abs(co_pred)
    return margin, margin  # (lower_margin, upper_margin)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    df = pd.read_excel(SRC)
    df["Line_Key"] = (df["Chap"].astype(str) + "-" + df["Prog"].astype(str) + "-" +
                      df["Reg"].astype(str)  + "-" + df["Proj"].astype(str) + "-" +
                      df["Lb"].astype(str))

    meta = df.drop_duplicates("Line_Key").set_index("Line_Key")[ID_COLS]
    keys = df["Line_Key"].unique()
    all_available_years = sorted(df["Year"].unique())

    # Validate on every year except the first two (need at least 2 training points)
    validation_years = all_available_years[2:]   # e.g. [2023, 2024, 2025]
    next_year        = all_available_years[-1] + 1

    best_model_for_pred: dict[str, str] = {}   # key -> model name (from last validation)
    best_mape_for_pred:  dict[str, float] = {}

    # ---------------------------------------------------------------- #
    #  Loop over each validation year                                   #
    # ---------------------------------------------------------------- #
    for val_year in validation_years:
        train_years = np.array([y for y in all_available_years if y < val_year])
        print(f"\n=== VALIDATION {val_year} (train: {train_years.tolist()}) ===")

        comp_rows = []
        best_rows = []

        for key in keys:
            grp = df[df["Line_Key"] == key].set_index("Year")
            base = {c: meta.loc[key, c] for c in ID_COLS}
            base["Line_Key"] = key

            co_train = np.array([grp.loc[y, "Credits_Ouverts_Vises"]
                                  if y in grp.index else np.nan for y in train_years])
            co_pred = predict_co(train_years, co_train, val_year)

            taux_train = np.array([
                (grp.loc[y, "Total_Engage_Vises"] / grp.loc[y, "Credits_Ouverts_Vises"]
                 if y in grp.index and grp.loc[y, "Credits_Ouverts_Vises"] not in (0, np.nan)
                 else np.nan)
                for y in train_years
            ])

            actual = grp.loc[val_year, "Total_Engage_Vises"] if val_year in grp.index else np.nan
            lo_m, hi_m = taux_interval(taux_train, co_pred)

            # -- Sheet 1: all models --
            comp_row = {**base, f"Total_Engage_Actual_{val_year}": actual}
            best_mape  = np.inf
            best_model = None
            best_pred  = np.nan

            for m in MODELS:
                t = predict_taux(m, train_years, taux_train, val_year)
                pred = co_pred * t if not (np.isnan(co_pred) or np.isnan(t)) else np.nan
                mape = (abs(pred - actual) / abs(actual) * 100
                        if not (np.isnan(pred) or np.isnan(actual) or actual == 0) else np.nan)
                comp_row[f"{m}_Prediction"] = round(pred, 2) if not np.isnan(pred) else np.nan
                comp_row[f"{m}_MAPE_Pct"]   = round(mape, 1) if not np.isnan(mape) else np.nan
                if not np.isnan(mape) and mape < best_mape:
                    best_mape, best_model, best_pred = mape, m, pred

            comp_row["Meilleur_Modele"] = best_model or "n/a"
            comp_row["Meilleur_MAPE"]   = round(best_mape, 1) if best_mape < np.inf else np.nan
            comp_rows.append(comp_row)

            # -- Sheet 2: best model detail --
            best_row = {**base,
                        f"Total_Engage_Actual_{val_year}":    actual,
                        f"Total_Engage_Prediction_{val_year}": round(best_pred, 2) if not np.isnan(best_pred) else np.nan,
                        "Intervalle_Bas":  round(max(0, best_pred - lo_m), 2) if not (np.isnan(best_pred) or np.isnan(lo_m)) else np.nan,
                        "Intervalle_Haut": round(best_pred + hi_m, 2) if not (np.isnan(best_pred) or np.isnan(hi_m)) else np.nan,
                        "Erreur_Abs":      round(best_pred - actual, 2) if not (np.isnan(best_pred) or np.isnan(actual)) else np.nan,
                        "Erreur_Pct":      round(best_mape, 1) if best_mape < np.inf else np.nan,
                        "Modele_Utilise":  best_model or "n/a",
                        "Fiabilite":       fiabilite_label(best_mape if best_mape < np.inf else np.nan)}
            best_rows.append(best_row)

            # Keep last validation year's best model for the prediction
            best_model_for_pred[key] = best_model or "average"
            best_mape_for_pred[key]  = best_mape if best_mape < np.inf else np.nan

        comp_df = pd.DataFrame(comp_rows)
        best_df = pd.DataFrame(best_rows)

        # Print summary
        for m in MODELS:
            col = f"{m}_MAPE_Pct"
            print(f"  {m:<20}: {comp_df[col].dropna().mean():.1f}%")
        print(f"  {'Best (auto)':<20}: {comp_df['Meilleur_MAPE'].dropna().mean():.1f}%")

        # Write per-year Excel with 2 sheets
        out_path = OUT_DIR / f"SituationChap_VALIDATION_{val_year}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            comp_df.to_excel(writer, sheet_name="Comparaison_Modeles", index=False)
            best_df.to_excel(writer, sheet_name="Meilleur_Modele",     index=False)
        print(f"  -> {out_path}")

    # ---------------------------------------------------------------- #
    #  Prediction for next year using best model per line              #
    # ---------------------------------------------------------------- #
    all_years = np.array(all_available_years)
    pred_rows = []

    for key in keys:
        grp = df[df["Line_Key"] == key].set_index("Year")
        row = {c: meta.loc[key, c] for c in ID_COLS}
        row["Line_Key"] = key

        co_all = np.array([grp.loc[y, "Credits_Ouverts_Vises"]
                           if y in grp.index else np.nan for y in all_years])
        co_pred = predict_co(all_years, co_all, next_year)

        taux_all = np.array([
            (grp.loc[y, "Total_Engage_Vises"] / grp.loc[y, "Credits_Ouverts_Vises"]
             if y in grp.index and grp.loc[y, "Credits_Ouverts_Vises"] not in (0, np.nan)
             else np.nan)
            for y in all_years
        ])

        best_model = best_model_for_pred.get(key, "average")
        t_pred     = predict_taux(best_model, all_years, taux_all, next_year)
        engage_pred = co_pred * t_pred if not (np.isnan(co_pred) or np.isnan(t_pred)) else np.nan

        lo_m, hi_m = taux_interval(taux_all, co_pred)
        mape_val = best_mape_for_pred.get(key, np.nan)

        row["Modele_Utilise"]              = best_model
        row["Validation_MAPE_Pct"]        = round(mape_val, 1) if not np.isnan(mape_val) else np.nan
        row["Credits_Ouverts_Pred"]        = round(co_pred, 2) if not np.isnan(co_pred) else np.nan
        row["Taux_Engagement_Pred"]        = round(t_pred, 4)  if not np.isnan(t_pred)  else np.nan
        row[f"Total_Engage_Pred_{next_year}"] = round(engage_pred, 2) if not np.isnan(engage_pred) else np.nan
        row["Intervalle_Bas"]              = round(max(0, engage_pred - lo_m), 2) if not (np.isnan(engage_pred) or np.isnan(lo_m)) else np.nan
        row["Intervalle_Haut"]             = round(engage_pred + hi_m, 2) if not (np.isnan(engage_pred) or np.isnan(hi_m)) else np.nan
        fiabilite  = fiabilite_label(mape_val)
        stabilite  = taux_cv_label(taux_all)
        row["Fiabilite"]                   = fiabilite
        row["Stabilite_Taux"]              = stabilite
        row["Confiance_Globale"]           = confiance_globale(fiabilite, stabilite)
        pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_path = OUT_DIR / f"SituationChap_PREVISION_BEST_{next_year}.xlsx"
    pred_df.to_excel(pred_path, index=False)
    print(f"\nPrévision {next_year} -> {pred_path}")


if __name__ == "__main__":
    main()
