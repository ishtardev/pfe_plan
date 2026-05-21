"""Predict Total_Engage_Vises for 2026 using only historical data (2021-2025).

Strategy (uses the budget identity you described):
  Taux d'engagement = Total_Engage / Credits_Ouverts
  This ratio is structurally more stable than raw amounts.

  Step 1 - Predict Credits_Ouverts_2026  via linear trend on 2021-2025
  Step 2 - Predict Taux_2026             via linear trend on 2021-2025
  Step 3 - Total_Engage_2026 = Credits_Ouverts_2026 x Taux_2026

Validation: same logic trained on 2021-2024, tested on actual 2025.

Outputs in 03_extracted/:
  SituationChap_VALIDATION_TAUX_2025.xlsx  -- error vs actual 2025
  SituationChap_PREVISION_ENGAGE_2026.xlsx -- 2026 predictions
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
SRC      = DATA_DIR / "03_extracted" / "SituationChap_LIGNES_BUDGETAIRES_COMMUNS.xlsx"
OUT_DIR  = DATA_DIR / "03_extracted"

ID_COLS = ["Chap", "Chap_Intitule", "Prog", "Prog_Intitule",
           "Reg",  "Reg_Intitule",  "Proj", "Proj_Intitule",
           "Lb",   "Intitule"]


def linear_predict(years: np.ndarray, values: np.ndarray, target_year: int):
    """OLS linear trend -> (prediction, lower_95, upper_95). Returns NaN on failure."""
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    x = years[mask].astype(float)
    y = values[mask].astype(float)
    a, b = np.polyfit(x, y, 1)
    pred = a * target_year + b
    sigma = np.std(y - (a * x + b), ddof=min(2, mask.sum() - 1))
    margin = 2.0 * sigma
    return pred, pred - margin, pred + margin


def reliability_label(cv: float) -> str:
    if np.isnan(cv):
        return "Insuffisant"
    if cv < 20:
        return "Fiable"
    if cv < 40:
        return "Acceptable"
    return "Incertain"


def build_rows(df: pd.DataFrame, train_years: np.ndarray, target_year: int) -> list[dict]:
    rows = []
    meta = df.drop_duplicates("Line_Key").set_index("Line_Key")[ID_COLS]

    for key in df["Line_Key"].unique():
        grp = df[df["Line_Key"] == key].set_index("Year")
        row = {c: meta.loc[key, c] for c in ID_COLS}
        row["Line_Key"] = key

        def _get(col, y):
            return grp.loc[y, col] if y in grp.index else np.nan

        # --- Predict each component via linear trend ---
        co_vals   = np.array([_get("Credits_Ouverts_Vises", y)    for y in train_years])
        vp_vals   = np.array([_get("Virements_En_Plus_Vises", y)  for y in train_years])
        vm_vals   = np.array([_get("Virements_En_Moins_Vises", y) for y in train_years])

        co_pred,  co_lo,  co_hi  = linear_predict(train_years, co_vals,  target_year)
        vp_pred,  _,      _      = linear_predict(train_years, vp_vals,  target_year)
        vm_pred,  _,      _      = linear_predict(train_years, vm_vals,  target_year)

        # Crédits disponibles = CO + Virt+ - Virt-  (for reference only)
        dispo_pred = (co_pred + vp_pred - vm_pred
                      if not any(np.isnan(v) for v in (co_pred, vp_pred, vm_pred))
                      else np.nan)
        dispo_lo = (co_lo + vp_pred - vm_pred
                    if not any(np.isnan(v) for v in (co_lo, vp_pred, vm_pred))
                    else np.nan)
        dispo_hi = (co_hi + vp_pred - vm_pred
                    if not any(np.isnan(v) for v in (co_hi, vp_pred, vm_pred))
                    else np.nan)

        # --- Taux d'engagement = Total_Engage / Credits_Ouverts ---
        # We use Credits_Ouverts (not disponibles) as the base because it is stable
        # and law-defined. Virements are noise that is already reflected in taux variation.
        taux_vals = np.array([
            (grp.loc[y, "Total_Engage_Vises"] / grp.loc[y, "Credits_Ouverts_Vises"]
             if y in grp.index and grp.loc[y, "Credits_Ouverts_Vises"] not in (0, np.nan)
             else np.nan)
            for y in train_years
        ])

        # Coefficient of variation -> reliability indicator
        taux_cv = (np.nanstd(taux_vals, ddof=1) / abs(np.nanmean(taux_vals)) * 100
                   if np.nansum(~np.isnan(taux_vals)) > 1 else np.nan)

        taux_pred, _, _ = linear_predict(train_years, taux_vals, target_year)
        taux_pred = float(np.clip(taux_pred, 0, 1)) if not np.isnan(taux_pred) else np.nan

        # --- Predicted Total_Engage = Credits_Ouverts_pred × Taux ---
        engage_pred = co_pred * taux_pred if not (np.isnan(co_pred) or np.isnan(taux_pred)) else np.nan
        engage_lo   = co_lo   * taux_pred if not (np.isnan(co_lo)   or np.isnan(taux_pred)) else np.nan
        engage_hi   = co_hi   * taux_pred if not (np.isnan(co_hi)   or np.isnan(taux_pred)) else np.nan

        row["Credits_Ouverts_Pred"]          = round(co_pred,    2) if not np.isnan(co_pred)    else np.nan
        row["Virements_Plus_Pred"]           = round(vp_pred,    2) if not np.isnan(vp_pred)    else np.nan
        row["Virements_Moins_Pred"]          = round(vm_pred,    2) if not np.isnan(vm_pred)    else np.nan
        row["Credits_Disponibles_Pred"]      = round(dispo_pred, 2) if not np.isnan(dispo_pred) else np.nan
        row["Taux_Engagement_Pred"]          = round(taux_pred,  4) if not np.isnan(taux_pred)  else np.nan
        row["Taux_CV_Pct"]                   = round(taux_cv,    1) if not np.isnan(taux_cv)    else np.nan
        row["Fiabilite"]                     = reliability_label(taux_cv)
        row["Total_Engage_Pred"]             = round(engage_pred, 2) if not np.isnan(engage_pred) else np.nan
        row["Total_Engage_Intervalle_Bas"]   = round(engage_lo,   2) if not np.isnan(engage_lo)   else np.nan
        row["Total_Engage_Intervalle_Haut"]  = round(engage_hi,   2) if not np.isnan(engage_hi)   else np.nan

        rows.append(row)
    return rows


def main():
    df = pd.read_excel(SRC)
    df["Line_Key"] = (df["Chap"].astype(str) + "-" + df["Prog"].astype(str) + "-" +
                      df["Reg"].astype(str)  + "-" + df["Proj"].astype(str) + "-" +
                      df["Lb"].astype(str))

    # ------------------------------------------------------------------ #
    #  VALIDATION: train 2021-2024, predict 2025, compare with actual     #
    # ------------------------------------------------------------------ #
    train_years = np.array([2021, 2022, 2023, 2024])
    val_rows = build_rows(df[df["Year"].isin(train_years)], train_years, 2025)
    val_df = pd.DataFrame(val_rows)

    # Merge actual 2025
    actual_2025 = (df[df["Year"] == 2025]
                   .set_index("Line_Key")[["Credits_Ouverts_Vises", "Total_Engage_Vises"]]
                   .rename(columns={"Credits_Ouverts_Vises": "Credits_Ouverts_Actual_2025",
                                    "Total_Engage_Vises":    "Total_Engage_Actual_2025"}))
    val_df = val_df.join(actual_2025, on="Line_Key")

    val_df["Engage_Erreur_Abs"] = (val_df["Total_Engage_Pred"] -
                                   val_df["Total_Engage_Actual_2025"]).round(2)
    val_df["Engage_Erreur_Pct"] = (
        val_df["Engage_Erreur_Abs"].abs() /
        val_df["Total_Engage_Actual_2025"].abs().replace(0, np.nan) * 100
    ).round(1)

    mape_all    = val_df["Engage_Erreur_Pct"].dropna().mean()
    mape_stable = val_df.loc[val_df["Fiabilite"] == "Fiable", "Engage_Erreur_Pct"].dropna().mean()

    print("\n=== VALIDATION 2025 ===")
    print(f"  MAPE toutes lignes   : {mape_all:.1f}%")
    print(f"  MAPE lignes Fiable   : {mape_stable:.1f}%")
    print(f"  Lignes Fiable        : {(val_df['Fiabilite']=='Fiable').sum()}")
    print(f"  Lignes Acceptable    : {(val_df['Fiabilite']=='Acceptable').sum()}")
    print(f"  Lignes Incertain     : {(val_df['Fiabilite']=='Incertain').sum()}")

    col_order = (ID_COLS + ["Line_Key",
                 "Credits_Ouverts_Pred", "Virements_Plus_Pred", "Virements_Moins_Pred",
                 "Credits_Disponibles_Pred", "Credits_Ouverts_Actual_2025",
                 "Taux_Engagement_Pred", "Taux_CV_Pct", "Fiabilite",
                 "Total_Engage_Pred", "Total_Engage_Actual_2025",
                 "Engage_Erreur_Abs", "Engage_Erreur_Pct",
                 "Total_Engage_Intervalle_Bas", "Total_Engage_Intervalle_Haut"])
    val_df = val_df[[c for c in col_order if c in val_df.columns]]
    val_path = OUT_DIR / "SituationChap_VALIDATION_TAUX_2025.xlsx"
    val_df.to_excel(val_path, index=False)
    print(f"\nDétail validation -> {val_path}")

    # ------------------------------------------------------------------ #
    #  PREDICTION 2026: train on full 2021-2025                           #
    # ------------------------------------------------------------------ #
    all_years = np.array([2021, 2022, 2023, 2024, 2025])
    pred_rows = build_rows(df, all_years, 2026)
    pred_df = pd.DataFrame(pred_rows)

    col_order_pred = (ID_COLS + ["Line_Key",
                      "Credits_Ouverts_Pred", "Virements_Plus_Pred", "Virements_Moins_Pred",
                      "Credits_Disponibles_Pred",
                      "Taux_Engagement_Pred", "Taux_CV_Pct", "Fiabilite",
                      "Total_Engage_Pred",
                      "Total_Engage_Intervalle_Bas", "Total_Engage_Intervalle_Haut"])
    pred_df = pred_df[[c for c in col_order_pred if c in pred_df.columns]]
    pred_path = OUT_DIR / "SituationChap_PREVISION_ENGAGE_2026.xlsx"
    pred_df.to_excel(pred_path, index=False)
    print(f"Prévision 2026      -> {pred_path}")


if __name__ == "__main__":
    main()
