"""Rapport complet consolidé — un seul fichier Excel avec tout dedans.

Sheets :
  1. Récapitulatif -- une ligne par LB : historique CO/Engage/Taux + prévision
                      avec intervalles, fiabilité, stabilité, confiance
  2. Validations   -- toutes les années de validation empilées (une ligne par
                      LB × année), avec prédit / réel / erreur / modèle utilisé

Output : 03_extracted/SituationChap_RAPPORT_COMPLET.xlsx
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = DATA_DIR / "03_extracted"

SRC_HIST    = OUT_DIR / "SituationChap_LIGNES_BUDGETAIRES_COMMUNS.xlsx"

# Auto-détection des fichiers disponibles
def find_file(pattern: str):
    matches = sorted(OUT_DIR.glob(pattern))
    return matches[-1] if matches else None

SRC_PREV  = find_file("SituationChap_PREVISION_BEST_*.xlsx")
SRC_TRIEN = find_file("SituationChap_PREVISION_TRIENNALE.xlsx")

ID_COLS = ["Chap", "Chap_Intitule", "Prog", "Prog_Intitule",
           "Reg",  "Reg_Intitule",  "Proj", "Proj_Intitule",
           "Lb",   "Intitule"]


def make_line_key(df):
    return (df["Chap"].astype(str) + "-" + df["Prog"].astype(str) + "-" +
            df["Reg"].astype(str)  + "-" + df["Proj"].astype(str) + "-" +
            df["Lb"].astype(str))


def build_recap(hist: pd.DataFrame,
                val_dfs: dict,
                prev_df: pd.DataFrame,
                trien_long: pd.DataFrame = None) -> pd.DataFrame:
    """Récapitulatif : une ligne par (LB × Année_Prévision).
    - Si triennal dispo : 3 lignes par LB (2026/2027/2028)
    - Sinon : 1 ligne par LB avec la prévision N+1 unique."""

    meta = hist.drop_duplicates("Line_Key").set_index("Line_Key")[ID_COLS]
    keys = meta.index.tolist()
    rows = []

    if trien_long is not None and not trien_long.empty:
        tl = trien_long.copy()
        tl["Line_Key"] = make_line_key(tl)
        for key in keys:
            sub = tl[tl["Line_Key"] == key]
            for _, r in sub.iterrows():
                row = {c: meta.loc[key, c] for c in ID_COLS}
                row["Line_Key"]          = key
                row["Année_Prévision"]   = int(r["Annee"])
                row["Horizon_Annees"]    = int(r["Horizon_Annees"])
                row["Engage_Prévu"]      = r.get("Total_Engage_Pred", np.nan)
                row["Int_Bas"]           = r.get("Intervalle_Bas", np.nan)
                row["Int_Haut"]          = r.get("Intervalle_Haut", np.nan)
                row["Modele_Utilise"]    = r.get("Modele_Utilise", "n/a")
                row["Validation_MAPE"]   = r.get("Validation_MAPE", np.nan)
                row["Fiabilite"]         = r.get("Fiabilite", "n/a")
                row["Stabilite_Taux"]    = r.get("Stabilite_Taux", "n/a")
                row["Confiance_Globale"] = r.get("Confiance_Globale", "n/a")
                rows.append(row)
        return pd.DataFrame(rows)

    # Fallback : une seule année (prev_df)
    for key in keys:
        row = {c: meta.loc[key, c] for c in ID_COLS}
        row["Line_Key"] = key
        prow = prev_df[prev_df["Line_Key"] == key]
        if len(prow) > 0:
            prow = prow.iloc[0]
            eng_col = [c for c in prow.index if c.startswith("Total_Engage_Pred_")]
            next_y  = eng_col[0].split("_")[-1] if eng_col else "?"
            row["Année_Prévision"]   = int(next_y) if next_y != "?" else np.nan
            row["Engage_Prévu"]      = prow.get(eng_col[0], np.nan) if eng_col else np.nan
            row["Int_Bas"]           = prow.get("Intervalle_Bas", np.nan)
            row["Int_Haut"]          = prow.get("Intervalle_Haut", np.nan)
            row["Modele_Utilise"]    = prow.get("Modele_Utilise", "n/a")
            row["Validation_MAPE"]   = prow.get("Validation_MAPE_Pct", np.nan)
            row["Fiabilite"]         = prow.get("Fiabilite", "n/a")
            row["Stabilite_Taux"]    = prow.get("Stabilite_Taux", "n/a")
            row["Confiance_Globale"] = prow.get("Confiance_Globale", "n/a")
        rows.append(row)

    return pd.DataFrame(rows)


def build_donnees(hist: pd.DataFrame,
                  val_dfs: dict,
                  prev_df: pd.DataFrame,
                  trien_long: pd.DataFrame = None) -> pd.DataFrame:
    """Long format: one row per LB × Année.
    Combines historical actuals, validation (actual + prediction), and forecast.
    Type column = 'Historique' | 'Validation' | 'Prévision'
    → directly usable as a fact table in Power BI."""

    meta     = hist.drop_duplicates("Line_Key").set_index("Line_Key")[ID_COLS]
    keys     = meta.index.tolist()
    val_years = set(val_dfs.keys())
    rows     = []

    for key in keys:
        grp     = hist[hist["Line_Key"] == key].set_index("Year")
        id_base = {c: meta.loc[key, c] for c in ID_COLS}
        id_base["Line_Key"] = key

        # Historical years (not validation, and not partial current year 2026)
        hist_only_years = [y for y in grp.index if y not in val_years and y != 2026]
        for y in hist_only_years:
            co  = grp.loc[y, "Credits_Ouverts_Vises"]
            eng = grp.loc[y, "Total_Engage_Vises"]
            taux = eng / co if co and not np.isnan(co) and co != 0 else np.nan
            rows.append({**id_base,
                "Année": y, "Type": "Historique",
                "CO": round(co, 2), "Engage_Reel": round(eng, 2),
                "Taux_Reel": round(taux, 4) if not np.isnan(taux) else np.nan,
                "Prediction": np.nan, "Int_Bas": np.nan, "Int_Haut": np.nan,
                "MAPE_Pct": np.nan, "Modele_Utilise": np.nan})

        # Validation years (actual + prediction side by side)
        for y, vdf in val_dfs.items():
            vrow = vdf[vdf["Line_Key"] == key]
            co  = grp.loc[y, "Credits_Ouverts_Vises"] if y in grp.index else np.nan
            eng = grp.loc[y, "Total_Engage_Vises"]    if y in grp.index else np.nan
            taux = eng / co if (co and not np.isnan(co) and co != 0
                                and not np.isnan(eng)) else np.nan
            pred = mape = int_bas = int_haut = np.nan
            modele = np.nan
            if len(vrow) > 0:
                v = vrow.iloc[0]
                pred_col = f"Total_Engage_Prediction_{y}"
                pred     = v.get(pred_col, np.nan)
                mape     = v.get("Erreur_Pct", np.nan)
                int_bas  = v.get("Intervalle_Bas", np.nan)
                int_haut = v.get("Intervalle_Haut", np.nan)
                modele   = v.get("Modele_Utilise", np.nan)
            rows.append({**id_base,
                "Année": y, "Type": "Validation",
                "CO": round(co, 2) if not np.isnan(co) else np.nan,
                "Engage_Reel": round(eng, 2) if not np.isnan(eng) else np.nan,
                "Taux_Reel": round(taux, 4) if not np.isnan(taux) else np.nan,
                "Prediction": round(pred, 2) if not np.isnan(pred) else np.nan,
                "Int_Bas": int_bas, "Int_Haut": int_haut,
                "MAPE_Pct": mape, "Modele_Utilise": modele})

        # Forecast — triennal si dispo, sinon prev_df (1 année)
        if trien_long is not None and not trien_long.empty:
            tl = trien_long.copy()
            tl["Line_Key"] = make_line_key(tl)
            sub = tl[tl["Line_Key"] == key]
            for _, r in sub.iterrows():
                rows.append({**id_base,
                    "Année": int(r["Annee"]), "Type": "Prévision",
                    "CO": r.get("Credits_Ouverts_Pred", np.nan),
                    "Engage_Reel": np.nan, "Taux_Reel": np.nan,
                    "Prediction": r.get("Total_Engage_Pred", np.nan),
                    "Int_Bas": r.get("Intervalle_Bas", np.nan),
                    "Int_Haut": r.get("Intervalle_Haut", np.nan),
                    "MAPE_Pct": np.nan,
                    "Modele_Utilise": r.get("Modele_Utilise", np.nan)})
        else:
            prow = prev_df[prev_df["Line_Key"] == key]
            if len(prow) > 0:
                p = prow.iloc[0]
                eng_col = [c for c in p.index if c.startswith("Total_Engage_Pred_")]
                if eng_col:
                    next_y = int(eng_col[0].split("_")[-1])
                    rows.append({**id_base,
                        "Année": next_y, "Type": "Prévision",
                        "CO": p.get("Credits_Ouverts_Pred", np.nan),
                        "Engage_Reel": np.nan, "Taux_Reel": np.nan,
                        "Prediction": p.get(eng_col[0], np.nan),
                        "Int_Bas": p.get("Intervalle_Bas", np.nan),
                        "Int_Haut": p.get("Intervalle_Haut", np.nan),
                        "MAPE_Pct": np.nan,
                        "Modele_Utilise": p.get("Modele_Utilise", np.nan)})

    return pd.DataFrame(rows)


def main():
    print("Chargement des données…")
    hist = pd.read_excel(SRC_HIST)
    hist["Line_Key"] = make_line_key(hist)

    all_years = sorted(hist["Year"].unique())
    val_years = [y for y in all_years[2:] if y != 2026]   # exclure année partielle

    val_dfs = {}
    for y in val_years:
        path = OUT_DIR / f"SituationChap_VALIDATION_{y}.xlsx"
        if path.exists():
            vdf = pd.read_excel(path, sheet_name="Meilleur_Modele")
            vdf["Line_Key"] = make_line_key(vdf)
            val_dfs[y] = vdf

    if SRC_PREV is None or not SRC_PREV.exists():
        print("ERREUR : fichier PREVISION_BEST introuvable. Lance d'abord script 05.")
        return
    prev_df = pd.read_excel(SRC_PREV)
    prev_df["Line_Key"] = make_line_key(prev_df)
    next_year = int(SRC_PREV.stem.split("_")[-1])
    print(f"  Prévision détectée : {next_year}")

    trien_df = None
    trien_long = None
    if SRC_TRIEN and SRC_TRIEN.exists():
        trien_df   = pd.read_excel(SRC_TRIEN, sheet_name="Prévision_Triennale")
        trien_df["Line_Key"] = make_line_key(trien_df)
        trien_long = pd.read_excel(SRC_TRIEN, sheet_name="Format_Long")

    print("Construction des tables…")
    recap_df   = build_recap(hist, val_dfs, prev_df, trien_long)
    donnees_df = build_donnees(hist, val_dfs, prev_df, trien_long)
    print(f"  Récapitulatif : {len(recap_df)} lignes")
    print(f"  Données       : {len(donnees_df)} lignes ({len(donnees_df['Année'].unique())} années)")

    out_path = OUT_DIR / "SituationChap_RAPPORT_COMPLET.xlsx"
    print(f"Écriture de {out_path}…")

    # Top lignes par consommation (agrégé sur 2021-2025)
    hist_val = donnees_df[donnees_df["Type"].isin(["Historique", "Validation"])]
    top_lignes = (hist_val.groupby(["Line_Key"] + ID_COLS)
                          .agg(Total_Engage_Reel=("Engage_Reel", "sum"),
                               Total_CO=("CO", "sum"),
                               Nb_Annees=("Année", "count"))
                          .reset_index())
    top_lignes["Taux_Moyen"] = (top_lignes["Total_Engage_Reel"] / top_lignes["Total_CO"]).round(2)
    top_lignes["Rang"] = top_lignes["Total_Engage_Reel"].rank(method="dense", ascending=False).astype(int)
    top_lignes = top_lignes.sort_values("Rang")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        recap_df.to_excel(writer,   sheet_name="Récapitulatif", index=False)
        donnees_out = donnees_df.copy()
        donnees_out["Année"] = donnees_out["Année"].astype(int)  # entier pour DAX
        donnees_out.to_excel(writer, sheet_name="Données",       index=False)
        top_lignes.to_excel(writer,  sheet_name="Top_Lignes",    index=False)

    print(f"\n-> {out_path}")
    print(f"   Sheets : Récapitulatif (dim. table) | Données (long format, Power BI)")


if __name__ == "__main__":
    main()
