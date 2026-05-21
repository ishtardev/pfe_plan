"""
validation_walk_forward.py
==========================
Validation walk-forward de la prévision triennale.

Principe :
    Pour chaque année cible T (par défaut 2024 et 2025) :
      - on n'utilise QUE les années < T pour l'apprentissage
      - on choisit la meilleure méthode par ligne sur les années antérieures
      - on prédit T avec cette méthode
      - on compare à la valeur réelle observée en T

Sorties (toutes en français) dans v2/data/03_forecast/ :
    Validation_Walk_Forward.xlsx  avec 3 feuilles :
        - Detail_Lignes  : Réel, Prévision, Écart, Écart %, par ligne × année cible
        - Synthese_Annee : agrégats par année cible (sMAPE moyen, médian, WAPE, biais)
        - Synthese_Globale : un seul résumé toutes années confondues

Run :
    python v2/code/validation_walk_forward.py
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT = ROOT / "data" / "03_forecast" / "Validation_Walk_Forward.xlsx"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_pipeline import METHODS, m_xgb, train_xgb, smape  # type: ignore

# Années cibles testées (2026 exclue car partielle — nous sommes encore dedans)
TARGET_YEARS = [2024, 2025]
MIN_HISTORY = 3            # nb d'années d'historique minimum requises


# --------------------------------------------------------------------
def load_lignes() -> pd.DataFrame:
    panel = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str,
                                        "Proj": str, "Lb": str})
    return panel[panel.Level == "Ligne"].copy()


def predict_one_year(lignes: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Pour chaque ligne :
      1) On entraîne XGB global sur years < target_year
      2) Sur les années (< target_year) AVANT l'an dernier connu, on choisit la
         méthode qui a le plus faible sMAPE en backtest interne.
      3) On prédit target_year avec cette méthode.
    """
    train_xgb(lignes[lignes.Year < target_year], target_year)
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        intitule = sub["Intitule"].iloc[-1]
        # On n'utilise que les années strictement antérieures à la cible
        avail = sub.loc[sub.index < target_year, "Total_Engage_Vises"]
        if len(avail) < MIN_HISTORY or target_year not in sub.index:
            continue

        # ----- sélection de la méthode sur backtest interne -----
        # On valide la méthode sur l'avant-dernière année connue,
        # avec entraînement sur les années encore plus anciennes.
        validation_year = int(avail.index.max())          # ex. si target=2025, validation=2024
        train_avail = avail.loc[avail.index < validation_year]
        actual_val = float(avail.loc[validation_year])
        if len(train_avail) < 2:
            best_method = "Naive"
        else:
            best_method, best_sm = "Naive", float("inf")
            for name, fn in METHODS.items():
                try:
                    m_xgb._current_key = key
                    pred_val = float(fn(train_avail, validation_year))
                    sm = smape(actual_val, pred_val)
                except Exception:
                    continue
                if sm < best_sm:
                    best_method, best_sm = name, sm

        # ----- prédiction finale pour target_year -----
        fn = METHODS.get(best_method, METHODS["Naive"])
        m_xgb._current_key = key
        try:
            pred = float(fn(avail, target_year))
        except Exception:
            pred = float(avail.iloc[-1])

        actual = float(sub.loc[target_year, "Total_Engage_Vises"])
        ecart = pred - actual
        ecart_pct = (100 * ecart / actual) if actual != 0 else np.nan

        rows.append({
            "Code_Ligne": key,
            "Intitule": intitule,
            "Annee_Cible": target_year,
            "Methode_Retenue": best_method,
            "Reel": actual,
            "Prevision": pred,
            "Ecart": ecart,
            "Ecart_Absolu": abs(ecart),
            "Ecart_Pct": ecart_pct,
            "sMAPE_Pct": smape(actual, pred),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
def synthese_par_annee(detail: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ty, grp in detail.groupby("Annee_Cible"):
        total_reel = grp["Reel"].sum()
        total_prev = grp["Prevision"].sum()
        wape = 100 * grp["Ecart_Absolu"].sum() / total_reel if total_reel else np.nan
        out.append({
            "Annee_Cible": ty,
            "Nb_Lignes": len(grp),
            "Total_Reel": round(total_reel),
            "Total_Prevision": round(total_prev),
            "Ecart_Global": round(total_prev - total_reel),
            "Ecart_Global_Pct": round(100 * (total_prev - total_reel) / total_reel, 2)
                                if total_reel else np.nan,
            "WAPE_Pct": round(wape, 2),
            "sMAPE_Moyen_Pct": round(grp["sMAPE_Pct"].mean(), 2),
            "sMAPE_Median_Pct": round(grp["sMAPE_Pct"].median(), 2),
            "Lignes_Erreur_inf_20pct": int((grp["sMAPE_Pct"] < 20).sum()),
            "Lignes_Erreur_sup_50pct": int((grp["sMAPE_Pct"] > 50).sum()),
        })
    return pd.DataFrame(out)


def synthese_globale(detail: pd.DataFrame) -> pd.DataFrame:
    total_reel = detail["Reel"].sum()
    total_prev = detail["Prevision"].sum()
    return pd.DataFrame([{
        "Nb_Predictions": len(detail),
        "Nb_Lignes_Uniques": detail["Code_Ligne"].nunique(),
        "Annees_Testees": ", ".join(map(str, sorted(detail["Annee_Cible"].unique()))),
        "Total_Reel": round(total_reel),
        "Total_Prevision": round(total_prev),
        "Ecart_Global": round(total_prev - total_reel),
        "Ecart_Global_Pct": round(100 * (total_prev - total_reel) / total_reel, 2)
                            if total_reel else np.nan,
        "WAPE_Pct": round(100 * detail["Ecart_Absolu"].sum() / total_reel, 2),
        "sMAPE_Moyen_Pct": round(detail["sMAPE_Pct"].mean(), 2),
        "sMAPE_Median_Pct": round(detail["sMAPE_Pct"].median(), 2),
        "Part_Lignes_Erreur_inf_20pct": f"{100*(detail['sMAPE_Pct']<20).mean():.1f}%",
        "Part_Lignes_Erreur_inf_50pct": f"{100*(detail['sMAPE_Pct']<50).mean():.1f}%",
    }])


# --------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  VALIDATION WALK-FORWARD  (Réel vs Prévision)")
    print("=" * 72)
    lignes = load_lignes()
    print(f"[data] {lignes['Line_Key'].nunique()} lignes uniques, "
          f"années disponibles : {sorted(lignes.Year.unique().tolist())}")
    print(f"[data] années cibles testées : {TARGET_YEARS}  (2026 exclue : partielle)")

    all_detail = []
    for ty in TARGET_YEARS:
        print(f"\n  --> Backtest année cible {ty}  (entraînement sur années < {ty})")
        df = predict_one_year(lignes, ty)
        print(f"      {len(df)} lignes prédites")
        all_detail.append(df)

    detail = pd.concat(all_detail, ignore_index=True)
    # tri lisible : par année puis par écart absolu décroissant (lignes qui ratent le plus)
    detail = detail.sort_values(["Annee_Cible", "Ecart_Absolu"], ascending=[True, False])

    synth_year = synthese_par_annee(detail)
    synth_glob = synthese_globale(detail)

    # ----- impression terminal -----
    print("\n" + "=" * 72)
    print("  SYNTHESE PAR ANNEE CIBLE")
    print("=" * 72)
    print(synth_year.to_string(index=False))

    print("\n" + "=" * 72)
    print("  SYNTHESE GLOBALE")
    print("=" * 72)
    print(synth_glob.T.to_string(header=False))

    print("\n" + "=" * 72)
    print("  TOP 15 ECARTS LES PLUS IMPORTANTS (en valeur absolue)")
    print("=" * 72)
    top = (detail.nlargest(15, "Ecart_Absolu")
                 [["Annee_Cible", "Code_Ligne", "Intitule",
                   "Methode_Retenue", "Reel", "Prevision", "Ecart", "Ecart_Pct"]]
                 .copy())
    top["Reel"] = top["Reel"].map("{:>15,.0f}".format)
    top["Prevision"] = top["Prevision"].map("{:>15,.0f}".format)
    top["Ecart"] = top["Ecart"].map("{:>+15,.0f}".format)
    top["Ecart_Pct"] = top["Ecart_Pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
    top["Intitule"] = top["Intitule"].astype(str).str.slice(0, 45)
    print(top.to_string(index=False))

    # ----- écriture Excel -----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        # synthèses en premier (l'utilisateur les voit en ouvrant le fichier)
        synth_glob.to_excel(xw, sheet_name="Synthese_Globale", index=False)
        synth_year.to_excel(xw, sheet_name="Synthese_Annee", index=False)
        detail.round(2).to_excel(xw, sheet_name="Detail_Lignes", index=False)

    print(f"\n[ok] Fichier écrit : {OUT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
