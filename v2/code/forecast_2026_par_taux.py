"""
forecast_2026_par_taux.py
=========================
Prévision du Total Engagé 2026 = Crédits Ouverts 2026 × Taux d'engagement prédit.

Pourquoi ?
    Les crédits ouverts sont fixés en début d'année, donc 2026 EST déjà connue
    sur cette colonne. Le taux d'engagement (Engage / Crédits) est historiquement
    très stable (97–102 % au global). On peut donc prédire Engage 2026 de façon
    bien plus robuste que par les méthodes time-series sur Engage seul.

Méthode :
    Pour chaque ligne :
      1) Taux historique = Engage / Crédits sur 2021..2025
      2) Taux prédit 2026 = médiane des 3 dernières années   (pondéré si dispo)
      3) Engage prédit 2026 = Crédits_Ouverts_2026 × Taux prédit 2026
      4) Comparaison vs valeur 2026 actuellement dans le panel

Sortie :
    v2/data/03_forecast/Prevision_2026_par_Taux.xlsx
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT = ROOT / "data" / "03_forecast" / "Prevision_2026_par_Taux.xlsx"

TARGET = 2026
HIST_END = 2025          # taux appris sur 2021..2025


def main():
    panel = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str,
                                        "Proj": str, "Lb": str})
    lignes = panel[panel.Level == "Ligne"].copy()
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        intitule = sub["Intitule"].iloc[-1]
        # Crédits ouverts 2026 (la valeur "déjà connue")
        if TARGET not in sub.index:
            continue
        credits_2026 = float(sub.loc[TARGET, "Credits_Ouverts_Vises"])
        engage_2026_actuel = float(sub.loc[TARGET, "Total_Engage_Vises"])

        # Historique strict < TARGET
        hist = sub.loc[sub.index < TARGET]
        hist = hist[hist["Credits_Ouverts_Vises"] > 0]
        if hist.empty or credits_2026 <= 0:
            continue
        taux_hist = (hist["Total_Engage_Vises"] / hist["Credits_Ouverts_Vises"]).clip(0, 2)

        # Taux prédit = médiane des 3 dernières années dispo (ou tout si <3 ans)
        recent = taux_hist.tail(3)
        taux_pred = float(recent.median())
        # Sécurité : taux dans [0, 1.2]
        taux_pred = max(0.0, min(1.2, taux_pred))

        prev_engage = credits_2026 * taux_pred
        ecart = prev_engage - engage_2026_actuel
        ecart_pct = (100 * ecart / engage_2026_actuel) if engage_2026_actuel else np.nan

        rows.append({
            "Code_Ligne": key,
            "Intitule": intitule,
            "Credits_Ouverts_2026": round(credits_2026),
            "Taux_2021": round(taux_hist.get(2021, np.nan)*100, 1) if 2021 in taux_hist.index else None,
            "Taux_2022": round(taux_hist.get(2022, np.nan)*100, 1) if 2022 in taux_hist.index else None,
            "Taux_2023": round(taux_hist.get(2023, np.nan)*100, 1) if 2023 in taux_hist.index else None,
            "Taux_2024": round(taux_hist.get(2024, np.nan)*100, 1) if 2024 in taux_hist.index else None,
            "Taux_2025": round(taux_hist.get(2025, np.nan)*100, 1) if 2025 in taux_hist.index else None,
            "Taux_Predit_2026_Pct": round(taux_pred*100, 1),
            "Prevision_Engage_2026": round(prev_engage),
            "Engage_2026_Actuel": round(engage_2026_actuel),
            "Ecart": round(ecart),
            "Ecart_Pct": round(ecart_pct, 2) if pd.notna(ecart_pct) else None,
        })

    detail = pd.DataFrame(rows).sort_values(
        "Prevision_Engage_2026", ascending=False).reset_index(drop=True)

    # --- Synthèse ---
    total_credits = detail["Credits_Ouverts_2026"].sum()
    total_prev = detail["Prevision_Engage_2026"].sum()
    total_act = detail["Engage_2026_Actuel"].sum()
    synth = pd.DataFrame([{
        "Nb_Lignes": len(detail),
        "Total_Credits_Ouverts_2026": round(total_credits),
        "Total_Engage_Prevu_2026": round(total_prev),
        "Total_Engage_Actuel_2026": round(total_act),
        "Ecart_Global": round(total_prev - total_act),
        "Ecart_Global_Pct": round(100*(total_prev - total_act)/total_act, 2) if total_act else None,
        "Taux_Engagement_Predit_Global_Pct": round(100*total_prev/total_credits, 2),
        "Taux_Engagement_Actuel_Global_Pct": round(100*total_act/total_credits, 2),
    }])

    # --- Affichage terminal ---
    print("=" * 78)
    print("  PRÉVISION 2026 PAR TAUX D'ENGAGEMENT")
    print("=" * 78)
    print(synth.T.to_string(header=False))
    print("\n" + "=" * 78)
    print("  TOP 15 LIGNES PAR PRÉVISION 2026")
    print("=" * 78)
    top = detail.head(15)[["Code_Ligne", "Intitule",
                           "Credits_Ouverts_2026", "Taux_Predit_2026_Pct",
                           "Prevision_Engage_2026", "Engage_2026_Actuel",
                           "Ecart_Pct"]].copy()
    top["Intitule"] = top["Intitule"].astype(str).str.slice(0, 40)
    top["Credits_Ouverts_2026"] = top["Credits_Ouverts_2026"].map("{:>15,.0f}".format)
    top["Prevision_Engage_2026"] = top["Prevision_Engage_2026"].map("{:>15,.0f}".format)
    top["Engage_2026_Actuel"] = top["Engage_2026_Actuel"].map("{:>15,.0f}".format)
    top["Ecart_Pct"] = top["Ecart_Pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
    print(top.to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        synth.to_excel(xw, sheet_name="Synthese", index=False)
        detail.to_excel(xw, sheet_name="Detail_Lignes", index=False)
    print(f"\n[ok] Fichier écrit : {OUT.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
