"""
update_2026_with_predictions.py
================================
Remplace les valeurs partielles 2026 par les prévisions du modèle Crédits×Taux,
afin que 2021→2026 forme un historique complet pour prédire 2027/2028/2029.

Ce que fait le script :
    1. Sauvegarde les fichiers actuels (.bak)
    2. Charge la prévision 2026 par taux (Prevision_2026_par_Taux.xlsx)
    3. Met à jour Total_Engage_Vises sur Year=2026 pour chaque ligne du panel stable
    4. Re-agrège les niveaux supérieurs (Chap/Prog/Reg/Proj) comme somme des lignes
    5. Sauvegarde le nouveau panel
    6. Met aussi à jour SituationChap-2026_clean.xlsx pour cohérence
"""
from __future__ import annotations
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "02_cleaned"
PANEL = CLEAN / "SituationChap_STABLE_PANEL.xlsx"
CLEAN26 = CLEAN / "SituationChap-2026_clean.xlsx"
PRED = ROOT / "data" / "03_forecast" / "Prevision_2026_par_Taux.xlsx"


def main():
    # --- 1. Backups ---
    for f in [PANEL, CLEAN26]:
        bak = f.with_suffix(f.suffix + ".bak")
        if not bak.exists():
            shutil.copy(f, bak)
            print(f"[bak] sauvegarde -> {bak.name}")

    # --- 2. Charger la prévision ---
    pred = pd.read_excel(PRED, sheet_name="Detail_Lignes",
                         dtype={"Code_Ligne": str})
    pred_map = dict(zip(pred["Code_Ligne"], pred["Prevision_Engage_2026"]))
    print(f"[load] {len(pred_map)} prévisions 2026 chargées")

    # --- 3. Mettre à jour le panel stable ---
    panel = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str,
                                        "Proj": str, "Lb": str, "Line_Key": str})
    # Normaliser les NaN string en None pour les merges
    for c in ["Chap", "Prog", "Reg", "Proj", "Lb"]:
        panel[c] = panel[c].where(panel[c].notna() & (panel[c] != "nan"), None)

    # Sauvegarde des anciennes valeurs 2026 ligne pour récap
    before = panel[(panel.Year == 2026) & (panel.Level == "Ligne")] \
                    ["Total_Engage_Vises"].sum()

    mask = (panel.Year == 2026) & (panel.Level == "Ligne")
    panel.loc[mask, "Total_Engage_Vises"] = (
        panel.loc[mask, "Line_Key"].map(pred_map)
                                   .fillna(panel.loc[mask, "Total_Engage_Vises"])
    )
    after = panel.loc[mask, "Total_Engage_Vises"].sum()
    print(f"[update] Total Engagé 2026 (lignes) :  {before:>15,.0f}  ->  {after:>15,.0f}")

    # --- 4. Re-agréger les niveaux supérieurs ---
    # Dans le panel, Chap/Prog/Reg/Proj sont remplis SEULEMENT au niveau correspondant.
    # On utilise donc Line_Key prefix : niveau Chapitre = key = Chap, niveau Programme = Chap-Prog, etc.
    lignes_2026 = panel[(panel.Year == 2026) & (panel.Level == "Ligne")].copy()
    # parts du Line_Key pour les lignes
    parts = lignes_2026["Line_Key"].str.split("-", expand=True).astype(str).fillna("")
    parts.columns = ["P0", "P1", "P2", "P3", "P4"]
    lignes_2026 = pd.concat([lignes_2026.reset_index(drop=True), parts.reset_index(drop=True)], axis=1)

    level_specs = [
        ("Chapitre",  ["P0"],                       "Chap"),
        ("Programme", ["P0", "P1"],                 None),
        ("Region",    ["P0", "P1", "P2"],           None),
        ("Projet",    ["P0", "P1", "P2", "P3"],     None),
    ]
    for level_name, part_cols, _ in level_specs:
        # sum by composite key
        sums_dict = (lignes_2026.assign(_key=lignes_2026[part_cols].agg("-".join, axis=1))
                                 .groupby("_key")["Total_Engage_Vises"].sum().to_dict())
        m = (panel.Year == 2026) & (panel.Level == level_name)
        if not m.any():
            continue
        # The aggregated rows store the key in their Line_Key column (already)
        keys = panel.loc[m, "Line_Key"]
        new_vals = keys.map(sums_dict).fillna(0).values
        panel.loc[m, "Total_Engage_Vises"] = new_vals
        print(f"[agg] niveau {level_name:<10}: {m.sum()} ligne(s) re-agrégée(s)  ({new_vals.sum():,.0f})")

    # --- 5. Sauvegarder ---
    panel.to_excel(PANEL, index=False)
    print(f"[save] panel mis à jour -> {PANEL.name}")

    # --- 6. Mettre à jour SituationChap-2026_clean.xlsx depuis le panel (source de vérité) ---
    clean26 = pd.read_excel(CLEAN26)
    panel_2026 = panel[panel.Year == 2026].copy()

    # Construire une clé d'appariement identique des deux côtés à partir de Chap/Prog/Reg/Proj/Lb + Level
    def build_key(df):
        cols = ["Chap", "Prog", "Reg", "Proj", "Lb"]
        out = df[cols].copy()
        for c in cols:
            # convertir float->int->str proprement, NaN -> ""
            out[c] = out[c].apply(
                lambda v: "" if pd.isna(v) else (
                    str(int(float(v))) if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit())
                    else str(v)
                )
            )
        return out.agg("|".join, axis=1) + "||" + df["Level"].astype(str)

    clean26["_K"] = build_key(clean26)
    panel_2026["_K"] = build_key(panel_2026)
    new_vals = dict(zip(panel_2026["_K"], panel_2026["Total_Engage_Vises"]))
    clean26["Total_Engage_Vises"] = clean26["_K"].map(new_vals).fillna(clean26["Total_Engage_Vises"])
    n_matched = clean26["_K"].isin(new_vals).sum()
    clean26 = clean26.drop(columns=["_K"])
    clean26.to_excel(CLEAN26, index=False)
    print(f"[save] cleaned 2026 mis à jour ({n_matched}/{len(clean26)} lignes synchronisées) -> {CLEAN26.name}")
    print(f"       Total Engagé 2026 (Ligne) dans clean26 : "
          f"{clean26[clean26.Level=='Ligne']['Total_Engage_Vises'].sum():,.0f}")

    # --- Vérification finale ---
    print("\n" + "=" * 72)
    print("  VERIFICATION : Total Engagé par année (panel stable, niveau Ligne)")
    print("=" * 72)
    final = pd.read_excel(PANEL)
    final_lignes = final[final.Level == "Ligne"]
    print(final_lignes.groupby("Year")["Total_Engage_Vises"].sum().round(0).to_string())
    print("\n[ok] 2026 est maintenant cohérent — utilisable comme historique pour 2027+")


if __name__ == "__main__":
    main()
