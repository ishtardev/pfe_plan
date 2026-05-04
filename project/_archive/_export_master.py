"""
Build one master Excel sheet with every dimension and every metric:
Année, Catégorie, Programme, Projet, Ligne, Région, LF, T1, T2, T3,
T4 réel, T4 prédit (juin), T4 prédit (sept.), Risque, MDH à risque,
Anomalie, Années d'historique, Écart T3 (pp)
"""
import pandas as pd
import numpy as np
import os

BASE     = r"C:\Users\Inann\Desktop\pfe_plan"
PROJ     = os.path.join(BASE, "project")

# ── 1. Load raw quarterly data ──────────────────────────────────────────────
# Use project/data_lines.csv — it has budget_type and more rows (3120)
df = pd.read_csv(os.path.join(PROJ, "data_lines.csv"))

# budget_type already present in project file
df["categorie"] = df["budget_type"]

# ── 2. Pivot quarters → T1, T2, T3, T4 columns ─────────────────────────────
pivot = (df.pivot_table(
            index=["year", "line_id", "programme", "programme_label",
                   "projet", "projet_label", "ligne", "ligne_label",
                   "region", "region_label", "type_ligne", "categorie",
                   "lf_mdh"],
            columns="quarter",
            values="taux",
            aggfunc="first")
          .reset_index())

# Rename quarter columns
pivot.columns.name = None
rename_q = {1: "T1 (%)", 2: "T2 (%)", 3: "T3 (%)", 4: "T4 reel (%)"}
pivot.rename(columns=rename_q, inplace=True)

# Convert taux to percentages (they are stored as 0–1 ratios)
for col in ["T1 (%)", "T2 (%)", "T3 (%)", "T4 reel (%)"]:
    if col in pivot.columns:
        pivot[col] = (pivot[col] * 100).round(1)

# ── 3. Load 2025 predictions and merge ──────────────────────────────────────
prev = pd.read_excel(os.path.join(BASE, "project", "previsions_2025_20260429.xlsx"))

# Rename for merging
prev = prev.rename(columns={
    "Programme":          "programme_label",
    "Ligne budgetaire":   "ligne_label",
    "Region":             "region_label",
    "Type budget":        "categorie",
    "T4 predit T2 (%)":   "T4 predit juin (%)",
    "T4 predit T3 (%)":   "T4 predit sept. (%)",
    "Risque":             "Risque",
    "MDH a risque":       "Credits a risque (MDH)",
    "Anomalie":           "Anomalie",
    "Annees d'historique":"Annees historique",
    "Ecart T3 (pp)":      "Ecart T3 (pp)",
})

pred_cols = ["programme_label", "ligne_label", "region_label",
             "T4 predit juin (%)", "T4 predit sept. (%)",
             "Risque", "Credits a risque (MDH)", "Anomalie",
             "Annees historique", "Ecart T3 (pp)"]

pivot = pivot.merge(
    prev[pred_cols],
    on=["programme_label", "ligne_label", "region_label"],
    how="left",
    suffixes=("", "_prev")
)

# Predictions only make sense for 2025 rows — blank them for other years
pred_only = ["T4 predit juin (%)", "T4 predit sept. (%)",
             "Risque", "Credits a risque (MDH)", "Anomalie",
             "Annees historique", "Ecart T3 (pp)"]
for col in pred_only:
    if col in pivot.columns:
        pivot.loc[pivot["year"] != 2025, col] = np.nan

# ── 4. Final column selection and rename ────────────────────────────────────
out = pivot[[
    "year", "categorie", "programme", "programme_label",
    "projet", "projet_label", "ligne", "ligne_label",
    "region", "region_label", "type_ligne", "lf_mdh",
    "T1 (%)", "T2 (%)", "T3 (%)", "T4 reel (%)",
    "T4 predit juin (%)", "T4 predit sept. (%)",
    "Risque", "Credits a risque (MDH)", "Anomalie",
    "Annees historique", "Ecart T3 (pp)",
]].copy()

out.rename(columns={
    "year":             "Annee",
    "categorie":        "Categorie budget",
    "programme":        "Code programme",
    "programme_label":  "Programme",
    "projet":           "Code projet",
    "projet_label":     "Projet",
    "ligne":            "Code ligne",
    "ligne_label":      "Ligne budgetaire",
    "region":           "Code region",
    "region_label":     "Region",
    "type_ligne":       "Type de ligne",
    "lf_mdh":           "LF (MDH)",
}, inplace=True)

out = out.sort_values(["Annee", "Categorie budget", "Programme",
                       "Region", "Ligne budgetaire"]).reset_index(drop=True)

# ── 5. Export ────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, "donnees_master.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    out.to_excel(writer, sheet_name="Donnees completes", index=False)

    # Auto-fit column widths
    ws = writer.sheets["Donnees completes"]
    for col_cells in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 45)

print(f"Done — {len(out)} rows exported to {out_path}")
print(f"Years: {sorted(out['Annee'].unique())}")
print(f"Columns: {out.columns.tolist()}")
