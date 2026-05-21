"""
Tableau récapitulatif historique — Performance par ligne budgétaire 2020-2025
Génère une feuille Excel avec l'historique complet + prédictions 2025 pour chaque ligne
"""

import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Charger données
df_lines = pd.read_csv("data_lines.csv")  # Historique 2020-2025
df_alert = pd.read_csv("alert_results_2025.csv")  # Prédictions 2025

# Récupérer les 2025 prédictions
pred_2025 = {}
for _, row in df_alert.iterrows():
    lid = row["ligne_label"]  # ou line_id si disponible
    pred_2025[lid] = {
        "taux_T2": row["taux_T2"],
        "pred_T4_direct": row["pred_T4_direct"],
        "pred_T4_rolling": row["pred_T4_rolling"],
        "actual_T4": row["actual_T4"],
        "z_score": row["z_score_T2"],
        "anomalie": row["anomalie_label"],
    }

# Construire table récapitulative par ligne
recap_data = []

for ligne_id in df_lines["line_id"].unique():
    line_hist = df_lines[df_lines["line_id"] == ligne_id].sort_values(["year", "quarter"])
    
    if line_hist.empty:
        continue
    
    # Infos ligne
    ligne_label = line_hist.iloc[0]["ligne_label"]
    programme = line_hist.iloc[0]["programme"]
    region = line_hist.iloc[0]["region_label"]
    type_ligne = line_hist.iloc[0]["type_ligne"]
    lf_mdh = line_hist.iloc[0]["lf_mdh"]
    
    # Historique T2/T4 par année
    years_data = {}
    for year in sorted(line_hist["year"].unique()):
        year_data = line_hist[line_hist["year"] == year]
        
        t2 = year_data[year_data["quarter"] == 2]["taux"].values
        t4 = year_data[year_data["quarter"] == 4]["taux"].values
        
        years_data[year] = {
            "T2": float(t2[0]) if len(t2) > 0 else np.nan,
            "T4": float(t4[0]) if len(t4) > 0 else np.nan,
        }
    
    # Construire une ligne pour chaque année
    for year in sorted(years_data.keys()):
        row_dict = {
            "line_id": ligne_id,
            "programme": programme,
            "ligne_label": ligne_label,
            "region_label": region,
            "type_ligne": type_ligne,
            "lf_mdh": lf_mdh,
            "year": year,
            "taux_T2": years_data[year]["T2"],
            "actual_T4": years_data[year]["T4"],
        }
        
        # 2025: ajouter prédictions
        if year == 2025 and ligne_label in pred_2025:
            pred = pred_2025[ligne_label]
            row_dict["pred_T4_direct"] = pred["pred_T4_direct"]
            row_dict["pred_T4_rolling"] = pred["pred_T4_rolling"]
            row_dict["ecart_rolling"] = pred["pred_T4_rolling"] - years_data[year]["T4"]
            row_dict["z_score_T2"] = pred["z_score"]
            row_dict["anomalie_label"] = pred["anomalie"]
        else:
            row_dict["pred_T4_direct"] = np.nan
            row_dict["pred_T4_rolling"] = np.nan
            row_dict["ecart_rolling"] = np.nan
            row_dict["z_score_T2"] = np.nan
            row_dict["anomalie_label"] = ""
        
        recap_data.append(row_dict)

recap_df = pd.DataFrame(recap_data)

# Export Excel
output_file = os.path.join("..", "projectv2", "historical_recap_2025.xlsx")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    recap_df.to_excel(writer, sheet_name="Historique", index=False)

print(f"✓ Créé {output_file}")
print(f"  {len(recap_df)} observations (lignes × années)")
print(f"  Années couverte: {recap_df['year'].min():.0f}-{recap_df['year'].max():.0f}")
print(f"\nColonnes:")
for i, col in enumerate(recap_df.columns, 1):
    print(f"  {i}. {col}")
