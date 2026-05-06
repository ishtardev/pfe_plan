import pandas as pd
import os

# Définir le chemin
project_dir = r'C:\Users\godde\OneDrive\Desktop\pfe_plan\projectv2'

# Charger les données
alert_results = pd.read_excel(os.path.join(project_dir, 'alert_results_2025.xlsx'))

# Vérifier les colonnes disponibles
print("Colonnes disponibles:")
print(alert_results.columns.tolist())
print("\nAperçu des données:")
print(alert_results.head())

# Créer un résumé par budget_type
summary = alert_results.groupby('budget_type').agg({
    'actual_T4': 'mean',           # T4 2025 réel moyen
    'pred_T4_direct': 'mean',      # Prédiction directe moyenne
    'pred_T4_rolling': 'mean'      # Prédiction rolling moyenne
}).reset_index()

# Renommer les colonnes
summary.columns = ['budget_type', 'T4_2025_réel', 'T4_2025_prédit_direct', 'T4_2025_prédit_rolling']

# Convertir en pourcentages (multiplier par 100 si nécessaire)
# Vérifier d'abord la plage des valeurs
print("\nValeurs avant conversion:")
print(summary)

# Multiplier par 100 pour convertir en pourcentages (peu importe la plage)
summary['T4_2025_réel'] = (summary['T4_2025_réel'] * 100).round(1)
summary['T4_2025_prédit_direct'] = (summary['T4_2025_prédit_direct'] * 100).round(1)
summary['T4_2025_prédit_rolling'] = (summary['T4_2025_prédit_rolling'] * 100).round(1)

# Sauvegarder en Excel
output_file = os.path.join(project_dir, 'summary_T4_2025_validation.xlsx')
summary.to_excel(output_file, index=False)

print("\n✓ Fichier créé:")
print(f"  {output_file}")
print(f"\nRésumé:")
print(summary)
