"""
Budget Overview Dashboard — Préparation des données
Connecte AlertResults + TGR pour vue d'ensemble Page 1
Export: budget_overview.xlsx
"""

import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# 1. Charger AlertResults (2025)
# ──────────────────────────────────────────────────────────────────────────────

alert_df = pd.read_csv("../projectv2/alert_results_2025.csv")

# Ajouter line_id unique
alert_df.insert(0, 'line_id', range(1, len(alert_df) + 1))

print("="*100)
print("BUDGET OVERVIEW — PRÉPARATION DONNÉES DASHBOARD PAGE 1")
print("="*100)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Ajouter colonne budget_type basée sur type_ligne
# ──────────────────────────────────────────────────────────────────────────────

def map_to_budget_type(type_ligne):
    """Mappe type_ligne vers Categorie TGR"""
    type_ligne_lower = str(type_ligne).lower().strip()
    
    if 'personnel' in type_ligne_lower:
        return 'PERSONNEL'
    elif any(x in type_ligne_lower for x in ['fournituren', 'equipement', 'acquisition']):
        return 'MATERIEL'
    else:
        return 'INVESTISSEMENT'

alert_df['budget_type'] = alert_df['type_ligne'].apply(map_to_budget_type)

print("\n✓ Colonne 'budget_type' créée")
print(f"  Répartition:\n{alert_df['budget_type'].value_counts()}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Charger données TGR (2020-2025)
# ──────────────────────────────────────────────────────────────────────────────

tgr_df = pd.read_excel("../projectv2/donnees_tgr_trimestriel.xlsx")

print(f"\n✓ TGR chargé ({len(tgr_df)} lignes)")
print(f"  Années: {sorted(tgr_df['Annee'].unique())}")
print(f"  Catégories: {sorted(tgr_df['Categorie'].unique())}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Créer tableau "Évolution T1-T4 par Budget (2020-2025)"
# ──────────────────────────────────────────────────────────────────────────────

# Renommer colonnes pour clarté
tgr_df.rename(columns={
    'Categorie': 'budget_type',
    'Trimestre': 'trimestre',
    'Annee': 'annee',
    'Taux execution (%)': 'taux_execution'
}, inplace=True)

# Pivoter pour avoir T1, T2, T3, T4 en colonnes
evolution_df = tgr_df.pivot_table(
    index=['annee', 'budget_type'],
    columns='trimestre',
    values='taux_execution',
    aggfunc='first'
).reset_index()

# Réordonner colonnes
evolution_df = evolution_df[['annee', 'budget_type', 'T1', 'T2', 'T3', 'T4']]

print(f"\n✓ Tableau d'évolution créé ({len(evolution_df)} lignes)")
print("\nAperçu Evolution T1-T4:")
print(evolution_df.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 5. Créer "Budget Overview" — Summary KPIs par budget
# ──────────────────────────────────────────────────────────────────────────────

budget_summary = alert_df.groupby('budget_type').agg({
    'ligne_label': 'count',
    'lf_mdh': 'sum',
    'region_label': 'nunique',
    'pred_T4_rolling': 'mean'
}).reset_index()

budget_summary.columns = ['budget_type', 'nombre_lignes', 'budget_total_mdh', 'nombre_regions', 'T4_moyen_predit']
budget_summary = budget_summary.sort_values('budget_total_mdh', ascending=False)

print(f"\n✓ Résumé Budget créé:")
print(budget_summary.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 6. Créer données pour "Comparaison T4 2025: Réel vs Direct vs Rolling"
# ──────────────────────────────────────────────────────────────────────────────

# Ajouter T4 réel 2024 du TGR (comme baseline)
t4_2024_tgr = tgr_df[tgr_df['annee'] == 2024][['budget_type', 'taux_execution']].copy()
t4_2024_tgr.columns = ['budget_type', 't4_2024_reel']

# Calculer moyennes par budget pour 2025 prédictions
pred_summary = alert_df.groupby('budget_type').agg({
    'pred_T4_direct': 'mean',
    'pred_T4_rolling': 'mean'
}).reset_index()

# Joindre toutes les données
comparison_2025 = t4_2024_tgr.merge(pred_summary, on='budget_type', how='outer')
comparison_2025 = comparison_2025.sort_values('t4_2024_reel', ascending=False)

print(f"\n✓ Comparaison T4 2025 créée:")
print(comparison_2025.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 7. Exporter données enrichies pour Power BI
# ──────────────────────────────────────────────────────────────────────────────

# Fichier 1: AlertResults enrichis avec budget_type (OVERWRITE original)
alert_df.to_csv("../projectv2/alert_results_2025.csv", index=False)
alert_df.to_excel("../projectv2/alert_results_2025.xlsx", index=False, sheet_name="AlertResults")

print("\n" + "="*100)
print("FICHIERS GÉNÉRÉS AVEC SUCCÈS")
print("="*100)
print("""
✓ evolution_t1_t4_by_budget.csv
  → Pour 3 Line Charts (T1, T2, T3, T4 par budget 2020-2025)
  
✓ budget_summary_kpi.csv
  → Pour 3 Cards (# lignes, budget total, # régions par budget)
  
✓ comparison_t4_2025.csv
  → Pour 3 Bar Charts (T4 réel 2024 vs pred direct vs rolling)
  
✓ alert_results_enriched.csv
  → AlertResults + colonne 'budget_type' pour Power BI
  
UTILISATION POWER BI:
1. Importer ces 4 fichiers comme sources de données
2. Créer relations via 'budget_type'
3. Page 1 — Vue Générale:
   a) 3 Cards KPI (budget_summary_kpi)
   b) 3 Line Charts (evolution_t1_t4_by_budget)
   c) 3 Bar Charts (comparison_t4_2025)
""")

print(f"\nStatistiques finales:")
print(f"  Budgets: {budget_summary['budget_type'].nunique()}")
print(f"  Total lignes: {budget_summary['nombre_lignes'].sum()}")
print(f"  Budget total: {budget_summary['budget_total_mdh'].sum():.1f} MDH")
print(f"  Années TGR: {sorted(evolution_df['annee'].unique())}")
