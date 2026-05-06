"""
Convertir fichiers CSV → XLSX pour Power BI
"""

import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

files = {
    '../projectv2/budget_summary_kpi.csv': '../projectv2/budget_summary_kpi.xlsx',
    '../projectv2/evolution_t1_t4_by_budget.csv': '../projectv2/evolution_t1_t4_by_budget.xlsx',
    '../projectv2/comparison_t4_2025.csv': '../projectv2/comparison_t4_2025.xlsx',
    '../projectv2/alert_results_enriched.csv': '../projectv2/alert_results_enriched.xlsx'
}

print("="*80)
print("CONVERSION CSV → XLSX")
print("="*80)

for csv_file, xlsx_file in files.items():
    df = pd.read_csv(csv_file)
    df.to_excel(xlsx_file, index=False, sheet_name='Data')
    print(f"\n✓ {os.path.basename(xlsx_file)}")
    print(f"  Lignes: {len(df)}, Colonnes: {len(df.columns)}")

print("\n" + "="*80)
print("TOUS LES FICHIERS XLSX GÉNÉRÉS ✓")
print("="*80)
print("""
Prêts pour Power BI:
  • budget_summary_kpi.xlsx
  • evolution_t1_t4_by_budget.xlsx
  • comparison_t4_2025.xlsx
  • alert_results_enriched.xlsx
""")
