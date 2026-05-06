import pandas as pd
import os

os.chdir("../projectv2")

# Load both files
alert = pd.read_excel('alert_results_2025.xlsx')[['line_id', 'ligne_label', 'budget_type']].drop_duplicates()
historical = pd.read_excel('historical_recap_2025.xlsx')

print("Before merge:")
print(f"Historical columns: {historical.columns.tolist()}")

# Merge to add budget_type
historical = historical.merge(alert[['line_id', 'budget_type']], on='line_id', how='left')

print("\nAfter merge:")
print(f"Historical columns: {historical.columns.tolist()}")

# Reorder columns - put budget_type early
cols = ['line_id', 'ligne_label', 'budget_type', 'region_label'] + [c for c in historical.columns if c not in ['line_id', 'ligne_label', 'budget_type', 'region_label']]
historical = historical[cols]

# Save back
historical.to_excel('historical_recap_2025.xlsx', index=False)

print(f"\n✓ Added budget_type to historical_recap_2025.xlsx")
print(f"  Budget types: {historical['budget_type'].unique()}")
print(f"  Sample:")
print(historical[['ligne_label', 'budget_type', 'year', 'taux_T2']].head(10))
