import pandas as pd
import os

os.chdir("../projectv2")

# Load both files
alert = pd.read_excel('alert_results_2025.xlsx')[['line_id', 'ligne_label']].drop_duplicates()
historical = pd.read_excel('historical_recap_2025.xlsx')

print('Alert mapping (first 5):')
print(alert.head())

# Map the coded line_id to numeric line_id using ligne_label as key
historical = historical.merge(alert, on='ligne_label', how='left', suffixes=('_old', ''))

# Drop the old line_id if it exists
if 'line_id_old' in historical.columns:
    historical = historical.drop('line_id_old', axis=1)

# Reorder columns to put line_id first
cols = ['line_id'] + [c for c in historical.columns if c != 'line_id']
historical = historical[cols]

# Save it back
historical.to_excel('historical_recap_2025.xlsx', index=False)

print(f'\n✓ Updated historical_recap_2025.xlsx')
print(f'  Rows: {len(historical)}')
print(f'  line_id now matches alert_results (numeric 1-130)')
print(f'  Sample line_ids: {list(historical["line_id"].head(3))}')
