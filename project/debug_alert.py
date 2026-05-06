import pandas as pd

# Charger les deux sources
df_alert = pd.read_csv('alert_results_2025.csv')
df_lines = pd.read_csv('data_lines.csv')

# Vérifier la structure de alert_results
print('=== ALERT_RESULTS ===')
print('Shape:', df_alert.shape)
print('Colonnes:', list(df_alert.columns))

# Quelle est la vraie clé unique?
print('\nCles potentielles:')
print('  ligne_label uniques:', df_alert['ligne_label'].nunique())
print('  (programme + ligne_label) uniques:', df_alert.groupby(['programme', 'ligne_label']).ngroups)
print('  (programme + ligne_label + region_label) uniques:', df_alert.groupby(['programme', 'ligne_label', 'region_label']).ngroups)

# Afficher un doublon
print('\nExemple de doublon ligne_label:')
dup = df_alert[df_alert['ligne_label'].duplicated(keep=False)].iloc[:4]
print(dup[['programme', 'ligne_label', 'region_label', 'lf_mdh']])

print('\n=== DATA_LINES ===')
print('Shape:', df_lines.shape)
print('Unique line_id:', df_lines['line_id'].nunique())
print('Years per line:', df_lines['year'].nunique())
print('Quarters per year:', df_lines['quarter'].nunique())
print('Rows per line_id:', df_lines.groupby('line_id').size().value_counts())

# Vérifier la correspondance
print('\n=== CORRESPONDANCE ===')
unique_line_ids_in_alert = set(df_alert['line_id'].dropna().unique()) if 'line_id' in df_alert.columns else set()
print('line_id dans alert_results:', len(unique_line_ids_in_alert))

unique_line_ids_in_lines = set(df_lines['line_id'].unique())
print('line_id dans data_lines:', len(unique_line_ids_in_lines))

missing = unique_line_ids_in_lines - unique_line_ids_in_alert
print('Manquant dans alert_results:', missing if missing else 'Rien')
