import pandas as pd
import os

os.chdir("../projectv2")

# Load all data
alert = pd.read_excel('alert_results_2025.xlsx')
historical = pd.read_excel('historical_recap_2025.xlsx')

print("Creating comprehensive master table...")

# Calculate historical statistics per ligne
hist_stats = historical.groupby('line_id').agg({
    'taux_T2': ['mean', 'std', 'min', 'max'],
    'actual_T4': ['mean', 'std'],
    'year': 'count'
}).reset_index()

# Flatten column names
hist_stats.columns = ['line_id', 'mean_T2_historical', 'std_T2_historical', 
                      'min_T2_historical', 'max_T2_historical',
                      'mean_T4_historical', 'std_T4_historical',
                      'num_years_data']

# Get 2025 historical data
hist_2025 = historical[historical['year'] == 2025][['line_id', 'taux_T2', 'actual_T4']].rename(
    columns={'taux_T2': 'T2_2025_actual', 'actual_T4': 'T4_2025_actual'}
)

# Merge everything
master = alert.copy()
master = master.merge(hist_stats, on='line_id', how='left')
master = master.merge(hist_2025, on='line_id', how='left')

# Add column descriptions for clarity
master.rename(columns={
    'taux_T2': 'T2_2025_predicted',
    'pred_T4_direct': 'T4_2025_pred_direct_method',
    'pred_T4_rolling': 'T4_2025_pred_rolling_method',
    'z_score_T2': 'z_score_T2_2025',
    'actual_T4': 'T4_2025_realized',
    'ecart_rolling': 'error_rolling_method_pct',
    'risk_label': 'risk_category',
    'anomalie_label': 'anomaly_type'
}, inplace=True)

# Reorder columns logically
cols_order = [
    # Identifiers
    'line_id', 'ligne_label', 'programme', 'region_label', 'type_ligne',
    # Budget category
    'budget_type',
    # Budget size
    'lf_mdh',
    # 2025 Actual vs Predictions
    'T2_2025_actual', 'T2_2025_predicted', 'T4_2025_realized', 
    'T4_2025_pred_direct_method', 'T4_2025_pred_rolling_method',
    # Technical ML metrics
    'z_score_T2_2025', 'error_rolling_method_pct', 'risk_category', 'anomaly_type',
    # Historical statistics
    'mean_T2_historical', 'std_T2_historical', 'min_T2_historical', 'max_T2_historical',
    'mean_T4_historical', 'std_T4_historical', 'num_years_data',
    # Remaining columns
    'remaining_mdh'
]

# Only include columns that exist
cols_order = [c for c in cols_order if c in master.columns]
master = master[cols_order]

# Save to Excel
master.to_excel('master_table_complete.xlsx', index=False, sheet_name='Master Data')

print(f"\n✓ Created master_table_complete.xlsx")
print(f"  Rows: {len(master)}")
print(f"  Columns: {len(master.columns)}")
print(f"\nColumn list:")
for i, col in enumerate(master.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSample row:")
print(master.iloc[0])
