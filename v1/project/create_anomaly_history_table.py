import pandas as pd
import os

os.chdir("../projectv2")

# Load files
alert = pd.read_excel('alert_results_2025.xlsx')
historical = pd.read_excel('historical_recap_2025.xlsx')

# Get only anomalies
anomalies = alert[alert['anomalie_label'].notna()][['line_id', 'ligne_label', 'region_label', 'z_score_T2', 'anomalie_label']].copy()

print(f"Found {len(anomalies)} anomalies")
print(anomalies)

# For each anomaly, get its historical T2 data (2020-2025)
result = []

for _, anom_row in anomalies.iterrows():
    line_id = anom_row['line_id']
    ligne_label = anom_row['ligne_label']
    region = anom_row['region_label']
    z_score = anom_row['z_score_T2']
    anomaly_type = anom_row['anomalie_label']
    
    # Get historical data for this line
    hist_data = historical[historical['line_id'] == line_id][['year', 'taux_T2']].sort_values('year')
    
    # Add a row for each year
    for _, hist_row in hist_data.iterrows():
        result.append({
            'line_id': line_id,
            'ligne_label': ligne_label,
            'region_label': region,
            'anomalie_label': anomaly_type,
            'z_score_2025': z_score,
            'year': int(hist_row['year']),
            'taux_T2': round(hist_row['taux_T2'], 4)
        })

# Convert to dataframe
anomaly_history = pd.DataFrame(result)

# Save to Excel
anomaly_history.to_excel('anomalies_with_history.xlsx', index=False)

print(f"\n✓ Created anomalies_with_history.xlsx")
print(f"  Rows: {len(anomaly_history)}")
print(f"  Columns: {anomaly_history.columns.tolist()}")
print(f"\nPreview:")
print(anomaly_history)
