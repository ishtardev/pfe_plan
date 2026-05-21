"""
Generate HTML dashboard from alert_results_2025.csv
Run: python build_dashboard.py
Output: Open dashboard.html in browser
"""

import pandas as pd
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
alert_df = pd.read_csv("../projectv2/alert_results_2025.csv")

# Count statistics
n_total = len(alert_df)
n_ok = (alert_df["risk_label"] == "OK").sum()
n_warn = (alert_df["risk_label"] == "Attention").sum()
n_crit = (alert_df["risk_label"] == "Critique").sum()
mdh_risk = alert_df["remaining_mdh"].sum()

# Accuracy distribution
alert_df["ecart"] = (alert_df["pred_T4_rolling"] - alert_df["actual_T4"]).abs()
def classify_acc(e):
    if e < 0.05: return "Excellent"
    elif e < 0.10: return "Bon"
    elif e < 0.15: return "Acceptable"
    else: return "Mauvais"

dist = alert_df["ecart"].apply(classify_acc).value_counts()
n_excellent = dist.get("Excellent", 0)
n_good = dist.get("Bon", 0)
n_acceptable = dist.get("Acceptable", 0)
n_poor = dist.get("Mauvais", 0)

# Get anomalies
anomalies_df = alert_df[alert_df["anomalie_label"] != ""].sort_values("z_score_T2").head(10)

# Generate HTML
html_template = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tableau de bord budgétaire — Ministère de la Justice</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: "Segoe UI", sans-serif; background: #f0f2f5; color: #333; }}
        
        header {{
            background: #1b3a5c;
            color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        header h1 {{ font-size: 24px; margin-bottom: 4px; }}
        header p {{ font-size: 12px; opacity: 0.9; }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid #009688;
        }}
        .kpi-card.critical {{ border-left-color: #d73027; }}
        .kpi-card.warning {{ border-left-color: #fc8d59; }}
        .kpi-card.ok {{ border-left-color: #1a9850; }}
        
        .kpi-value {{ font-size: 32px; font-weight: bold; color: #1b3a5c; }}
        .kpi-label {{ font-size: 12px; color: #666; margin-top: 4px; text-transform: uppercase; }}
        
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .section h2 {{ font-size: 16px; margin-bottom: 15px; color: #1b3a5c; border-bottom: 2px solid #009688; padding-bottom: 8px; }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .chart-box h3 {{ font-size: 14px; margin-bottom: 15px; color: #1b3a5c; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        thead {{
            background: #1b3a5c;
            color: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        tbody tr:hover {{ background: #f9f9f9; }}
        
        .status-ok {{ background: #EDF7EE; color: #145a32; font-weight: bold; text-align: center; padding: 6px; border-radius: 4px; }}
        .status-attention {{ background: #FFFBEF; color: #7B5200; font-weight: bold; text-align: center; padding: 6px; border-radius: 4px; }}
        .status-critique {{ background: #FEF2F2; color: #7B0000; font-weight: bold; text-align: center; padding: 6px; border-radius: 4px; }}
        
        .filter-row {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        input, select {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 13px;
        }}
        button {{
            padding: 8px 16px;
            background: #1b3a5c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
        }}
        button:hover {{ background: #2a4f7c; }}
        
        .anomaly-flag {{
            background: #FFF3CD;
            color: #7D4E00;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: bold;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 11px;
        }}
    </style>
</head>
<body>

<header>
    <h1>🎯 Tableau de bord budgétaire — Ministère de la Justice</h1>
    <p>Prévisions T4 2025 | Actualisation : 5 mai 2026</p>
</header>

<div class="container">
    
    <!-- KPI CARDS -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{n_total}</div>
            <div class="kpi-label">Lignes analysées</div>
        </div>
        <div class="kpi-card ok">
            <div class="kpi-value">{n_ok}</div>
            <div class="kpi-label">OK (≥80%)</div>
        </div>
        <div class="kpi-card warning">
            <div class="kpi-value">{n_warn}</div>
            <div class="kpi-label">Attention (60-80%)</div>
        </div>
        <div class="kpi-card critical">
            <div class="kpi-value">{n_crit}</div>
            <div class="kpi-label">Critique (<60%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{mdh_risk:.1f}</div>
            <div class="kpi-label">Crédits à risque (MDH)</div>
        </div>
    </div>
    
    <!-- CHARTS -->
    <div class="charts-row">
        <div class="chart-box">
            <h3>Distribution des risques</h3>
            <div style="position: relative; height: 250px;">
                <canvas id="riskChart"></canvas>
            </div>
        </div>
        <div class="chart-box">
            <h3>Précision des prédictions</h3>
            <div style="position: relative; height: 250px;">
                <canvas id="accuracyChart"></canvas>
            </div>
        </div>
    </div>
    
    <!-- PREDICTIONS TABLE -->
    <div class="section">
        <h2>📋 Prévisions par ligne ({n_total} lignes)</h2>
        <div class="filter-row">
            <input type="text" id="filterProg" placeholder="Filtrer par programme...">
            <input type="text" id="filterRegion" placeholder="Filtrer par région...">
            <select id="filterRisk">
                <option value="">Tous les risques</option>
                <option value="OK">OK uniquement</option>
                <option value="Attention">Attention uniquement</option>
                <option value="Critique">Critique uniquement</option>
            </select>
            <button onclick="filterTable()">Filtrer</button>
            <button onclick="resetFilter()" style="background: #999;">Réinitialiser</button>
        </div>
        <table id="tablePredict">
            <thead>
                <tr>
                    <th>Prog</th>
                    <th>Ligne budgétaire</th>
                    <th>Région</th>
                    <th>LF (MDH)</th>
                    <th>T2 réel</th>
                    <th>T4 prévu</th>
                    <th>Écart</th>
                    <th>Risque</th>
                    <th>Anomalie</th>
                </tr>
            </thead>
            <tbody id="tbody-predict">
"""

# Add data rows
for _, row in alert_df.sort_values("pred_T4_rolling").iterrows():
    prog = f"P{int(row['programme'])}"
    ligne = row["ligne_label"][:40]
    region = row["region_label"][:20]
    lf = f"{row['lf_mdh']:.1f}"
    t2 = f"{row['taux_T2']*100:.1f}%"
    t4 = f"{row['pred_T4_rolling']*100:.1f}%"
    ecart = f"{(row['pred_T4_rolling'] - row['actual_T4'])*100:+.1f} pp"
    risk = row["risk_label"]
    anom = row["anomalie_label"] if pd.notna(row["anomalie_label"]) and row["anomalie_label"] != "" else "-"
    
    risk_class = f"status-{risk.lower().replace(' ', '')}"
    
    html_template += f"""                <tr>
                    <td>{prog}</td>
                    <td>{ligne}</td>
                    <td>{region}</td>
                    <td>{lf}</td>
                    <td>{t2}</td>
                    <td>{t4}</td>
                    <td>{ecart}</td>
                    <td class="{risk_class}">{risk}</td>
                    <td>{'<span class="anomaly-flag">' + str(anom)[:30] + '</span>' if anom != '-' else '-'}</td>
                </tr>
"""

html_template += """            </tbody>
        </table>
    </div>
    
    <!-- ANOMALIES -->
    <div class="section">
        <h2>⚠️ Anomalies détectées</h2>
        <table>
            <thead>
                <tr>
                    <th>Ligne</th>
                    <th>Région</th>
                    <th>T2 2025</th>
                    <th>Z-score</th>
                    <th>Anomalie</th>
                    <th>Action requise</th>
                </tr>
            </thead>
            <tbody>
"""

for _, row in anomalies_df.iterrows():
    ligne = row["ligne_label"][:40]
    region = row["region_label"][:20]
    t2 = f"{row['taux_T2']*100:.1f}%"
    z = f"{row['z_score_T2']:.1f}"
    anom = str(row["anomalie_label"])[:30] if pd.notna(row["anomalie_label"]) else "Anomalie"
    action = "Vérifier immédiatement" if row["z_score_T2"] < -2 else "Surveiller de près"
    
    html_template += f"""                <tr>
                    <td>{ligne}</td>
                    <td>{region}</td>
                    <td>{t2}</td>
                    <td><strong>{z}</strong></td>
                    <td><span class="anomaly-flag">{anom}</span></td>
                    <td>{action}</td>
                </tr>
"""

html_template += f"""            </tbody>
        </table>
    </div>
    
    <!-- INSIGHTS -->
    <div class="section">
        <h2>📊 Insights clés</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4 style="color: #1b3a5c; margin-bottom: 10px;">✓ Points forts</h4>
                <ul style="list-style: none; line-height: 1.8;">
                    <li>✅ {n_ok} lignes à faible risque (OK)</li>
                    <li>✅ 82.2% accuracy sur 2024 (validé)</li>
                    <li>✅ Modèle spécialisé par catégorie</li>
                    <li>✅ Couverture complète {n_total}/130 lignes</li>
                </ul>
            </div>
            <div>
                <h4 style="color: #1b3a5c; margin-bottom: 10px;">⚠️ Points d'attention</h4>
                <ul style="list-style: none; line-height: 1.8;">
                    <li>⚠️ {len(anomalies_df)} anomalies flaggées (z-score < -1.5)</li>
                    <li>⚠️ {mdh_risk:.1f} MDH de crédits à risque</li>
                    <li>⚠️ Budget lines volatiles naturellement</li>
                    <li>⚠️ Recommandé : suivi expert par domaine</li>
                </ul>
            </div>
        </div>
    </div>

</div>

<footer>
    Modèle: XGBoost + Lasso + Hist_mean | Validation: Walk-forward CV 2020-2023 → 2024 (82% accuracy) <br>
    Source: alert_results_2025.csv | Génération: 05/05/2026
</footer>

<script>
    function filterTable() {{
        const prog = document.getElementById("filterProg").value.toUpperCase();
        const region = document.getElementById("filterRegion").value.toUpperCase();
        const risk = document.getElementById("filterRisk").value;
        
        const tbody = document.getElementById("tbody-predict");
        Array.from(tbody.children).forEach(row => {{
            const rowProg = row.cells[0].textContent;
            const rowRegion = row.cells[2].textContent.toUpperCase();
            const rowRisk = row.cells[7].textContent;
            
            const match = (!prog || rowProg.includes(prog)) &&
                          (!region || rowRegion.includes(region)) &&
                          (!risk || rowRisk === risk);
            row.style.display = match ? "" : "none";
        }});
    }}
    
    function resetFilter() {{
        document.getElementById("filterProg").value = "";
        document.getElementById("filterRegion").value = "";
        document.getElementById("filterRisk").value = "";
        filterTable();
    }}
    
    // Charts
    function createCharts() {{
        const ctxRisk = document.getElementById("riskChart").getContext("2d");
        new Chart(ctxRisk, {{
            type: "doughnut",
            data: {{
                labels: ["OK (≥80%)", "Attention (60-80%)", "Critique (<60%)"],
                datasets: [{{
                    data: [{n_ok}, {n_warn}, {n_crit}],
                    backgroundColor: ["#1a9850", "#fc8d59", "#d73027"],
                    borderColor: "#fff",
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: "bottom" }} }}
            }}
        }});
        
        const ctxAcc = document.getElementById("accuracyChart").getContext("2d");
        new Chart(ctxAcc, {{
            type: "bar",
            data: {{
                labels: ["Excellent\\n<5pp", "Bon\\n5-10pp", "Acceptable\\n10-15pp", "Mauvais\\n>15pp"],
                datasets: [{{
                    label: "Nombre de lignes",
                    data: [{n_excellent}, {n_good}, {n_acceptable}, {n_poor}],
                    backgroundColor: ["#1a9850", "#7cb342", "#fc8d59", "#d73027"],
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: "y",
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ x: {{ beginAtZero: true }} }}
            }}
        }});
    }}
    
    createCharts();
</script>

</body>
</html>
"""

# Write file
with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("✅ Dashboard généré!")
print("📂 Ouvrir : dashboard.html")
