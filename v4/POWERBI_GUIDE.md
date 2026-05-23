# Power BI Dashboard Guide

## Files to load

Go to **Get Data → Excel Workbook** and load these 2 files from `03_extracted/`:

| File | Sheets to import |
|---|---|
| `SituationChap_RAPPORT_COMPLET.xlsx` | `Récapitulatif`, `Données` |
| `SituationChap_ANOMALIES.xlsx` | `Anomalies_Annuelles`, `Lignes_Anormales` |

> In the Navigator window, only check the sheets listed above — ignore everything else.

---

## Page 1 — Vue d'ensemble

**Source:** `Données` sheet

This sheet is already in long format (one row per budget line × year), perfect for Power BI.

| Visual | Type | Fields |
|---|---|---|
| KPI cards (3x) | Card | `CO` (sum), `Engage_Reel` (sum), `Taux_Reel` (average) |
| Budget par chapitre | Bar chart | X: `Chap_Intitule`, Y: `CO` (sum) |
| Évolution du taux | Line chart | X: `Année`, Y: `Taux_Reel` (average) |
| Filtre année | Slicer | `Année` |
| Filtre chapitre | Slicer | `Chap_Intitule` |

> Tip: add a filter on the KPI cards to show only the most recent year (2026) by default.

---

## Page 2 — Prévisions

**Source:** `Récapitulatif` sheet

| Visual | Type | Fields |
|---|---|---|
| Prévisions par chapitre | Bar chart | X: `Chap_Intitule`, Y: `Engage_Prévu` (sum) |
| Tableau détaillé | Table | `Intitule`, `Année_Prévision`, `Engage_Prévu`, `Int_Bas`, `Int_Haut`, `Fiabilite`, `Modele_Utilise` |
| Filtre fiabilité | Slicer | `Fiabilite` (values: Fiable / Acceptable / Incertain) |
| Filtre année prévue | Slicer | `Année_Prévision` (values: 2026 / 2027 / 2028) |

> Tip: color the `Fiabilite` column using conditional formatting:
> - Fiable → green
> - Acceptable → orange  
> - Incertain → red

---

## Page 3 — Anomalies

**Source:** `Anomalies_Annuelles` and `Lignes_Anormales` sheets

| Visual | Type | Fields | Note |
|---|---|---|---|
| Nb anomalies temporelles | Card | Count of rows where `Anomalie_Annee = Oui` | Add filter on visual |
| Nb lignes anormales (IF) | Card | Count of rows where `Est_Anomalie_IF = Oui` | Add filter on visual |
| Tableau Z-score | Table | `Intitule`, `Year`, `Taux_Engagement`, `Z_Score`, `Anomalie_Annee` | Filter: `Anomalie_Annee = Oui` |
| Tableau Isolation Forest | Table | `Intitule`, `Rang_Anomalie`, `Score_Anomalie`, `Est_Anomalie_IF` | Sort by `Rang_Anomalie` ascending |

> Tip: on the Z-score table, use conditional formatting on `Z_Score` — a diverging color scale (blue → white → red) makes it very readable.

---

## Relationships (optional but useful)

If you want slicers on Page 1 to filter across pages, link the tables by `Line_Key`:
- `Anomalies_Annuelles[Line_Key]` → `Lignes_Anormales[Line_Key]`

The `Données` and `Récapitulatif` sheets share `Chap`, `Prog`, `Reg`, `Proj`, `Lb` — you can create a relationship on a concatenated key if needed, but for a thesis dashboard it's not required.

---

## Column reference

### `Données` (historique)
| Column | Description |
|---|---|
| `Année` | Year (2021–2026) |
| `Type` | historique or prévision |
| `CO` | Crédits Ouverts |
| `Engage_Reel` | Engagement réel |
| `Taux_Reel` | Taux d'engagement réel |
| `Prediction` | Valeur prédite (for validation rows) |
| `MAPE_Pct` | Error % between predicted and real |

### `Récapitulatif` (prévisions)
| Column | Description |
|---|---|
| `Année_Prévision` | 2026, 2027, or 2028 |
| `Engage_Prévu` | Predicted engagement |
| `Int_Bas` / `Int_Haut` | 95% confidence interval |
| `Fiabilite` | Fiable / Acceptable / Incertain |
| `Stabilite_Taux` | How stable the engagement rate is historically |
| `Confiance_Globale` | Overall confidence score |

### `Anomalies_Annuelles` (Z-score)
| Column | Description |
|---|---|
| `Year` | Year of the anomaly |
| `Taux_Engagement` | Engagement rate that year |
| `Z_Score` | How many std deviations from the line's average |
| `Anomalie_Annee` | Oui / Non |

### `Lignes_Anormales` (Isolation Forest)
| Column | Description |
|---|---|
| `Score_Anomalie` | More negative = more abnormal |
| `Rang_Anomalie` | 1 = most abnormal line overall |
| `Est_Anomalie_IF` | Oui / Non |
