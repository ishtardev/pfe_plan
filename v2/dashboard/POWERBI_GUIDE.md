# Power BI Integration Guide

Companion document for `predictions_for_powerbi.xlsx` (downloaded from the HTML app via *Télécharger tout (Power BI)*).

This is a build recipe — paste each block into Power BI Desktop and the report
will look like the screenshots in the thesis.

---

## 1. Data source

In Power BI Desktop:

1. **Home → Get data → Excel workbook** → select `predictions_for_powerbi.xlsx`.
2. Tick **all six sheets**:
   - `Forecast` (one row per ligne — main fact table)
   - `Backtest` (one row per (ligne × method) — accuracy)
   - `Leaderboard` (one row per method — summary)
   - `StablePanel` (long-format history, one row per (ligne × year))
   - `Anomalies_Extraction` (extraction-rule flags)
   - `Anomalies_Business` (business-rule flags)
3. Click **Transform Data** to open Power Query.

---

## 2. Power Query (M) transformations

Paste these into the Advanced Editor of each query.

### Forecast

```m
let
    Source = Excel.Workbook(File.Contents(DataFile), null, true),
    Sheet = Source{[Item="Forecast",Kind="Sheet"]}[Data],
    Promoted = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"Line_Key", type text}, {"Chapitre", type text},
        {"Programme", type text}, {"Region", type text},
        {"Intitule", type text}, {"Best_Method", type text},
        {"Backtest_Winner_Method", type text},
        {"Forecast_2027", Int64.Type}, {"Forecast_Low", Int64.Type},
        {"Forecast_High", Int64.Type}, {"Conformal_Q", type number},
        {"Bias_Scale", type number}, {"Delta_vs_ref_pct", type number}
    }),
    AddBandWidth = Table.AddColumn(Typed, "Band_Width_pct",
        each if [Forecast_2027] > 0
             then ([Forecast_High] - [Forecast_Low]) / [Forecast_2027] * 100
             else null, type number)
in
    AddBandWidth
```

### StablePanel (long → keep as-is)

```m
let
    Source = Excel.Workbook(File.Contents(DataFile), null, true),
    Sheet = Source{[Item="StablePanel",Kind="Sheet"]}[Data],
    Promoted = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(Promoted, {
        {"Year", Int64.Type}, {"Line_Key", type text},
        {"Total_Engage_Vises", Int64.Type},
        {"Credits_Ouverts_Vises", Int64.Type},
        {"Total_Credits", Int64.Type}
    })
in
    Typed
```

### Backtest / Leaderboard / Anomalies\_\*

Use the same pattern: load sheet → promote headers → set types. Set `Method`
and `Rule` columns as `text`, `sMAPE_pct` and `Abs_Error` as `number`.

### Parameter

Create a Power Query parameter named **`DataFile`** (type Text) with the
default value `C:\path\to\predictions_for_powerbi.xlsx`. Reference it
everywhere with `File.Contents(DataFile)` (already done above). End users
can then point to their local copy without editing the M code.

---

## 3. Data model

| Relationship | From → To | Cardinality |
|---|---|---|
| `Forecast[Line_Key] → StablePanel[Line_Key]` | One-to-many | Single-direction |
| `Forecast[Line_Key] → Backtest[Line_Key]` | One-to-many | Single-direction |
| `Forecast[Line_Key] → Anomalies_Extraction[Line_Key]` | One-to-many | Single |
| `Forecast[Line_Key] → Anomalies_Business[Line_Key]` | One-to-many | Single |
| `Leaderboard[Method] → Backtest[Method]` | One-to-many | Single |

Mark `StablePanel[Year]` as a date hierarchy parent if you want to slice by
year easily — or create a separate Calendar table.

---

## 4. DAX measures

Paste each block in the **Modeling → New measure** dialog. Group them in a
*Measures* table for tidiness.

### Core KPIs

```dax
Total Forecast 2027 =
SUM ( Forecast[Forecast_2027] )

Total Engage 2025 =
CALCULATE (
    SUM ( StablePanel[Total_Engage_Vises] ),
    StablePanel[Year] = 2025
)

Growth vs 2025 % =
DIVIDE (
    [Total Forecast 2027] - [Total Engage 2025],
    [Total Engage 2025]
) * 100
```

### Accuracy

```dax
Median sMAPE =
MEDIANX ( Backtest, Backtest[sMAPE_pct] )

Lines with CI =
CALCULATE (
    COUNTROWS ( Forecast ),
    NOT ISBLANK ( Forecast[Forecast_Low] )
)

Lines flagged extraction =
DISTINCTCOUNT ( Anomalies_Extraction[Line_Key] )

Lines flagged business =
DISTINCTCOUNT ( Anomalies_Business[Line_Key] )
```

### Conformal band readout

```dax
Conformal Q =
MAX ( Forecast[Conformal_Q] )

Avg band width % =
AVERAGEX (
    FILTER ( Forecast, Forecast[Forecast_2027] > 0 ),
    DIVIDE (
        Forecast[Forecast_High] - Forecast[Forecast_Low],
        Forecast[Forecast_2027]
    ) * 100
)
```

---

## 5. Report layout (5 pages)

### Page 1 — *Synthèse executive*

- 4 KPI cards (top row): `Total Forecast 2027`, `Total Engage 2025`,
  `Growth vs 2025 %`, `Median sMAPE`.
- Donut chart: `Forecast 2027` by `Chapitre`.
- Map / treemap by `Region` (drill to `Programme`).
- Slicer: `Chapitre`, `Region`.

### Page 2 — *Forecast par ligne*

- Matrix visual: rows = `Chapitre` → `Programme` → `Intitule`. Values =
  `Engage 2025`, `Forecast 2027`, `Forecast_Low`, `Forecast_High`,
  `Delta_vs_ref_pct`, `Best_Method`.
- Conditional formatting on `Delta_vs_ref_pct` (red < -20%, green > +20%).
- Drill-through page → page 5.

### Page 3 — *Précision du modèle*

- Horizontal bar: `Median sMAPE` by `Method` (from `Leaderboard`).
- Histogram: count of lignes by sMAPE bucket (`Backtest` table filtered on
  the winning method).
- Scatter: x = `Actual_2025`, y = predicted (need to build this from
  `Backtest` cross-joined with `StablePanel`).

### Page 4 — *Anomalies*

- Table from `Anomalies_Business` with `Severity` icon column.
- Filter pane on `Rule`, `Severity`.
- Card: `Lines flagged business`.

### Page 5 — *Détail ligne* (drill-through target)

- Drill-through field: `Forecast[Line_Key]`.
- Line chart: `Total_Engage_Vises` by `Year` from `StablePanel` + an
  annotation point for `Forecast_2027` with error bars (Lo/Hi).
- Card: `Best_Method`, `Conformal Q`.

---

## 6. Refresh workflow

1. Re-run the HTML app with a new SitGen workbook.
2. Click *Télécharger tout (Power BI)* → overwrite the local
   `predictions_for_powerbi.xlsx`.
3. In Power BI Desktop: **Home → Refresh** (or set a scheduled refresh once
   the file lives on OneDrive/SharePoint).

The DataFile parameter is the only thing that changes between users; the
queries and model are fixed.

---

## 7. Optional: theme

Save the following as `ministry_theme.json` then **View → Themes → Browse**:

```json
{
  "name": "Ministère de la Justice",
  "dataColors": ["#0a3d62", "#1e88e5", "#10b981", "#f59e0b", "#dc2626",
                 "#6b7280", "#7c3aed", "#0891b2"],
  "background": "#ffffff",
  "foreground": "#111827",
  "tableAccent": "#0a3d62"
}
```
