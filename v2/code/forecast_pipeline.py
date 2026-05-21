"""
forecast_pipeline.py
====================
End-to-end forecast pipeline driven by the manager.

Usage:
    python forecast_pipeline.py --target-year 2027 --horizon 3 [--skip-convert]

What it does:
    1. (optional) Converts raw .xls to .xlsx (skip if files are already .xlsx).
    2. Extracts the budget data from v2/data/01_raw/*.xlsx into a clean panel.
    3. Auto-detects which years of history are available.
    4. Runs anomaly detection (rules + IsolationForest).
    5. Runs a per-line backtest tournament across 5 methods
       (Naive, Mean3, Trend, Ridge, XGB-global) and picks the best per line.
    6. Forecasts target_year, target_year+1, ..., target_year+horizon-1
       recursively (each year's prediction feeds the next).
    7. Writes ONE Excel file with three sheets:
         - Synthese        : totals by year × programme + grand totals
         - Detail_Lignes   : per-line forecast for every horizon year
         - Methodes        : per-line best method + backtest sMAPE + ML risk

Output file:
    v2/data/04_manager_output/Forecast_Triennale_<start>_<end>.xlsx
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "01_raw"
PANEL = DATA / "02_cleaned" / "SituationChap_STABLE_PANEL.xlsx"
OUT_DIR = DATA / "04_manager_output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
CODE = Path(__file__).resolve().parent


# =====================================================================
# 1. Forecast methods (each takes a Series indexed by Year, returns float)
# =====================================================================
def m_naive(hist: pd.Series, target_year: int) -> float:
    return float(hist.iloc[-1])


def m_mean3(hist: pd.Series, target_year: int) -> float:
    return float(hist.tail(3).mean())


def m_trend(hist: pd.Series, target_year: int) -> float:
    x = hist.index.values.astype(float).reshape(-1, 1)
    y = hist.values.astype(float)
    if len(x) < 2:
        return float(y[-1])
    model = LinearRegression().fit(x, y)
    return max(float(model.predict([[target_year]])[0]), 0.0)


def m_ridge(hist: pd.Series, target_year: int) -> float:
    years = hist.index.values.astype(int)
    vals = hist.values.astype(float)
    Xs, ys = [], []
    for i in range(2, len(vals)):
        Xs.append([vals[i - 1], vals[i - 2], vals[:i].mean(),
                   np.polyfit(years[:i], vals[:i], 1)[0]])
        ys.append(vals[i])
    if len(Xs) < 2:
        return m_trend(hist, target_year)
    model = Ridge(alpha=1.0).fit(np.array(Xs), np.array(ys))
    feat = [[vals[-1], vals[-2], vals.mean(), np.polyfit(years, vals, 1)[0]]]
    return max(float(model.predict(feat)[0]), 0.0)


# XGB lookup populated globally before tournament + forecast
XGB_CACHE: dict[tuple[str, int], float] = {}


def m_xgb(hist: pd.Series, target_year: int) -> float:
    key = getattr(m_xgb, "_current_key", None)
    if key is not None and (key, target_year) in XGB_CACHE:
        return XGB_CACHE[(key, target_year)]
    return m_trend(hist, target_year)


METHODS = {"Naive": m_naive, "Mean3": m_mean3, "Trend": m_trend, "Ridge": m_ridge}
if HAS_XGB:
    METHODS["XGB"] = m_xgb


# =====================================================================
# 2. Metrics
# =====================================================================
def smape(actual: float, pred: float) -> float:
    d = (abs(actual) + abs(pred)) / 2
    return 0.0 if d == 0 else 100 * abs(actual - pred) / d


# =====================================================================
# 3. Global XGB training (one model per target year)
# =====================================================================
def train_xgb(lignes: pd.DataFrame, target_year: int) -> None:
    if not HAS_XGB:
        return
    line_ids = {k: i for i, k in enumerate(sorted(lignes["Line_Key"].unique()))}
    rows_train, rows_pred = [], []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub[sub.Year < target_year].sort_values("Year")
        if len(sub) < 3:
            continue
        vals = sub["Total_Engage_Vises"].values.astype(float)
        years = sub["Year"].values.astype(int)
        lid = line_ids[key]
        for i in range(2, len(vals)):
            rows_train.append([lid, years[i], vals[i - 1], vals[i - 2],
                               vals[:i].mean(), np.polyfit(years[:i], vals[:i], 1)[0],
                               vals[i]])
        rows_pred.append([key, lid, target_year, vals[-1], vals[-2], vals.mean(),
                          np.polyfit(years, vals, 1)[0]])
    if not rows_train:
        return
    train = pd.DataFrame(rows_train, columns=["lid", "yr", "l1", "l2", "mp", "sl", "y"])
    pred  = pd.DataFrame(rows_pred,  columns=["Line_Key", "lid", "yr", "l1", "l2", "mp", "sl"])
    feat = ["lid", "yr", "l1", "l2", "mp", "sl"]
    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.9, reg_lambda=1.0, random_state=42, verbosity=0)
    model.fit(train[feat], train["y"])
    preds = model.predict(pred[feat])
    for k, p in zip(pred["Line_Key"], preds):
        XGB_CACHE[(k, target_year)] = max(float(p), 0.0)


# =====================================================================
# 4. Tournament: pick best method per line on backtest
# =====================================================================
def run_tournament(lignes: pd.DataFrame, complete_years: list[int]) -> pd.DataFrame:
    """Backtest the 2 most recent complete years; pick lowest mean sMAPE per line."""
    validation_years = complete_years[-2:]
    records = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        intitule = sub["Intitule"].iloc[-1]
        for vy in validation_years:
            hist = sub.loc[sub.index < vy, "Total_Engage_Vises"]
            if len(hist) < 2 or vy not in sub.index:
                continue
            actual = float(sub.loc[vy, "Total_Engage_Vises"])
            for name, fn in METHODS.items():
                try:
                    m_xgb._current_key = key
                    pred = fn(hist, vy)
                except Exception:
                    continue
                records.append({"Line_Key": key, "Intitule": intitule,
                                "Method": name, "Year": vy, "sMAPE": smape(actual, pred)})
    val = pd.DataFrame(records)
    if val.empty:
        return pd.DataFrame(columns=["Line_Key", "Best_Method", "Backtest_sMAPE"])
    agg = val.groupby(["Line_Key", "Method"])["sMAPE"].mean().reset_index()
    idx = agg.groupby("Line_Key")["sMAPE"].idxmin()
    return (agg.loc[idx]
               .rename(columns={"Method": "Best_Method", "sMAPE": "Backtest_sMAPE"})
               .reset_index(drop=True))


# =====================================================================
# 5. Multi-year forecast (recursive)
# =====================================================================
def forecast_horizons(lignes: pd.DataFrame, best: pd.DataFrame,
                      target_years: list[int]) -> pd.DataFrame:
    """For each line, forecast each target year. Predictions for year t feed t+1."""
    best_map = dict(zip(best["Line_Key"], best["Best_Method"]))
    rows = []
    for key, sub in lignes.groupby("Line_Key"):
        sub = sub.sort_values("Year").set_index("Year")
        intitule = sub["Intitule"].iloc[-1]
        # Line_Key format: Chap-Prog-Reg-Proj-Lb  -> derive Chap/Prog from key
        parts = str(key).split("-")
        chap = parts[0] if len(parts) >= 1 else ""
        prog = parts[1] if len(parts) >= 2 else ""
        hist = sub["Total_Engage_Vises"].copy()
        method_name = best_map.get(key, "Naive")
        fn = METHODS.get(method_name, m_naive)
        row = {"Line_Key": key, "Intitule": intitule, "Chap": chap, "Prog": prog,
               "Best_Method": method_name,
               "Last_Year": int(hist.index.max()),
               "Last_Engage": float(hist.iloc[-1])}
        for ty in target_years:
            m_xgb._current_key = key
            try:
                p = fn(hist, ty)
            except Exception:
                p = float(hist.iloc[-1])
            row[f"Forecast_{ty}"] = p
            # Append prediction to history so next horizon uses it (recursive)
            hist = pd.concat([hist, pd.Series([p], index=[ty])])
        rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# 6. Anomaly score (IsolationForest only — lightweight)
# =====================================================================
def anomaly_scores(lignes: pd.DataFrame) -> pd.DataFrame:
    df = lignes.copy()
    eps = 1e-9
    df["taux_engagement"] = df["Total_Engage_Vises"] / (df["Credits_Ouverts_Vises"] + eps)
    df = df.sort_values(["Line_Key", "Year"])
    df["delta_yoy"] = df.groupby("Line_Key")["Total_Engage_Vises"].pct_change().fillna(0.0)
    df["volatility"] = df.groupby("Line_Key")["taux_engagement"].transform("std").fillna(0.0)
    feats = ["taux_engagement", "delta_yoy", "volatility"]
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = StandardScaler().fit_transform(df[feats])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42).fit(X)
    df["iso_score"] = -iso.score_samples(X)
    # One score per Line_Key = max across years
    return (df.groupby("Line_Key")["iso_score"].max()
              .rename("ML_Risk_Score").reset_index())


# =====================================================================
# 7. Orchestration
# =====================================================================
def maybe_run_convert() -> None:
    """Run 01_convert if there are .xls files without a matching .xlsx."""
    xls = list(RAW.rglob("*.xls"))
    needs = [p for p in xls if not p.with_suffix(".xlsx").exists()]
    if not needs:
        print(f"[skip] convert step: all .xls already converted ({len(xls)} files)")
        return
    print(f"[run]  convert step: {len(needs)} .xls -> .xlsx")
    subprocess.run([sys.executable, str(CODE / "01_convert_xls_to_xlsx.py")], check=True)


def run_extract() -> None:
    print("[run]  extract step")
    subprocess.run([sys.executable, str(CODE / "02_extract_budget_data.py")], check=True)


def load_panel() -> tuple[pd.DataFrame, list[int]]:
    """Load the stable panel and return (lignes, complete_years)."""
    if not PANEL.exists():
        raise SystemExit(f"Panel not found: {PANEL}")
    panel = pd.read_excel(PANEL, dtype={"Chap": str, "Prog": str, "Reg": str,
                                        "Proj": str, "Lb": str})
    lignes = panel[panel.Level == "Ligne"].copy()
    # A year is "complete" if engagement total is materially > 0 across the dataset.
    # In practice the partial last year shows much lower totals; flag it.
    yearly_total = lignes.groupby("Year")["Total_Engage_Vises"].sum()
    median_total = yearly_total.median()
    complete = sorted(int(y) for y, v in yearly_total.items() if v >= 0.5 * median_total)
    print(f"[info] years available    : {sorted(lignes.Year.unique().tolist())}")
    print(f"[info] complete years     : {complete}")
    return lignes[lignes.Year.isin(complete)].copy(), complete


def build_synthese(detail: pd.DataFrame, target_years: list[int]) -> pd.DataFrame:
    forecast_cols = [f"Forecast_{y}" for y in target_years]
    by_prog = (detail.groupby(["Chap", "Prog"], dropna=False)[forecast_cols + ["Last_Engage"]]
                     .sum().round(0).reset_index())
    by_prog["Nb_Lignes"] = detail.groupby(["Chap", "Prog"]).size().values
    total = pd.DataFrame({"Chap": ["TOTAL"], "Prog": [""],
                          "Nb_Lignes": [len(detail)],
                          "Last_Engage": [detail["Last_Engage"].sum().round(0)]})
    for c in forecast_cols:
        total[c] = [detail[c].sum().round(0)]
    return pd.concat([by_prog, total], ignore_index=True)


def build_html_viewer(detail: pd.DataFrame, synthese: pd.DataFrame,
                      methodes: pd.DataFrame, target_years: list[int]) -> Path:
    """Render a self-contained HTML viewer of the triennale forecast.
    No internet, no dependencies — pure HTML+CSS+JS with embedded JSON."""
    last_year = int(detail["Last_Year"].iloc[0])
    payload = {
        "target_years": target_years,
        "last_year": last_year,
        "last_total": float(detail["Last_Engage"].sum()),
        "forecast_totals": {str(y): float(detail[f"Forecast_{y}"].sum()) for y in target_years},
        "synthese": synthese.fillna("").to_dict(orient="records"),
        "lignes": detail.fillna("").to_dict(orient="records"),
        "methodes": methodes.fillna("").to_dict(orient="records"),
        "method_distribution": methodes["Best_Method"].value_counts().to_dict(),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, default=str)

    forecast_cols_html = "".join(f"<th>Forecast {y}</th>" for y in target_years)
    synthese_cols_html = "".join(f"<th>{y}</th>" for y in target_years)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Forecast Triennale {target_years[0]}–{target_years[-1]}</title>
<style>
  :root {{
    --bg:#f5f6fa; --panel:#fff; --ink:#1e293b; --muted:#64748b;
    --border:#e2e8f0; --accent:#1e40af; --pos:#15803d; --neg:#b91c1c;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         background:var(--bg); color:var(--ink); }}
  header {{ background:var(--accent); color:#fff; padding:20px 32px; }}
  header h1 {{ margin:0; font-size:22px; }}
  header .sub {{ opacity:0.85; margin-top:4px; }}
  main {{ padding:24px 32px; max-width:1400px; margin:0 auto; }}
  .kpis {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr));
           gap:16px; margin-bottom:24px; }}
  .kpi {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
          padding:16px; }}
  .kpi .lbl {{ font-size:12px; color:var(--muted); text-transform:uppercase;
               letter-spacing:0.5px; }}
  .kpi .val {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .kpi .delta {{ font-size:13px; margin-top:4px; }}
  .delta.pos {{ color:var(--pos); }} .delta.neg {{ color:var(--neg); }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
            padding:20px; margin-bottom:20px; }}
  .panel h3 {{ margin:0 0 12px 0; }}
  table {{ width:100%; border-collapse: collapse; font-size:13px; }}
  th, td {{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--border); }}
  th {{ background:#f8fafc; color:var(--muted); font-weight:600;
        text-transform:uppercase; letter-spacing:0.3px; font-size:11px; }}
  th.left, td.left {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:4px;
            font-size:11px; font-weight:600; background:#e2e8f0; color:#334155; }}
  .badge.risk-h {{ background:#fee2e2; color:#991b1b; }}
  .badge.risk-m {{ background:#fef3c7; color:#92400e; }}
  .badge.risk-l {{ background:#dcfce7; color:#166534; }}
  .toolbar {{ display:flex; gap:12px; margin-bottom:12px; }}
  input[type=text] {{ flex:1; padding:8px 12px; border:1px solid var(--border);
                      border-radius:6px; font-size:13px; }}
  .scrollbox {{ max-height:560px; overflow:auto; border:1px solid var(--border);
                border-radius:6px; }}
  .scrollbox thead th {{ position:sticky; top:0; z-index:1; }}
  .tabs {{ display:flex; gap:4px; margin-bottom:16px; border-bottom:2px solid var(--border); }}
  .tab {{ padding:10px 18px; cursor:pointer; border:none; background:none;
          color:var(--muted); font-weight:600; font-size:14px; }}
  .tab.active {{ color:var(--accent); border-bottom:2px solid var(--accent);
                 margin-bottom:-2px; }}
  .tab-panel {{ display:none; }} .tab-panel.active {{ display:block; }}
  footer {{ text-align:center; padding:20px; color:var(--muted); font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>Prévision budgétaire — Programmation triennale {target_years[0]} – {target_years[-1]}</h1>
  <div class="sub">Modèles sélectionnés par backtest · Données 100% locales · Aucune connexion réseau</div>
</header>
<main>
  <div class="panel" style="background:#fef9c3;border-color:#fde047;padding:12px 16px">
    <b>Note :</b> les modèles ont été choisis à partir de données simulées. Une mise à jour est nécessaire lors de la disponibilité des données réelles.
  </div>
  <div class="kpis" id="kpis"></div>

  <div class="tabs">
    <button class="tab active" data-tab="syn">Synthèse par Programme</button>
    <button class="tab" data-tab="lig">Détail par Ligne ({len(detail)})</button>
    <button class="tab" data-tab="met">Méthodes & Performance</button>
  </div>

  <div class="panel tab-panel active" id="tab-syn">
    <h3>Synthèse par Chapitre × Programme</h3>
    <div class="scrollbox">
      <table>
        <thead><tr>
          <th class="left">Chapitre</th><th class="left">Programme</th><th>Lignes</th>
          <th>{last_year} (réel)</th>{synthese_cols_html}
        </tr></thead>
        <tbody id="synBody"></tbody>
      </table>
    </div>
  </div>

  <div class="panel tab-panel" id="tab-lig">
    <h3>Détail par Ligne budgétaire</h3>
    <div class="toolbar">
      <input type="text" id="lineFilter" placeholder="Filtrer (code ou intitulé)..." />
    </div>
    <div class="scrollbox">
      <table>
        <thead><tr>
          <th class="left">Code</th><th class="left">Intitulé</th>
          <th>{last_year} (réel)</th>{forecast_cols_html}
          <th>Méthode</th><th>Backtest sMAPE</th><th>Risque ML</th>
        </tr></thead>
        <tbody id="ligBody"></tbody>
      </table>
    </div>
  </div>

  <div class="panel tab-panel" id="tab-met">
    <h3>Distribution des méthodes choisies</h3>
    <div id="methodDist" style="margin-bottom:20px;"></div>
    <h3>Performance par ligne (trié par précision)</h3>
    <div class="scrollbox">
      <table>
        <thead><tr>
          <th class="left">Code</th><th class="left">Intitulé</th>
          <th>Méthode</th><th>Backtest sMAPE</th><th>Risque ML</th>
        </tr></thead>
        <tbody id="metBody"></tbody>
      </table>
    </div>
  </div>
</main>
<footer>Généré par forecast_pipeline.py · Tout est en local, aucune donnée n'a quitté votre poste.</footer>

<script>
const DATA = {payload_json};
const TY = DATA.target_years;
const fmt = n => (n == null || n === "" || isNaN(n)) ? "—"
  : new Intl.NumberFormat("fr-FR", {{maximumFractionDigits:0}}).format(n);
const fmtPct = n => (n == null || isNaN(n)) ? "—"
  : (n >= 0 ? "+" : "") + n.toFixed(1) + "%";

// KPIs
const kpiEl = document.getElementById("kpis");
kpiEl.innerHTML = `<div class="kpi"><div class="lbl">${{DATA.last_year}} (réel)</div>
                   <div class="val">${{fmt(DATA.last_total)}}</div>
                   <div class="delta">MAD</div></div>` +
  TY.map(y => {{
    const v = DATA.forecast_totals[y];
    const d = (v - DATA.last_total) / DATA.last_total * 100;
    return `<div class="kpi"><div class="lbl">Prévision ${{y}}</div>
            <div class="val">${{fmt(v)}}</div>
            <div class="delta ${{d>=0?'pos':'neg'}}">${{fmtPct(d)}} vs ${{DATA.last_year}}</div></div>`;
  }}).join("");

// Synthese
function renderSyn() {{
  const rows = DATA.synthese.map(r => {{
    const isTotal = String(r.Chap) === "TOTAL";
    const tds = [`<td class="left"><b>${{r.Chap}}</b></td>`,
                 `<td class="left">${{r.Prog || ""}}</td>`,
                 `<td>${{fmt(r.Nb_Lignes)}}</td>`,
                 `<td>${{fmt(r.Last_Engage)}}</td>`];
    TY.forEach(y => tds.push(`<td><b>${{fmt(r["Forecast_"+y])}}</b></td>`));
    return `<tr${{isTotal?' style="background:#f1f5f9;font-weight:700"':''}}>${{tds.join("")}}</tr>`;
  }});
  document.getElementById("synBody").innerHTML = rows.join("");
}}
renderSyn();

// Lignes
function riskBadge(s) {{
  if (s == null || s === "") return '<span class="badge">—</span>';
  const v = parseFloat(s);
  if (v >= 0.45) return `<span class="badge risk-h">${{(v*100).toFixed(0)}}</span>`;
  if (v >= 0.30) return `<span class="badge risk-m">${{(v*100).toFixed(0)}}</span>`;
  return `<span class="badge risk-l">${{(v*100).toFixed(0)}}</span>`;
}}
function renderLig() {{
  const q = document.getElementById("lineFilter").value.toLowerCase();
  const filtered = DATA.lignes.filter(l =>
    !q || String(l.Line_Key).toLowerCase().includes(q)
       || String(l.Intitule).toLowerCase().includes(q));
  document.getElementById("ligBody").innerHTML = filtered.map(l => {{
    const tds = [`<td class="left"><span class="badge">${{l.Line_Key}}</span></td>`,
                 `<td class="left">${{l.Intitule}}</td>`,
                 `<td>${{fmt(l.Last_Engage)}}</td>`];
    TY.forEach(y => tds.push(`<td><b>${{fmt(l["Forecast_"+y])}}</b></td>`));
    tds.push(`<td><span class="badge">${{l.Best_Method}}</span></td>`);
    tds.push(`<td>${{l.Backtest_sMAPE == null ? "—" : Number(l.Backtest_sMAPE).toFixed(1) + "%"}}</td>`);
    tds.push(`<td>${{riskBadge(l.ML_Risk_Score)}}</td>`);
    return `<tr>${{tds.join("")}}</tr>`;
  }}).join("");
}}
renderLig();
document.getElementById("lineFilter").addEventListener("input", renderLig);

// Methodes
const md = DATA.method_distribution;
document.getElementById("methodDist").innerHTML = Object.entries(md)
  .map(([k,v]) => `<span class="badge" style="margin-right:8px;padding:6px 12px;font-size:13px">${{k}}: ${{v}} lignes</span>`)
  .join("");
document.getElementById("metBody").innerHTML = DATA.methodes.map(m => `
  <tr><td class="left"><span class="badge">${{m.Line_Key}}</span></td>
      <td class="left">${{m.Intitule}}</td>
      <td><span class="badge">${{m.Best_Method}}</span></td>
      <td>${{m.Backtest_sMAPE == null ? "—" : Number(m.Backtest_sMAPE).toFixed(1) + "%"}}</td>
      <td>${{riskBadge(m.ML_Risk_Score)}}</td></tr>`).join("");

// Tabs
document.querySelectorAll(".tab").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  }});
}});
</script>
</body>
</html>"""
    html_path = OUT_DIR / f"Forecast_Triennale_{target_years[0]}_{target_years[-1]}.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Manager-driven multi-year budget forecast")
    ap.add_argument("--target-year", type=int, required=True,
                    help="First year to forecast (e.g. 2027)")
    ap.add_argument("--horizon", type=int, default=3,
                    help="Number of years to forecast (default 3 = triennale)")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip the .xls -> .xlsx conversion step")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip the extraction step (use existing panel)")
    args = ap.parse_args()

    target_years = list(range(args.target_year, args.target_year + args.horizon))
    print(f"\n=== Forecast pipeline for {target_years} ===\n")

    if not args.skip_convert:
        maybe_run_convert()
    if not args.skip_extract:
        run_extract()

    lignes, complete_years = load_panel()
    if max(complete_years) >= args.target_year:
        print(f"[warn] target year {args.target_year} is ≤ last complete year "
              f"{max(complete_years)} — you are 'forecasting' the past.")

    # Pre-train XGB for every horizon
    if HAS_XGB:
        print("[run]  pre-training XGBoost per horizon year")
        for ty in target_years:
            train_xgb(lignes, ty)

    print("[run]  tournament (backtest)")
    best = run_tournament(lignes, complete_years)
    method_dist = best["Best_Method"].value_counts()
    print("       method distribution:")
    for m, n in method_dist.items():
        print(f"         {m:<8s} : {n}")

    print("[run]  forecasting horizons")
    detail = forecast_horizons(lignes, best, target_years)

    print("[run]  anomaly scoring (IsolationForest)")
    risk = anomaly_scores(lignes)

    detail = detail.merge(best[["Line_Key", "Backtest_sMAPE"]], on="Line_Key", how="left") \
                   .merge(risk, on="Line_Key", how="left")
    detail["ML_Risk_Score"] = detail["ML_Risk_Score"].fillna(0.0).round(3)
    detail["Backtest_sMAPE"] = detail["Backtest_sMAPE"].round(2)
    detail = detail.sort_values(f"Forecast_{target_years[0]}", ascending=False)

    synthese = build_synthese(detail, target_years)
    methodes = best.merge(risk, on="Line_Key", how="left") \
                   .merge(detail[["Line_Key", "Intitule"]], on="Line_Key", how="left") \
                   .sort_values("Backtest_sMAPE")[
                       ["Line_Key", "Intitule", "Best_Method",
                        "Backtest_sMAPE", "ML_Risk_Score"]]

    fname = f"Forecast_Triennale_{target_years[0]}_{target_years[-1]}.xlsx"
    out_path = OUT_DIR / fname
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        synthese.to_excel(w, sheet_name="Synthese", index=False)
        detail.to_excel(w, sheet_name="Detail_Lignes", index=False)
        methodes.to_excel(w, sheet_name="Methodes", index=False)

    html_path = build_html_viewer(detail, synthese, methodes, target_years)

    print(f"\n=== Done ===")
    print(f"Excel  : {out_path}")
    print(f"HTML   : {html_path}")
    print(f"Lines  : {len(detail)}")
    print(f"Years  : {target_years}")
    print("\nSum of forecasts:")
    for ty in target_years:
        print(f"   {ty}: {detail[f'Forecast_{ty}'].sum():,.0f} MAD")
    print(f"   ({int(detail['Last_Year'].iloc[0])} actual: {detail['Last_Engage'].sum():,.0f} MAD)")


if __name__ == "__main__":
    main()
