"""
manager_app.py
==============
Tiny local web app so the manager can:
  1. See which years of data are already loaded.
  2. Upload new yearly raw files (SituationChap-YYYY.xls / .xlsx).
  3. Pick the target year + horizon, launch the forecast pipeline.
  4. Download the generated Excel and open the HTML viewer.

Everything runs on http://127.0.0.1:5000 — no internet, no cloud.
Launch with:
    python v2\\code\\manager_app.py
"""
from __future__ import annotations
import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

from flask import (Flask, request, redirect, url_for, send_from_directory,
                   render_template_string, flash, abort)

# --- When frozen by PyInstaller, the .exe doubles as the pipeline runner ----
# Pattern: re-invoke "ForecastManager.exe --run-pipeline ..." and hand off
# to forecast_pipeline.main(). This avoids needing python on the manager PC.
if "--run-pipeline" in sys.argv:
    sys.argv.remove("--run-pipeline")
    # When frozen, pipeline scripts are extracted to sys._MEIPASS
    if getattr(sys, "frozen", False):
        sys.path.insert(0, str(Path(sys._MEIPASS) / "v2" / "code"))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import forecast_pipeline
    forecast_pipeline.main()
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "01_raw"
OUT = ROOT / "data" / "04_manager_output"
CODE = Path(__file__).resolve().parent
PIPELINE = CODE / "forecast_pipeline.py"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT = {".xls", ".xlsx"}
YEAR_RE = re.compile(r"(20\d{2})")  # extract 4-digit year from filename

app = Flask(__name__)
app.secret_key = "pfe-local-only"
RUN_LOG: list[str] = []  # last pipeline run output


# ---------------------------------------------------------------- helpers
def detected_years() -> list[int]:
    years = set()
    for p in RAW.glob("*.xls*"):
        m = YEAR_RE.search(p.stem)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def latest_outputs() -> list[dict]:
    items = []
    for p in sorted(OUT.glob("Forecast_Triennale_*"), key=lambda x: x.stat().st_mtime, reverse=True):
        items.append({
            "name": p.name,
            "size_kb": p.stat().st_size // 1024,
            "kind": "html" if p.suffix == ".html" else "xlsx",
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return items


# ----------------------------------------------------------------- pages
PAGE = r"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<title>Forecast Manager</title>
<style>
:root{--bg:#f5f6fa;--panel:#fff;--ink:#1e293b;--muted:#64748b;
      --border:#e2e8f0;--accent:#1e40af;--pos:#15803d;--neg:#b91c1c;}
*{box-sizing:border-box} body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--ink)}
header{background:var(--accent);color:#fff;padding:20px 32px}
header h1{margin:0;font-size:22px} header .sub{opacity:.85;margin-top:4px}
main{padding:24px 32px;max-width:1100px;margin:0 auto}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;
  padding:20px;margin-bottom:20px}
.panel h2{margin:0 0 12px 0;font-size:16px;color:var(--accent)}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:end}
.field{flex:1;min-width:160px}
label{display:block;font-size:12px;color:var(--muted);margin-bottom:4px;
  text-transform:uppercase;letter-spacing:.4px}
input[type=number],input[type=file],select{width:100%;padding:8px 10px;
  border:1px solid var(--border);border-radius:6px;font-size:14px;background:#fff}
button{padding:10px 18px;background:var(--accent);color:#fff;border:none;
  border-radius:6px;font-weight:600;cursor:pointer;font-size:14px}
button:hover{filter:brightness(1.1)}
button.secondary{background:#fff;color:var(--accent);border:1px solid var(--accent)}
.badge{display:inline-block;padding:2px 10px;border-radius:4px;
  background:#dbeafe;color:#1e40af;font-size:12px;font-weight:600;margin:2px}
.badge.miss{background:#fee2e2;color:#991b1b}
.flash{padding:10px 14px;border-radius:6px;margin-bottom:12px;font-size:14px}
.flash.ok{background:#dcfce7;color:#166534;border:1px solid #86efac}
.flash.err{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
th{background:#f8fafc;color:var(--muted);font-weight:600;text-transform:uppercase;
  letter-spacing:.3px;font-size:11px}
a.btn{text-decoration:none;padding:5px 10px;border-radius:4px;
  background:#dbeafe;color:#1e40af;font-size:12px;font-weight:600;margin-right:4px}
pre.log{background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;
  font-size:12px;max-height:300px;overflow:auto;white-space:pre-wrap}
.muted{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi{background:#f8fafc;border:1px solid var(--border);border-radius:6px;padding:12px}
.kpi .v{font-size:20px;font-weight:700} .kpi .l{font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.5px}
footer{text-align:center;padding:18px;color:var(--muted);font-size:12px}
</style></head><body>
<header>
  <h1>Forecast Manager — Programmation triennale</h1>
  <div class="sub">Uploadez les données annuelles, choisissez l'année cible, lancez le forecast. 100% local.</div>
</header>
<main>
{% with messages = get_flashed_messages(with_categories=true) %}
  {% for cat, msg in messages %}<div class="flash {{cat}}">{{msg}}</div>{% endfor %}
{% endwith %}

<div class="panel">
  <h2>1. Données disponibles</h2>
  <div class="grid">
    <div class="kpi"><div class="l">Années détectées</div><div class="v">{{years|length}}</div></div>
    <div class="kpi"><div class="l">Première année</div><div class="v">{{years[0] if years else "—"}}</div></div>
    <div class="kpi"><div class="l">Dernière année</div><div class="v">{{years[-1] if years else "—"}}</div></div>
  </div>
  <p style="margin-top:10px">
    {% for y in years %}<span class="badge">{{y}}</span>{% endfor %}
    {% if not years %}<span class="badge miss">Aucune année — uploadez au moins 3 fichiers</span>{% endif %}
  </p>
  <p class="muted">Fichiers attendus : <code>SituationChap-YYYY.xls</code> ou <code>.xlsx</code>
  (TGR — Situation des dépenses par chapitre).</p>
</div>

<div class="panel">
  <h2>2. Charger un nouveau fichier annuel</h2>
  <form method="post" action="{{url_for('upload')}}" enctype="multipart/form-data" class="row">
    <div class="field" style="flex:2">
      <label>Fichier (.xls / .xlsx)</label>
      <input type="file" name="file" accept=".xls,.xlsx" required>
    </div>
    <div class="field">
      <label>Année (optionnel)</label>
      <input type="number" name="year" min="2000" max="2099" placeholder="Auto-détecté">
    </div>
    <button type="submit">Uploader</button>
  </form>
  <p class="muted" style="margin-top:8px">L'année est auto-détectée à partir du nom de fichier
  (ex. <code>SituationChap-2027.xls</code>). Sinon précisez-la.</p>
</div>

<div class="panel">
  <h2>3. Lancer le forecast</h2>
  <form method="post" action="{{url_for('run')}}" class="row">
    <div class="field">
      <label>Année cible</label>
      <input type="number" name="target_year" min="2020" max="2099"
             value="{{(years[-1] if years else 2026)+1}}" required>
    </div>
    <div class="field">
      <label>Horizon (années)</label>
      <input type="number" name="horizon" min="1" max="5" value="3" required>
    </div>
    <div class="field">
      <label>Re-extraire la donnée</label>
      <select name="extract">
        <option value="auto" selected>Auto (si modifications)</option>
        <option value="force">Forcer (long)</option>
        <option value="skip">Sauter (rapide)</option>
      </select>
    </div>
    <button type="submit"{% if years|length < 3 %} disabled{% endif %}>
      Lancer le pipeline →
    </button>
  </form>
  {% if years|length < 3 %}
    <p class="muted" style="color:var(--neg)">Il faut au moins 3 années d'historique pour lancer le forecast.</p>
  {% endif %}
</div>

{% if log %}
<div class="panel">
  <h2>Dernier journal d'exécution</h2>
  <pre class="log">{{log}}</pre>
</div>
{% endif %}

<div class="panel">
  <h2>4. Résultats générés</h2>
  {% if outputs %}
  <table>
    <thead><tr><th>Fichier</th><th>Type</th><th>Date</th><th>Taille</th><th>Actions</th></tr></thead>
    <tbody>
    {% for o in outputs %}
      <tr>
        <td><b>{{o.name}}</b></td>
        <td><span class="badge">{{o.kind|upper}}</span></td>
        <td>{{o.mtime}}</td>
        <td>{{o.size_kb}} KB</td>
        <td>
          {% if o.kind == 'html' %}
            <a class="btn" href="{{url_for('view_output', filename=o.name)}}" target="_blank">Ouvrir</a>
          {% endif %}
          <a class="btn" href="{{url_for('download', filename=o.name)}}">Télécharger</a>
        </td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
    <p class="muted">Aucun résultat encore. Lancez le pipeline ci-dessus.</p>
  {% endif %}
</div>
</main>
<footer>Application locale · Aucune donnée ne quitte votre poste.</footer>
</body></html>
"""


# ----------------------------------------------------------------- routes
@app.route("/")
def home():
    return render_template_string(
        PAGE,
        years=detected_years(),
        outputs=latest_outputs(),
        log="\n".join(RUN_LOG[-200:]) if RUN_LOG else "",
    )


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Aucun fichier sélectionné.", "err")
        return redirect(url_for("home"))
    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        flash(f"Extension non autorisée : {suffix}. Acceptées : .xls, .xlsx", "err")
        return redirect(url_for("home"))

    # Determine year
    year_field = (request.form.get("year") or "").strip()
    if year_field.isdigit():
        year = int(year_field)
    else:
        m = YEAR_RE.search(Path(f.filename).stem)
        if not m:
            flash("Impossible de détecter l'année dans le nom de fichier. Précisez-la.", "err")
            return redirect(url_for("home"))
        year = int(m.group(1))

    target_name = f"SituationChap-{year}{suffix}"
    dest = RAW / target_name
    f.save(dest)
    flash(f"Fichier enregistré : {target_name} ({dest.stat().st_size // 1024} KB).", "ok")
    return redirect(url_for("home"))


@app.route("/run", methods=["POST"])
def run():
    try:
        target_year = int(request.form["target_year"])
        horizon = int(request.form["horizon"])
    except (KeyError, ValueError):
        flash("Année cible ou horizon invalide.", "err")
        return redirect(url_for("home"))

    extract_mode = request.form.get("extract", "auto")
    # When frozen as .exe, sys.executable IS the exe — we hand off to it
    # via the --run-pipeline sentinel handled at the top of this file.
    # When running as a script, sys.executable is python and we run the .py.
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--run-pipeline",
               "--target-year", str(target_year),
               "--horizon", str(horizon)]
    else:
        cmd = [sys.executable, str(PIPELINE),
               "--target-year", str(target_year),
               "--horizon", str(horizon)]
    # Convert/Extract gating:
    # - "skip": both skipped (fastest, assumes panel exists)
    # - "auto": skip-convert (xlsx exist), let extract run
    # - "force": run everything
    if extract_mode == "skip":
        cmd += ["--skip-convert", "--skip-extract"]
    elif extract_mode == "auto":
        cmd += ["--skip-convert"]
    # "force": no flags

    RUN_LOG.clear()
    RUN_LOG.append(f"$ {' '.join(cmd)}")
    RUN_LOG.append(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] démarrage...\n")
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT.parent),
                              capture_output=True, text=True, timeout=600)
        RUN_LOG.append(proc.stdout or "")
        if proc.stderr:
            RUN_LOG.append("\n[stderr]\n" + proc.stderr)
        if proc.returncode == 0:
            flash(f"Forecast terminé pour {target_year}–{target_year + horizon - 1}.", "ok")
        else:
            flash(f"Le pipeline a échoué (code {proc.returncode}). Voir le journal.", "err")
    except subprocess.TimeoutExpired:
        RUN_LOG.append("\n[timeout] dépassement de 10 minutes.")
        flash("Le pipeline a dépassé le délai (10 min).", "err")
    except Exception as e:
        RUN_LOG.append(f"\n[exception] {e}")
        flash(f"Erreur : {e}", "err")
    return redirect(url_for("home"))


@app.route("/output/<path:filename>")
def view_output(filename):
    # Safety: only serve files inside OUT
    full = (OUT / filename).resolve()
    if not str(full).startswith(str(OUT.resolve())) or not full.exists():
        abort(404)
    return send_from_directory(OUT, filename)


@app.route("/download/<path:filename>")
def download(filename):
    full = (OUT / filename).resolve()
    if not str(full).startswith(str(OUT.resolve())) or not full.exists():
        abort(404)
    return send_from_directory(OUT, filename, as_attachment=True)


if __name__ == "__main__":
    print("=" * 60)
    print(" Forecast Manager — ouvrez http://127.0.0.1:5000 dans un navigateur")
    print("=" * 60)
    # Auto-open the default browser ~1s after server starts
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
