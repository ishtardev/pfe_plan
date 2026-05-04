"""
Système de Pilotage Budgétaire — Ministère de la Justice
Application bureau 100% hors ligne — Prévision T4 au 30 juin 2025

Lancer : python app.py
"""

import os
import json
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG      = "#F0F2F5"
C_HEADER  = "#1B3A5C"
C_ACCENT  = "#009688"
C_WHITE   = "#FFFFFF"
C_OK      = "#1a9850"
C_WARN    = "#fc8d59"
C_CRIT    = "#d73027"
C_CARD_BG = "#FFFFFF"

# ── Audit log ────────────────────────────────────────────────────────────────
def _run_audit_log(df, log_path="audit_log.txt", cache_path="predictions_cache.json"):
    """Compare T4 predictions with the previous run; log changes >= 5 pp."""
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = {str(r.line_id): round(float(r.pred_T4), 4) for _, r in df.iterrows()}

    changes = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                prev = json.load(f)
            for lid, pred in cur.items():
                if lid in prev:
                    delta = abs(pred - prev[lid])
                    if delta >= 0.05:
                        changes.append((lid, prev[lid], pred, delta))
        except Exception:
            pass

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cur, f)
    except Exception:
        pass

    if changes:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] {len(changes)} changement(s) >= 5 pp :\n")
                for lid, old, new, delta in changes:
                    arrow = "▲" if new > old else "▼"
                    f.write(f"  Ligne {lid}: {old:.1%} -> {new:.1%}  ({arrow}{delta*100:.1f} pp)\n")
        except Exception:
            pass

    return changes


# ── Modèle ────────────────────────────────────────────────────────────────────
def build_predictions():
    df = pd.read_csv("data_lines.csv")

    # Encodages
    type_map = {"acquisitions": 0, "equipements": 1, "etudes": 2,
                "fournitures": 3, "travaux": 4, "personnel": 5}
    prog_map   = {300: 0, 301: 1, 302: 2, 303: 3}
    budget_map = {"INVESTISSEMENT": 0, "MATERIEL": 1, "PERSONNEL": 2}

    df["type_enc"]   = df["type_ligne"].map(type_map)
    df["prog_enc"]   = df["programme"].map(prog_map)
    df["budget_enc"] = df["budget_type"].map(budget_map)
    # region is stored as int (0-12) in CSV — use directly
    df["reg_enc"]  = df["region"].astype(int)

    YEARS = sorted(df["year"].unique())

    # ── Historical T3 lookup ──────────────────────────────────────────────────
    # For each (line_id, year), compute the mean T3 taux across years strictly
    # before `year` — captures whether a line historically back-loads between
    # T2 and T3, with no leakage into CV folds.
    _t3 = (df[df["quarter"] == 3][["line_id", "year", "taux"]]
             .sort_values(["line_id", "year"]))
    _t3_lookup: dict = {}
    for _lid, _grp in _t3.groupby("line_id", sort=False):
        _yrs  = _grp["year"].tolist()
        _vals = _grp["taux"].tolist()
        for _i, _yr in enumerate(_yrs):
            _t3_lookup[(_lid, _yr)] = float(np.mean(_vals[:_i])) if _i > 0 else np.nan

    rows = []
    for year in YEARS:
        prev_year = year - 1
        if prev_year not in YEARS:
            continue
        for lid in df["line_id"].unique():
            cur  = df[(df["line_id"] == lid) & (df["year"] == year)].set_index("quarter")
            prev = df[(df["line_id"] == lid) & (df["year"] == prev_year)].set_index("quarter")
            if len(cur) < 4 or len(prev) < 4:
                continue
            rows.append({
                "line_id":      lid,
                "year":         year,
                "taux_T2":      cur.loc[2,  "taux"],
                "taux_T1":      cur.loc[1,  "taux"],
                "taux_T4_lag":  prev.loc[4, "taux"],
                "taux_T2_lag":  prev.loc[2, "taux"],
                "hist_avg_T3":  _t3_lookup.get((lid, year), np.nan),
                "taux_T3":      cur.loc[3,  "taux"],
                "taux_T3_lag":  prev.loc[3, "taux"],
                "lf_ratio":     cur.loc[2,  "lf_ratio"],

                "lf_share":     cur.loc[2,  "lf_share"],
                "type_enc":     cur.loc[2,  "type_enc"],
                "prog_enc":     cur.loc[2,  "prog_enc"],
                "reg_enc":      cur.loc[2,  "reg_enc"],
                "lf_mdh":           cur.loc[2,  "lf_mdh"],
                "ligne_label":      cur.loc[2,  "ligne_label"],
                "type_ligne":       cur.loc[2,  "type_ligne"],
                "programme":        cur.loc[2,  "programme"],
                "programme_label":  cur.loc[2,  "programme_label"],
                "region_label":     cur.loc[2,  "region_label"],
                "budget_type":      cur.loc[2,  "budget_type"],
                "budget_enc":       cur.loc[2,  "budget_enc"],
                "target_T4":    cur.loc[4,  "taux"],
            })

    pred_df = pd.DataFrame(rows)
    # Fill hist_avg_T3 NaN (lines with no prior T3 data) with global mean
    _global_t3 = pred_df["hist_avg_T3"].mean()
    pred_df["hist_avg_T3"] = pred_df["hist_avg_T3"].fillna(_global_t3)

    # Number of distinct years per line (used for reliability warnings)
    years_per_line = pred_df.groupby("line_id")["year"].nunique()

    FEATURES = ["taux_T2", "taux_T1", "taux_T4_lag", "taux_T2_lag", "hist_avg_T3",
                "lf_ratio", "lf_share", "type_enc", "prog_enc", "reg_enc", "budget_enc"]
    # T3-augmented feature set (September prediction — real Q3 rate available)
    FEATURES_T3 = FEATURES + ["taux_T3", "taux_T3_lag"]

    xgb_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, reg_alpha=0.1, random_state=42, verbosity=0)

    # Walk-forward CV — strictly past years only, no future data leakage
    # test 2022 → train on 2021 only
    # test 2023 → train on 2021-2022
    # test 2024 → train on 2021-2023
    train_years  = [y for y in YEARS if 2021 <= y < 2025]
    lasso_params = dict(alpha=0.01, max_iter=5000)
    all_cv_years = sorted(pred_df["year"].unique())
    test_years_cv = [y for y in all_cv_years if y < 2025
                     and (pred_df["year"] < y).any()]
    loyo_per_cat = {"INVESTISSEMENT": [], "MATERIEL": [], "PERSONNEL": []}

    for y in test_years_cv:
        tr = pred_df[pred_df["year"] < y]   # strictly past — no leakage
        te = pred_df[pred_df["year"] == y]

        # INVESTISSEMENT → XGBoost
        tr_i = tr[tr["budget_type"] == "INVESTISSEMENT"]
        te_i = te[te["budget_type"] == "INVESTISSEMENT"]
        if len(tr_i) >= 2 and len(te_i) >= 1:
            m = XGBRegressor(**xgb_params)
            m.fit(tr_i[FEATURES], tr_i["target_T4"])
            loyo_per_cat["INVESTISSEMENT"].append(
                np.sqrt(mean_squared_error(te_i["target_T4"],
                        m.predict(te_i[FEATURES]).clip(0, 1.30))))

        # MATERIEL → Lasso (walk-forward validated: 2/3 folds, RMSE 0.066 vs Hist_mean 0.089)
        tr_m = tr[tr["budget_type"] == "MATERIEL"]
        te_m = te[te["budget_type"] == "MATERIEL"]
        if len(tr_m) >= 2 and len(te_m) >= 1:
            m = Pipeline([("sc", StandardScaler()), ("reg", Lasso(**lasso_params))])
            m.fit(tr_m[FEATURES], tr_m["target_T4"])
            loyo_per_cat["MATERIEL"].append(
                np.sqrt(mean_squared_error(te_m["target_T4"],
                        np.clip(m.predict(te_m[FEATURES]), 0, 1.30))))

        # PERSONNEL → moyenne historique par ligne (past years only)
        tr_p = tr[tr["budget_type"] == "PERSONNEL"]
        te_p = te[te["budget_type"] == "PERSONNEL"]
        if len(tr_p) >= 1 and len(te_p) >= 1:
            hist_p  = tr_p.groupby("line_id")["target_T4"].mean()
            preds_p = te_p["line_id"].map(hist_p).fillna(tr_p["target_T4"].mean()).clip(0, 1.30)
            loyo_per_cat["PERSONNEL"].append(
                np.sqrt(mean_squared_error(te_p["target_T4"], preds_p)))

    wf_cv_summary = {
        "INVESTISSEMENT": {"model": "XGBoost", "rmse": round(np.mean(loyo_per_cat["INVESTISSEMENT"]), 4)},
        "MATERIEL":       {"model": "Lasso",    "rmse": round(np.mean(loyo_per_cat["MATERIEL"]),       4)},
        "PERSONNEL":      {"model": "Hist. moy","rmse": round(np.mean(loyo_per_cat["PERSONNEL"]),      4)},
    }

    # Modèles finaux → prévision 2025
    train_full = pred_df[pred_df["year"].isin(train_years)]
    test_2025  = pred_df[pred_df["year"] == 2025].copy()
    test_2025["pred_T4"] = np.nan

    # INVESTISSEMENT: XGBoost
    tr_i    = train_full[train_full["budget_type"] == "INVESTISSEMENT"]
    xgb_inv = XGBRegressor(**xgb_params)
    xgb_inv.fit(tr_i[FEATURES], tr_i["target_T4"])
    mask_i  = test_2025["budget_type"] == "INVESTISSEMENT"
    test_2025.loc[mask_i, "pred_T4"] = xgb_inv.predict(
        test_2025.loc[mask_i, FEATURES]).clip(0, 1.30)

    # MATERIEL: Lasso
    tr_m      = train_full[train_full["budget_type"] == "MATERIEL"]
    lasso_mat = Pipeline([("sc", StandardScaler()), ("reg", Lasso(**lasso_params))])
    lasso_mat.fit(tr_m[FEATURES], tr_m["target_T4"])
    mask_m    = test_2025["budget_type"] == "MATERIEL"
    test_2025.loc[mask_m, "pred_T4"] = np.clip(
        lasso_mat.predict(test_2025.loc[mask_m, FEATURES]), 0, 1.30)

    # PERSONNEL: moyenne historique par ligne
    tr_p      = train_full[train_full["budget_type"] == "PERSONNEL"]
    hist_p    = tr_p.groupby("line_id")["target_T4"].mean()
    overall_p = tr_p["target_T4"].mean()
    mask_p    = test_2025["budget_type"] == "PERSONNEL"
    test_2025.loc[mask_p, "pred_T4"] = (
        test_2025.loc[mask_p, "line_id"].map(hist_p).fillna(overall_p).clip(0, 1.30).values
    )
    # ── T3 predictions (September — real Q3 execution rate available) ────────────
    # The T2 model (June) predicts T4 without knowing T3.
    # The T3 model retrains with taux_T3 added, giving a more accurate estimate.
    test_2025["pred_T4_T2"] = test_2025["pred_T4"]   # save June prediction
    test_2025["pred_T4_T3"] = np.nan

    tr_i_t3 = train_full[train_full["budget_type"] == "INVESTISSEMENT"]
    xgb_t3  = XGBRegressor(**xgb_params)
    xgb_t3.fit(tr_i_t3[FEATURES_T3], tr_i_t3["target_T4"])
    test_2025.loc[mask_i, "pred_T4_T3"] = xgb_t3.predict(
        test_2025.loc[mask_i, FEATURES_T3]).clip(0, 1.30)

    tr_m_t3  = train_full[train_full["budget_type"] == "MATERIEL"]
    lasso_t3 = Pipeline([("sc", StandardScaler()), ("reg", Lasso(**lasso_params))])
    lasso_t3.fit(tr_m_t3[FEATURES_T3], tr_m_t3["target_T4"])
    test_2025.loc[mask_m, "pred_T4_T3"] = np.clip(
        lasso_t3.predict(test_2025.loc[mask_m, FEATURES_T3]), 0, 1.30)

    # PERSONNEL: hist mean is T3-independent — reuse T2 prediction
    test_2025.loc[mask_p, "pred_T4_T3"] = test_2025.loc[mask_p, "pred_T4_T2"]

    # Active prediction = T3-based (uses real Q3 data — more accurate)
    test_2025["pred_T4"] = test_2025["pred_T4_T3"]
    # ── Z-score anomaly detection ─────────────────────────────────────────
    # Compare each line's 2025 T2 taux against that same line's own
    # historical T2 taux across training years (per-line baseline).
    hist = (train_full.groupby("line_id")["taux_T2"]
                      .agg(["mean", "std"])
                      .rename(columns={"mean": "hist_mean_T2", "std": "hist_std_T2"}))
    test_2025 = test_2025.join(hist, on="line_id")
    test_2025["hist_std_T2"] = test_2025["hist_std_T2"].fillna(0.01).clip(lower=0.005)
    test_2025["z_score_T2"] = ((test_2025["taux_T2"] - test_2025["hist_mean_T2"])
                                / test_2025["hist_std_T2"]).round(2)
    test_2025["anomalie"] = test_2025["z_score_T2"] < -1.5

    def _anomalie_label(row):
        if not row["anomalie"]:
            return ""
        t = row["taux_T2"]
        z = row["z_score_T2"]
        if t == 0.0:
            return "⚠ Exécution nulle"
        elif z < -3.0:
            return "⚠ Sous-exécution critique"
        elif z < -2.0:
            return "⚠ Sous-exécution sévère"
        else:
            return "⚠ Sous-exécution modérée"

    test_2025["anomalie_label"] = test_2025.apply(_anomalie_label, axis=1)

    def classify(t):
        if t < 0.60: return "Critique",  C_CRIT
        if t < 0.80: return "Attention", C_WARN
        return           "OK",           C_OK

    labels, colors = zip(*test_2025["pred_T4"].apply(classify))
    test_2025["risk_label"] = list(labels)
    test_2025["risk_color"] = list(colors)
    test_2025["credit_risque_mdh"] = (
        test_2025.apply(lambda r: r["lf_mdh"] * max(0, 0.80 - r["pred_T4"])
                        if r["risk_label"] != "OK" else 0.0, axis=1)
    ).round(2)

    test_2025 = test_2025.sort_values("pred_T4").reset_index(drop=True)

    # ── History depth & reliability warning ───────────────────────────────
    test_2025["hist_years_count"] = (test_2025["line_id"]
                                     .map(years_per_line)
                                     .fillna(1).astype(int))
    test_2025["low_history"] = test_2025["hist_years_count"] < 3

    # ── Audit log (compare with previous run, write to audit_log.txt) ─────
    build_ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    _run_audit_log(test_2025)

    return test_2025, pred_df, wf_cv_summary, build_ts


# ── Application ───────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Système de Pilotage Budgétaire — Ministère de la Justice")
        self.geometry("1280x780")
        self.minsize(900, 600)
        self.configure(bg=C_BG)

        # ── TTK global style ──────────────────────────────────────────────────
        _s = ttk.Style(self)
        _s.theme_use("clam")
        _s.configure("TNotebook",     background=C_BG, borderwidth=0)
        _s.configure("TNotebook.Tab", font=("Segoe UI", 10), padding=[16, 6],
                     background="#D6DAE0", foreground="#444")
        _s.map("TNotebook.Tab",
               background=[("selected", C_HEADER)],
               foreground=[("selected", C_WHITE)])
        _s.configure("Treeview.Heading",
                     background=C_HEADER, foreground=C_WHITE,
                     font=("Segoe UI", 9, "bold"), relief="flat")
        _s.map("Treeview.Heading", background=[("active", "#2A4F7C")])
        _s.configure("Treeview",
                     rowheight=28, font=("Segoe UI", 9),
                     fieldbackground=C_WHITE, background=C_WHITE,
                     foreground="#222")
        _s.configure("TScrollbar",
                     background="#CDD2D8", troughcolor=C_BG, relief="flat")
        _s.configure("TFrame", background=C_BG)

        self._build_header()
        self._build_status("Chargement du modèle en cours…")
        self.update()
        self.after(100, self._load)

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        h = tk.Frame(self, bg=C_HEADER, height=56)
        h.pack(fill="x")
        h.pack_propagate(False)
        tk.Label(h,
                 text="  Ministère de la Justice — Système d'Alerte Précoce Budgétaire  |  Exercice 2025",
                 bg=C_HEADER, fg=C_WHITE,
                 font=("Segoe UI", 12, "bold")).pack(side="left", pady=14)
        tk.Frame(self, bg=C_ACCENT, height=3).pack(fill="x")

    def _build_status(self, msg):
        self._status_lbl = tk.Label(self, text=msg, bg=C_BG,
                                    font=("Segoe UI", 10), fg="#555")
        self._status_lbl.pack(pady=8)

    # ── Load model ────────────────────────────────────────────────────────────
    def _load(self):
        try:
            self.data, self.all_data, self.wf_cv_summary, self.build_ts = build_predictions()
            self._status_lbl.destroy()
            self._build_main()
        except Exception as e:
            self._status_lbl.config(text=f"Erreur : {e}", fg=C_CRIT)

    # ── Main layout ───────────────────────────────────────────────────────────
    def _build_main(self):
        df = self.data

        # ── KPI cards ─────────────────────────────────────────────────────────
        card_row = tk.Frame(self, bg=C_BG)
        card_row.pack(fill="x", padx=18, pady=(6, 4))

        n_ok    = (df["risk_label"] == "OK").sum()
        n_warn  = (df["risk_label"] == "Attention").sum()
        n_crit  = (df["risk_label"] == "Critique").sum()
        mdh     = df["credit_risque_mdh"].sum()
        n_anom  = df["anomalie"].sum()

        kpis = [
            ("Lignes analysées",  str(len(df)),          C_HEADER,  ""),
            ("OK  ≥ 80 %",        str(n_ok),             C_OK,      ""),
            ("Attention 60–80 %", str(n_warn),           C_WARN,    ""),
            ("Critique  < 60 %",  str(n_crit),           C_CRIT,    ""),
            ("Crédits à risque",  f"{mdh:.1f} MDH",      "#7D3C98", ""),
            ("Anomalies ⚠",       str(n_anom),           "#B7950B", "z-score < −2"),
        ]
        for title, val, color, sub in kpis:
            border = tk.Frame(card_row, bg="#D0D5DD", width=174, height=86)
            border.pack(side="left", padx=4)
            border.pack_propagate(False)
            f = tk.Frame(border, bg=C_WHITE)
            f.pack(fill="both", expand=True, padx=1, pady=1)
            tk.Frame(f, bg=color, height=4).pack(fill="x")
            tk.Label(f, text=val, bg=C_WHITE, fg=color,
                     font=("Segoe UI", 20, "bold")).pack(pady=(5, 0))
            tk.Label(f, text=title, bg=C_WHITE, fg="#555",
                     font=("Segoe UI", 8)).pack()
            if sub:
                tk.Label(f, text=sub, bg=C_WHITE, fg="#999",
                         font=("Segoe UI", 7)).pack()

        # Precision card — one line per specialised model
        ls = self.wf_cv_summary
        p_clr = "#2E6DA4"
        pbox = tk.Frame(card_row, bg="#D0D5DD", width=214, height=86)
        pbox.pack(side="left", padx=4)
        pbox.pack_propagate(False)
        prec_f = tk.Frame(pbox, bg=C_WHITE)
        prec_f.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Frame(prec_f, bg=p_clr, height=4).pack(fill="x")
        tk.Label(prec_f, text="Précision — Walk-forward CV", bg=C_WHITE, fg=p_clr,
                 font=("Segoe UI", 7, "bold")).pack(pady=(4, 1))
        for cat, short in [("INVESTISSEMENT", "Invest. \u2192 XGBoost"),
                           ("MATERIEL",       "Mat\u00e9r.  \u2192 Lasso  "),
                           ("PERSONNEL",      "Person. \u2192 Hist.moy")]:
            tk.Label(prec_f,
                     text=f"{short}   RMSE {ls[cat]['rmse']:.3f}",
                     bg=C_WHITE, fg="#555",
                     font=("Segoe UI", 7)).pack()

        # ── Timestamp ─────────────────────────────────────────────────────────
        tk.Label(self,
                 text=f"Prévisions générées le {self.build_ts}",
                 bg=C_BG, fg="#888", font=("Segoe UI", 8),
                 anchor="e").pack(fill="x", padx=20, pady=(0, 2))

        # ── Notebook ──────────────────────────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(2, 8))

        self._tab_table(nb, df)
        self._tab_chart(nb, df)
        self._tab_region(nb, df)

    # ── Tab 1 : Tableau ───────────────────────────────────────────────────────
    def _tab_table(self, nb, df):
        frame = ttk.Frame(nb)
        nb.add(frame, text="  Tableau de bord  ")

        # Filters
        fbar = tk.Frame(frame, bg=C_WHITE, relief="groove", bd=1)
        fbar.pack(fill="x", padx=10, pady=(8, 0))

        def lbl(parent, text):
            tk.Label(parent, text=text, bg=C_WHITE,
                     font=("Segoe UI", 9)).pack(side="left", padx=(10, 2))

        lbl(fbar, "Programme :")
        prog_var = tk.StringVar(value="Tous")
        prog_labels = {int(r.programme): r.programme_label
                       for _, r in df[["programme", "programme_label"]].drop_duplicates().iterrows()}
        progs = ["Tous"] + sorted([f"P{p} – {prog_labels[p]}" for p in prog_labels])
        ttk.Combobox(fbar, textvariable=prog_var, values=progs,
                     width=30, state="readonly").pack(side="left")

        lbl(fbar, "Région :")
        reg_var = tk.StringVar(value="Toutes")
        regs = ["Toutes"] + sorted(df["region_label"].unique().tolist())
        ttk.Combobox(fbar, textvariable=reg_var, values=regs,
                     width=22, state="readonly").pack(side="left")

        lbl(fbar, "Risque :")
        risk_var = tk.StringVar(value="Tous")
        ttk.Combobox(fbar, textvariable=risk_var,
                     values=["Tous", "Critique", "Attention", "OK"],
                     width=10, state="readonly").pack(side="left")

        lbl(fbar, "Budget :")
        budget_var = tk.StringVar(value="Tous")
        ttk.Combobox(fbar, textvariable=budget_var,
                     values=["Tous", "INVESTISSEMENT", "MATERIEL", "PERSONNEL"],
                     width=16, state="readonly").pack(side="left")

        tk.Button(fbar, text="Filtrer", bg=C_HEADER, fg=C_WHITE,
                  relief="flat", font=("Segoe UI", 9, "bold"), padx=12,
                  cursor="hand2",
                  command=lambda: refresh()).pack(side="left", padx=14)

        tk.Button(fbar, text="⬇ Exporter", bg="#27AE60", fg=C_WHITE,
                  relief="flat", font=("Segoe UI", 9, "bold"), padx=12,
                  cursor="hand2",
                  command=lambda: _export()).pack(side="left", padx=4)

        count_var = tk.StringVar()
        tk.Label(fbar, textvariable=count_var, bg=C_WHITE,
                 font=("Segoe UI", 9), fg="#555").pack(side="right", padx=10)

        # Treeview
        cols = ("Programme", "Ligne budgétaire", "Région",
                "LF (MDH)", "T2 réel", "T4 (juin)", "T4 (sept.)", "T4 réel", "Écart (pp)", "Risque", "MDH à risque", "Anomalie")
        widths = [220, 260, 190, 72, 72, 72, 78, 72, 72, 82, 88, 175]

        tree_frame = tk.Frame(frame)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=22)
        for col, w in zip(cols, widths):
            tree.heading(col, text=col,
                         command=lambda c=col: _sort(c))
            tree.column(col, width=w,
                        anchor="center" if w < 200 else "w")

        tree.tag_configure("ecart_pos", foreground="#145a32")
        tree.tag_configure("ecart_neg", foreground="#922b21")
        tree.tag_configure("OK",        background="#EDF7EE", foreground="#145a32")
        tree.tag_configure("Attention", background="#FFFBEF", foreground="#7B5200")
        tree.tag_configure("Critique",  background="#FEF2F2", foreground="#7B0000")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        def _fmt_anomalie(r):
            parts = []
            if r.anomalie:
                parts.append(f"{r.anomalie_label}  (z={r.z_score_T2:.1f})")
            if r.low_history:
                parts.append(f"⚠ Hist. {r.hist_years_count} an(s)")
            return "  |  ".join(parts)

        def _export():
            path = filedialog.asksaveasfilename(
                parent=self,
                defaultextension=".xlsx",
                filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv")],
                initialfile=f"previsions_2025_{datetime.date.today().strftime('%Y%m%d')}.xlsx",
            )
            if not path:
                return
            exp = df[["programme_label", "ligne_label", "region_label", "budget_type",
                       "lf_mdh", "taux_T2", "pred_T4_T2", "pred_T4", "target_T4", "risk_label",
                       "credit_risque_mdh", "anomalie_label", "hist_years_count"]].copy()
            exp.columns = ["Programme", "Ligne budgetaire", "Region", "Type budget",
                           "LF (MDH)", "T2 realise (%)", "T4 predit T2 (%)", "T4 predit T3 (%)", "T4 reel (%)", "Risque",
                           "MDH a risque", "Anomalie", "Annees d'historique"]
            exp["T2 realise (%)"]   = (exp["T2 realise (%)"]   * 100).round(1)
            exp["T4 predit T2 (%)"] = (exp["T4 predit T2 (%)"] * 100).round(1)
            exp["T4 predit T3 (%)"] = (exp["T4 predit T3 (%)"] * 100).round(1)
            exp["T4 reel (%)"]      = (exp["T4 reel (%)"]      * 100).round(1)
            exp["Ecart T3 (pp)"]    = (exp["T4 predit T3 (%)"] - exp["T4 reel (%)"]).round(1)
            try:
                if path.endswith(".xlsx"):
                    exp.to_excel(path, index=False)
                else:
                    exp.to_csv(path, index=False, encoding="utf-8-sig")
                messagebox.showinfo("Export reussi", f"Fichier exporte :\n{path}", parent=self)
            except Exception as exc:
                messagebox.showerror("Erreur export", str(exc), parent=self)

        sort_state = {"col": None, "rev": False}

        def _sort(col):
            col_map = {
                "T2 réel": "taux_T2_val", "T4 (juin)": "pred_T4_T2",
                "T4 (sept.)": "pred_T4", "LF (MDH)": "lf_mdh_val",
                "MDH à risque": "credit_risque_mdh_val"
            }
            items = [(tree.set(k, col), k) for k in tree.get_children()]
            rev = sort_state["col"] == col and not sort_state["rev"]
            try:
                items.sort(key=lambda t: float(t[0].replace("%","").replace(" MDH","")), reverse=rev)
            except Exception:
                items.sort(key=lambda t: t[0], reverse=rev)
            for i, (_, k) in enumerate(items):
                tree.move(k, "", i)
            sort_state["col"] = col
            sort_state["rev"] = rev

        def refresh():
            for row in tree.get_children():
                tree.delete(row)
            filtered = df.copy()
            if prog_var.get() != "Tous":
                p = int(prog_var.get().split(" ")[0].replace("P", ""))
                filtered = filtered[filtered["programme"] == p]
            if reg_var.get() != "Toutes":
                filtered = filtered[filtered["region_label"] == reg_var.get()]
            if risk_var.get() != "Tous":
                filtered = filtered[filtered["risk_label"] == risk_var.get()]
            if budget_var.get() != "Tous":
                filtered = filtered[filtered["budget_type"] == budget_var.get()]

            count_var.set(f"{len(filtered)} ligne(s) affichée(s)")
            for _, r in filtered.iterrows():
                ecart_val = (r.pred_T4 - r.target_T4) * 100
                ecart_str = f"{ecart_val:+.1f} pp"
                tree.insert("", "end", iid=str(r.line_id), values=(
                    f"P{int(r.programme)} – {r.programme_label}",
                    r.ligne_label,
                    r.region_label,
                    f"{r.lf_mdh:.1f}",
                    f"{r.taux_T2:.1%}",
                    f"{r.pred_T4_T2:.1%}",
                    f"{r.pred_T4:.1%}",
                    f"{r.target_T4:.1%}",
                    ecart_str,
                    r.risk_label,
                    f"{r.credit_risque_mdh:.2f}",
                    _fmt_anomalie(r),
                ), tags=(r.risk_label,))

        refresh()
        tree.bind("<Double-1>", lambda e, t=tree: self._show_line_detail(t, e))

    # ── Détail d'une ligne (double-clic) ──────────────────────────────────────
    def _show_line_detail(self, tree, event):
        item = tree.identify_row(event.y)
        if not item:
            return
        line_id = item

        hist = self.all_data[self.all_data["line_id"] == line_id].sort_values("year").copy()
        cur  = self.data[self.data["line_id"] == line_id]
        if hist.empty:
            return

        info      = hist.iloc[-1]
        anomalie  = bool(cur.iloc[0]["anomalie"])    if not cur.empty else False
        z_score   = float(cur.iloc[0]["z_score_T2"]) if not cur.empty else None
        pred_t4    = float(cur.iloc[0]["pred_T4"])    if not cur.empty else None   # T3-based
        pred_t4_t2 = float(cur.iloc[0]["pred_T4_T2"]) if not cur.empty else None   # T2-based

        win = tk.Toplevel(self)
        win.title(f"Détail — {info.ligne_label[:60]}")
        win.geometry("720x560")
        win.configure(bg=C_BG)
        win.grab_set()

        tk.Label(win,
                 text=f"P{int(info.programme)} | {info.ligne_label} | {info.region_label}",
                 bg=C_BG, font=("Segoe UI", 10, "bold"),
                 wraplength=700).pack(pady=(10, 3), padx=10)

        if anomalie:
            anom_label = cur.iloc[0]["anomalie_label"] if not cur.empty else "⚠ Anomalie"
            tk.Label(win,
                     text=f"{anom_label}  —  taux T2 2025 anormalement bas  (z = {z_score:.1f})",
                     bg="#FFF3CD", fg="#7D4E00",
                     font=("Segoe UI", 9, "bold"), padx=10, pady=4
                     ).pack(fill="x", padx=10, pady=(0, 4))

        years   = hist["year"].tolist()
        t2_vals = (hist["taux_T2"]    * 100).tolist()
        t4_vals = (hist["target_T4"]  * 100).tolist()

        hist_tr = hist[hist["year"] < 2025]
        mean_t2 = hist_tr["taux_T2"].mean() * 100
        std_t2  = hist_tr["taux_T2"].std()  * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
        fig.patch.set_facecolor(C_BG)

        bar_colors = [C_CRIT if (y == 2025 and anomalie) else "#4A90D9" for y in years]
        ax1.bar(years, t2_vals, color=bar_colors, alpha=0.85, zorder=3)
        ax1.axhline(mean_t2, color="navy", linestyle="--", linewidth=1.2,
                    label=f"Moy. {mean_t2:.1f}%")
        if not np.isnan(std_t2) and std_t2 > 0:
            ax1.axhspan(mean_t2 - 1.5 * std_t2, mean_t2 + 1.5 * std_t2,
                        alpha=0.12, color="navy", label="\u00b11.5\u03c3")
        ax1.set_title("Taux T2 par ann\u00e9e", fontsize=9)
        ax1.set_xlabel("Ann\u00e9e", fontsize=8)
        ax1.set_ylabel("Taux (%)", fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.set_facecolor("#FAFAFA")
        ax1.set_xticks(years)

        ax2.bar(years, t4_vals, color=C_OK, alpha=0.75, label="T4 r\u00e9alis\u00e9", zorder=3)
        if pred_t4_t2 is not None:
            ax2.bar([2025 - 0.22], [pred_t4_t2 * 100], color="#4A90D9", alpha=0.88,
                    label="T4 pr\u00e9vu T2 (juin)", zorder=4, width=0.26)
        if pred_t4 is not None:
            ax2.bar([2025 + 0.22], [pred_t4 * 100], color=C_WARN, alpha=0.88,
                    label="T4 pr\u00e9vu T3 (sept.)", zorder=4, width=0.26)
        ax2.axhline(80, color="navy", linestyle="--", linewidth=1.0, label="Cible 80%")
        ax2.set_title("Taux T4 par ann\u00e9e", fontsize=9)
        ax2.set_xlabel("Ann\u00e9e", fontsize=8)
        ax2.set_ylabel("Taux (%)", fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.set_facecolor("#FAFAFA")
        ax2.set_xticks(years)

        plt.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", padx=10, pady=(4, 4))

        # summary table
        tcols   = ("Ann\u00e9e", "T2 r\u00e9alis\u00e9", "T4 r\u00e9alis\u00e9", "T4 T2 (juin)", "T4 T3 (sept.)", "\u00c9cart T3")
        twidths = [55, 90, 90, 80, 80, 72]
        inner = tk.Frame(win, bg=C_BG)
        inner.pack(fill="x", padx=10, pady=(0, 10))
        st = ttk.Treeview(inner, columns=tcols, show="headings", height=len(hist))
        for col, w in zip(tcols, twidths):
            st.heading(col, text=col)
            st.column(col, width=w, anchor="center")
        st.tag_configure("anom", background="#FFF3CD", foreground="#7D4E00")
        st.tag_configure("cur",  background="#EBF5FB")
        for _, rd in hist.iterrows():
            y     = int(rd["year"])
            t2    = f"{rd['taux_T2']:.1%}"
            t4r   = f"{rd['target_T4']:.1%}"
            t4pt2 = f"{pred_t4_t2:.1%}" if (y == 2025 and pred_t4_t2 is not None) else "\u2014"
            t4pt3 = f"{pred_t4:.1%}"    if (y == 2025 and pred_t4   is not None) else "\u2014"
            if y == 2025 and pred_t4 is not None:
                ecart = f"{(pred_t4 - rd['target_T4'])*100:+.1f} pp"
            else:
                ecart = "\u2014"
            tag = "anom" if (y == 2025 and anomalie) else ("cur" if y == 2025 else "")
            st.insert("", "end", values=(y, t2, t4r, t4pt2, t4pt3, ecart), tags=(tag,))
        st.pack(fill="x")

    # ── Tab 2 : Graphique barres ──────────────────────────────────────────────
    def _tab_chart(self, nb, df):
        frame = ttk.Frame(nb)
        nb.add(frame, text="  Graphique  ")

        # ── Scrollable container so the tall figure isn't squashed ───────
        outer = tk.Frame(frame, bg=C_BG)
        outer.pack(fill="both", expand=True)

        scroll_canvas = tk.Canvas(outer, bg=C_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical",
                            command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(scroll_canvas, bg=C_BG)
        inner_id = scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_e):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            # make inner frame match canvas width so figure fills horizontally
            scroll_canvas.itemconfigure(inner_id, width=e.width)
        scroll_canvas.bind("<Configure>", _on_canvas_config)

        # Mouse-wheel scrolling (Windows / Linux)
        def _on_wheel(e):
            delta = -1 * (e.delta // 120) if e.delta else (-1 if e.num == 4 else 1)
            scroll_canvas.yview_scroll(delta, "units")
        scroll_canvas.bind_all("<MouseWheel>", _on_wheel)
        scroll_canvas.bind_all("<Button-4>",  _on_wheel)
        scroll_canvas.bind_all("<Button-5>",  _on_wheel)

        # ── Figure (tall: 0.28 inch per line) ────────────────────────────
        n = len(df)
        fig_h = max(6, n * 0.28)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor("#FAFAFA")

        BAR_H = 0.35
        for idx, (_, row) in enumerate(df.iterrows()):
            y = idx
            ax.barh(y + BAR_H / 2, row.pred_T4 * 100,
                    height=BAR_H, color=row.risk_color, alpha=0.88, zorder=3)
            ax.barh(y - BAR_H / 2, row.taux_T2 * 100,
                    height=BAR_H, color="#4A90D9", alpha=0.60, zorder=3)
            ax.text(row.pred_T4 * 100 + 0.5, y + BAR_H / 2,
                    f"{row.pred_T4:.0%}", va="center", fontsize=7)

        ax.axvline(80, color="navy", linestyle="--", linewidth=1.3,
                   label="Seuil cible 80 %", zorder=4)
        ax.axvline(60, color=C_CRIT, linestyle=":",  linewidth=1.0,
                   label="Seuil critique 60 %", zorder=4)

        labels = [
            f"P{int(r.programme)} | {r.ligne_label[:32]} ({r.region_label[:18]})"
            for _, r in df.iterrows()
        ]
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_ylim(-1, n)
        ax.invert_yaxis()
        ax.set_xlim(0, 115)
        ax.set_xlabel("Taux d'exécution (%)", fontsize=9)
        ax.set_title("Prévision T4 par ligne budgétaire — 30 juin 2025\n"
                     "(barres sombres = T4 prévu, barres bleues = T2 réalisé)",
                     fontsize=10, pad=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        # generous left margin for long labels
        fig.subplots_adjust(left=0.32, right=0.97, top=0.985, bottom=0.02)

        mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
        mpl_canvas.draw()
        widget = mpl_canvas.get_tk_widget()
        # Force the widget to render at the figure's native pixel height
        # so each row gets ~28 px instead of being squashed into the tab.
        px_h = int(fig_h * fig.dpi)
        widget.configure(height=px_h)
        widget.pack(fill="both", expand=True)

    # ── Tab 3 : Synthèse par région ───────────────────────────────────────────
    def _tab_region(self, nb, df):
        frame = ttk.Frame(nb)
        nb.add(frame, text="  Synthèse par région  ")

        summary = (df.groupby("region_label")
                     .agg(
                         n_lignes=("ligne_label", "count"),
                         n_attention=("risk_label", lambda x: (x == "Attention").sum()),
                         n_critique=("risk_label", lambda x: (x == "Critique").sum()),
                         mdh_risque=("credit_risque_mdh", "sum"),
                         pred_moy=("pred_T4", "mean"),
                     )
                     .reset_index()
                     .sort_values("mdh_risque", ascending=False))

        cols = ("Région", "Lignes", "Attention", "Critique",
                "MDH à risque", "Taux prévu moyen")
        widths = [230, 65, 80, 75, 110, 140]

        tree = ttk.Treeview(frame, columns=cols, show="headings", height=25)
        for col, w in zip(cols, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center" if w < 200 else "w")

        tree.tag_configure("alerte", background="#FFFBEF", foreground="#7B5200")
        tree.tag_configure("danger", background="#FEF2F2", foreground="#7B0000")
        tree.tag_configure("normal", background="#EDF7EE", foreground="#145a32")

        for _, r in summary.iterrows():
            tag = "danger" if r.n_critique > 0 else \
                  "alerte" if r.n_attention > 0 else "normal"
            tree.insert("", "end", values=(
                r.region_label,
                int(r.n_lignes),
                int(r.n_attention),
                int(r.n_critique),
                f"{r.mdh_risque:.2f}",
                f"{r.pred_moy:.1%}",
            ), tags=(tag,))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True, padx=10, pady=10)


# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
