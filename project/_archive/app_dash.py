"""
Système de Pilotage Budgétaire — Ministère de la Justice
Application Dash — v2 Redesign Moderne

Lancer : python app_dash_modern.py
Ouvrir  : http://127.0.0.1:8050
"""

import os
import io
import base64
import json
import datetime

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
C_BG          = "#F5F7FA"
C_HDR_DARK    = "#1A2E55"
C_HDR_MID     = "#1A2E55"
C_ACCENT      = "#2060C8"
C_OK          = "#15803D"
C_OK_BG       = "#F0FDF4"
C_WARN        = "#B45309"
C_WARN_BG     = "#FFFBEB"
C_CRIT        = "#B91C1C"
C_CRIT_BG     = "#FFF5F5"
C_BLUE        = "#1D4ED8"
C_PURPLE      = "#5B21B6"
C_MUTED       = "#64748B"
C_BORDER      = "#DDE3ED"
C_TEXT        = "#1E293B"
C_TEXT_LT     = "#94A3B8"
C_CARD        = "#FFFFFF"
C_SIDEBAR      = "#0D1F33"
C_SIDEBAR_ACT  = "#1A3A5A"
C_SIDEBAR_TXT  = "#8CAEC4"

FONT      = "'DM Sans', 'Segoe UI', sans-serif"
FONT_MONO = "'DM Mono', 'Consolas', monospace"

_NAV_ITEMS = [
    ("tab-table",  "bi-grid-fill",       "Tableau de bord"),
    ("alerts",     "bi-bell-fill",        "Alertes & Risques"),
    ("prog",       "bi-collection-fill",  "Programmes"),
    ("analyses",   "bi-bar-chart-fill",   "Analyses détaillées"),
    ("region",     "bi-map-fill",         "Synthèse régionale"),
    ("exports",    "bi-download",         "Exports & Rapports"),
]
_TAB_MAP = {
    "tab-table": "tab-table", "alerts": "tab-alerts", "prog": "tab-prog",
    "analyses": "tab-chart",  "region": "tab-region", "exports": "tab-table",
}
def _nav_style(is_active):
    return {
        "display": "flex", "alignItems": "center", "gap": "9px",
        "padding": "7px 14px", "cursor": "pointer", "borderRadius": "5px",
        "color": "#FFFFFF" if is_active else C_SIDEBAR_TXT,
        "background": C_SIDEBAR_ACT if is_active else "transparent",
        "margin": "1px 8px", "fontSize": "0.77rem",
        "fontWeight": "600" if is_active else "400",
    }

_lbl = {"fontSize": "0.76rem", "fontWeight": "600", "color": C_MUTED, "marginBottom": "3px"}
_fd  = {"display": "flex", "flexDirection": "column"}


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def _run_audit_log(df, log_path="audit_log.txt", cache_path="predictions_cache.json"):
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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def build_predictions():
    df = pd.read_csv("data_lines.csv")

    type_map   = {"acquisitions": 0, "equipements": 1, "etudes": 2,
                  "fournitures": 3, "travaux": 4, "personnel": 5}
    prog_map   = {300: 0, 301: 1, 302: 2, 303: 3}
    budget_map = {"INVESTISSEMENT": 0, "MATERIEL": 1, "PERSONNEL": 2}

    df["type_enc"]   = df["type_ligne"].map(type_map)
    df["prog_enc"]   = df["programme"].map(prog_map)
    df["budget_enc"] = df["budget_type"].map(budget_map)
    df["reg_enc"]    = df["region"].astype(int)

    YEARS = sorted(df["year"].unique())

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
                "line_id": lid, "year": year,
                "taux_T2": cur.loc[2, "taux"], "taux_T1": cur.loc[1, "taux"],
                "taux_T4_lag": prev.loc[4, "taux"], "taux_T2_lag": prev.loc[2, "taux"],
                "hist_avg_T3": _t3_lookup.get((lid, year), np.nan),
                "taux_T3": cur.loc[3, "taux"], "taux_T3_lag": prev.loc[3, "taux"],
                "lf_ratio": cur.loc[2, "lf_ratio"], "lf_share": cur.loc[2, "lf_share"],
                "type_enc": cur.loc[2, "type_enc"], "prog_enc": cur.loc[2, "prog_enc"],
                "reg_enc": cur.loc[2, "reg_enc"], "lf_mdh": cur.loc[2, "lf_mdh"],
                "ligne_label": cur.loc[2, "ligne_label"], "type_ligne": cur.loc[2, "type_ligne"],
                "programme": cur.loc[2, "programme"], "programme_label": cur.loc[2, "programme_label"],
                "region_label": cur.loc[2, "region_label"], "budget_type": cur.loc[2, "budget_type"],
                "budget_enc": cur.loc[2, "budget_enc"], "target_T4": cur.loc[4, "taux"],
            })

    pred_df = pd.DataFrame(rows)
    _global_t3 = pred_df["hist_avg_T3"].mean()
    pred_df["hist_avg_T3"] = pred_df["hist_avg_T3"].fillna(_global_t3)
    years_per_line = pred_df.groupby("line_id")["year"].nunique()

    FEATURES    = ["taux_T2", "taux_T1", "taux_T4_lag", "taux_T2_lag", "hist_avg_T3",
                   "lf_ratio", "lf_share", "type_enc", "prog_enc", "reg_enc", "budget_enc"]
    FEATURES_T3 = FEATURES + ["taux_T3", "taux_T3_lag"]

    xgb_params   = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                        subsample=0.8, reg_alpha=0.1, random_state=42, verbosity=0)
    lasso_params = dict(alpha=0.01, max_iter=5000)
    train_years  = [y for y in YEARS if 2021 <= y < 2025]

    all_cv_years  = sorted(pred_df["year"].unique())
    test_years_cv = [y for y in all_cv_years if y < 2025 and (pred_df["year"] < y).any()]
    loyo_per_cat  = {"INVESTISSEMENT": [], "MATERIEL": [], "PERSONNEL": []}

    for y in test_years_cv:
        tr = pred_df[pred_df["year"] < y]
        te = pred_df[pred_df["year"] == y]
        tr_i = tr[tr["budget_type"] == "INVESTISSEMENT"]
        te_i = te[te["budget_type"] == "INVESTISSEMENT"]
        if len(tr_i) >= 2 and len(te_i) >= 1:
            m = XGBRegressor(**xgb_params)
            m.fit(tr_i[FEATURES], tr_i["target_T4"])
            loyo_per_cat["INVESTISSEMENT"].append(
                np.sqrt(mean_squared_error(te_i["target_T4"],
                        m.predict(te_i[FEATURES]).clip(0, 1.30))))
        tr_m = tr[tr["budget_type"] == "MATERIEL"]
        te_m = te[te["budget_type"] == "MATERIEL"]
        if len(tr_m) >= 2 and len(te_m) >= 1:
            m = Pipeline([("sc", StandardScaler()), ("reg", Lasso(**lasso_params))])
            m.fit(tr_m[FEATURES], tr_m["target_T4"])
            loyo_per_cat["MATERIEL"].append(
                np.sqrt(mean_squared_error(te_m["target_T4"],
                        np.clip(m.predict(te_m[FEATURES]), 0, 1.30))))
        tr_p = tr[tr["budget_type"] == "PERSONNEL"]
        te_p = te[te["budget_type"] == "PERSONNEL"]
        if len(tr_p) >= 1 and len(te_p) >= 1:
            hist_p  = tr_p.groupby("line_id")["target_T4"].mean()
            preds_p = te_p["line_id"].map(hist_p).fillna(tr_p["target_T4"].mean()).clip(0, 1.30)
            loyo_per_cat["PERSONNEL"].append(
                np.sqrt(mean_squared_error(te_p["target_T4"], preds_p)))

    wf_cv_summary = {
        "INVESTISSEMENT": {"model": "XGBoost",   "rmse": round(np.mean(loyo_per_cat["INVESTISSEMENT"]), 4)},
        "MATERIEL":       {"model": "Lasso",      "rmse": round(np.mean(loyo_per_cat["MATERIEL"]),       4)},
        "PERSONNEL":      {"model": "Hist. moy",  "rmse": round(np.mean(loyo_per_cat["PERSONNEL"]),      4)},
    }

    train_full = pred_df[pred_df["year"].isin(train_years)]
    test_2025  = pred_df[pred_df["year"] == 2025].copy()
    test_2025["pred_T4"] = np.nan

    tr_i    = train_full[train_full["budget_type"] == "INVESTISSEMENT"]
    xgb_inv = XGBRegressor(**xgb_params)
    xgb_inv.fit(tr_i[FEATURES], tr_i["target_T4"])
    mask_i  = test_2025["budget_type"] == "INVESTISSEMENT"
    test_2025.loc[mask_i, "pred_T4"] = xgb_inv.predict(
        test_2025.loc[mask_i, FEATURES]).clip(0, 1.30)

    tr_m      = train_full[train_full["budget_type"] == "MATERIEL"]
    lasso_mat = Pipeline([("sc", StandardScaler()), ("reg", Lasso(**lasso_params))])
    lasso_mat.fit(tr_m[FEATURES], tr_m["target_T4"])
    mask_m    = test_2025["budget_type"] == "MATERIEL"
    test_2025.loc[mask_m, "pred_T4"] = np.clip(
        lasso_mat.predict(test_2025.loc[mask_m, FEATURES]), 0, 1.30)

    tr_p      = train_full[train_full["budget_type"] == "PERSONNEL"]
    hist_p    = tr_p.groupby("line_id")["target_T4"].mean()
    overall_p = tr_p["target_T4"].mean()
    mask_p    = test_2025["budget_type"] == "PERSONNEL"
    test_2025.loc[mask_p, "pred_T4"] = (
        test_2025.loc[mask_p, "line_id"].map(hist_p).fillna(overall_p).clip(0, 1.30).values
    )

    test_2025["pred_T4_T2"] = test_2025["pred_T4"]
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
    test_2025.loc[mask_p, "pred_T4_T3"] = test_2025.loc[mask_p, "pred_T4_T2"]
    test_2025["pred_T4"] = test_2025["pred_T4_T3"]

    hist = (train_full.groupby("line_id")["taux_T2"]
                      .agg(["mean", "std"])
                      .rename(columns={"mean": "hist_mean_T2", "std": "hist_std_T2"}))
    test_2025 = test_2025.join(hist, on="line_id")
    test_2025["hist_std_T2"] = test_2025["hist_std_T2"].fillna(0.01).clip(lower=0.005)
    test_2025["z_score_T2"]  = ((test_2025["taux_T2"] - test_2025["hist_mean_T2"])
                                 / test_2025["hist_std_T2"]).round(2)
    test_2025["anomalie"] = test_2025["z_score_T2"] < -1.5

    def _anomalie_label(row):
        if not row["anomalie"]: return ""
        t, z = row["taux_T2"], row["z_score_T2"]
        if t == 0.0:     return "Exécution nulle"
        elif z < -3.0:   return "Sous-exécution critique"
        elif z < -2.0:   return "Sous-exécution sévère"
        else:            return "Sous-exécution modérée"

    test_2025["anomalie_label"] = test_2025.apply(_anomalie_label, axis=1)

    def classify(t):
        if t < 0.60: return "Critique"
        if t < 0.80: return "Attention"
        return "OK"

    test_2025["risk_label"] = test_2025["pred_T4"].apply(classify)
    test_2025["credit_risque_mdh"] = (
        test_2025.apply(lambda r: r["lf_mdh"] * max(0, 0.80 - r["pred_T4"])
                        if r["risk_label"] != "OK" else 0.0, axis=1)
    ).round(2)
    test_2025 = test_2025.sort_values("pred_T4").reset_index(drop=True)
    test_2025["hist_years_count"] = (test_2025["line_id"]
                                     .map(years_per_line).fillna(1).astype(int))
    test_2025["low_history"] = test_2025["hist_years_count"] < 3

    build_ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    _run_audit_log(test_2025)

    return test_2025, pred_df, wf_cv_summary, build_ts


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("⏳ Chargement du modèle…")
DF, ALL_DATA, CV_SUMMARY, BUILD_TS = build_predictions()
print(f"✅ Prêt — {len(DF)} lignes chargées")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_pct(v): return f"{v:.1%}"
def fmt_pp(v):  return f"{v:+.1f} pp"

def _risk_colors(label):
    return {
        "OK":        (C_OK,   C_OK_BG),
        "Attention": (C_WARN, C_WARN_BG),
        "Critique":  (C_CRIT, C_CRIT_BG),
    }.get(label, (C_MUTED, C_BG))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + TOPBAR
# ══════════════════════════════════════════════════════════════════════════════
def build_sidebar(n_alerts=0, active_nav="tab-table"):
    nav_items = []
    for nav_id, icon_cls, label in _NAV_ITEMS:
        badge = n_alerts if (nav_id == "alerts" and n_alerts > 0) else None
        nav_items.append(html.Div([
            html.I(className=f"bi {icon_cls}", style={
                "fontSize": "0.82rem", "width": "16px", "textAlign": "center", "flexShrink": "0",
            }),
            html.Span(label, style={"flex": "1"}),
            *([html.Span(str(badge), style={
                "fontSize": "0.58rem", "fontWeight": "700", "color": "#fff",
                "background": C_CRIT, "borderRadius": "10px",
                "padding": "1px 5px", "minWidth": "15px", "textAlign": "center",
            })] if badge else []),
        ], id=f"nav-{nav_id}", n_clicks=0, style=_nav_style(nav_id == active_nav)))

    return html.Div([
        html.Div([
            html.Div(html.I(className="bi bi-shield-fill-check", style={"fontSize": "1.1rem", "color": C_ACCENT}), style={
                "width": "34px", "height": "34px", "borderRadius": "8px",
                "background": "rgba(32,96,200,.18)", "border": "1px solid rgba(32,96,200,.28)",
                "display": "flex", "alignItems": "center", "justifyContent": "center", "flexShrink": "0",
            }),
            html.Div([
                html.Div("SYSTÈME D’ALERTE", style={"fontSize": "0.62rem", "fontWeight": "800", "color": "#fff", "lineHeight": "1.15", "letterSpacing": "0.3px"}),
                html.Div("PRÉCOCE BUDGÉTAIRE", style={"fontSize": "0.62rem", "fontWeight": "800", "color": "#fff", "lineHeight": "1.15", "letterSpacing": "0.3px"}),
            ], style={"marginLeft": "10px"}),
        ], style={
            "display": "flex", "alignItems": "center", "padding": "13px 14px 11px",
            "borderBottom": "1px solid rgba(255,255,255,.07)", "marginBottom": "6px",
        }),
        *nav_items,
        html.Div(style={"flex": "1"}),
        html.Div([
            html.Div([
                html.I(className="bi bi-gear-fill", style={"marginRight": "8px", "fontSize": "0.78rem"}),
                "Paramètres",
            ], style={
                "display": "flex", "alignItems": "center", "padding": "6px 14px",
                "margin": "0 8px", "fontSize": "0.75rem", "color": C_SIDEBAR_TXT, "cursor": "pointer", "borderRadius": "5px",
            }),
            html.Div(style={"borderTop": "1px solid rgba(255,255,255,.07)", "margin": "6px 0 0"}),
            html.Div([
                html.Div("À propos du système", style={"fontSize": "0.63rem", "fontWeight": "700", "color": "#CBD5E1", "marginBottom": "3px"}),
                html.Div("Outil d’aide à la décision pour le suivi budgétaire en temps réel.", style={"fontSize": "0.58rem", "color": C_SIDEBAR_TXT, "lineHeight": "1.4"}),
                html.Div("Ministère de la Justice", style={"fontSize": "0.58rem", "color": C_SIDEBAR_TXT, "marginTop": "5px", "fontWeight": "500"}),
                html.Div("Direction des Affaires Financières", style={"fontSize": "0.58rem", "color": C_SIDEBAR_TXT}),
                html.Div("© 2025 — Exercice 2025", style={"fontSize": "0.55rem", "color": "rgba(255,255,255,.22)", "marginTop": "6px"}),
            ], style={"padding": "8px 14px 12px"}),
        ]),
    ], style={
        "width": "192px", "minWidth": "192px",
        "background": C_SIDEBAR, "height": "100vh",
        "position": "sticky", "top": "0",
        "display": "flex", "flexDirection": "column",
        "overflowY": "auto", "boxShadow": "2px 0 10px rgba(0,0,0,.2)",
        "flexShrink": "0",
    })


def build_topbar(build_ts, n_alerts=0):
    return html.Div([
        html.Div([
            html.Span("Ministère de la Justice, Direction des Affaires Financières", style={
                "fontWeight": "700", "fontSize": "0.9rem", "color": C_TEXT,
            }),
            html.Span(" — Exercice 2025", style={
                "fontSize": "0.84rem", "color": C_MUTED, "fontWeight": "400",
            }),
        ]),
        html.Div([
            html.I(className="bi bi-calendar3", style={"fontSize": "0.82rem", "color": C_MUTED, "marginRight": "6px"}),
            html.Div([
                html.Div("Dernière mise à jour", style={"fontSize": "0.57rem", "color": C_MUTED, "lineHeight": "1"}),
                html.Div(build_ts, style={"fontSize": "0.72rem", "fontWeight": "700", "color": C_TEXT, "lineHeight": "1.3"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "3px"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "9px 20px", "background": C_CARD,
        "borderBottom": f"1px solid {C_BORDER}",
        "boxShadow": "0 1px 4px rgba(0,0,0,.05)", "flexShrink": "0",
    })


# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
def kpi_card(value, label, subtitle, icon, color, bg, progress=None):
    """Professional KPI card with top accent border."""
    return html.Div([
        # Icon badge
        html.Div(
            html.I(className=f"bi {icon}", style={"fontSize": "0.88rem", "color": color}),
            style={
                "position": "absolute", "top": "8px", "right": "10px",
                "width": "26px", "height": "26px", "borderRadius": "6px",
                "background": bg, "display": "flex",
                "alignItems": "center", "justifyContent": "center",
            },
        ),
        # Value
        html.Div(value, style={
            "fontSize": "1.55rem", "fontWeight": "700", "color": color,
            "letterSpacing": "-0.5px", "lineHeight": "1",
            "fontVariantNumeric": "tabular-nums", "marginBottom": "5px",
        }),
        # Label
        html.Div(label, style={
            "fontSize": "0.7rem", "fontWeight": "700", "color": C_TEXT,
            "textTransform": "uppercase", "letterSpacing": "0.5px",
        }),
        # Subtitle
        html.Div(subtitle, style={
            "fontSize": "0.66rem", "color": C_MUTED, "marginTop": "3px",
            "lineHeight": "1.3",
        }),

        # Progress bar (optional)
        *([
            html.Div([
                html.Div(style={
                    "width": f"{min(100,max(0,progress))}%", "height": "100%",
                    "background": color, "borderRadius": "2px",
                })
            ], style={
                "height": "3px", "background": C_BORDER,
                "borderRadius": "2px", "marginTop": "8px",
            })
        ] if progress is not None else []),

    ], style={
        "position": "relative",
        "background": C_CARD,
        "border": f"1px solid {C_BORDER}",
        "borderTop": f"3px solid {color}",
        "borderRadius": "6px",
        "padding": "10px 12px 8px",
        "flex": "1",
        "minWidth": "110px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.05)",
    })


def model_precision_card(cv):
    """Walk-forward CV precision card — redesigned."""
    def rmse_color(v):
        if v < 0.05: return C_OK
        if v < 0.10: return C_WARN
        return C_CRIT

    entries = [
        ("Investissement", "XGBoost",  cv["INVESTISSEMENT"]["rmse"]),
        ("Matériel",       "Lasso",    cv["MATERIEL"]["rmse"]),
        ("Personnel",      "Hist.moy", cv["PERSONNEL"]["rmse"]),
    ]

    rows = []
    for cat, model, rmse in entries:
        rc = rmse_color(rmse)
        rows.append(html.Div([
            html.Div([
                html.Span(cat,   style={"fontSize": "0.72rem", "color": C_TEXT,
                                        "fontWeight": "600"}),
                html.Span(f" / {model}", style={"fontSize": "0.68rem", "color": C_MUTED}),
            ]),
            html.Span(f"RMSE {rmse:.3f}", style={
                "fontSize": "0.7rem", "fontWeight": "700", "color": rc,
                "background": f"{rc}15", "padding": "1px 7px",
                "borderRadius": "10px",
            }),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "4px 0",
            "borderBottom": f"1px solid {C_BORDER}",
        }))

    return html.Div([
        html.Div([
            html.Div("Précision du modèle", style={
                "fontSize": "0.7rem", "fontWeight": "700", "color": C_TEXT,
                "textTransform": "uppercase", "letterSpacing": "0.6px",
            }),
            html.Div("Validation walk-forward par exercice", style={
                "fontSize": "0.64rem", "color": C_MUTED, "marginTop": "2px",
            }),
        ], style={"marginBottom": "8px"}),
        html.Div(rows),
    ], style={
        "background": C_CARD,
        "border": f"1px solid {C_BORDER}",
        "borderTop": f"3px solid {C_BLUE}",
        "borderRadius": "6px",
        "padding": "10px 12px 8px",
        "minWidth": "200px",
        "boxShadow": "0 1px 3px rgba(0,0,0,.05)",
    })


def build_kpi_row(df, cv):
    n_ok   = int((df["risk_label"] == "OK").sum())
    n_warn = int((df["risk_label"] == "Attention").sum())
    n_crit = int((df["risk_label"] == "Critique").sum())
    total  = len(df)
    mdh    = df["credit_risque_mdh"].sum()
    n_anom = int(df["anomalie"].sum())
    pct_ok = (n_ok / total * 100) if total else 0

    return html.Div([
        # ── KPI cards ─────────────────────────────────────────────────────────
        html.Div([
            kpi_card(str(total), "Lignes analysées",
                     "Dotations budgétaires suivies",
                     "bi-clipboard2-data-fill", C_BLUE, f"{C_BLUE}08"),
            kpi_card(str(n_ok), "Conformes ≥ 80 %",
                     "Exécution prévisionnelle dans les objectifs",
                     "bi-check-circle-fill", C_OK, C_OK_BG, progress=pct_ok),
            kpi_card(str(n_warn), "Attention 60–80 %",
                     "Risque modéré — suivi renforcé conseillé",
                     "bi-exclamation-triangle-fill", C_WARN, C_WARN_BG),
            kpi_card(str(n_crit), "Critique < 60 %",
                     "Intervention requise — sous-exécution grave",
                     "bi-x-circle-fill", C_CRIT, C_CRIT_BG),
            kpi_card(f"{mdh:.1f} MDH", "Crédits à risque",
                     "Montant estimé sous le seuil d'alerte (80%)",
                     "bi-graph-down-arrow", C_PURPLE, f"{C_PURPLE}08"),
            kpi_card(str(n_anom), "Anomalies détectées",
                     "Z-score < −1.5 vs historique T2",
                     "bi-activity", "#D97706", C_WARN_BG),
        ], style={
            "display": "flex", "gap": "10px", "flex": "1", "flexWrap": "wrap",
        }),
        # ── Precision card ────────────────────────────────────────────────────
        model_precision_card(cv),
    ], style={
        "display": "flex", "gap": "8px", "padding": "8px 16px 6px",
        "alignItems": "stretch",
    })


# ══════════════════════════════════════════════════════════════════════════════
# RISK LEGEND STRIP
# ══════════════════════════════════════════════════════════════════════════════
def risk_legend():
    items = [
        ("● OK ≥ 80 %",         C_OK,   "Exécution conforme aux objectifs annuels"),
        ("● Attention 60–80 %", C_WARN, "Risque de sous-exécution — surveillance requise"),
        ("● Critique < 60 %",   C_CRIT, "Sous-exécution grave — intervention budgétaire nécessaire"),
    ]
    return html.Div([
        html.Span("Niveaux de risque :", style={
            "fontSize": "0.7rem", "fontWeight": "700", "color": C_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "marginRight": "14px", "flexShrink": "0",
        }),
        *[
            html.Div([
                html.Span(label, style={
                    "fontSize": "0.72rem", "fontWeight": "700", "color": color,
                    "marginRight": "5px",
                }),
                html.Span(f"— {desc}", style={"fontSize": "0.7rem", "color": C_MUTED}),
            ], style={"marginRight": "20px", "display": "flex", "alignItems": "center"})
            for label, color, desc in items
        ],
        html.Div([
            html.Span("T2", style={"fontWeight": "700", "color": C_TEXT}),
            " = Taux d'exécution réel au 30 juin  ·  ",
            html.Span("T4", style={"fontWeight": "700", "color": C_TEXT}),
            " = Prévision fin d'exercice (31 déc.)  ·  ",
            html.Span("MDH", style={"fontWeight": "700", "color": C_TEXT}),
            " = Millions de dirhams",
        ], style={
            "fontSize": "0.68rem", "color": C_TEXT_LT,
            "marginLeft": "auto", "flexShrink": "0",
            "borderLeft": f"1px solid {C_BORDER}", "paddingLeft": "14px",
        }),
    ], style={
        "display": "flex", "alignItems": "center", "flexWrap": "nowrap",
        "overflowX": "auto",
        "padding": "5px 16px",
        "background": C_CARD,
        "borderTop": f"1px solid {C_BORDER}",
        "borderBottom": f"1px solid {C_BORDER}",
        "gap": "4px",
    })


# ══════════════════════════════════════════════════════════════════════════════
# TABLE TAB
# ══════════════════════════════════════════════════════════════════════════════
def build_table_data(df):
    rows = []
    for _, r in df.iterrows():
        anom_parts = []
        if r.anomalie:
            anom_parts.append(f"⚠ {r.anomalie_label} (z={r.z_score_T2:.1f})")
        if r.low_history:
            anom_parts.append(f"⚠ Hist. limitée ({r.hist_years_count} an(s))")
        rows.append({
            "line_id":     str(r.line_id),
            "Programme":   f"P{int(r.programme)} – {r.programme_label}",
            "Ligne":       r.ligne_label,
            "Région":      r.region_label,
            "Budget":      r.budget_type,
            "LF (MDH)":    round(r.lf_mdh, 1),
            "T2 réel":     f"{r.taux_T2:.1%}",
            "T4 (juin)":   f"{r.pred_T4_T2:.1%}",
            "T4 (sept.)":  f"{r.pred_T4:.1%}",
            "T4 réel":     f"{r.target_T4:.1%}",
            "Écart (pp)":  fmt_pp((r.pred_T4 - r.target_T4) * 100),
            "Risque":      r.risk_label,
            "MDH risque":  round(r.credit_risque_mdh, 2),
            "Anomalie":    "  ·  ".join(anom_parts),
            "_risk":       r.risk_label,
        })
    return rows


TABLE_COLS = [
    {"name": "Programme",   "id": "Programme",  "type": "text"},
    {"name": "Ligne",       "id": "Ligne",      "type": "text"},
    {"name": "Région",      "id": "Région",     "type": "text"},
    {"name": "Budget",      "id": "Budget",     "type": "text"},
    {"name": "LF (MDH)",    "id": "LF (MDH)",   "type": "numeric"},
    {"name": "T2 réel ★",   "id": "T2 réel",    "type": "text"},
    {"name": "T4 prévu juin","id": "T4 (juin)",  "type": "text"},
    {"name": "T4 prévu sept.","id":"T4 (sept.)", "type": "text"},
    {"name": "T4 réel",     "id": "T4 réel",    "type": "text"},
    {"name": "Écart (pp)",  "id": "Écart (pp)", "type": "text"},
    {"name": "Risque",      "id": "Risque",     "type": "text"},
    {"name": "MDH risque",  "id": "MDH risque", "type": "numeric"},
    {"name": "Anomalie",    "id": "Anomalie",   "type": "text"},
]


def tab_table():
    prog_labels = {int(r.programme): r.programme_label
                   for _, r in DF[["programme", "programme_label"]].drop_duplicates().iterrows()}
    progs   = [{"label": "Tous les programmes", "value": "Tous"}] + \
              [{"label": f"P{p} – {prog_labels[p]}", "value": str(p)} for p in sorted(prog_labels)]
    regions = [{"label": "Toutes les régions", "value": "Toutes"}] + \
              [{"label": r, "value": r} for r in sorted(DF["region_label"].unique())]
    risques = [{"label": l, "value": l} for l in ["Tous", "Critique", "Attention", "OK"]]
    budgets = [{"label": l, "value": l} for l in ["Tous", "INVESTISSEMENT", "MATERIEL", "PERSONNEL"]]

    dd_style = {"fontSize": "0.79rem", "minWidth": "160px"}
    lbl_style = {"fontSize": "0.65rem", "fontWeight": "700", "color": C_MUTED,
                 "textTransform": "uppercase", "letterSpacing": "0.4px", "marginBottom": "3px"}

    filter_bar = html.Div([
        html.Div([
            html.Label("Programme", style=lbl_style),
            dcc.Dropdown(progs,   value="Tous",    id="dd-prog",   clearable=False, style=dd_style),
        ], style=_fd),
        html.Div([
            html.Label("Région", style=lbl_style),
            dcc.Dropdown(regions, value="Toutes",  id="dd-reg",    clearable=False, style=dd_style),
        ], style=_fd),
        html.Div([
            html.Label("Risque", style=lbl_style),
            dcc.Dropdown(risques, value="Tous",    id="dd-risk",   clearable=False,
                         style={"fontSize": "0.79rem", "minWidth": "110px"}),
        ], style=_fd),
        html.Div([
            html.Label("Budget", style=lbl_style),
            dcc.Dropdown(budgets, value="Tous",    id="dd-budget", clearable=False,
                         style={"fontSize": "0.79rem", "minWidth": "140px"}),
        ], style=_fd),
        html.Div([
            html.Button(
                "Exporter CSV",
                id="btn-export", n_clicks=0, style={
                "background": C_BLUE, "color": "#fff", "border": "none",
                "borderRadius": "5px", "padding": "6px 14px",
                "cursor": "pointer", "fontWeight": "600", "fontSize": "0.78rem",
            }),
            dcc.Download(id="download-csv"),
        ], style={"alignSelf": "flex-end"}),
        html.Div(id="row-count", style={
            "alignSelf": "center", "marginLeft": "auto",
            "color": C_MUTED, "fontSize": "0.78rem", "fontWeight": "600",
        }),
    ], style={
        "display": "flex", "alignItems": "flex-end", "gap": "8px",
        "flexWrap": "wrap", "padding": "8px 12px",
        "background": C_CARD,
        "border": f"1px solid {C_BORDER}",
        "borderRadius": "6px", "margin": "6px 12px 0",
        "boxShadow": "0 1px 3px rgba(0,0,0,.04)",
    })

    detail_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
        dbc.ModalBody(id="modal-body"),
    ], id="modal-detail", size="xl", is_open=False)

    table = dash_table.DataTable(
        id="main-table",
        columns=TABLE_COLS,
        data=build_table_data(DF),
        sort_action="native",
        filter_action="none",
        page_size=20,
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_header={
            "backgroundColor": C_HDR_DARK, "color": "#fff",
            "fontWeight": "700", "fontSize": "0.78rem",
            "border": "none", "padding": "9px 10px",
            "textTransform": "uppercase", "letterSpacing": "0.4px",
            "fontFamily": FONT,
        },
        style_cell={
            "fontSize": "0.82rem", "padding": "7px 10px",
            "border": f"1px solid {C_BORDER}",
            "fontFamily": FONT,
            "overflow": "hidden", "textOverflow": "ellipsis", "maxWidth": "220px",
            "color": C_TEXT,
        },
        style_cell_conditional=[
            {"if": {"column_id": "Programme"},  "minWidth": "200px", "textAlign": "left"},
            {"if": {"column_id": "Ligne"},       "minWidth": "190px", "textAlign": "left"},
            {"if": {"column_id": "Anomalie"},    "minWidth": "180px", "textAlign": "left", "color": C_WARN},
        ] + [
            {"if": {"column_id": c}, "textAlign": "center", "fontFamily": FONT_MONO,
             "fontWeight": "600"}
            for c in ["T2 réel", "T4 (juin)", "T4 (sept.)", "T4 réel", "Écart (pp)"]
        ],
        style_data_conditional=(
            [{"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFC"}] +
            [{"if": {"filter_query": '{_risk} = "OK"'},
              "backgroundColor": C_OK_BG, "color": "#064E3B"}] +
            [{"if": {"filter_query": '{_risk} = "Attention"'},
              "backgroundColor": C_WARN_BG, "color": "#78350F"}] +
            [{"if": {"filter_query": '{_risk} = "Critique"'},
              "backgroundColor": C_CRIT_BG, "color": "#7F1D1D"}] +
            [{"if": {"state": "selected"},
              "backgroundColor": "#DBEAFE", "border": f"1px solid {C_BLUE}"}]
        ),
        row_selectable=False,
        selected_rows=[],
        tooltip_delay=0,
        tooltip_duration=None,
    )

    return html.Div([
        filter_bar,
        html.Div(table, style={"margin": "6px 12px"}),
        detail_modal,
    ])


# ══════════════════════════════════════════════════════════════════════════════
# CHART TAB
# ══════════════════════════════════════════════════════════════════════════════
def build_bar_chart(df):
    color_map = {"OK": C_OK, "Attention": C_WARN, "Critique": C_CRIT}
    labels = [f"P{int(r.programme)} | {r.ligne_label[:35]} ({r.region_label[:16]})"
              for _, r in df.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=df["taux_T2"] * 100,
        name="T2 réel (30 juin)", orientation="h",
        marker_color=C_BLUE, opacity=0.40,
        hovertemplate="<b>%{y}</b><br>T2 réel : %{x:.1f} %<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=df["pred_T4"] * 100,
        name="T4 prévu (31 déc.)", orientation="h",
        marker_color=[color_map[r] for r in df["risk_label"]],
        opacity=0.90,
        hovertemplate="<b>%{y}</b><br>T4 prévu : %{x:.1f} %<extra></extra>",
    ))
    fig.add_vline(x=80, line_dash="dash", line_color=C_HDR_DARK, line_width=1.5,
                  annotation_text="Cible 80 %", annotation_position="top right",
                  annotation_font_color=C_HDR_DARK, annotation_font_size=11)
    fig.add_vline(x=60, line_dash="dot", line_color=C_CRIT, line_width=1.5,
                  annotation_text="Seuil critique 60 %", annotation_position="bottom right",
                  annotation_font_color=C_CRIT, annotation_font_size=11)
    fig.update_layout(
        barmode="overlay",
        height=max(520, len(df) * 26),
        margin=dict(l=10, r=20, t=50, b=20),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_CARD,
        font=dict(family=FONT, size=11, color=C_TEXT),
        legend=dict(
            orientation="h", y=1.04, x=1, xanchor="right",
            bgcolor="rgba(255,255,255,.85)", bordercolor=C_BORDER, borderwidth=1,
        ),
        title=dict(
            text="Prévision T4 par ligne budgétaire — Exercice 2025",
            font=dict(size=14, color=C_HDR_DARK, family=FONT),
            x=0.01,
        ),
        xaxis=dict(
            title="Taux d'exécution (%)", range=[0, 120],
            gridcolor="#EEF0F3", zeroline=False,
            ticksuffix=" %",
        ),
        yaxis=dict(tickfont=dict(size=9.5)),
    )
    return fig


def tab_chart():
    return html.Div([
        html.Div([
            html.Span("Comparaison T2 réel vs T4 prévu par ligne budgétaire", style={
                "fontSize": "0.85rem", "fontWeight": "700", "color": C_MUTED,
                "textTransform": "uppercase", "letterSpacing": "0.4px",
            }),
            html.Span(" — Cliquez sur une barre pour zoomer", style={
                "fontSize": "0.75rem", "color": C_TEXT_LT, "marginLeft": "10px",
            }),
        ], style={"padding": "14px 16px 0", "display": "flex", "alignItems": "center"}),
        dcc.Graph(
            id="bar-chart",
            figure=build_bar_chart(DF),
            style={"height": "100%"},
            config={"displayModeBar": True, "scrollZoom": True},
        ),
    ], style={"overflowY": "auto", "height": "calc(100vh - 260px)"})


# ══════════════════════════════════════════════════════════════════════════════
# REGION TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_region():
    summary = (DF.groupby("region_label")
                 .agg(
                     n_lignes    = ("ligne_label",       "count"),
                     n_attention = ("risk_label", lambda x: (x == "Attention").sum()),
                     n_critique  = ("risk_label", lambda x: (x == "Critique").sum()),
                     mdh_risque  = ("credit_risque_mdh", "sum"),
                     pred_moy    = ("pred_T4",           "mean"),
                 )
                 .reset_index()
                 .sort_values("mdh_risque", ascending=False))

    rows = []
    for _, r in summary.iterrows():
        tag = "danger" if r.n_critique > 0 else ("alerte" if r.n_attention > 0 else "normal")
        rows.append({
            "Région":           r.region_label,
            "Lignes":           int(r.n_lignes),
            "Attention":        int(r.n_attention),
            "Critique":         int(r.n_critique),
            "MDH à risque":     round(r.mdh_risque, 2),
            "Taux prévu moyen": f"{r.pred_moy:.1%}",
            "_tag": tag,
        })

    cols = [{"name": c, "id": c} for c in
            ["Région", "Lignes", "Attention", "Critique", "MDH à risque", "Taux prévu moyen"]]

    table = dash_table.DataTable(
        columns=cols, data=rows,
        sort_action="native",
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_header={
            "backgroundColor": C_HDR_DARK, "color": "#fff",
            "fontWeight": "700", "fontSize": "0.8rem",
            "border": "none", "padding": "12px 14px",
            "textTransform": "uppercase", "letterSpacing": "0.4px",
            "fontFamily": FONT,
        },
        style_cell={
            "fontSize": "0.84rem", "padding": "10px 14px",
            "border": f"1px solid {C_BORDER}",
            "fontFamily": FONT, "color": C_TEXT,
        },
        style_cell_conditional=[
            {"if": {"column_id": "Région"}, "textAlign": "left", "minWidth": "200px"},
        ] + [
            {"if": {"column_id": c}, "textAlign": "center", "fontWeight": "600"}
            for c in ["Lignes", "Attention", "Critique"]
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFC"},
            {"if": {"filter_query": '{_tag} = "danger"'},
             "backgroundColor": C_CRIT_BG, "color": "#7F1D1D"},
            {"if": {"filter_query": '{_tag} = "alerte"'},
             "backgroundColor": C_WARN_BG, "color": "#78350F"},
            {"if": {"filter_query": '{_tag} = "normal"'},
             "backgroundColor": C_OK_BG,   "color": "#064E3B"},
        ],
    )

    bubble = px.scatter(
        summary, x="pred_moy", y="region_label",
        size=summary["mdh_risque"].clip(lower=0.1),
        color=summary["n_critique"].apply(lambda x: "Critique" if x > 0 else "Conforme"),
        color_discrete_map={"Critique": C_CRIT, "Conforme": C_OK},
        hover_data={"mdh_risque": ":.2f", "n_attention": True, "n_critique": True},
        labels={"pred_moy": "Taux d'exécution prévu moyen", "region_label": "Région"},
        title="Vue régionale — taux moyen prévu & crédits à risque (taille des bulles = MDH à risque)",
    )
    bubble.update_layout(
        paper_bgcolor=C_BG, plot_bgcolor=C_CARD,
        font=dict(family=FONT, color=C_TEXT),
        margin=dict(l=10, r=20, t=50, b=20),
        showlegend=True,
        xaxis=dict(tickformat=".0%", gridcolor="#EEF0F3",
                   title="Taux d'exécution prévu moyen (T4)"),
        height=440,
        title=dict(font=dict(size=13, color=C_HDR_DARK)),
    )

    return html.Div([
        html.Div([
            html.Div("Synthèse par région administrative", style={
                "fontSize": "0.82rem", "fontWeight": "700",
                "color": C_MUTED, "textTransform": "uppercase",
                "letterSpacing": "0.4px",
            }),
        ], style={"padding": "14px 16px 8px"}),
        html.Div(table, style={"margin": "0 16px 16px"}),
        dcc.Graph(figure=bubble, config={"displayModeBar": False},
                  style={"margin": "0 16px 16px"}),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# ALERTES TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_alerts():
    df_alerts = DF[DF["anomalie"] | (DF["risk_label"] == "Critique")].copy()
    if df_alerts.empty:
        return html.Div("Aucune alerte détectée.", style={"padding": "32px", "color": C_MUTED, "fontSize": "0.9rem"})

    rows = []
    for _, r in df_alerts.sort_values("credit_risque_mdh", ascending=False).iterrows():
        rows.append({
            "Programme": f"P{int(r.programme)} \u2013 {r.programme_label}",
            "Ligne":     r.ligne_label,
            "R\u00e9gion": r.region_label,
            "Risque":    r.risk_label,
            "T2 r\u00e9el": f"{r.taux_T2:.1%}",
            "T4 pr\u00e9vu": f"{r.pred_T4:.1%}",
            "MDH \u00e0 risque": round(r.credit_risque_mdh, 2),
            "Anomalie": r.anomalie_label if r.anomalie else "",
        })

    cols = [{"name": c, "id": c} for c in
            ["Programme", "Ligne", "R\u00e9gion", "Risque", "T2 r\u00e9el", "T4 pr\u00e9vu", "MDH \u00e0 risque", "Anomalie"]]

    table = dash_table.DataTable(
        id="alerts-table",
        columns=cols, data=rows,
        sort_action="native",
        page_size=25,
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_header={
            "backgroundColor": C_HDR_DARK, "color": "#fff",
            "fontWeight": "700", "fontSize": "0.78rem",
            "border": "none", "padding": "9px 10px",
            "textTransform": "uppercase", "letterSpacing": "0.4px", "fontFamily": FONT,
        },
        style_cell={
            "fontSize": "0.82rem", "padding": "7px 10px",
            "border": f"1px solid {C_BORDER}", "fontFamily": FONT,
            "overflow": "hidden", "textOverflow": "ellipsis", "maxWidth": "220px", "color": C_TEXT,
        },
        style_cell_conditional=[
            {"if": {"column_id": "Programme"}, "minWidth": "200px", "textAlign": "left"},
            {"if": {"column_id": "Ligne"},     "minWidth": "180px", "textAlign": "left"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFC"},
            {"if": {"filter_query": '{Risque} = "Critique"'}, "backgroundColor": C_CRIT_BG, "color": "#7F1D1D"},
            {"if": {"filter_query": '{Risque} = "Attention"'}, "backgroundColor": C_WARN_BG, "color": "#78350F"},
        ],
    )

    summary = html.Div([
        html.Span(f"{len(df_alerts)} ligne(s) en alerte", style={
            "fontSize": "0.82rem", "fontWeight": "700", "color": C_CRIT, "marginRight": "16px",
        }),
        html.Span(f"MDH total \u00e0 risque : {df_alerts['credit_risque_mdh'].sum():.1f} MDH", style={
            "fontSize": "0.82rem", "color": C_MUTED,
        }),
    ], style={"padding": "10px 16px 6px", "display": "flex", "alignItems": "center"})

    return html.Div([
        html.Div("Alertes & Risques \u2014 Lignes n\u00e9cessitant une attention", style={
            "fontSize": "0.82rem", "fontWeight": "700", "color": C_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.4px",
            "padding": "14px 16px 4px",
        }),
        summary,
        html.Div(table, style={"margin": "0 12px 16px"}),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PROGRAMMES TAB
# ══════════════════════════════════════════════════════════════════════════════
def tab_prog():
    summary = (DF.groupby(["programme", "programme_label"])
                 .agg(
                     n_lignes   = ("ligne_label", "count"),
                     n_ok       = ("risk_label", lambda x: (x == "OK").sum()),
                     n_warn     = ("risk_label", lambda x: (x == "Attention").sum()),
                     n_crit     = ("risk_label", lambda x: (x == "Critique").sum()),
                     lf_total   = ("lf_mdh", "sum"),
                     mdh_risque = ("credit_risque_mdh", "sum"),
                     pred_moy   = ("pred_T4", "mean"),
                 )
                 .reset_index()
                 .sort_values("programme"))

    rows = []
    for _, r in summary.iterrows():
        tag = "critique" if r.n_crit > 0 else ("attention" if r.n_warn > 0 else "ok")
        rows.append({
            "Programme": f"P{int(r.programme)} \u2013 {r.programme_label}",
            "Lignes": int(r.n_lignes),
            "OK": int(r.n_ok),
            "Attention": int(r.n_warn),
            "Critique": int(r.n_crit),
            "LF total (MDH)": round(r.lf_total, 1),
            "MDH \u00e0 risque": round(r.mdh_risque, 2),
            "T4 moyen pr\u00e9vu": f"{r.pred_moy:.1%}",
            "_tag": tag,
        })

    cols = [{"name": c, "id": c} for c in
            ["Programme", "Lignes", "OK", "Attention", "Critique", "LF total (MDH)", "MDH \u00e0 risque", "T4 moyen pr\u00e9vu"]]

    table = dash_table.DataTable(
        id="prog-table",
        columns=cols, data=rows,
        sort_action="native",
        style_table={"overflowX": "auto", "borderRadius": "10px", "overflow": "hidden"},
        style_header={
            "backgroundColor": C_HDR_DARK, "color": "#fff",
            "fontWeight": "700", "fontSize": "0.78rem",
            "border": "none", "padding": "9px 10px",
            "textTransform": "uppercase", "letterSpacing": "0.4px", "fontFamily": FONT,
        },
        style_cell={
            "fontSize": "0.82rem", "padding": "7px 10px",
            "border": f"1px solid {C_BORDER}", "fontFamily": FONT, "color": C_TEXT,
        },
        style_cell_conditional=[
            {"if": {"column_id": "Programme"}, "minWidth": "240px", "textAlign": "left"},
        ] + [
            {"if": {"column_id": c}, "textAlign": "center", "fontWeight": "600"}
            for c in ["Lignes", "OK", "Attention", "Critique"]
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#FAFBFC"},
            {"if": {"filter_query": '{_tag} = "critique"'}, "backgroundColor": C_CRIT_BG, "color": "#7F1D1D"},
            {"if": {"filter_query": '{_tag} = "attention"'}, "backgroundColor": C_WARN_BG, "color": "#78350F"},
            {"if": {"filter_query": '{_tag} = "ok"'}, "backgroundColor": C_OK_BG, "color": "#064E3B"},
        ],
    )

    return html.Div([
        html.Div("Synth\u00e8se par programme budg\u00e9taire", style={
            "fontSize": "0.82rem", "fontWeight": "700", "color": C_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.4px",
            "padding": "14px 16px 8px",
        }),
        html.Div(table, style={"margin": "0 12px 16px"}),
    ])


# ══════════════════════════════════════════════════════════════════════════════
def build_detail(line_id):
    hist = ALL_DATA[ALL_DATA["line_id"] == line_id].sort_values("year").copy()
    cur  = DF[DF["line_id"] == line_id]
    if hist.empty or cur.empty:
        return "Aucune donnée.", ""

    info       = cur.iloc[0]
    anomalie   = bool(info["anomalie"])
    z_score    = float(info["z_score_T2"])
    pred_t4    = float(info["pred_T4"])
    pred_t4_t2 = float(info["pred_T4_T2"])

    title = f"P{int(info.programme)} · {info.ligne_label} · {info.region_label}"

    years   = hist["year"].tolist()
    t2_vals = (hist["taux_T2"]   * 100).tolist()
    t4_vals = (hist["target_T4"] * 100).tolist()
    hist_tr = hist[hist["year"] < 2025]
    mean_t2 = hist_tr["taux_T2"].mean() * 100
    std_t2  = hist_tr["taux_T2"].std()  * 100

    _chart_layout = dict(
        paper_bgcolor=C_CARD, plot_bgcolor="#FAFBFC",
        font=dict(family=FONT, size=11, color=C_TEXT),
        margin=dict(l=10, r=10, t=38, b=20),
        showlegend=False,
        xaxis=dict(tickvals=years, gridcolor=C_BORDER),
    )

    bar_colors = [C_CRIT if (y == 2025 and anomalie) else C_BLUE for y in years]
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=years, y=t2_vals, marker_color=bar_colors,
                          name="T2 réel", hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig1.add_hline(y=mean_t2, line_dash="dash", line_color=C_HDR_DARK, line_width=1.2,
                   annotation_text=f"Moy. {mean_t2:.1f} %")
    if not np.isnan(std_t2) and std_t2 > 0:
        fig1.add_hrect(y0=mean_t2 - 1.5*std_t2, y1=mean_t2 + 1.5*std_t2,
                       fillcolor=C_BLUE, opacity=0.06, line_width=0)
    fig1.update_layout(title=dict(text="Taux T2 par année", font=dict(size=12, color=C_HDR_DARK)),
                       height=260, yaxis=dict(ticksuffix=" %"), **_chart_layout)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=[y for y in years if y < 2025],
                          y=[v for v, y in zip(t4_vals, years) if y < 2025],
                          marker_color=C_OK, opacity=0.75, name="T4 réalisé",
                          hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig2.add_trace(go.Bar(x=[2024.7], y=[pred_t4_t2*100], width=0.25,
                          marker_color=C_BLUE, name="T4 prévu T2 (juin)"))
    fig2.add_trace(go.Bar(x=[2025.3], y=[pred_t4*100], width=0.25,
                          marker_color=C_WARN, name="T4 prévu T3 (sept.)"))
    if 2025 in years:
        idx = years.index(2025)
        fig2.add_trace(go.Bar(x=[2025], y=[t4_vals[idx]], width=0.25,
                              marker_color=C_OK, opacity=0.75, showlegend=False))
    fig2.add_hline(y=80, line_dash="dash", line_color=C_HDR_DARK, line_width=1.2,
                   annotation_text="Cible 80 %")
    fig2.update_layout(title=dict(text="Taux T4 par année", font=dict(size=12, color=C_HDR_DARK)),
                       height=260, barmode="overlay",
                       yaxis=dict(ticksuffix=" %"),
                       legend=dict(orientation="h", y=1.18, font=dict(size=10)),
                       showlegend=True,
                       **{k: v for k, v in _chart_layout.items() if k != "showlegend"})

    tbl_rows = []
    for _, rd in hist.iterrows():
        y = int(rd["year"])
        tbl_rows.append({
            "Année":         y,
            "T2 réalisé":    f"{rd['taux_T2']:.1%}",
            "T4 réalisé":    f"{rd['target_T4']:.1%}",
            "T4 prévu juin": f"{pred_t4_t2:.1%}" if y == 2025 else "—",
            "T4 prévu sept.":f"{pred_t4:.1%}"    if y == 2025 else "—",
            "Écart T3":      fmt_pp((pred_t4 - rd["target_T4"])*100) if y == 2025 else "—",
            "_cur": "yes" if y == 2025 else "no",
        })

    detail_table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in
                 ["Année", "T2 réalisé", "T4 réalisé",
                  "T4 prévu juin", "T4 prévu sept.", "Écart T3"]],
        data=tbl_rows,
        style_table={"overflowX": "auto", "borderRadius": "8px", "overflow": "hidden"},
        style_header={"backgroundColor": C_HDR_DARK, "color": "#fff",
                      "fontWeight": "700", "fontSize": "0.78rem",
                      "border": "none", "padding": "10px 12px", "fontFamily": FONT},
        style_cell={"fontSize": "0.82rem", "padding": "8px 12px",
                    "border": f"1px solid {C_BORDER}", "textAlign": "center",
                    "fontFamily": FONT_MONO, "color": C_TEXT},
        style_data_conditional=[
            {"if": {"filter_query": '{_cur} = "yes"'},
             "backgroundColor": "#EFF6FF", "fontWeight": "700",
             "borderLeft": f"3px solid {C_BLUE}"},
        ],
    )

    anom_banner = html.Div([
        html.Span("⚠ ", style={"fontSize": "1.1rem"}),
        f"{info['anomalie_label']} — T2 anormalement bas (z-score = {z_score:.1f})",
    ], style={
        "background": "#FFFBEB", "color": "#78350F",
        "padding": "10px 16px", "borderRadius": "8px",
        "fontWeight": "600", "fontSize": "0.88rem",
        "border": f"1px solid {C_WARN}40",
        "borderLeft": f"3px solid {C_WARN}",
        "marginBottom": "14px",
    }) if anomalie else None

    body = html.Div([
        anom_banner,
        html.Div([
            dcc.Graph(figure=fig1, config={"displayModeBar": False}, style={"flex": "1"}),
            dcc.Graph(figure=fig2, config={"displayModeBar": False}, style={"flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
        html.Div(detail_table, style={"marginTop": "14px"}),
    ])

    return body, title


# ══════════════════════════════════════════════════════════════════════════════
# APP + CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Pilotage Budgétaire — MJ",
)

app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      body {
        font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
        margin: 0; background: #F5F7FA;
      }
      /* Tab bar */
      .dash-tabs { border-bottom: 1px solid #DDE3ED!important; }
      .dash-tab {
        border: none!important; background: #F5F7FA!important;
        color: #64748B!important; font-weight: 600!important;
        font-size: 0.75rem!important; padding: 6px 16px!important;
        border-bottom: 2px solid transparent!important;
        transition: color .15s, border-color .15s!important;
        font-family: 'DM Sans', sans-serif!important;
        letter-spacing: 0.2px!important;
      }
      .dash-tab:hover { color: #1A2E55!important; background: #EEF1F6!important; }
      .dash-tab--selected {
        color: #1A2E55!important;
        border-bottom: 2px solid #2060C8!important;
        background: #fff!important;
      }
      /* Dropdown */
      .Select-control { border-radius: 6px!important; border: 1px solid #DDE3ED!important; }
      .Select-control:hover { border-color: #2060C8!important; }
      /* Table pagination */
      .previous-page, .next-page, .last-page, .first-page {
        font-family: 'DM Sans', sans-serif!important;
        font-size: 0.82rem!important;
        color: #0D1B2A!important;
      }
      /* Table filter row */
      input.dash-filter--case { font-family: 'DM Sans', sans-serif!important; }
      /* Scrollbar */
      ::-webkit-scrollbar { width: 6px; height: 6px; }
      ::-webkit-scrollbar-track { background: #F5F7FA; }
      ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
      ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
      /* Export button hover */
      #btn-export:hover { opacity: .88; transition: .15s; }
      /* Sidebar nav hover */
      [id^="nav-"]:hover { background: rgba(255,255,255,.08)!important; color: #fff!important; }
      [id^="nav-"] { transition: background .15s, color .15s; }
      /* Dropdown menu must escape overflow containers */
      .Select-menu-outer { z-index: 9999 !important; }
      .VirtualizedSelectOption { font-size: 0.82rem !important; }
      /* Compact tab bar */
      .dash-tabs .dash-tab { min-height: 32px !important; }
      .dash-tabs .rc-tabs-tab-btn { min-height: 32px !important; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
'''

app.layout = html.Div([

    # ── Sidebar ───────────────────────────────────────────────────────────────
    build_sidebar(n_alerts=int(DF["anomalie"].sum()), active_nav="tab-table"),

    # ── Main column ───────────────────────────────────────────────────────────
    html.Div([
        build_topbar(BUILD_TS, n_alerts=int(DF["anomalie"].sum())),
        build_kpi_row(DF, CV_SUMMARY),
        risk_legend(),
        dcc.Tabs(id="tabs", value="tab-table", children=[
            dcc.Tab(label="Tableau de bord",   value="tab-table",
                    style={"padding":"6px 18px","fontSize":"0.77rem","fontWeight":"600","color":"#64748B","minHeight":"32px"},
                    selected_style={"padding":"6px 18px","fontSize":"0.77rem","fontWeight":"700","color":"#1A2E55","borderBottom":"2px solid #2060C8","background":"#fff","minHeight":"32px"}),
            dcc.Tab(label="Graphique",          value="tab-chart",
                    style={"padding":"6px 18px","fontSize":"0.77rem","fontWeight":"600","color":"#64748B","minHeight":"32px"},
                    selected_style={"padding":"6px 18px","fontSize":"0.77rem","fontWeight":"700","color":"#1A2E55","borderBottom":"2px solid #2060C8","background":"#fff","minHeight":"32px"}),
            dcc.Tab(label="Synthèse régionale", value="tab-region",
                    style={"padding":"6px 18px","fontSize":"0.77rem","fontWeight":"600","color":"#64748B","minHeight":"32px"},
                    selected_style={"padding":"6px 18px","fontSize":"0.77rem","fontWeight":"700","color":"#1A2E55","borderBottom":"2px solid #2060C8","background":"#fff","minHeight":"32px"}),
            # Hidden tabs — accessible via sidebar nav only
            dcc.Tab(value="tab-alerts", label="",
                    style={"display":"none"}, selected_style={"display":"none"}),
            dcc.Tab(value="tab-prog",   label="",
                    style={"display":"none"}, selected_style={"display":"none"}),
        ], style={"margin": "0", "background": "#F5F7FA", "borderBottom": f"1px solid #DDE3ED"}),
        html.Div(id="tab-content"),
    ], style={
        "flex": "1", "display": "flex", "flexDirection": "column",
        "minWidth": "0",
    }),

], style={"display": "flex", "minHeight": "100vh", "fontFamily": FONT})


# ══════════════════════════════════════════════════════════════════════════════
# CALLBACKS  (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-table":  return tab_table()
    if tab == "tab-chart":  return tab_chart()
    if tab == "tab-region": return tab_region()
    if tab == "tab-alerts": return tab_alerts()
    if tab == "tab-prog":   return tab_prog()
    return html.Div()


@app.callback(
    Output("tabs", "value"),
    [Input(f"nav-{n}", "n_clicks") for n, _, _ in _NAV_ITEMS],
    prevent_initial_call=True,
)
def sidebar_click(*args):
    from dash import ctx
    triggered = ctx.triggered_id or ""
    nav_id = triggered.replace("nav-", "")
    return _TAB_MAP.get(nav_id, "tab-table")


@app.callback(
    [Output(f"nav-{n}", "style") for n, _, _ in _NAV_ITEMS],
    Input("tabs", "value"),
)
def update_nav_styles(active_tab):
    nav_map = {
        "tab-table":  "tab-table",
        "tab-chart":  "analyses",
        "tab-region": "region",
        "tab-alerts": "alerts",
        "tab-prog":   "prog",
    }
    active = nav_map.get(active_tab, "tab-table")
    return [_nav_style(n == active) for n, _, _ in _NAV_ITEMS]


@app.callback(
    Output("main-table", "data"),
    Output("row-count",  "children"),
    Input("dd-prog",   "value"),
    Input("dd-reg",    "value"),
    Input("dd-risk",   "value"),
    Input("dd-budget", "value"),
)
def filter_table(prog, reg, risk, budget):
    filtered = DF.copy()
    if prog   != "Tous":    filtered = filtered[filtered["programme"]    == int(prog)]
    if reg    != "Toutes":  filtered = filtered[filtered["region_label"] == reg]
    if risk   != "Tous":    filtered = filtered[filtered["risk_label"]   == risk]
    if budget != "Tous":    filtered = filtered[filtered["budget_type"]  == budget]
    return build_table_data(filtered), f"{len(filtered)} ligne(s) affichée(s)"


@app.callback(
    Output("modal-detail", "is_open"),
    Output("modal-body",   "children"),
    Output("modal-title",  "children"),
    Input("main-table", "active_cell"),
    State("main-table", "data"),
    prevent_initial_call=True,
)
def open_detail(active_cell, data):
    if not active_cell or not data:
        return False, "", ""
    row     = data[active_cell["row"]]
    line_id = row["line_id"]
    body, title = build_detail(line_id)
    return True, body, title


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("main-table", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, data):
    if not data:
        return dash.no_update
    exp = pd.DataFrame(data).drop(columns=["line_id", "_risk"], errors="ignore")
    return dcc.send_data_frame(exp.to_csv, "previsions_2025.csv",
                               index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import webbrowser, threading
    def _open():
        import time; time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:8050")
    threading.Thread(target=_open, daemon=True).start()
    app.run(debug=False, host="127.0.0.1", port=8050)
