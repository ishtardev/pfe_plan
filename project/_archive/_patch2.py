with open('app_dash.py', encoding='utf-8') as f:
    content = f.read()

# The original (uncorrupted) part is from 0 to 49852
orig = content[:49852]

SEP = '\u2550' * 78
header_start = 14053          # start of "# ═══\n# HEADER\n" block
kpi_section  = orig.find('\n# KPI CARDS\n')   # 15484

NEW = f'''# {SEP}
# SIDEBAR + TOPBAR
# {SEP}
def build_sidebar(n_alerts=0, active_nav="tab-table"):
    nav_items = []
    for nav_id, icon_cls, label in _NAV_ITEMS:
        badge = n_alerts if (nav_id == "alerts" and n_alerts > 0) else None
        nav_items.append(html.Div([
            html.I(className=f"bi {{icon_cls}}", style={{
                "fontSize": "0.82rem", "width": "16px", "textAlign": "center", "flexShrink": "0",
            }}),
            html.Span(label, style={{"flex": "1"}}),
            *([html.Span(str(badge), style={{
                "fontSize": "0.58rem", "fontWeight": "700", "color": "#fff",
                "background": C_CRIT, "borderRadius": "10px",
                "padding": "1px 5px", "minWidth": "15px", "textAlign": "center",
            }})] if badge else []),
        ], id=f"nav-{{nav_id}}", n_clicks=0, style=_nav_style(nav_id == active_nav)))

    return html.Div([
        html.Div([
            html.Div(html.I(className="bi bi-shield-fill-check", style={{"fontSize": "1.1rem", "color": C_ACCENT}}), style={{
                "width": "34px", "height": "34px", "borderRadius": "8px",
                "background": "rgba(32,96,200,.18)", "border": "1px solid rgba(32,96,200,.28)",
                "display": "flex", "alignItems": "center", "justifyContent": "center", "flexShrink": "0",
            }}),
            html.Div([
                html.Div("SYST\u00c8ME D\u2019ALERTE", style={{"fontSize": "0.62rem", "fontWeight": "800", "color": "#fff", "lineHeight": "1.15", "letterSpacing": "0.3px"}}),
                html.Div("PR\u00c9COCE BUDG\u00c9TAIRE", style={{"fontSize": "0.62rem", "fontWeight": "800", "color": "#fff", "lineHeight": "1.15", "letterSpacing": "0.3px"}}),
            ], style={{"marginLeft": "10px"}}),
        ], style={{
            "display": "flex", "alignItems": "center", "padding": "13px 14px 11px",
            "borderBottom": "1px solid rgba(255,255,255,.07)", "marginBottom": "6px",
        }}),
        *nav_items,
        html.Div(style={{"flex": "1"}}),
        html.Div([
            html.Div([
                html.I(className="bi bi-gear-fill", style={{"marginRight": "8px", "fontSize": "0.78rem"}}),
                "Param\u00e8tres",
            ], style={{
                "display": "flex", "alignItems": "center", "padding": "6px 14px",
                "margin": "0 8px", "fontSize": "0.75rem", "color": C_SIDEBAR_TXT, "cursor": "pointer", "borderRadius": "5px",
            }}),
            html.Div(style={{"borderTop": "1px solid rgba(255,255,255,.07)", "margin": "6px 0 0"}}),
            html.Div([
                html.Div("\u00c0 propos du syst\u00e8me", style={{"fontSize": "0.63rem", "fontWeight": "700", "color": "#CBD5E1", "marginBottom": "3px"}}),
                html.Div("Outil d\u2019aide \u00e0 la d\u00e9cision pour le suivi budg\u00e9taire en temps r\u00e9el.", style={{"fontSize": "0.58rem", "color": C_SIDEBAR_TXT, "lineHeight": "1.4"}}),
                html.Div("Minist\u00e8re de la Justice", style={{"fontSize": "0.58rem", "color": C_SIDEBAR_TXT, "marginTop": "5px", "fontWeight": "500"}}),
                html.Div("Direction des Affaires Financi\u00e8res", style={{"fontSize": "0.58rem", "color": C_SIDEBAR_TXT}}),
                html.Div("\u00a9 2025 \u2014 Exercice 2025", style={{"fontSize": "0.55rem", "color": "rgba(255,255,255,.22)", "marginTop": "6px"}}),
            ], style={{"padding": "8px 14px 12px"}}),
        ]),
    ], style={{
        "width": "192px", "minWidth": "192px",
        "background": C_SIDEBAR, "height": "100vh",
        "position": "sticky", "top": "0",
        "display": "flex", "flexDirection": "column",
        "overflowY": "auto", "boxShadow": "2px 0 10px rgba(0,0,0,.2)",
        "flexShrink": "0",
    }})


def build_topbar(build_ts, n_alerts=0):
    return html.Div([
        html.Div([
            html.Span("Minist\u00e8re de la Justice, Direction des Affaires Financi\u00e8res", style={{
                "fontWeight": "700", "fontSize": "0.9rem", "color": C_TEXT,
            }}),
            html.Span(" \u2014 Exercice 2025", style={{
                "fontSize": "0.84rem", "color": C_MUTED, "fontWeight": "400",
            }}),
        ]),
        html.Div([
            html.I(className="bi bi-calendar3", style={{"fontSize": "0.82rem", "color": C_MUTED, "marginRight": "6px"}}),
            html.Div([
                html.Div("Derni\u00e8re mise \u00e0 jour", style={{"fontSize": "0.57rem", "color": C_MUTED, "lineHeight": "1"}}),
                html.Div(build_ts, style={{"fontSize": "0.72rem", "fontWeight": "700", "color": C_TEXT, "lineHeight": "1.3"}}),
            ]),
            html.Div([
                html.I(className="bi bi-bell-fill", style={{"fontSize": "0.88rem", "color": C_MUTED}}),
                *([html.Span(str(n_alerts), style={{
                    "position": "absolute", "top": "-4px", "right": "-5px",
                    "fontSize": "0.5rem", "fontWeight": "800", "color": "#fff",
                    "background": C_CRIT, "borderRadius": "10px",
                    "padding": "0 3px", "lineHeight": "11px", "minWidth": "11px", "textAlign": "center",
                }})] if n_alerts > 0 else []),
            ], style={{"position": "relative", "cursor": "pointer", "marginLeft": "14px", "padding": "4px"}}),
        ], style={{"display": "flex", "alignItems": "center", "gap": "3px"}}),
    ], style={{
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "9px 20px", "background": C_CARD,
        "borderBottom": f"1px solid {{C_BORDER}}",
        "boxShadow": "0 1px 4px rgba(0,0,0,.05)", "flexShrink": "0",
    }})

'''

fixed = orig[:header_start] + NEW + orig[kpi_section:]
with open('app_dash.py', 'w', encoding='utf-8') as f:
    f.write(fixed)
print(f'Done. New file size: {len(fixed):,}')
