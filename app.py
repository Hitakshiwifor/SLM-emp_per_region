# app.py
from pathlib import Path
from copy import deepcopy
import os, json
import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash_deck

# -------------------- Paths (edit as needed) --------------------
PARQUET_PATH = Path(r"D:\Hitakshi\Dashboard\slm-dashboard\data\eu_labor_force_small.parquet")
GEO_PATHS = {
    2: Path(r"D:\Hitakshi\Dashboard\slm-dashboard\data\NUTS_RG_20M_2024_4326_LEVL_0.geojson"),
    3: Path(r"D:\Hitakshi\Dashboard\slm-dashboard\data\NUTS_RG_20M_2024_4326_LEVL_1.geojson"),
    4: Path(r"D:\Hitakshi\Dashboard\slm-dashboard\data\NUTS_RG_20M_2024_4326_LEVL_2.geojson"),
}
APP_TITLE = "Skilled Labor Monitor"
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")
THEME = dbc.themes.FLATLY

# -------------------- Load data --------------------
df = pd.read_parquet(PARQUET_PATH)
need = {"region_level","region_code","year","sex_code","age_code","age_level","emp"}
miss = need - set(df.columns)
if miss:
    raise ValueError(f"Parquet missing columns: {sorted(miss)}")

# Preload GeoJSONs
GEO = {}
for lvl, p in GEO_PATHS.items():
    with open(p, "r", encoding="utf-8") as f:
        GEO[lvl] = json.load(f)

# -------------------- Filter options --------------------
def uniq_sorted(series):
    s = pd.Series(series).dropna().unique().tolist()
    return sorted(s)

years_all     = sorted(map(int, uniq_sorted(df["year"])))
sex_all       = list(map(str, uniq_sorted(df["sex_code"]))) or ["T"]
age_code_all  = list(map(str, uniq_sorted(df["age_code"]))) or ["TOTAL"]
age_level_all = list(map(str, uniq_sorted(df["age_level"]))) or ["Y"]
levels_all    = [2,3,4]

default_year      = years_all[-16] if len(years_all) >= 16 else years_all[-1]
default_sex       = sex_all[2] if len(sex_all) > 2 else sex_all[-1]
default_age_code  = age_code_all[1] if len(age_code_all) > 1 else age_code_all[0]
default_age_level = age_level_all[0]
default_level     = 2

# -------------------- Helpers --------------------
def norm_code(x):
    if pd.isna(x): return None
    return str(x).strip().upper()

def initial_view_state():
    return dict(latitude=54.0, longitude=15.0, zoom=3.35,
                minZoom=2.0, maxZoom=10.5, pitch=0, bearing=0)

def emp_map_for(year, sex_code, age_code, age_level, region_level) -> dict:
    sub = df[
        (df["region_level"] == int(region_level)) &
        (df["year"].astype(int) == int(year)) &
        (df["sex_code"].astype(str) == str(sex_code)) &
        (df["age_code"].astype(str) == str(age_code)) &
        (df["age_level"].astype(str) == str(age_level))
    ].copy()
    if sub.empty:
        return {}
    sub = sub.sort_values("region_code")
    return sub.groupby("region_code")["emp"].last().rename(index=norm_code).to_dict()

def enrich_geo_quantile_colors_with_flat_tooltip(geo: dict, emp_map: dict):
    g = deepcopy(geo)
    palette = [
        [239,243,255],[198,219,239],[158,202,225],
        [107,174,214],[66,146,198],[33,113,181],[8,81,156],
    ]
    num_classes = len(palette)
    vals = np.array([v for v in emp_map.values() if pd.notna(v)], dtype=float)
    if len(vals) >= 2:
        bins = np.quantile(vals, np.linspace(0,1,num_classes+1)).astype(float)
        for i in range(1,len(bins)):
            if bins[i] <= bins[i-1]:
                bins[i] = bins[i-1] + 1e-9
    else:
        bins = np.array([0,1,2,3,4,5,6,7], dtype=float)

    def rgba_for(v):
        if v is None or pd.isna(v):
            return [230,230,230,190]
        idx = np.searchsorted(bins, float(v), side="right") - 1
        idx = max(0, min(idx, num_classes-1))
        r,g_,b = palette[idx]
        return [int(r),int(g_),int(b),210]

    for feat in g.get("features", []):
        props = feat.setdefault("properties", {})
        nuts_id = norm_code(props.get("NUTS_ID") or props.get("nuts_id") or props.get("ID"))
        name = props.get("NAME_LATN") or props.get("NAME_2021") or props.get("CNTR_NAME") or nuts_id or ""
        v = emp_map.get(nuts_id)
        emp_val = None if v is None or pd.isna(v) else float(v)
        emp_disp = "N/A" if emp_val is None else f"{emp_val:,.0f}"
        props["fillColor"] = rgba_for(emp_val)
        feat["region_name"] = name
        feat["emp_disp"] = emp_disp
    return g

def deck_spec(geojson_enriched: dict):
    layer = {
        "@@type": "GeoJsonLayer",
        "id": "nuts-layer",
        "data": geojson_enriched,
        "pickable": True,
        "stroked": True,
        "filled": True,
        "autoHighlight": True,
        "opacity": 0.97,
        "getFillColor": "@@=properties.fillColor",
        "getLineColor": [160,160,160],
        "lineWidthMinPixels": 0.75,
    }
    spec = {
        "initialViewState": initial_view_state(),
        "layers": [layer],
        "controller": True,
    }
    spec["mapStyle"] = (
        "mapbox://styles/mapbox/light-v11" if MAPBOX_TOKEN
        else "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    )
    return spec

# -------------------- App & global styles --------------------
app = dash.Dash(__name__, external_stylesheets=[THEME], title=APP_TITLE,
                suppress_callback_exceptions=True)
server = app.server

# Glassy header + nav-pill CSS (hover/focus/active) + Bootstrap Icons
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
    {%css%}
    <style>
      body { background-color: rgb(217,217,217); }
      .glass {
        background: rgba(255,255,255,0.78);
        border: 1px solid #e6e9ef;
        backdrop-filter: saturate(160%) blur(10px);
        -webkit-backdrop-filter: saturate(160%) blur(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,.08);
      }
      .nav-pill {
        padding: 8px 14px;
        border-radius: 999px;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
        margin-left: 8px;
        color: #003662;
        background: #ffffff;
        border: 1px solid #e6e9ef;
        transition: all .18s ease;
        display: inline-flex; align-items: center; gap: 8px;
        white-space: nowrap;
      }
      .nav-pill:hover { transform: translateY(-1px); box-shadow: 0 6px 14px rgba(0,0,0,.06); }
      .nav-pill:focus { outline: 0; box-shadow: 0 0 0 3px rgba(0,54,98,.18); }
      .nav-pill.active { background: #e7eff6; border-color: #cbd8e6; }
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
"""

# -------------------- Common header with right-aligned nav --------------------
NAV_ITEMS = [
    (" Introduction", "bi-house-door", "/"),
    (" Sankey", "bi-diagram-3", "/sankey"),
    (" Tables", "bi-table", "/tables"),
    (" Map", "bi-geo-alt", "/map"),
]

def make_header(pathname="/"):
    def link(lbl, icon, href):
        active = (href == pathname) or (href == "/" and pathname in ["/", "", None])
        cls = "nav-pill" + (" active" if active else "")
        return dcc.Link(html.Span([html.I(className=f"bi {icon}"), lbl]), href=href, className=cls)

    return html.Div(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Img(src="/assets/SLM Dark.png",
                                         style={"height":"38px","marginRight":"14px"}),
                                html.Span(
                                    APP_TITLE,
                                    className="fw-semibold",
                                    style={"fontSize":"28px","color":"#003662","whiteSpace":"nowrap"},
                                ),
                            ],
                            className="d-flex align-items-center"
                        ),
                        md="auto"
                    ),
                    dbc.Col(
                        html.Div([link(l,i,h) for (l,i,h) in NAV_ITEMS],
                                 className="d-flex justify-content-end align-items-center flex-wrap"),
                        align="center"
                    ),
                ],
                className="align-items-center g-2"
            ),
            fluid=True,
            style={"paddingTop":"10px","paddingBottom":"10px"}
        ),
        className="glass",
        style={
            "position":"sticky","top":"8px","zIndex":"30","borderRadius":"16px","marginBottom":"10px"
        }
    )

# -------------------- YOUR MAP PAGE UI (unchanged) --------------------
filter_bar = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col([dbc.Label("Year", className="text-muted small"), dcc.Dropdown(
                    id="year-dd",
                    options=[{"label": str(y), "value": int(y)} for y in years_all],
                    value=default_year, clearable=False,
                )], md=2),
                dbc.Col([dbc.Label("Sex", className="text-muted small"), dcc.Dropdown(
                    id="sex-dd",
                    options=[{"label": s, "value": s} for s in sex_all],
                    value=default_sex, clearable=False,
                )], md=2),
                dbc.Col([dbc.Label("Age code", className="text-muted small"), dcc.Dropdown(
                    id="agecode-dd",
                    options=[{"label": a, "value": a} for a in age_code_all],
                    value=default_age_code, clearable=False,
                )], md=3),
                dbc.Col([dbc.Label("Age level", className="text-muted small"), dcc.Dropdown(
                    id="agelevel-dd",
                    options=[{"label": a, "value": a} for a in age_level_all],
                    value=default_age_level, clearable=False,
                )], md=2),
                dbc.Col([dbc.Label("Region level", className="text-muted small"), dcc.Dropdown(
                    id="level-dd",
                    options=[
                        {"label": "Countries", "value": 2},
                        {"label": "Major Socio-economic Regions", "value": 3},
                        {"label": "Medium-sized administrative regions", "value": 4},
                    ],
                    value=default_level, clearable=False,
                )], md=3),
            ],
            className="g-3 align-items-end"
        )
    ),
    style={
        "borderRadius":"16px",
        "boxShadow":"0 2px 12px rgba(0,0,0,.06)",
        "backgroundColor":"#ffffff"
    },
    className="my-3"
)

map_card = dbc.Card(
    dbc.CardBody(
        html.Div(
            id="deck-wrap",
            style={
                "height":"70vh",
                "borderRadius":"16px",
                "overflow":"hidden",
                "border":"1px solid #e6e9ef"
            }
        )
    ),
    style={
        "borderRadius":"20px",
        "boxShadow":"0 4px 20px rgba(0,0,0,.10)",
        "backgroundColor":"#ffffff"
    },
    className="mb-3"
)

# -------------------- Simple other pages --------------------
def intro_page():
    return dbc.Container(
        dbc.Card(
            dbc.CardBody([
                html.H3("Introduction", style={"color":"#003662","marginBottom":"8px"}),
                html.P("Overview of the Skilled Labor Monitor and how to use each page."),
                html.Ul([
                    html.Li("Interactive Map to explore regional employment."),
                    html.Li("Sankey view for composition insights (coming)."),
                    html.Li("Tables for sortable details (coming)."),
                ], className="mb-0")
            ]),
            style={"borderRadius":"20px","boxShadow":"0 4px 20px rgba(0,0,0,.10)","backgroundColor":"#ffffff"},
            className="my-3"
        ),
        fluid=True
    )

def sankey_page():
    return dbc.Container(
        dbc.Card(
            dbc.CardBody([
                html.H3("Sankey (placeholder)", style={"color":"#003662"}),
                html.P("Add your Sankey visualization here."),
            ]),
            style={"borderRadius":"20px","boxShadow":"0 4px 20px rgba(0,0,0,.10)","backgroundColor":"#ffffff"},
            className="my-3"
        ),
        fluid=True
    )

def tables_page():
    return dbc.Container(
        dbc.Card(
            dbc.CardBody([
                html.H3("Tables (placeholder)", style={"color":"#003662"}),
                html.P("Add your tables here."),
            ]),
            style={"borderRadius":"20px","boxShadow":"0 4px 20px rgba(0,0,0,.10)","backgroundColor":"#ffffff"},
            className="my-3"
        ),
        fluid=True
    )

# -------------------- Routing --------------------
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="header-slot"),
    html.Div(id="page-slot"),
])

@app.callback(
    Output("header-slot","children"),
    Output("page-slot","children"),
    Input("url","pathname"),
)
def render_page(pathname):
    header = make_header(pathname or "/")
    if pathname in ["/", "", "/intro", "/home"]:
        return header, intro_page()
    if pathname == "/sankey":
        return header, sankey_page()
    if pathname == "/tables":
        return header, tables_page()
    if pathname == "/map":
        # Your original map page UI (unchanged)
        return header, dbc.Container([filter_bar, map_card, html.Div(style={"height":"12px"})], fluid=True)
    return header, dbc.Container(html.H4("404 — Page not found", className="mt-4"), fluid=True)

# -------------------- Map callback (unchanged) --------------------
@app.callback(
    Output("deck-wrap", "children"),
    Input("year-dd", "value"),
    Input("sex-dd", "value"),
    Input("agecode-dd", "value"),
    Input("agelevel-dd", "value"),
    Input("level-dd", "value"),
)
def update_map(year, sex_code, age_code, age_level, region_level):
    region_level = int(region_level)
    geo_for_level = GEO.get(region_level)
    if geo_for_level is None:
        return html.Div("No GeoJSON for selected region level.", className="text-danger")

    emp_by_code = emp_map_for(year, sex_code, age_code, age_level, region_level)
    gj = enrich_geo_quantile_colors_with_flat_tooltip(geo_for_level, emp_by_code)
    spec = deck_spec(gj)
    return dash_deck.DeckGL(
        spec,
        id="deck-gl",
        mapboxKey=MAPBOX_TOKEN,
        tooltip={
            "html": "<b>{region_name}</b><br/>Emp: {emp_disp}",
            "style": {
                "backgroundColor": "rgba(0,0,0,.78)",
                "color": "#fff",
                "padding": "6px 8px",
                "borderRadius": "6px",
                "fontSize": "12px"
            }
        },
        style={"width": "100%", "height": "100%"}
    )

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True, port=8051)
