import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash_deck
from ..data.loader import load_df, load_geo
from ..data.geo import norm_code
from ..config import MAPBOX_TOKEN

df = load_df()
GEO = load_geo()

def uniq_sorted(series):
    s = pd.Series(series).dropna().unique().tolist()
    return sorted(s)

years_all     = sorted(map(int, uniq_sorted(df["year"])))
sex_all       = list(map(str, uniq_sorted(df["sex_code"]))) or ["T"]
age_code_all  = list(map(str, uniq_sorted(df["age_code"]))) or ["TOTAL"]
age_level_all = list(map(str, uniq_sorted(df["age_level"]))) or ["Y"]

default_year      = years_all[-16] if len(years_all) >= 16 else years_all[-1]
default_sex       = sex_all[2] if len(sex_all) > 2 else sex_all[-1]
default_age_code  = age_code_all[1] if len(age_code_all) > 1 else age_code_all[0]
default_age_level = age_level_all[0]
default_level     = 2

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
    g = dict(geo)
    palette = [[239,243,255],[198,219,239],[158,202,225],[107,174,214],[66,146,198],[33,113,181],[8,81,156]]
    num_classes = len(palette)
    vals = np.array([v for v in emp_map.values() if pd.notna(v)], dtype=float)
    if len(vals) >= 2:
        bins = np.quantile(vals, np.linspace(0,1,num_classes+1)).astype(float)
        for i in range(1,len(bins)):
            if bins[i] <= bins[i-1]: bins[i] = bins[i-1] + 1e-9
    else:
        bins = np.array([0,1,2,3,4,5,6,7], dtype=float)

    def rgba_for(v):
        if v is None or pd.isna(v): return [230,230,230,190]
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
        "pickable": True, "stroked": True, "filled": True, "autoHighlight": True,
        "opacity": 0.97,
        "getFillColor": "@@=properties.fillColor",
        "getLineColor": [160,160,160],
        "lineWidthMinPixels": 0.75,
    }
    return {
        "initialViewState": initial_view_state(),
        "layers": [layer],
        "controller": True,
        "mapStyle": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    }

# ---------- UI (unchanged) ----------
filter_bar = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col([dbc.Label("Year", className="text-muted small"), dcc.Dropdown(
                    id="year-dd", options=[{"label": str(y), "value": int(y)} for y in years_all],
                    value=default_year, clearable=False)], md=2),
                dbc.Col([dbc.Label("Sex", className="text-muted small"), dcc.Dropdown(
                    id="sex-dd", options=[{"label": s, "value": s} for s in sex_all],
                    value=default_sex, clearable=False)], md=2),
                dbc.Col([dbc.Label("Age code", className="text-muted small"), dcc.Dropdown(
                    id="agecode-dd", options=[{"label": a, "value": a} for a in age_code_all],
                    value=default_age_code, clearable=False)], md=3),
                dbc.Col([dbc.Label("Age level", className="text-muted small"), dcc.Dropdown(
                    id="agelevel-dd", options=[{"label": a, "value": a} for a in age_level_all],
                    value=default_age_level, clearable=False)], md=2),
                dbc.Col([dbc.Label("Region level", className="text-muted small"), dcc.Dropdown(
                    id="level-dd", options=[
                        {"label":"Countries","value":2},
                        {"label":"Major Socio-economic Regions","value":3},
                        {"label":"Medium-sized administrative regions","value":4},
                    ], value=default_level, clearable=False)], md=3),
            ],
            className="g-3 align-items-end"
        )
    ),
    style={"borderRadius":"16px","boxShadow":"0 2px 12px rgba(0,0,0,.06)","backgroundColor":"#ffffff"},
    className="my-3 card-lite",
)

map_card = dbc.Card(
    dbc.CardBody(html.Div(
        id="deck-wrap", className="map-frame",
        style={"height":"70vh","borderRadius":"16px","overflow":"hidden","border":"1px solid #e6e9ef"}
    )),
    style={"borderRadius":"20px","boxShadow":"0 4px 20px rgba(0,0,0,.10)","backgroundColor":"#ffffff"},
    className="mb-3"
)

def layout():
    return [filter_bar, map_card, html.Div(style={"height":"12px"})]

def register_callbacks(app):
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
                "style": {"backgroundColor":"rgba(0,0,0,.78)","color":"#fff",
                          "padding":"6px 8px","borderRadius":"6px","fontSize":"12px"}
            },
            style={"width":"100%","height":"100%"},
        )
