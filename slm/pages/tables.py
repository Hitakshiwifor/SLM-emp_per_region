# slm/pages/tables.py
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

# -------- project loader / geo config ----------
from ..data.loader import load_df  # cached parquet loader

GEO_PATHS = None
try:
    from ..config import GEO_PATHS as _GEO_PATHS_FROM_CONFIG
    GEO_PATHS = _GEO_PATHS_FROM_CONFIG
except Exception:
    pass

_get_geo_by_level = None
try:
    from ..geo import get_geo_by_level as _get_geo_by_level  # get_geo_by_level(level)->dict
except Exception:
    _get_geo_by_level = None

# -------- dataset + column detection ----------
df = load_df()

REG_LEVEL_COL_CAND = ["region_level", "geo_level", "geo_levels"]
REG_CODE_COL_CAND  = ["region_code", "geo_code", "NUTS_ID", "nuts_id", "REGION_CODE"]
EMP_COL_CAND       = ["emp", "employment", "Employed", "EMP"]

def _detect_col(cands, frame=df):
    for c in cands:
        if c in frame.columns:
            return c
    return None

REG_LEVEL_COL = _detect_col(REG_LEVEL_COL_CAND)
REG_CODE_COL  = _detect_col(REG_CODE_COL_CAND)
EMP_COL       = _detect_col(EMP_COL_CAND)
YEAR_COL      = "year"

REG_LEVELS = [
    {"label": "Countries", "value": 2},
    {"label": "NUTS 1",   "value": 3},
    {"label": "NUTS 2",   "value": 4},
]
DEFAULT_LEVEL = 2

CURRENT_YEAR = 2025
TARGET_YEAR = 2040

# -------- GeoJSON helpers ----------
@lru_cache(maxsize=16)
def _load_geojson_for_level(level: int) -> dict:
    if _get_geo_by_level:
        gj = _get_geo_by_level(int(level))
        if gj:
            return gj
    if GEO_PATHS and int(level) in GEO_PATHS:
        p = GEO_PATHS[int(level)]
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}

def _geo_features(level: int):
    return _load_geojson_for_level(level).get("features", [])

@lru_cache(maxsize=3)
def _code_to_name(level: int):
    m = {}
    for ft in _geo_features(level):
        pr = ft.get("properties", {})
        code = str(pr.get("NUTS_ID") or pr.get("nuts_id") or pr.get("id") or "")
        if not code:
            continue
        name = pr.get("NAME_LATN") or pr.get("name_latn") or pr.get("NAME_ENGL") or pr.get("NAME") or code
        m[code] = str(name)
    return m

@lru_cache(maxsize=3)
def _code_to_cntr(level: int):
    m = {}
    for ft in _geo_features(level):
        pr = ft.get("properties", {})
        code = str(pr.get("NUTS_ID") or pr.get("nuts_id") or pr.get("id") or "")
        if not code:
            continue
        cntr = pr.get("CNTR_CODE") or ""
        m[code] = str(cntr)
    return m

# -------- dataset presence by year/level ----------
def _dataset_codes_for(level: int):
    if REG_LEVEL_COL is None or REG_CODE_COL is None:
        return set()
    sub = df[df[REG_LEVEL_COL].astype(int) == int(level)][[REG_CODE_COL]].dropna()
    return set(sub[REG_CODE_COL].astype(str).unique())

# -------- dropdown options (NAME_LATN) ----------
def _countries_options():
    names = _code_to_name(2)
    ds_codes = _dataset_codes_for(2)
    return sorted(({"label": names.get(c, c), "value": c} for c in ds_codes), key=lambda x: x["label"])

def _nuts1_options(selected_countries=None):
    ds_codes = _dataset_codes_for(3)
    if not ds_codes:
        return []
    names = _code_to_name(3)
    cntr3 = _code_to_cntr(3)
    if selected_countries:
        cntr2 = _code_to_cntr(2)
        selected_countries = selected_countries if isinstance(selected_countries, list) else [selected_countries]
        cntr_filter = {cntr2.get(c, "") for c in selected_countries}
        cntr_filter.discard("")
        ds_codes = {c for c in ds_codes if cntr3.get(c, "") in cntr_filter}
    return sorted(({"label": names.get(c, c), "value": c} for c in ds_codes), key=lambda x: x["label"])

def _nuts2_options(selected_countries=None, selected_nuts1=None):
    ds_codes = _dataset_codes_for(4)
    if not ds_codes:
        return []
    names = _code_to_name(4)
    cntr4 = _code_to_cntr(4)

    if selected_countries:
        cntr2 = _code_to_cntr(2)
        selected_countries = selected_countries if isinstance(selected_countries, list) else [selected_countries]
        cntr_filter = {cntr2.get(c, "") for c in selected_countries}
        cntr_filter.discard("")
        ds_codes = {c for c in ds_codes if cntr4.get(c, "") in cntr_filter}

    if selected_nuts1:
        sel1 = set(selected_nuts1 if isinstance(selected_nuts1, list) else [selected_nuts1])
        ds_codes = {c for c in ds_codes if any(c.startswith(p) for p in sel1)}

    return sorted(({"label": names.get(c, c), "value": c} for c in ds_codes), key=lambda x: x["label"])

# -------- filtering + aggregation ----------
def _ensure_levels(sub: pd.DataFrame) -> pd.DataFrame:
    if "age_level" in sub.columns:
        sub = sub[sub["age_level"].astype(str) == "2"]
    if "sex_level" in sub.columns:
        sub = sub[sub["sex_level"].astype(str) == "2"]
    return sub

def _deepest_mask(sel_cntr, sel_nuts1, sel_nuts2):
    if sel_nuts2:
        level = 4
        codes = set(sel_nuts2 if isinstance(sel_nuts2, list) else [sel_nuts2])
        return (df[REG_LEVEL_COL].astype(int) == level) & (df[REG_CODE_COL].astype(str).isin(codes))
    if sel_nuts1:
        level = 3
        codes = set(sel_nuts1 if isinstance(sel_nuts1, list) else [sel_nuts1])
        return (df[REG_LEVEL_COL].astype(int) == level) & (df[REG_CODE_COL].astype(str).isin(codes))
    if sel_cntr:
        level = 2
        codes = set(sel_cntr if isinstance(sel_cntr, list) else [sel_cntr])
        return (df[REG_LEVEL_COL].astype(int) == level) & (df[REG_CODE_COL].astype(str).isin(codes))
    # no geo selection → use countries to avoid double counting
    return df[REG_LEVEL_COL].astype(int) == 2

@lru_cache(maxsize=256)
def _rows_for_table(region_level: int, sel_countries_key: str, sel_nuts1_key: str, sel_nuts2_key: str):
    """Return a tidy frame with one row per region at `region_level`, with sparkline series."""
    def _decode(k):
        return None if k == "ALL" else k.split("|")

    sel_cntr = _decode(sel_countries_key)
    sel_n1   = _decode(sel_nuts1_key)
    sel_n2   = _decode(sel_nuts2_key)

    mask = _deepest_mask(sel_cntr, sel_n1, sel_n2)
    sub = df.loc[mask, [YEAR_COL, REG_LEVEL_COL, REG_CODE_COL, EMP_COL]].copy()
    sub = _ensure_levels(sub)

    target_level = int(region_level)

    # Which codes exist at the target level?
    at_level = df[REG_LEVEL_COL].astype(int) == target_level
    codes_at_level = set(df.loc[at_level, REG_CODE_COL].astype(str).unique())

    # Map any code to its ancestor at target level via prefix
    def _to_level_code(code: str) -> str:
        if code in codes_at_level:
            return code
        for i in range(len(code), 1, -1):
            pref = code[:i]
            if pref in codes_at_level:
                return pref
        return code  # fallback (filtered out if not in codes_at_level)

    sub["row_code"] = sub[REG_CODE_COL].astype(str).map(_to_level_code)
    sub = sub[sub["row_code"].isin(codes_at_level)]

    # Aggregate to yearly totals for each row
    ts = sub.groupby(["row_code", YEAR_COL], as_index=False)[EMP_COL].sum()

    rows = []
    for code, g in ts.sort_values(["row_code", YEAR_COL]).groupby("row_code", sort=False):
        years = g[YEAR_COL].astype(int).tolist()
        vals  = g[EMP_COL].astype(float).tolist()
        yrs_np = g[YEAR_COL].to_numpy()
        vals_np = g[EMP_COL].to_numpy()

        def pick_value(target_year: int) -> float:
            if target_year in g[YEAR_COL].values:
                return float(g.loc[g[YEAR_COL] == target_year, EMP_COL].sum())
            # nearest year fallback
            idx = int(np.argmin(np.abs(yrs_np - target_year)))
            return float(vals_np[idx])

        emp_curr = pick_value(CURRENT_YEAR)
        emp_2040 = pick_value(TARGET_YEAR)

        rows.append({
            "code": code,
            "current": emp_curr,
            "y2040":  emp_2040,
            "series": (years, vals),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    name_map = _code_to_name(target_level)
    out["region"] = out["code"].map(lambda c: name_map.get(c, c))
    return out.sort_values("current", ascending=False).head(30)

# -------- tiny sparkline figure ----------
def _sparkline(x, y):
    color = "#003662"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines+markers",
            line=dict(width=1.5, color=color),
            marker=dict(size=4, color=color),
            hovertemplate="%{y:,.0f} in %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=44,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# -------- UI ----------
def _about_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("About this table", className="mb-2"),
                html.P(
                    "This view shows employment by region with a quick time-series sparkline for each row. "
                    "Use the filters above to narrow by country or NUTS levels. The first column updates to "
                    "Countries, NUTS 1, or NUTS 2 depending on the Region level you choose."
                ),
                html.H6("How to read it", className="mt-3 mb-2"),
                html.Ul(
                    [
                        html.Li("Currently Employed shows the value for Employed population in ongoing year 2025 ."),
                        html.Li("2040 Employed shows the value for Employed population in 2040."),
                        html.Li("Hover the sparkline to see the exact value and year for that point."),
                    ],
                    className="mb-0",
                ),
            ]
        ),
        style={
            "borderRadius": "16px",
            "boxShadow": "0 4px 20px rgba(0,0,0,.10)",
            "backgroundColor": "#ffffff",
            "minHeight": "100%",
        },
    )

def _filter_bar():
    DD_STYLE = {"zIndex": 2000}
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col([  # Region level (small)
                        dbc.Label("Region level", className="text-muted small"),
                        dcc.Dropdown(
                            id="tbl-level-dd",
                            options=REG_LEVELS,
                            value=DEFAULT_LEVEL,
                            clearable=False,
                            style=DD_STYLE,
                        )
                    ], md=2),
                    dbc.Col([  # Countries (wide)
                        dbc.Label("Countries", className="text-muted small"),
                        dcc.Dropdown(
                            id="tbl-country-dd",
                            options=[],
                            value=None,
                            multi=True,
                            placeholder="All countries",
                            clearable=True,
                            style=DD_STYLE,
                        )
                    ], md=4),
                    dbc.Col([  # NUTS 1 (medium)
                        dbc.Label("Major Socio-economic Regions", className="text-muted small"),
                        dcc.Dropdown(
                            id="tbl-nuts1-dd",
                            options=[],
                            value=None,
                            multi=True,
                            placeholder="All NUTS 1",
                            clearable=True,
                            style=DD_STYLE,
                        )
                    ], md=3),
                    dbc.Col([  # NUTS 2 (medium)
                        dbc.Label("Medium-sized administrative regions", className="text-muted small"),
                        dcc.Dropdown(
                            id="tbl-nuts2-dd",
                            options=[],
                            value=None,
                            multi=True,
                            placeholder="All NUTS 2",
                            clearable=True,
                            style=DD_STYLE,
                        )
                    ], md=3),
                ],
                className="g-3 align-items-end"
            )
        ),
        style={"borderRadius": "16px", "boxShadow": "0 2px 12px rgba(0,0,0,.06)", "backgroundColor": "#ffffff"},
        className="mb-3"
    )

def _table_card():
    header = html.Thead(
        html.Tr([
            html.Th(id="tbl-region-header", children="Region"),   # dynamic header
            html.Th(f"Currently Employed ({CURRENT_YEAR})", style={"width": "180px", "textAlign": "right"}),
            html.Th(f"{TARGET_YEAR} Employed", style={"width": "160px", "textAlign": "right"}),
            html.Th("Metric over Time"),
        ])
    )
    body = html.Tbody(id="tbl-body")
    return dbc.Card(
        dbc.CardBody(dbc.Table([header, body], bordered=False, hover=True, responsive=True, className="align-middle")),
        style={"borderRadius": "20px", "boxShadow": "0 4px 20px rgba(0,0,0,.10)", "backgroundColor": "#ffffff"},
        className="mb-3"
    )

def layout():
    return dbc.Container(
        [
            _filter_bar(),
            dbc.Row(
                [
                    dbc.Col(_about_card(), lg=4, md=5, className="mb-3"),
                    dbc.Col(_table_card(), lg=8, md=7, className="mb-3"),
                ],
                className="g-3",
            ),
            html.Div(style={"height": "12px"}),
        ],
        fluid=True,
    )

# -------- callbacks ----------
def register_callbacks(app):
    # Populate Countries
    @app.callback(
        Output("tbl-country-dd", "options"),
        Output("tbl-country-dd", "value"),
        Input("tbl-level-dd", "value"),  # keeps cache fresh
        State("tbl-country-dd", "value"),
        prevent_initial_call=False
    )
    def _update_countries(level, cur_val):
        opts = _countries_options()
        if not cur_val:
            return opts, None
        valid = {o["value"] for o in opts}
        kept = [v for v in (cur_val if isinstance(cur_val, list) else [cur_val]) if v in valid]
        return opts, (kept if kept else None)

    # Populate NUTS1 (depends on selected countries)
    @app.callback(
        Output("tbl-nuts1-dd", "options"),
        Output("tbl-nuts1-dd", "value"),
        Input("tbl-country-dd", "value"),
        State("tbl-nuts1-dd", "value"),
        prevent_initial_call=False
    )
    def _update_nuts1(sel_countries, cur_val):
        opts = _nuts1_options(sel_countries)
        if not cur_val:
            return opts, None
        valid = {o["value"] for o in opts}
        kept = [v for v in (cur_val if isinstance(cur_val, list) else [cur_val]) if v in valid]
        return opts, (kept if kept else None)

    # Populate NUTS2 (depends on selected countries + nuts1)
    @app.callback(
        Output("tbl-nuts2-dd", "options"),
        Output("tbl-nuts2-dd", "value"),
        Input("tbl-country-dd", "value"),
        Input("tbl-nuts1-dd", "value"),
        State("tbl-nuts2-dd", "value"),
        prevent_initial_call=False
    )
    def _update_nuts2(sel_countries, sel_nuts1, cur_val):
        opts = _nuts2_options(sel_countries, sel_nuts1)
        if not cur_val:
            return opts, None
        valid = {o["value"] for o in opts}
        kept = [v for v in (cur_val if isinstance(cur_val, list) else [cur_val]) if v in valid]
        return opts, (kept if kept else None)

    # Build table rows (+ set region header text)
    @app.callback(
        Output("tbl-body", "children"),
        Output("tbl-region-header", "children"),
        Input("tbl-level-dd", "value"),
        Input("tbl-country-dd", "value"),
        Input("tbl-nuts1-dd", "value"),
        Input("tbl-nuts2-dd", "value"),
    )
    def _update_table(level, sel_countries, sel_nuts1, sel_nuts2):
        def _key(v):
            if not v:
                return "ALL"
            v = v if isinstance(v, list) else [v]
            return "|".join(sorted(map(str, v)))

        rows = _rows_for_table(
            int(level),
            _key(sel_countries), _key(sel_nuts1), _key(sel_nuts2)
        )

        region_header = {2: "Countries", 3: "NUTS 1", 4: "NUTS 2"}.get(int(level), "Region")

        name_map = _code_to_name(int(level))
        body_rows = []
        for _, r in rows.iterrows():
            x, y = r["series"]
            fig = _sparkline(x, y)
            body_rows.append(
                html.Tr([
                    html.Td(name_map.get(r["code"], r["code"])),
                    html.Td(f"{r['current']:,.0f}" if pd.notna(r["current"]) else "—", style={"textAlign": "right"}),
                    html.Td(f"{r['y2040']:,.0f}" if pd.notna(r["y2040"]) else "—", style={"textAlign": "right"}),
                    html.Td(dcc.Graph(
                        id={"type": "spark", "code": r["code"]},
                        figure=fig,
                        config={"displayModeBar": False},  # hover enabled
                        style={"height": "44px"}
                    )),
                ])
            )
        return body_rows, region_header
