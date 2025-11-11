# pages/sankey.py
from functools import lru_cache
import json
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

# ---- project loaders / config ----
from ..data.loader import load_df  # cached parquet loader

# Optional: paths/helper from your project
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

# ----------------------------------
df = load_df()

# Robust column detection (dataset side)
REG_LEVEL_COL_CAND = ["region_level", "geo_level", "geo_levels"]
REG_CODE_COL_CAND  = ["region_code", "geo_code", "NUTS_ID", "nuts_id", "REGION_CODE"]
SEX_CODE_COL_CAND  = ["sex_code", "sex", "SEX"]
AGE_CODE_COL_CAND  = ["age_code", "age", "AGE_CODE"]
EMP_COL_CAND       = ["emp", "employment", "Employed", "EMP"]

def _detect_col(cands, frame=df):
    for c in cands:
        if c in frame.columns:
            return c
    return None

REG_LEVEL_COL = _detect_col(REG_LEVEL_COL_CAND)
REG_CODE_COL  = _detect_col(REG_CODE_COL_CAND)
SEX_CODE_COL  = _detect_col(SEX_CODE_COL_CAND)
AGE_CODE_COL  = _detect_col(AGE_CODE_COL_CAND)
EMP_COL       = _detect_col(EMP_COL_CAND)

# ------------------------- basics / defaults -------------------------
def _uniq_sorted(series):
    s = pd.Series(series).dropna().unique().tolist()
    try:
        s = sorted(map(int, s))
    except Exception:
        s = sorted(map(str, s))
    return s

YEARS_ALL = _uniq_sorted(df["year"])
DEFAULT_YEAR = 2025 if 2025 in YEARS_ALL else YEARS_ALL[-1]

METRIC_OPTIONS = [
    {"label": "Sex", "value": "sex"},
    {"label": "Age code", "value": "age"},
]
DEFAULT_METRIC = "sex"

# ------------------------- GeoJSON loading -------------------------
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
        cntr = pr.get("CNTR_CODE") or ""  # e.g., DE, FR
        m[code] = str(cntr)
    return m

# ------------------------- utility filters -------------------------
def _ensure_constant_levels(sub: pd.DataFrame) -> pd.DataFrame:
    if "age_level" in sub.columns:
        sub = sub[sub["age_level"].astype(str) == "2"]
    if "sex_level" in sub.columns:
        sub = sub[sub["sex_level"].astype(str) == "2"]
    return sub

# dataset presence by level/year
def _dataset_codes_for(year: int, level: int):
    if REG_LEVEL_COL is None or REG_CODE_COL is None:
        return set()
    sub = df[
        (df["year"].astype(int) == int(year)) &
        (df[REG_LEVEL_COL].astype(int) == int(level))
    ][[REG_CODE_COL]].dropna()
    return set(sub[REG_CODE_COL].astype(str).unique())

# ------------------------- dropdown options (NAME_LATN) -------------------------
def _countries_options(year: int):
    ds_codes = _dataset_codes_for(year, 2)
    names = _code_to_name(2)
    return sorted(
        ({"label": names.get(c, c), "value": c} for c in ds_codes),
        key=lambda x: x["label"]
    )

def _nuts1_options(year: int, selected_countries=None):
    ds_codes = _dataset_codes_for(year, 3)
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

def _nuts2_options(year: int, selected_countries=None, selected_nuts1=None):
    ds_codes = _dataset_codes_for(year, 4)
    if not ds_codes:
        return []
    names = _code_to_name(4)
    cntr4 = _code_to_cntr(4)

    # Filter by countries if provided
    if selected_countries:
        cntr2 = _code_to_cntr(2)
        selected_countries = selected_countries if isinstance(selected_countries, list) else [selected_countries]
        cntr_filter = {cntr2.get(c, "") for c in selected_countries}
        cntr_filter.discard("")
        ds_codes = {c for c in ds_codes if cntr4.get(c, "") in cntr_filter}

    # If NUTS1 provided, keep only NUTS2 that fall under selected NUTS1 (prefix rule)
    if selected_nuts1:
        sel1 = set(selected_nuts1 if isinstance(selected_nuts1, list) else [selected_nuts1])
        ds_codes = {c for c in ds_codes if any(c.startswith(p) for p in sel1)}

    return sorted(({"label": names.get(c, c), "value": c} for c in ds_codes), key=lambda x: x["label"])

# ------------------------- aggregation (cached) -------------------------
@lru_cache(maxsize=1024)
def _aggregate_for_sankey(year: int, metric: str,
                          countries_key: str, nuts1_key: str, nuts2_key: str):
    """
    Deepest selection wins (NUTS2 > NUTS1 > Countries). If none selected,
    we use level=2 (countries) to avoid double counting across levels.
    """
    # Decode selections
    sel_nuts2 = None if nuts2_key == "ALL" else set(nuts2_key.split("|"))
    sel_nuts1 = None if nuts1_key == "ALL" else set(nuts1_key.split("|"))
    sel_cntr  = None if countries_key == "ALL" else set(countries_key.split("|"))

    # Decide target level & constrain rows
    if sel_nuts2:
        target_level = 4
        mask = (df["year"].astype(int) == year) & (df[REG_LEVEL_COL].astype(int) == target_level) & \
               (df[REG_CODE_COL].astype(str).isin(sel_nuts2))
    elif sel_nuts1:
        target_level = 3
        mask = (df["year"].astype(int) == year) & (df[REG_LEVEL_COL].astype(int) == target_level) & \
               (df[REG_CODE_COL].astype(str).isin(sel_nuts1))
    elif sel_cntr:
        target_level = 2
        mask = (df["year"].astype(int) == year) & (df[REG_LEVEL_COL].astype(int) == target_level) & \
               (df[REG_CODE_COL].astype(str).isin(sel_cntr))
    else:
        # No geo selections: safest is level 2 to avoid double counting
        target_level = 2
        mask = (df["year"].astype(int) == year) & (df[REG_LEVEL_COL].astype(int) == target_level)

    sub = df.loc[mask].copy()
    sub = _ensure_constant_levels(sub)

    # Aggregate left side by metric
    if metric == "sex":
        if SEX_CODE_COL is None or EMP_COL is None:
            return [], []
        s = sub.copy()
        s[SEX_CODE_COL] = s[SEX_CODE_COL].astype(str)
        s = s[s[SEX_CODE_COL].isin(["M", "F"])]
        grp = s.groupby(SEX_CODE_COL, dropna=True)[EMP_COL].sum().sort_values(ascending=False)
        labels = ["Male" if k == "M" else "Female" for k in grp.index.tolist()]
        values = grp.values.astype(float).tolist()
        return labels, values

    # metric == "age"
    if AGE_CODE_COL is None or EMP_COL is None:
        return [], []
    s = sub.copy()
    s[AGE_CODE_COL] = s[AGE_CODE_COL].astype(str)
    grp = s.groupby(AGE_CODE_COL, dropna=True)[EMP_COL].sum().sort_values(ascending=False)
    labels = grp.index.tolist()
    values = grp.values.astype(float).tolist()
    return labels, values

# ------------------------- figure -------------------------
def _sankey_figure(left_labels, left_values, title_txt="Employment composition"):
    if not left_labels:
        return go.Figure(layout=dict(
            height=560,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=12, r=12, t=28, b=12),
            annotations=[dict(text="No data for the current filters.", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False,
                              font=dict(size=14, color="#6c757d"))]
        ))

    right_label = "Employed"
    node_labels = [*left_labels, right_label]
    right_idx = len(left_labels)

    sources = list(range(len(left_labels)))
    targets = [right_idx] * len(left_labels)
    values = left_values

    # Brighter palette
    bright = [
        "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
        "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#edc948"
    ]
    node_colors = [bright[i % len(bright)] for i in range(len(left_labels))] + ["#003662"]
    link_colors = ["rgba(0,54,98,0.25)"] * len(values)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels, color=node_colors, pad=18, thickness=18,
            line=dict(color="rgba(0,0,0,0.15)", width=1),
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources, target=targets, value=values, color=link_colors,
            hovertemplate="%{source.label} → %{target.label}<br><b>%{value:,.0f}</b><extra></extra>",
        )
    ))

    fig.update_layout(
        title=dict(text=title_txt, x=0.01, y=0.98, font=dict(size=16, color="#003662")),
        height=560, margin=dict(l=12, r=12, t=36, b=12),
        paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(255,255,255,1)",
        font=dict(size=12)
    )
    return fig

# ------------------------- UI -------------------------
def _filter_bar():
    # Single row, compact Year & Metric, wider geo filters
    DD_STYLE = {"zIndex": 2000}  # dropdown menus above cards

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col([  # Year (small)
                        dbc.Label("Year", className="text-muted small"),
                        dcc.Dropdown(
                            id="sankey-year-dd",
                            options=[{"label": str(y), "value": int(y)} for y in YEARS_ALL],
                            value=DEFAULT_YEAR, clearable=False, style=DD_STYLE
                        )
                    ], md=1),

                    dbc.Col([  # Countries (wide)
                        dbc.Label("Countries", className="text-muted small"),
                        dcc.Dropdown(
                            id="sankey-country-dd",
                            options=[], value=None, multi=True,
                            placeholder="All countries", clearable=True, style=DD_STYLE
                        )
                    ], md=4),

                    dbc.Col([  # NUTS 1 (medium)
                        dbc.Label("Major Socio-economic Regions", className="text-muted small"),
                        dcc.Dropdown(
                            id="sankey-nuts1-dd",
                            options=[], value=None, multi=True,
                            placeholder="All NUTS 1", clearable=True, style=DD_STYLE
                        )
                    ], md=3),

                    dbc.Col([  # NUTS 2 (medium)
                        dbc.Label("Medium-sized administrative regions", className="text-muted small"),
                        dcc.Dropdown(
                            id="sankey-nuts2-dd",
                            options=[], value=None, multi=True,
                            placeholder="All NUTS 2", clearable=True, style=DD_STYLE
                        )
                    ], md=3),

                    dbc.Col([  # Metric (small)
                        dbc.Label("Metric", className="text-muted small"),
                        dcc.Dropdown(
                            id="sankey-metric-dd",
                            options=METRIC_OPTIONS, value=DEFAULT_METRIC,
                            clearable=False, style=DD_STYLE
                        )
                    ], md=1),
                ],
                className="g-3 align-items-end"
            )
        ),
        style={"borderRadius": "16px", "boxShadow": "0 2px 12px rgba(0,0,0,.06)", "backgroundColor": "#ffffff"},
        className="mb-3"
    )

def _explainer_card():
    return dbc.Card(
        dbc.CardBody([
            html.H5("What this Sankey shows", className="mb-2", style={"color": "#003662"}),
            html.P(
                "See how total employed people split by your chosen metric on the left and flow into a single "
                "‘Employed’ node on the right. Drill down by Countries → NUTS 1 → NUTS 2. "
                "Deepest selection filters the data (NUTS2 > NUTS1 > Countries). "
                "Fixed at age_level = 2 and sex_level = 2.", className="text-muted"
            ),
            html.Ul([
                html.Li("Left = chosen metric groups (e.g., Male/Female or age bands)."),
                html.Li("Right = total employed (sum of selected groups)."),
                html.Li("Flows = employment totals for each group."),
            ], className="mb-0")
        ]),
        style={"borderRadius": "16px", "boxShadow": "0 2px 12px rgba(0,0,0,.06)", "backgroundColor": "#ffffff"},
        className="mb-3"
    )

def _chart_card():
    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                id="sankey-graph",
                config={"displayModeBar": False},
                style={"height": "560px"}
            )
        ),
        style={"borderRadius": "20px", "boxShadow": "0 4px 20px rgba(0,0,0,.10)", "backgroundColor": "#ffffff"},
        className="mb-3"
    )

def layout():
    return dbc.Container([
        _filter_bar(),
        dbc.Row([
            dbc.Col(_explainer_card(), md=4),
            dbc.Col(_chart_card(), md=8),
        ], className="g-3"),
        html.Div(style={"height": "12px"})
    ], fluid=True)

# ------------------------- callbacks -------------------------
def register_callbacks(app):
    # Countries options (by Year)
    @app.callback(
        Output("sankey-country-dd", "options"),
        Output("sankey-country-dd", "value"),
        Input("sankey-year-dd", "value"),
        State("sankey-country-dd", "value"),
        prevent_initial_call=False
    )
    def _update_countries(year, cur_val):
        opts = _countries_options(int(year))
        if not cur_val:
            return opts, None
        valid = {o["value"] for o in opts}
        kept = [v for v in (cur_val if isinstance(cur_val, list) else [cur_val]) if v in valid]
        return opts, (kept if kept else None)

    # NUTS1 options depend on Year + Countries
    @app.callback(
        Output("sankey-nuts1-dd", "options"),
        Output("sankey-nuts1-dd", "value"),
        Input("sankey-year-dd", "value"),
        Input("sankey-country-dd", "value"),
        State("sankey-nuts1-dd", "value"),
        prevent_initial_call=False
    )
    def _update_nuts1(year, sel_countries, cur_val):
        opts = _nuts1_options(int(year), sel_countries)
        if not cur_val:
            return opts, None
        valid = {o["value"] for o in opts}
        kept = [v for v in (cur_val if isinstance(cur_val, list) else [cur_val]) if v in valid]
        return opts, (kept if kept else None)

    # NUTS2 options depend on Year + Countries (+ NUTS1 if present)
    @app.callback(
        Output("sankey-nuts2-dd", "options"),
        Output("sankey-nuts2-dd", "value"),
        Input("sankey-year-dd", "value"),
        Input("sankey-country-dd", "value"),
        Input("sankey-nuts1-dd", "value"),
        State("sankey-nuts2-dd", "value"),
        prevent_initial_call=False
    )
    def _update_nuts2(year, sel_countries, sel_nuts1, cur_val):
        opts = _nuts2_options(int(year), sel_countries, sel_nuts1)
        if not cur_val:
            return opts, None
        valid = {o["value"] for o in opts}
        kept = [v for v in (cur_val if isinstance(cur_val, list) else [cur_val]) if v in valid]
        return opts, (kept if kept else None)

    # Sankey figure — deepest selection wins (NUTS2 > NUTS1 > Countries)
    @app.callback(
        Output("sankey-graph", "figure"),
        Input("sankey-year-dd", "value"),
        Input("sankey-metric-dd", "value"),
        Input("sankey-country-dd", "value"),
        Input("sankey-nuts1-dd", "value"),
        Input("sankey-nuts2-dd", "value"),
    )
    def _update_sankey(year, metric, sel_countries, sel_nuts1, sel_nuts2):
        def _key(v):
            if not v:
                return "ALL"
            v = v if isinstance(v, list) else [v]
            return "|".join(sorted(map(str, v)))

        labels, values = _aggregate_for_sankey(
            int(year), str(metric),
            _key(sel_countries), _key(sel_nuts1), _key(sel_nuts2)
        )

        title_bits = [
            f"Year {year}",
            "Metric: Sex" if metric == "sex" else "Metric: Age code",
            "age_level = 2", "sex_level = 2",
        ]
        return _sankey_figure(labels, values, " | ".join(title_bits))
