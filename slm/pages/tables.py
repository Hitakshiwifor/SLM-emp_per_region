# pages/tables.py
from functools import lru_cache
from pathlib import Path
import os

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State

# -------- project config / loader ----------
from ..config import DATA_DIR
from ..data.loader import load_df  # should load main.supply_per_region

# ========= 1. DUCKDB CONNECTION (for shapes.*) ===============================

# Path to qa.duckdb; override via env var if needed
DB_PATH = Path(os.environ.get("SLM_DUCKDB", DATA_DIR / "qa.duckdb"))

@lru_cache(maxsize=1)
def get_con() -> duckdb.DuckDBPyConnection:
    """Shared read-only DuckDB connection."""
    return duckdb.connect(DB_PATH, read_only=True)

# ========= 2. DATASET + COLUMN DETECTION =====================================

df = load_df()  # expected to be main.supply_per_region

# We explicitly want region_code
if "region_code" not in df.columns:
    raise ValueError("main.supply_per_region must contain a 'region_code' column.")

REGION_COL = "region_code"

# Detect year + metric columns flexibly
YEAR_COL_CAND   = ["year", "time", "TIME"]
EMP_COL_CAND    = ["emp", "EMP", "employment"]
UNE_COL_CAND    = ["une", "UNE", "unemp", "unemployment"]
SUPPLY_COL_CAND = ["supply", "SUPPLY", "labour_supply", "lab_supply"]


def _detect_col(cands, frame=df):
    for c in cands:
        if c in frame.columns:
            return c
    return None


YEAR_COL   = _detect_col(YEAR_COL_CAND)
EMP_COL    = _detect_col(EMP_COL_CAND)
UNE_COL    = _detect_col(UNE_COL_CAND)
SUPPLY_COL = _detect_col(SUPPLY_COL_CAND)

if YEAR_COL is None:
    raise ValueError("Could not detect year column in main.supply_per_region.")

# Map metric key -> actual column name in fact table
_METRIC_TO_COL = {}
if EMP_COL:
    _METRIC_TO_COL["emp"] = EMP_COL
if UNE_COL:
    _METRIC_TO_COL["une"] = UNE_COL
if SUPPLY_COL:
    _METRIC_TO_COL["supply"] = SUPPLY_COL

if not _METRIC_TO_COL:
    raise ValueError("No metric columns (emp, une, supply) found in main.supply_per_region.")

METRIC_LABELS = {
    "emp": "Employment (emp)",
    "une": "Unemployment (une)",
    "supply": "Labour supply (supply)",
}

METRIC_OPTIONS = [
    {"label": METRIC_LABELS[m], "value": m}
    for m in ["emp", "une", "supply"]
    if m in _METRIC_TO_COL
]
DEFAULT_METRIC_KEY = METRIC_OPTIONS[0]["value"]

CURRENT_YEAR = 2025  # "Metric today"
TARGET_YEAR  = 2040  # "Metric in 2040"

# ========= 3. SHAPES DIMENSION: region_code -> NAME_LATN =====================

@lru_cache(maxsize=1)
def _regioncode_to_name() -> dict:
    """
    Read shapes.NUTS_RG_60M_2024_4326 from qa.duckdb and build:
        NUTS_ID -> NAME_LATN

    We restrict to LEVL_CODE = 2 (NUTS2 regions).
    region_code in main.supply_per_region is assumed to match NUTS_ID.
    """
    con = get_con()
    shp = con.sql(
        """
        SELECT DISTINCT NUTS_ID, NAME_LATN
        FROM shapes.NUTS_RG_60M_2024_4326
        WHERE LEVL_CODE = 2
        """
    ).df()

    shp["NUTS_ID"] = shp["NUTS_ID"].astype(str)
    shp["NAME_LATN"] = shp["NAME_LATN"].astype(str)

    return dict(zip(shp["NUTS_ID"], shp["NAME_LATN"]))


@lru_cache(maxsize=1)
def _all_location_codes():
    """All region_code values present in the fact table."""
    return sorted(df[REGION_COL].dropna().astype(str).unique())


def _location_options():
    """
    Dropdown options for Location:
      - value: region_code
      - label: NAME_LATN from shapes.*
    """
    codes = _all_location_codes()
    name_map = _regioncode_to_name()
    opts = [
        {"label": name_map.get(c, c), "value": c}
        for c in codes
    ]
    return sorted(opts, key=lambda x: x["label"])

# ========= 4. DATA PREP FOR TABLE ===========================================

def _ensure_levels(sub: pd.DataFrame) -> pd.DataFrame:
    """
    To be consistent with other pages:
      - if age_level exists, restrict to 2
      - if sex_level exists, restrict to 2
    """
    if "age_level" in sub.columns:
        sub = sub[sub["age_level"].astype(str) == "2"]
    if "sex_level" in sub.columns:
        sub = sub[sub["sex_level"].astype(str) == "2"]
    return sub


def _year_clip(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep only years between 2000 and 2040."""
    sub = sub.copy()
    sub[YEAR_COL] = sub[YEAR_COL].astype(int)
    return sub[(sub[YEAR_COL] >= 2000) & (sub[YEAR_COL] <= 2040)]


@lru_cache(maxsize=32)
def _rows_for_table(metric_key: str) -> pd.DataFrame:
    """
    Precompute one tidy DataFrame per metric:
      - one row per region_code
      - 'current'   : value at 2025 (or nearest year)
      - 'y2040'     : value at 2040 (or nearest year)
      - 'series_x'  : list of years (2000–2040)
      - 'series_y'  : list of metric values
      - 'region_name': NAME_LATN from shapes.NUTS_RG_60M_2024_4326
    """
    if metric_key not in _METRIC_TO_COL:
        return pd.DataFrame([])

    metric_col = _METRIC_TO_COL[metric_key]

    sub = df[[YEAR_COL, REGION_COL, metric_col]].copy()
    sub = _ensure_levels(sub)
    sub = _year_clip(sub)

    if sub.empty:
        return pd.DataFrame([])

    sub[YEAR_COL] = sub[YEAR_COL].astype(int)
    sub[REGION_COL] = sub[REGION_COL].astype(str)

    # yearly totals per region_code
    agg = (
        sub
        .groupby([REGION_COL, YEAR_COL], as_index=False)[metric_col]
        .sum()
    )

    rows = []
    for code, g in agg.sort_values([REGION_COL, YEAR_COL]).groupby(REGION_COL, sort=False):
        years = g[YEAR_COL].astype(int).tolist()
        vals  = g[metric_col].astype(float).tolist()

        yrs_np = g[YEAR_COL].to_numpy()
        val_np = g[metric_col].to_numpy()

        def pick(target_year: int) -> float:
            # exact year if available (sum all rows for that year)
            if target_year in g[YEAR_COL].values:
                return float(g.loc[g[YEAR_COL] == target_year, metric_col].sum())
            # nearest year fallback
            if len(g) == 0:
                return float("nan")
            idx = int(np.argmin(np.abs(yrs_np - target_year)))
            return float(val_np[idx])

        val_current = pick(CURRENT_YEAR)
        val_2040    = pick(TARGET_YEAR)

        rows.append(
            {
                "code": code,
                "current": val_current,
                "y2040": val_2040,
                "series_x": years,
                "series_y": vals,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # attach NAME_LATN labels
    name_map = _regioncode_to_name()
    out["region_name"] = out["code"].map(lambda c: name_map.get(c, c))

    # keep all regions; sort by Metric today
    out = out.sort_values("current", ascending=False)
    return out.reset_index(drop=True)

# ========= 5. DATA PREP FOR RISK MATRIX =====================================

@lru_cache(maxsize=8)
def _risk_data_for_year(year: int) -> pd.DataFrame:
    """
    Build per-region data for the risk matrix scatter:
      - supply (x)
      - une (y)
      - region_name (for hover)
    """
    if SUPPLY_COL is None or UNE_COL is None:
        return pd.DataFrame([])

    sub = df[[YEAR_COL, REGION_COL, SUPPLY_COL, UNE_COL]].copy()
    sub = _ensure_levels(sub)
    sub[YEAR_COL] = sub[YEAR_COL].astype(int)
    sub = sub[sub[YEAR_COL] == int(year)]

    if sub.empty:
        return pd.DataFrame([])

    sub[REGION_COL] = sub[REGION_COL].astype(str)

    agg = (
        sub
        .groupby(REGION_COL, as_index=False)[[SUPPLY_COL, UNE_COL]]
        .sum()
    )

    agg.rename(
        columns={
            SUPPLY_COL: "supply",
            UNE_COL: "une"
        },
        inplace=True,
    )

    name_map = _regioncode_to_name()
    agg["region_name"] = agg[REGION_COL].map(lambda c: name_map.get(c, c))

    return agg


def _risk_matrix_figure(selected_locations):
    """Create the risk matrix scatter figure."""
    data = _risk_data_for_year(CURRENT_YEAR)

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    text="Risk matrix not available (missing supply/une data)",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Filter by selected locations if any
    if selected_locations:
        sel = selected_locations if isinstance(selected_locations, list) else [selected_locations]
        sel = set(map(str, sel))
        data = data[data[REGION_COL].isin(sel)].copy()

    if data.empty:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    text="No regions match the current filters.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Quadrant thresholds (medians)
    qx = float(data["supply"].median())
    qy = float(data["une"].median())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["supply"],
            y=data["une"],
            mode="markers",
            marker=dict(size=8, color="#003662", opacity=0.8),
            customdata=np.stack([data["region_name"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>"
                          "Supply: %{x:,.0f}<br>"
                          "Unemployed: %{y:,.0f}<extra></extra>",
        )
    )

    # Quadrant lines
    fig.add_shape(
        type="line",
        x0=qx, x1=qx,
        y0=data["une"].min(), y1=data["une"].max(),
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=data["supply"].min(), x1=data["supply"].max(),
        y0=qy, y1=qy,
        line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
    )

    # Quadrant labels (simple annotations)
    xs = data["supply"]
    ys = data["une"]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    fig.add_annotation(
        x=(x_min + qx) / 2,
        y=(qy + y_max) / 2,
        text="Lower supply<br>Higher unemployment",
        showarrow=False,
        font=dict(size=10),
    )
    fig.add_annotation(
        x=(qx + x_max) / 2,
        y=(qy + y_max) / 2,
        text="Higher supply<br>Higher unemployment",
        showarrow=False,
        font=dict(size=10),
    )
    fig.add_annotation(
        x=(x_min + qx) / 2,
        y=(y_min + qy) / 2,
        text="Lower supply<br>Lower unemployment",
        showarrow=False,
        font=dict(size=10),
    )
    fig.add_annotation(
        x=(qx + x_max) / 2,
        y=(y_min + qy) / 2,
        text="Higher supply<br>Lower unemployment",
        showarrow=False,
        font=dict(size=10),
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=f"Supply ({CURRENT_YEAR})",
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis=dict(
            title=f"Unemployed ({CURRENT_YEAR})",
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        showlegend=False,
    )

    return fig

# ========= 6. SPARKLINE FIGURE ==============================================

def _sparkline(x, y):
    """
    Small inline sparkline:
      - colour: #003662
      - hover: "<value> in <year>"
    """
    color = "#003662"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
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

# ========= 7. UI COMPONENTS ==================================================

def _risk_card():
    """Left-hand card: risk matrix scatter + info icon."""
    return dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H5(
                                "Risk matrix: supply vs unemployment",
                                className="mb-0",
                                style={"color": "#003662"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.I(className="bi bi-info-circle"),
                                id="tbl-help-btn",
                                outline=True,
                                color="secondary",
                                size="sm",
                                className="ms-2",
                                style={"borderRadius": "999px"},
                            ),
                            width="auto",
                        ),
                    ],
                    className="g-1 align-items-center mb-2",
                    justify="between",
                ),
                dcc.Graph(
                    id="tbl-risk-scatter",
                    config={"displayModeBar": False},
                    style={"height": "340px"},
                ),
                # Popover with explanation for the whole page
                dbc.Popover(
                    [
                        dbc.PopoverHeader("How to use this page"),
                        dbc.PopoverBody(
                            [
                                html.P(
                                    "This view combines a risk matrix and a time series table "
                                    "to help you prioritise regions.",
                                    className="mb-2",
                                ),
                                html.H6("Risk matrix", className="mt-1"),
                                html.Ul(
                                    [
                                        html.Li(
                                            f"Each point is a region in {CURRENT_YEAR}, with "
                                            "supply on the x-axis and unemployed on the y-axis."
                                        ),
                                        html.Li(
                                            "The quadrant lines (medians) split regions into "
                                            "higher / lower supply and higher / lower unemployment."
                                        ),
                                        html.Li(
                                            "Top-right: large labour pool but also high unemployment "
                                            "(latent talent). Bottom-right: large pool and low unemployment "
                                            "(highly utilised)."
                                        ),
                                        html.Li(
                                            "Use the Location filter to focus on specific regions."
                                        ),
                                    ]
                                ),
                                html.H6("Time series table", className="mt-2"),
                                html.Ul(
                                    [
                                        html.Li(
                                            f"Metric today shows the selected metric in {CURRENT_YEAR} "
                                            "(or nearest year)."
                                        ),
                                        html.Li(
                                            f"Metric in {TARGET_YEAR} shows the same metric in {TARGET_YEAR} "
                                            "(or nearest year)."
                                        ),
                                        html.Li(
                                            "Metric over time is a sparkline from 2000–2040; "
                                            "hover to see exact values and years."
                                        ),
                                        html.Li(
                                            "By default you see the top 30 locations by Metric today. "
                                            "Select locations in the filter to see others."
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    id="tbl-help-popover",
                    target="tbl-help-btn",
                    is_open=False,
                    placement="bottom",
                    trigger="click",
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
                    dbc.Col(
                        [
                            dbc.Label("Location", className="text-muted small"),
                            dcc.Dropdown(
                                id="tbl-location-dd",
                                options=[],     # filled via callback
                                value=None,
                                multi=True,
                                placeholder="All locations",
                                clearable=True,
                                style=DD_STYLE,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Metric", className="text-muted small"),
                            dcc.Dropdown(
                                id="tbl-metric-dd",
                                options=METRIC_OPTIONS,
                                value=DEFAULT_METRIC_KEY,
                                clearable=False,
                                style=DD_STYLE,
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-3 align-items-end",
            )
        ),
        style={
            "borderRadius": "16px",
            "boxShadow": "0 2px 12px rgba(0,0,0,.06)",
            "backgroundColor": "#ffffff",
        },
        className="mb-3",
    )


def _table_card():
    header = html.Thead(
        html.Tr(
            [
                html.Th("Location"),
                html.Th("Metric today", style={"width": "180px", "textAlign": "right"}),
                html.Th("Metric in 2040", style={"width": "160px", "textAlign": "right"}),
                html.Th("Metric over time"),
            ]
        )
    )
    body = html.Tbody(id="tbl-body")

    return dbc.Card(
        dbc.CardBody(
            dbc.Table(
                [header, body],
                bordered=False,
                hover=True,
                responsive=True,
                className="align-middle",
            )
        ),
        style={
            "borderRadius": "20px",
            "boxShadow": "0 4px 20px rgba(0,0,0,.10)",
            "backgroundColor": "#ffffff",
        },
        className="mb-3",
    )


def layout():
    return dbc.Container(
        [
            _filter_bar(),
            dbc.Row(
                [
                    dbc.Col(_risk_card(), lg=4, md=5, className="mb-3"),
                    dbc.Col(_table_card(), lg=8, md=7, className="mb-3"),
                ],
                className="g-3",
            ),
            html.Div(style={"height": "12px"}),
        ],
        fluid=True,
    )

# ========= 8. CALLBACKS ======================================================

def register_callbacks(app):
    # Populate Location dropdown (options + keep valid selection)
    @app.callback(
        Output("tbl-location-dd", "options"),
        Output("tbl-location-dd", "value"),
        Input("tbl-metric-dd", "value"),  # just to trigger on load
        State("tbl-location-dd", "value"),
        prevent_initial_call=False,
    )
    def _populate_locations(metric_value, current_value):
        opts = _location_options()
        if not current_value:
            return opts, None

        valid = {o["value"] for o in opts}
        current_list = current_value if isinstance(current_value, list) else [current_value]
        kept = [v for v in current_list if v in valid]
        return opts, (kept if kept else None)

    # Build table rows when Metric / Location change
    @app.callback(
        Output("tbl-body", "children"),
        Input("tbl-metric-dd", "value"),
        Input("tbl-location-dd", "value"),
    )
    def _update_table(metric_key, selected_locations):
        metric_key = metric_key or DEFAULT_METRIC_KEY
        rows_all = _rows_for_table(metric_key)

        if rows_all.empty:
            return [
                html.Tr(
                    html.Td(
                        "No data available for this metric.",
                        colSpan=4,
                        className="text-muted",
                    )
                )
            ]

        # If user selected locations -> show exactly those (no limit)
        if selected_locations:
            sel = selected_locations if isinstance(selected_locations, list) else [selected_locations]
            sel = set(map(str, sel))
            rows = rows_all[rows_all["code"].isin(sel)].copy()
        else:
            # No selection: show ONLY top 30 by "Metric today" (current)
            rows = rows_all.sort_values("current", ascending=False).head(30).copy()

        # Sort by Metric today for a consistent table order
        rows = rows.sort_values("current", ascending=False)

        body_rows = []
        for _, r in rows.iterrows():
            x = r["series_x"]
            y = r["series_y"]
            fig = _sparkline(x, y)

            body_rows.append(
                html.Tr(
                    [
                        html.Td(r["region_name"]),
                        html.Td(
                            f"{r['current']:,.0f}" if pd.notna(r["current"]) else "—",
                            style={"textAlign": "right"},
                        ),
                        html.Td(
                            f"{r['y2040']:,.0f}" if pd.notna(r["y2040"]) else "—",
                            style={"textAlign": "right"},
                        ),
                        html.Td(
                            dcc.Graph(
                                id={"type": "spark", "code": r["code"], "metric": metric_key},
                                figure=fig,
                                config={"displayModeBar": False},
                                style={"height": "44px"},
                            )
                        ),
                    ]
                )
            )

        return body_rows

    # Update risk matrix scatter when Location (or Metric just to re-trigger) changes
    @app.callback(
        Output("tbl-risk-scatter", "figure"),
        Input("tbl-location-dd", "value"),
        Input("tbl-metric-dd", "value"),  # not used in logic, but keeps chart in sync
    )
    def _update_risk_scatter(selected_locations, metric_key):
        return _risk_matrix_figure(selected_locations)

    # Toggle help popover
    @app.callback(
        Output("tbl-help-popover", "is_open"),
        Input("tbl-help-btn", "n_clicks"),
        State("tbl-help-popover", "is_open"),
        prevent_initial_call=False,
    )
    def _toggle_help(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open
