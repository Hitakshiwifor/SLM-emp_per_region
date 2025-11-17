# pages/sankey.py
from functools import lru_cache
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

# ---- project loaders / config ----
from ..data.loader import load_df
from ..config import GEO_PATHS

# Load once, already aggregated + cached inside load_df()
df = load_df().copy()

# --- basic column names (from loader) ---
YEAR_COL = "year"
REGION_COL = "region_code"
SEX_COL = "sex_code"
AGE_COL = "age_code"
EMP_COL = "emp"
UNE_COL = "une"
SUP_COL = "supply"

# --------- dropdown base values ---------
YEARS_ALL = sorted(int(y) for y in df[YEAR_COL].dropna().unique())
DEFAULT_YEAR = 2025 if 2025 in YEARS_ALL else (YEARS_ALL[-1] if YEARS_ALL else None)
SEX_CODES = [c for c in ["F", "M"] if c in set(df[SEX_COL].astype(str).unique())]

AGE_CODES = sorted(df[AGE_COL].astype(str).dropna().unique())


@lru_cache(maxsize=1)
def _region_name_map() -> dict:
    """Map NUTS2 region codes to human readable NAME_LATN using the level-2 GeoJSON.

    This mirrors the logic used on the map page: we read NUTS_ID and NAME_LATN
    from the NUTS GeoJSON (which itself comes from the same source as the
    shapes.NUTS_RG_60M_2024_4326 table).
    """
    p: Path | None = GEO_PATHS.get(4)
    if not p or not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        gj = json.load(f)
    mapping = {}
    for ft in gj.get("features", []):
        pr = ft.get("properties", {})
        code = str(
            pr.get("NUTS_ID") or pr.get("nuts_id") or pr.get("id") or ""
        ).strip().upper()
        if not code:
            continue
        name = (
            pr.get("NAME_LATN")
            or pr.get("name_latn")
            or pr.get("NAME_ENGL")
            or pr.get("NAME")
            or code
        )
        mapping[code] = str(name)
    return mapping


def _region_options_for_year(year: int):
    """Location dropdown options (value = NUTS2 code, label = NAME_LATN)."""
    sub = df[df[YEAR_COL].astype(int) == int(year)]
    codes = sorted(sub[REGION_COL].astype(str).dropna().unique())
    names = _region_name_map()
    return [
        {"label": names.get(c, c), "value": c}
        for c in codes
    ]


# =================== Layout helpers ===================
def _filter_card():
    dd_style = {"zIndex": 2000}

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    # Year
                    dbc.Col(
                        [
                            dbc.Label("Year", className="text-muted small"),
                            dcc.Dropdown(
                                id="sankey-year-dd",
                                options=[{"label": str(y), "value": int(y)} for y in YEARS_ALL],
                                value=DEFAULT_YEAR,
                                clearable=False,
                                style=dd_style,
                            ),
                        ],
                        md=2,
                    ),
                    # Location (NUTS2)
                    dbc.Col(
                        [
                            dbc.Label("Location", className="text-muted small"),
                            dcc.Dropdown(
                                id="sankey-location-dd",
                                options=[],
                                value=None,
                                multi=True,
                                placeholder="All locations",
                                style=dd_style,
                            ),
                        ],
                        md=4,
                    ),
                    # Gender
                    dbc.Col(
                        [
                            dbc.Label("Gender", className="text-muted small"),
                            dcc.Dropdown(
                                id="sankey-sex-dd",
                                options=[
                                    {"label": "Female", "value": "F"},
                                    {"label": "Male", "value": "M"},
                                ],
                                value=None,  # None = both
                                multi=True,
                                placeholder="All genders",
                                style=dd_style,
                            ),
                        ],
                        md=3,
                    ),
                    # Age group
                    dbc.Col(
                        [
                            dbc.Label("Age group", className="text-muted small"),
                            dcc.Dropdown(
                                id="sankey-age-dd",
                                options=[
                                    {"label": a.replace("Y", "Age "), "value": a}
                                    for a in AGE_CODES
                                ],
                                value=None,
                                multi=True,
                                placeholder="All age groups",
                                style=dd_style,
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="g-2 align-items-end",
            )
        ),
        className="shadow-sm mb-3",
    )


def _explainer_card():
    return dbc.Card(
        dbc.CardBody(
            [
                
                html.P(
                    [
                        "This Sankey chart shows how the workforce is distributed across ",
                        html.B("gender"), ", ",
                        html.B("age groups"), " and three key labor market metrics: ",
                        html.B("Employment"), ", ",
                        html.B("Unemployment"), " and ",
                        html.B("Labour supply"), ".",
                    ],
                    className="small text-muted",
                ),
                html.Ul(
                    [
                        html.Li("The thickness of each flow represents the underlying value.", className="small"),
                        html.Li("Left: Female / Male.", className="small"),
                        html.Li("Middle: Age bands (e.g. Y15-24, Y25-34).", className="small"),
                        html.Li("Right: Employment (emp), Unemployment (une) and Labour supply (supply).", className="small"),
                    ],
                    className="small text-muted",
                ),
                html.P(
                    "Use the filters above to focus on a specific year, location, gender or age group.",
                    className="small text-muted mt-2",
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


def _chart_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Gender / age composition", className="card-title mb-3"),
                dcc.Graph(
                    id="sankey-graph",
                    figure=go.Figure(),
                    config={
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                    style={"height": "520px"},
                ),
            ]
        ),
        className="shadow-sm h-100",
    )


# =================== Sankey construction ===================
def _build_sankey_figure(sub: pd.DataFrame, year: int) -> go.Figure:
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data for selected filters (Year {year})",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    # Clean codes and keep only F/M
    s = sub.copy()
    s[SEX_COL] = s[SEX_COL].astype(str).str.strip()
    s[AGE_COL] = s[AGE_COL].astype(str).str.strip()
    s = s[s[SEX_COL].isin(SEX_CODES)]

    if s.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data for selected filters (Year {year})",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    # Sex nodes (left)
    sex_nodes = [("F", "Female"), ("M", "Male")]
    sex_nodes = [p for p in sex_nodes if p[0] in set(s[SEX_COL])]
    sex_labels = [lbl for _, lbl in sex_nodes]

    # Age nodes (middle)
    age_vals = sorted(s[AGE_COL].unique().tolist())
    age_labels = [a.replace("Y", "Age ") for a in age_vals]

    # Metric nodes (right)
    metric_keys = ["emp", "une", "supply"]
    metric_labels = ["Employment (emp)", "Unemployment (une)", "Labour supply (supply)"]

    # Build node index mapping
    node_labels = sex_labels + age_labels + metric_labels

    sex_index = {code: i for i, (code, _) in enumerate(sex_nodes)}
    age_index = {a: len(sex_labels) + i for i, a in enumerate(age_vals)}
    metric_index = {
        "emp": len(sex_labels) + len(age_labels) + 0,
        "une": len(sex_labels) + len(age_labels) + 1,
        "supply": len(sex_labels) + len(age_labels) + 2,
    }

    sources = []
    targets = []
    values = []

    # 1) flows: Sex -> Age, weighted by labour supply
    if SUP_COL in s.columns:
        grp_sex_age = (
            s.groupby([SEX_COL, AGE_COL], observed=True)[SUP_COL]
            .sum()
            .reset_index()
        )
        for _, row in grp_sex_age.iterrows():
            sc = row[SEX_COL]
            ag = row[AGE_COL]
            val = float(row[SUP_COL])
            if val <= 0:
                continue
            if sc not in sex_index or ag not in age_index:
                continue
            sources.append(sex_index[sc])
            targets.append(age_index[ag])
            values.append(val)

    # 2) flows: Age -> metrics (emp / une / supply)
    grp_age = (
        s.groupby(AGE_COL, observed=True)[[EMP_COL, UNE_COL, SUP_COL]]
        .sum()
        .reset_index()
    )

    for _, row in grp_age.iterrows():
        ag = row[AGE_COL]
        if ag not in age_index:
            continue
        for key, col_name in [("emp", EMP_COL), ("une", UNE_COL), ("supply", SUP_COL)]:
            val = float(row[col_name])
            if val <= 0:
                continue
            sources.append(age_index[ag])
            targets.append(metric_index[key])
            values.append(val)

    if not sources:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data for selected filters (Year {year})",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        return fig

    # Colors: softer palette for nodes, subtle links
    base_colors = [
        "#003662", "#f58518", "#54a24b", "#e45756", "#72b7b2",
        "#b279a2", "#ff9da6", "#9d755d", "#bab0ac", "#edc948"
    ]
    node_colors = [base_colors[i % len(base_colors)] for i in range(len(node_labels))]
    link_colors = ["rgba(0,54,98,0.25)" for _ in values]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=18,
                line=dict(color="white", width=0.5),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    )

    fig.update_layout(
        title=f"Gender → Age → Labour market status | Year {year}",
        font=dict(size=11),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# =================== Public layout + callbacks ===================
def layout():
    return dbc.Container(
        [
            _filter_card(),
            dbc.Row(
                [
                    dbc.Col(_explainer_card(), md=4),
                    dbc.Col(_chart_card(), md=8),
                ],
                className="g-3",
            ),
            html.Div(style={"height": "12px"}),
        ],
        fluid=True,
    )


def register_callbacks(app):
    # Update location options when year changes
    @app.callback(
        Output("sankey-location-dd", "options"),
        Output("sankey-location-dd", "value"),
        Input("sankey-year-dd", "value"),
        prevent_initial_call=False,
    )
    def _update_locations(year):
        if year is None:
            return [], None
        opts = _region_options_for_year(int(year))
        # don't auto-select anything → None means "all"
        return opts, None

    # Main figure
    @app.callback(
        Output("sankey-graph", "figure"),
        Input("sankey-year-dd", "value"),
        Input("sankey-location-dd", "value"),
        Input("sankey-sex-dd", "value"),
        Input("sankey-age-dd", "value"),
        prevent_initial_call=False,
    )
    def _update_sankey(year, loc_value, sex_vals, age_vals):
        if year is None:
            return go.Figure()

        sub = df[df[YEAR_COL].astype(int) == int(year)].copy()

        # Normalise filter values to lists
        def _to_list(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple, set)):
                return list(v)
            return [v]

        loc_list = _to_list(loc_value)
        sex_list = _to_list(sex_vals)
        age_list = _to_list(age_vals)

        if loc_list:
            sub = sub[sub[REGION_COL].isin(loc_list)]
        if sex_list:
            sub = sub[sub[SEX_COL].astype(str).isin([str(x) for x in sex_list])]
        if age_list:
            sub = sub[sub[AGE_COL].astype(str).isin([str(x) for x in age_list])]

        return _build_sankey_figure(sub, int(year))
