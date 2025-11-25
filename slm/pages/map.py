# pages/map.py

from functools import lru_cache
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash_deck

from ..data.loader import load_df, load_geo
from ..data.geo import norm_code
from ..config import MAPBOX_TOKEN


# ========= 1. DATA & BASIC CONSTANTS ========================================

df = load_df()
GEO_ALL = load_geo()

REGION_LEVEL_FIXED = 4  # NUTS2 only

# ---- helper: safe unique sorted ----
def _uniq_sorted(series):
    s = pd.Series(series).dropna().unique().tolist()
    try:
        s = sorted(map(int, s))
    except Exception:
        s = sorted(map(str, s))
    return s

# metric columns (emp / une / supply)
EMP_COL = None
for c in ["emp", "employment", "EMP"]:
    if c in df.columns:
        EMP_COL = c
        break

UNE_COL = None
for c in ["une", "unemployment", "UNE"]:
    if c in df.columns:
        UNE_COL = c
        break

SUP_COL = None
for c in ["supply", "labour_supply", "SUP"]:
    if c in df.columns:
        SUP_COL = c
        break

if EMP_COL is None:
    raise ValueError("Dataset needs an employment column (emp / employment / EMP).")

# ---- filter values from main.supply_per_region -----------------------------
YEARS_ALL = _uniq_sorted(df["year"])
DEFAULT_YEAR = 2025 if 2025 in YEARS_ALL else YEARS_ALL[-1]

SEX_ALL = sorted(df["sex_code"].astype(str).dropna().unique().tolist())
AGE_CODES_ALL = sorted(df["age_code"].astype(str).dropna().unique().tolist())

# Only use regions that actually appear in the fact table at NUTS2 level
mask_nuts2 = df["region_level"].astype(int) == REGION_LEVEL_FIXED
REGION_CODES_ALL = sorted(
    df.loc[mask_nuts2, "region_code"].astype(str).dropna().unique().tolist()
)


# ========= 2. GEO HELPERS (names + shapes) ==================================

@lru_cache(maxsize=1)
def _geo_for_level(level: int) -> dict:
    """Return GeoJSON for a given NUTS level."""
    return GEO_ALL.get(int(level), {"type": "FeatureCollection", "features": []})


@lru_cache(maxsize=1)
def _code_to_name_nuts2() -> dict:
    """
    Map region_code -> NAME_LATN using the NUTS2 GeoJSON.
    This gives us nice labels for the Location dropdown and tooltips.
    """
    geo = _geo_for_level(REGION_LEVEL_FIXED)
    mapping = {}
    for ft in geo.get("features", []):
        props = ft.get("properties", {})
        code = norm_code(
            props.get("NUTS_ID")
            or props.get("nuts_id")
            or props.get("ID")
            or props.get("id")
        )
        if not code:
            continue
        name = (
            props.get("NAME_LATN")
            or props.get("NAME_2021")
            or props.get("NAME_ENGL")
            or props.get("NAME")
            or code
        )
        mapping[code] = str(name)
    return mapping


@lru_cache(maxsize=1)
def _code_to_cntr_nuts2() -> dict:
    """region_code -> 2-letter country code."""
    geo = _geo_for_level(REGION_LEVEL_FIXED)
    out = {}
    for ft in geo.get("features", []):
        props = ft.get("properties", {})
        code = norm_code(
            props.get("NUTS_ID")
            or props.get("nuts_id")
            or props.get("ID")
            or props.get("id")
        )
        if not code:
            continue
        cntr = props.get("CNTR_CODE") or ""
        out[code] = str(cntr)
    return out


@lru_cache(maxsize=1)
def _code_to_centroid_nuts2() -> dict:
    """Approx centroid of each polygon (for hotspot dot placement)."""
    geo = _geo_for_level(REGION_LEVEL_FIXED)
    out = {}
    for ft in geo.get("features", []):
        props = ft.get("properties", {})
        code = norm_code(
            props.get("NUTS_ID")
            or props.get("nuts_id")
            or props.get("ID")
            or props.get("id")
        )
        geom = ft.get("geometry") or {}
        coords_all = []

        if geom.get("type") == "Polygon":
            for ring in geom.get("coordinates", []):
                coords_all.extend(ring)
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                for ring in poly:
                    coords_all.extend(ring)

        if not coords_all or not code:
            continue

        xs, ys = zip(*coords_all)
        lon = float(sum(xs) / len(xs))
        lat = float(sum(ys) / len(ys))
        out[code] = [lon, lat]
    return out


@lru_cache(maxsize=1)
def _location_options():
    """
    Dropdown options for Location: label = NAME_LATN, value = region_code.
    Only includes regions that actually appear in main.supply_per_region.
    """
    names = _code_to_name_nuts2()
    opts = []
    for code in REGION_CODES_ALL:
        c_norm = norm_code(code)
        label = names.get(c_norm, c_norm)
        opts.append({"label": label, "value": code})
    return sorted(opts, key=lambda x: x["label"])


# ========= 3. FILTERED BASE & METRIC AGGREGATION ============================

def _ensure_levels(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Keep default harmonised levels, to be consistent with Sankey/Tables:
    age_level = 2, sex_level = 2 (when present).
    """
    if "age_level" in sub.columns:
        sub = sub[sub["age_level"].astype(str) == "2"]
    if "sex_level" in sub.columns:
        sub = sub[sub["sex_level"].astype(str) == "2"]
    return sub


def _filtered_base(year, sex_code, age_code, locations):
    """
    Common filtered subset used for:
      - summary cards,
      - age donut,
      - hotspot points.
    """
    sub = df.copy()
    sub = sub[sub["region_level"].astype(int) == REGION_LEVEL_FIXED]
    sub = sub[sub["year"].astype(int) == int(year)]

    if sex_code is not None:
        sub = sub[sub["sex_code"].astype(str) == str(sex_code)]

    if age_code not in (None, "ALL"):
        sub = sub[sub["age_code"].astype(str) == str(age_code)]

    sub = _ensure_levels(sub)

    if locations:
        if isinstance(locations, str):
            locations = [locations]
        sub = sub[sub["region_code"].astype(str).isin(locations)]

    return sub


@lru_cache(maxsize=512)
def _metrics_by_region(
    year: int,
    sex_code: str,
    age_code: str,
    locations_key: str,
):
    """
    Return a dict: {region_code_norm: {'emp':.., 'une':.., 'supply':..}}.
    """
    if locations_key == "ALL":
        selected_locs = None
    else:
        selected_locs = locations_key.split("|")

    sub = _filtered_base(year, sex_code, age_code, selected_locs)

    if sub.empty:
        return {}

    metrics_cols = [c for c in ["emp", "une", "supply"] if c in sub.columns]
    agg = (
        sub.groupby("region_code", as_index=False)[metrics_cols]
        .sum()
        .sort_values("region_code")
    )

    result = {}
    for _, row in agg.iterrows():
        code = norm_code(row["region_code"])
        result[code] = {m: float(row[m]) for m in metrics_cols}
    return result


def _summary_totals(sub: pd.DataFrame):
    """
    Aggregate to overall employment / unemployment / supply totals.
    """
    def _safe(col):
        if col and col in sub.columns:
            return float(sub[col].fillna(0).sum())
        return None

    emp = _safe(EMP_COL)
    une = _safe(UNE_COL)
    sup = _safe(SUP_COL)
    return emp, une, sup


# ========= 4. AGE STRUCTURE DONUT (ACTIVE 25–65 & YOUTH 15–24) ==============

def _age_bucket(code: str):
    """Classify an age_code into 'active', 'youth', or None."""
    c = str(code)
    if "15-24" in c:
        return "youth"
    if any(
        k in c
        for k in [
            "25-64",
            "25-65",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
        ]
    ):
        return "active"
    return None


def _empty_age_donut():
    fig = go.Figure()
    fig.update_layout(
        height=170,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text="No age data",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=11, color="#6c757d"),
            )
        ],
    )
    return fig


def _blue_shades(n: int):
    """Generate n shades around base #003662 for active age bands."""
    base = np.array([0, 54, 98], dtype=float)
    shades = []
    for i in range(n):
        factor = 0.4 + 0.6 * (i / max(n - 1, 1))  # in [0.4, 1.0]
        col = base * factor + 255 * (1 - factor) * 0.1
        col = np.clip(col, 0, 255).astype(int)
        shades.append(f"rgb({col[0]},{col[1]},{col[2]})")
    return shades


def _age_structure_donut(sub: pd.DataFrame):
    """
    Single donut:
      - each slice = one age band,
      - youth 15–24 = teal,
      - active 25–65 bands use different shades of #003662.
    """
    if EMP_COL not in sub.columns or "age_code" not in sub.columns:
        return _empty_age_donut(), 0.0, 0.0

    tmp = sub[["age_code", EMP_COL]].copy()
    tmp = tmp.dropna(subset=["age_code"])
    tmp["age_code"] = tmp["age_code"].astype(str)
    tmp["bucket"] = tmp["age_code"].apply(_age_bucket)
    tmp = tmp[tmp["bucket"].isin(["active", "youth"])]

    if tmp.empty:
        return _empty_age_donut(), 0.0, 0.0

    grp = (
        tmp.groupby(["age_code", "bucket"], as_index=False)[EMP_COL]
        .sum()
        .sort_values(["bucket", "age_code"])
    )

    labels = grp["age_code"].tolist()
    values = grp[EMP_COL].astype(float).tolist()

    buckets = grp["bucket"].tolist()
    active_indices = [i for i, b in enumerate(buckets) if b == "active"]
    youth_indices = [i for i, b in enumerate(buckets) if b == "youth"]

    colors = ["#003662"] * len(labels)
    for i in youth_indices:
        colors[i] = "#72b7b2"
    active_shades = _blue_shades(len(active_indices))
    for idx, i in enumerate(active_indices):
        colors[i] = active_shades[idx]

    active_total = float(grp.loc[grp["bucket"] == "active", EMP_COL].sum())
    youth_total = float(grp.loc[grp["bucket"] == "youth", EMP_COL].sum())

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.65,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(width=0)),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig, active_total, youth_total


# ========= 5. HOTSPOT POINTS (TOP 3 REGIONS PER COUNTRY) ====================

COLOR_EMP = [0, 54, 98, 220]    # deep blue
COLOR_UNE = [228, 87, 86, 220]  # red
COLOR_SUP = [245, 133, 24, 220] # orange


def _hotspot_points(sub: pd.DataFrame, top_per_country: int = 3):
    """
    For each country in the filtered data, pick the TOP `top_per_country` NUTS2
    regions based on score = supply + unemployment.
    """
    if sub.empty:
        return []

    cols = [c for c in [EMP_COL, UNE_COL, SUP_COL] if c and c in sub.columns]
    if not cols:
        return []

    agg = sub[["region_code"] + cols].groupby("region_code", as_index=False).sum()

    score = np.zeros(len(agg), dtype=float)
    if SUP_COL and SUP_COL in agg.columns:
        score = score + agg[SUP_COL].fillna(0.0).to_numpy(float)
    if UNE_COL and UNE_COL in agg.columns:
        score = score + agg[UNE_COL].fillna(0.0).to_numpy(float)
    agg["score"] = score

    code2cntr = _code_to_cntr_nuts2()
    code2name = _code_to_name_nuts2()
    code2centroid = _code_to_centroid_nuts2()

    agg["country"] = agg["region_code"].astype(str).map(
        lambda c: code2cntr.get(norm_code(c), "")
    )
    agg["region_name"] = agg["region_code"].astype(str).map(
        lambda c: code2name.get(norm_code(c), c)
    )

    max_score = float(agg["score"].max()) if np.isfinite(agg["score"].max()) else 1.0
    if max_score <= 0:
        max_score = 1.0

    points = []

    for cntr in sorted(agg["country"].dropna().unique()):
        g = agg[agg["country"] == cntr].sort_values("score", ascending=False).head(top_per_country)
        if g.empty:
            continue

        for _, row in g.iterrows():
            code = str(row["region_code"])
            c_norm = norm_code(code)
            pos = code2centroid.get(c_norm)
            if not pos:
                continue

            emp_val = float(row.get(EMP_COL, 0.0)) if EMP_COL in row.index else 0.0
            une_val = float(row.get(UNE_COL, 0.0)) if UNE_COL in row.index else 0.0
            sup_val = float(row.get(SUP_COL, 0.0)) if SUP_COL in row.index else 0.0

            vals = {"emp": emp_val, "une": une_val, "sup": sup_val}
            dom = max(vals, key=lambda k: vals[k])

            if dom == "emp":
                color = COLOR_EMP
                dom_label = "Employment hotspot"
            elif dom == "une":
                color = COLOR_UNE
                dom_label = "Unemployment hotspot"
            else:
                color = COLOR_SUP
                dom_label = "Supply hotspot"

            radius = 5000.0 + 18000.0 * (float(row["score"]) / max_score)

            points.append(
                dict(
                    position=pos,
                    color=color,
                    radius=float(radius),
                    region_code=c_norm,
                    region_name=str(row["region_name"]),
                    country=str(row["country"]),
                    dominant_label=dom_label,
                    emp_val=emp_val,
                    une_val=une_val,
                    sup_val=sup_val,
                )
            )

    return points


# ========= 6. COLORING & TOOLTIP ENRICHMENT FOR POLYGONS ====================

def _enrich_geo_with_metrics(
    geo: dict,
    metrics_map: dict,
    color_field: str = "supply",
) -> dict:
    """Attach fillColor and tooltip fields to each feature."""
    g = dict(geo)
    features = g.get("features", [])
    names = _code_to_name_nuts2()

    palette = [
        [239, 243, 255],
        [198, 219, 239],
        [158, 202, 225],
        [107, 174, 214],
        [66, 146, 198],
        [33, 113, 181],
        [8, 81, 156],
    ]
    num_classes = len(palette)

    vals = []
    for v_dict in metrics_map.values():
        v = v_dict.get(color_field)
        if v is not None and not pd.isna(v):
            vals.append(float(v))
    vals = np.array(vals, dtype=float)

    if len(vals) >= 2:
        bins = np.quantile(vals, np.linspace(0, 1, num_classes + 1)).astype(float)
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + 1e-9
    else:
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)

    def _rgba_for(v):
        if v is None or pd.isna(v):
            return [230, 230, 230, 190]
        idx = np.searchsorted(bins, float(v), side="right") - 1
        idx = max(0, min(idx, num_classes - 1))
        r, g_, b = palette[idx]
        return [int(r), int(g_), int(b), 210]

    for ft in features:
        props = ft.setdefault("properties", {})
        code = norm_code(
            props.get("NUTS_ID")
            or props.get("nuts_id")
            or props.get("ID")
            or props.get("id")
        )
        metrics = metrics_map.get(code, {})
        val_color = metrics.get(color_field)

        emp_val = metrics.get("emp")
        une_val = metrics.get("une")
        sup_val = metrics.get("supply")

        def fmt(v):
            return "N/A" if v is None or pd.isna(v) else f"{v:,.0f}"

        ft["region_name"] = names.get(code, code or "")
        ft["emp_disp"] = fmt(emp_val)
        ft["une_disp"] = fmt(une_val)
        ft["supply_disp"] = fmt(sup_val)

        props["fillColor"] = _rgba_for(val_color)

    g["features"] = features
    return g


# ========= 7. DECK.GL SPEC (CHOROPLETH + HOTSPOT DOTS) ======================

def _initial_view_state():
    return dict(
        latitude=54.0,
        longitude=15.0,
        zoom=3.35,
        minZoom=2.0,
        maxZoom=10.5,
        pitch=0,
        bearing=0,
    )


def _deck_spec(geojson_enriched: dict, points: list):
    geo_layer = {
        "@@type": "GeoJsonLayer",
        "id": "nuts-layer",
        "data": geojson_enriched,
        "pickable": True,
        "stroked": True,
        "filled": True,
        "autoHighlight": True,
        "opacity": 0.97,
        "getFillColor": "@@=properties.fillColor",
        "getLineColor": [160, 160, 160],
        "lineWidthMinPixels": 0.75,
    }
    hotspot_layer = {
        "@@type": "ScatterplotLayer",
        "id": "hotspot-layer",
        "data": points,
        "pickable": True,
        "radiusUnits": "meters",
        "getPosition": "@@=position",
        "getRadius": "@@=radius",
        "getFillColor": "@@=color",
        "getLineColor": [255, 255, 255],
        "lineWidthMinPixels": 1,
        "opacity": 0.9,
    }
    layers = [geo_layer]
    if points:
        layers.append(hotspot_layer)

    return {
        "initialViewState": _initial_view_state(),
        "layers": layers,
        "controller": True,
        "mapStyle": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    }


# ========= 8. UI LAYOUT (FILTERS + CARDS + MAP) =============================

def _filter_bar():
    DD_STYLE = {"zIndex": 2000}
    sex_options = [
        {
            "label": "Female (F)" if s == "F" else "Male (M)" if s == "M" else s,
            "value": s,
        }
        for s in SEX_ALL
    ]

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Year", className="text-muted small"),
                            dcc.Dropdown(
                                id="map-year-dd",
                                options=[{"label": str(y), "value": int(y)} for y in YEARS_ALL],
                                value=DEFAULT_YEAR,
                                clearable=False,
                                style=DD_STYLE,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Sex", className="text-muted small"),
                            dcc.Dropdown(
                                id="map-sex-dd",
                                options=sex_options,
                                value=SEX_ALL[0] if SEX_ALL else None,
                                clearable=False,
                                style=DD_STYLE,
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Age Groups", className="text-muted small"),
                            dcc.Dropdown(
                                id="map-agecode-dd",
                                options=[{"label": a, "value": a} for a in AGE_CODES_ALL],
                                value=AGE_CODES_ALL[0] if AGE_CODES_ALL else None,
                                clearable=False,
                                style=DD_STYLE,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Location", className="text-muted small"),
                            dcc.Dropdown(
                                id="map-location-dd",
                                options=_location_options(),
                                value=None,
                                multi=True,
                                placeholder="All regions",
                                clearable=True,
                                style=DD_STYLE,
                            ),
                        ],
                        md=5,
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
        className="my-3 card-lite",
    )


def _metric_pill(icon_class, title, value_id, bg_color):
    return dbc.Card(
        dbc.CardBody(
            html.Div(
                [
                    html.Div(
                        html.I(className=f"bi {icon_class}", style={"color": "#ffffff", "fontSize": "16px"}),
                        style={
                            "width": "36px",
                            "height": "36px",
                            "borderRadius": "12px",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "marginRight": "8px",
                            "background": bg_color,
                        },
                    ),
                    html.Div(
                        [
                            html.Div(title, className="text-muted small"),
                            html.Div(
                                "—",
                                id=value_id,
                                style={"fontWeight": 600, "fontSize": "16px"},
                            ),
                        ]
                    ),
                ],
                className="d-flex align-items-center",
            )
        ),
        style={
            "borderRadius": "16px",
            "border": "none",
            "boxShadow": "0 2px 10px rgba(0,0,0,.04)",
            "backgroundColor": "#ffffff",
        },
        className="mb-2",
    )


def _summary_column():
    return html.Div(
        [
            _metric_pill("bi-people-fill", "Total employment", "map-emp-total",
                         "linear-gradient(135deg,#4e79a7,#003662)"),
            _metric_pill("bi-graph-up", "Total supply", "map-supply-total",
                         "linear-gradient(135deg,#f6b26b,#f58518)"),
            _metric_pill("bi-exclamation-triangle-fill", "Total unemployment",
                         "map-une-total",
                         "linear-gradient(135deg,#ff8c8c,#e45756)"),
        ]
    )


def _age_donut_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Age structure", className="text-muted small mb-1"),
                dcc.Graph(
                    id="map-age-donut",
                    figure=_empty_age_donut(),
                    config={"displayModeBar": False},
                    style={"height": "170px"},
                ),
                html.Div(
                    [
                        html.Span("Active 25–65: ", className="text-muted small me-1"),
                        html.Span("—", id="map-active-count", className="fw-semibold me-3"),
                        html.Span("Age 15–24: ", className="text-muted small me-1"),
                        html.Span("—", id="map-youth-count", className="fw-semibold"),
                    ],
                    className="mt-1",
                ),
            ]
        ),
        style={
            "borderRadius": "16px",
            "border": "none",
            "boxShadow": "0 2px 10px rgba(0,0,0,.04)",
            "backgroundColor": "#ffffff",
        },
    )


def _empty_hotspot_donut():
    fig = go.Figure()
    fig.update_layout(
        height=170,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(
                text="Hover a hotspot",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=11, color="#6c757d"),
            )
        ],
    )
    return fig


def _hotspot_donut_figure(emp_val, une_val):
    total = emp_val + une_val
    if total <= 0:
        return _empty_hotspot_donut()

    fig = go.Figure(
        go.Pie(
            labels=["Employment", "Unemployment"],
            values=[emp_val, une_val],
            hole=0.6,
            sort=False,
            direction="clockwise",
            marker=dict(colors=["#003662", "#e45756"], line=dict(width=0)),
            textinfo="label+percent",
        )
    )
    fig.update_layout(
        height=170,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _region_donut_card():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Hotspot breakdown", className="text-muted small mb-1"),
                dcc.Graph(
                    id="map-hotspot-donut",
                    figure=_empty_hotspot_donut(),
                    config={"displayModeBar": False},
                    style={"height": "170px"},
                ),
                html.Div(
                    "Hover a hotspot region",
                    id="map-hotspot-donut-title",
                    className="fw-semibold small mt-1",
                ),
            ]
        ),
        style={
            "borderRadius": "16px",
            "border": "none",
            "boxShadow": "0 2px 10px rgba(0,0,0,.04)",
            "backgroundColor": "#ffffff",
        },
    )


def _map_card():
    return dbc.Card(
        dbc.CardBody(
            html.Div(
                id="deck-wrap",
                className="map-frame",
                style={
                    "height": "68vh",
                    "borderRadius": "16px",
                    "overflow": "hidden",
                    "border": "1px solid #e6e9ef",
                },
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
    return [
        _filter_bar(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        _summary_column(),
                        html.Div(style={"height": "8px"}),
                        dbc.Row(
                            [
                                dbc.Col(_age_donut_card(), md=6, xs=12),
                                dbc.Col(_region_donut_card(), md=6, xs=12),
                            ],
                            className="g-2",
                        ),
                    ],
                    lg=4,
                    md=5,
                    className="mb-3",
                ),
                dbc.Col(
                    _map_card(),
                    lg=8,
                    md=7,
                    className="mb-3",
                ),
            ],
            className="g-3",
        ),
        html.Div(style={"height": "8px"}),
    ]


# ========= 9. CALLBACKS =====================================================

def register_callbacks(app):
    @app.callback(
        Output("deck-wrap", "children"),
        Output("map-emp-total", "children"),
        Output("map-supply-total", "children"),
        Output("map-une-total", "children"),
        Output("map-age-donut", "figure"),
        Output("map-active-count", "children"),
        Output("map-youth-count", "children"),
        Input("map-year-dd", "value"),
        Input("map-sex-dd", "value"),
        Input("map-agecode-dd", "value"),
        Input("map-location-dd", "value"),
    )
    def update_map(year, sex_code, age_code, locations):
        if year is None:
            return (
                html.Div("No year selected.", className="text-danger"),
                "—",
                "—",
                "—",
                _empty_age_donut(),
                "—",
                "—",
            )

        geo_for_level = _geo_for_level(REGION_LEVEL_FIXED)
        if not geo_for_level.get("features"):
            return (
                html.Div("No GeoJSON for NUTS2 regions.", className="text-danger"),
                "—",
                "—",
                "—",
                _empty_age_donut(),
                "—",
                "—",
            )

        if not locations:
            locations_key = "ALL"
        else:
            if isinstance(locations, str):
                locations = [locations]
            locations_key = "|".join(sorted(map(str, locations)))

        base = _filtered_base(year, sex_code, age_code, locations)

        emp_tot, une_tot, sup_tot = _summary_totals(base)

        def fmt(v):
            return "—" if v is None else f"{v:,.0f}"

        emp_txt = fmt(emp_tot)
        une_txt = fmt(une_tot)
        sup_txt = fmt(sup_tot)

        # Age donut: ignore age filter to show full 15–65 structure
        pie_base = _filtered_base(year, sex_code, None, locations)
        age_fig, active_total, youth_total = _age_structure_donut(pie_base)
        active_txt = f"{active_total:,.0f}"
        youth_txt = f"{youth_total:,.0f}"

        metrics_map = _metrics_by_region(
            int(year),
            str(sex_code) if sex_code is not None else None,
            str(age_code) if age_code is not None else None,
            locations_key,
        )

        gj = _enrich_geo_with_metrics(geo_for_level, metrics_map, color_field="supply")

        points = _hotspot_points(base, top_per_country=3)

        spec = _deck_spec(gj, points)

        deck_component = dash_deck.DeckGL(
            spec,
            id="deck-gl",
            mapboxKey=MAPBOX_TOKEN,
            enableEvents=["hover"],
            tooltip={
                "html": (
                    "<b>{region_name}</b>"
                    "<br/>Employed: {emp_disp}"
                    "<br/>Unemployed: {une_disp}"
                    "<br/>Supply: {supply_disp}"
                ),
                "style": {
                    "backgroundColor": "rgba(0,0,0,.78)",
                    "color": "#fff",
                    "padding": "6px 8px",
                    "borderRadius": "6px",
                    "fontSize": "12px",
                },
            },
            style={"width": "100%", "height": "100%"},
        )

        return (
            deck_component,
            emp_txt,
            sup_txt,
            une_txt,
            age_fig,
            active_txt,
            youth_txt,
        )

    @app.callback(
        Output("map-hotspot-donut", "figure"),
        Output("map-hotspot-donut-title", "children"),
        Input("deck-gl", "hoverInfo"),
        prevent_initial_call=False,
    )
    def _update_hotspot_donut(hover_info):
        if not hover_info or not hover_info.get("object"):
            return _empty_hotspot_donut(), "Hover a hotspot region"

        obj = hover_info["object"]

        # Only react to hotspot points (they carry emp_val / une_val)
        if "emp_val" not in obj or "une_val" not in obj:
            return _empty_hotspot_donut(), "Hover a hotspot region"

        region_name = obj.get("region_name", "")
        country = obj.get("country", "")
        emp_val = float(obj.get("emp_val", 0.0))
        une_val = float(obj.get("une_val", 0.0))

        title = f"{region_name} ({country})"
        fig = _hotspot_donut_figure(emp_val, une_val)
        return fig, title
