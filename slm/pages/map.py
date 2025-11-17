# pages/map.py

from functools import lru_cache
import numpy as np
import pandas as pd

from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import dash_deck

from ..data.loader import load_df, load_geo
from ..data.geo import norm_code
from ..config import MAPBOX_TOKEN


# ========= 1. DATA & BASIC CONSTANTS ========================================

# main.supply_per_region – already wired to qa.duckdb via load_df()
df = load_df()

# GeoJSON shapes (backed by shapes.NUTS_RG_60M_2024_4326 / GEO_PATHS in your config)
GEO_ALL = load_geo()

# We only have/need NUTS2 in this file (region_level = 4)
REGION_LEVEL_FIXED = 4

# ---- helper: safe unique sorted ----
def _uniq_sorted(series):
    s = pd.Series(series).dropna().unique().tolist()
    try:
        s = sorted(map(int, s))
    except Exception:
        s = sorted(map(str, s))
    return s


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
    # Sort by label for a nice UX
    return sorted(opts, key=lambda x: x["label"])


# ========= 3. METRICS AGGREGATION (EMP / UNE / SUPPLY) ======================

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


@lru_cache(maxsize=512)
def _metrics_by_region(
    year: int,
    sex_code: str,
    age_code: str,
    locations_key: str,
):
    """
    Return a dict: {region_code_norm: {'emp':.., 'une':.., 'supply':..}}.

    - Filters to region_level = 4 (NUTS2).
    - Filters by year, sex_code, age_code.
    - Optional filter by subset of region codes (Location dropdown).
    """
    # Decode locations (string key for cache)
    if locations_key == "ALL":
        selected_locs = None
    else:
        selected_locs = locations_key.split("|")

    sub = df.copy()

    # Fix level at NUTS2
    sub = sub[sub["region_level"].astype(int) == REGION_LEVEL_FIXED]

    # Filters
    sub = sub[sub["year"].astype(int) == int(year)]
    if sex_code is not None:
        sub = sub[sub["sex_code"].astype(str) == str(sex_code)]
    if age_code is not None:
        sub = sub[sub["age_code"].astype(str) == str(age_code)]

    sub = _ensure_levels(sub)

    if selected_locs:
        sub = sub[sub["region_code"].astype(str).isin(selected_locs)]

    if sub.empty:
        return {}

    # Aggregate metrics at region_code level
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


# ========= 4. COLORING & TOOLTIP ENRICHMENT =================================

def _enrich_geo_with_metrics(
    geo: dict,
    metrics_map: dict,
    color_field: str = "supply",
) -> dict:
    """
    Attach fillColor and tooltip fields to each feature.

    - Color is based on the chosen metric (default: supply).
    - Tooltip shows region name and all available metrics (emp, une, supply).
    """
    g = dict(geo)  # shallow copy
    features = g.get("features", [])
    names = _code_to_name_nuts2()

    # Palette (light → dark blue)
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

    # Collect values for chosen color metric
    vals = []
    for v_dict in metrics_map.values():
        v = v_dict.get(color_field)
        if v is not None and not pd.isna(v):
            vals.append(float(v))
    vals = np.array(vals, dtype=float)

    if len(vals) >= 2:
        bins = np.quantile(vals, np.linspace(0, 1, num_classes + 1)).astype(float)
        # Ensure strictly increasing
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + 1e-9
    else:
        # Fallback dummy bins
        bins = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)

    def _rgba_for(v):
        if v is None or pd.isna(v):
            return [230, 230, 230, 190]  # light grey for missing
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

        # Nice formatted strings
        def fmt(v):
            return "N/A" if v is None or pd.isna(v) else f"{v:,.0f}"

        ft["region_name"] = names.get(code, code or "")
        ft["emp_disp"] = fmt(emp_val)
        ft["une_disp"] = fmt(une_val)
        ft["supply_disp"] = fmt(sup_val)

        props["fillColor"] = _rgba_for(val_color)

    g["features"] = features
    return g


# ========= 5. DECK.GL SPEC ==================================================

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


def _deck_spec(geojson_enriched: dict):
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
        "getLineColor": [160, 160, 160],
        "lineWidthMinPixels": 0.75,
    }
    return {
        "initialViewState": _initial_view_state(),
        "layers": [layer],
        "controller": True,
        "mapStyle": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    }


# ========= 6. UI LAYOUT (same look & feel) ==================================

def _filter_bar():
    DD_STYLE = {"zIndex": 2000}  # keep dropdown menus above cards
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
                                options=[
                                    {"label": str(y), "value": int(y)} for y in YEARS_ALL
                                ],
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
                                options=[
                                    {"label": a, "value": a} for a in AGE_CODES_ALL
                                ],
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
                                value=None,  # None = "All regions"
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


def _map_card():
    return dbc.Card(
        dbc.CardBody(
            html.Div(
                id="deck-wrap",
                className="map-frame",
                style={
                    "height": "70vh",
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
        _map_card(),
        html.Div(style={"height": "12px"}),
    ]


# ========= 7. CALLBACKS =====================================================

def register_callbacks(app):
    @app.callback(
        Output("deck-wrap", "children"),
        Input("map-year-dd", "value"),
        Input("map-sex-dd", "value"),
        Input("map-agecode-dd", "value"),
        Input("map-location-dd", "value"),
    )
    def update_map(year, sex_code, age_code, locations):
        # Safeguards
        if year is None:
            return html.Div("No year selected.", className="text-danger")

        geo_for_level = _geo_for_level(REGION_LEVEL_FIXED)
        if not geo_for_level.get("features"):
            return html.Div("No GeoJSON for NUTS2 regions.", className="text-danger")

        # Encode locations as a cacheable key
        if not locations:
            locations_key = "ALL"
        else:
            if isinstance(locations, str):
                locations = [locations]
            locations_key = "|".join(sorted(map(str, locations)))

        metrics_map = _metrics_by_region(
            int(year),
            str(sex_code) if sex_code is not None else None,
            str(age_code) if age_code is not None else None,
            locations_key,
        )

        # Enrich shapes with metrics – color by "supply" (from supply_per_region)
        gj = _enrich_geo_with_metrics(geo_for_level, metrics_map, color_field="supply")
        spec = _deck_spec(gj)

        # Deck.gl component with a compact, informative tooltip
        return dash_deck.DeckGL(
            spec,
            id="deck-gl",
            mapboxKey=MAPBOX_TOKEN,
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
