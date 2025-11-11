from pathlib import Path
import os
import dash_bootstrap_components as dbc

# Base paths (env vars override these if set)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SLM_DATA_DIR", PROJECT_ROOT / "data"))

PARQUET_PATH = Path(os.environ.get(
    "SLM_PARQUET",
    DATA_DIR / "eu_labor_force_small.parquet"
))

GEO_PATHS = {
    2: Path(os.environ.get("SLM_GEO_L0", DATA_DIR / "NUTS_RG_20M_2024_4326_LEVL_0.geojson")),
    3: Path(os.environ.get("SLM_GEO_L1", DATA_DIR / "NUTS_RG_20M_2024_4326_LEVL_1.geojson")),
    4: Path(os.environ.get("SLM_GEO_L2", DATA_DIR / "NUTS_RG_20M_2024_4326_LEVL_2.geojson")),
}

APP_TITLE = "Skilled Labor Monitor"
MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")
THEME = dbc.themes.FLATLY
PORT = int(os.environ.get("PORT", "8051"))
