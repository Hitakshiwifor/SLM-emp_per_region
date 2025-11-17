# slm/data/loader.py

from functools import lru_cache
from pathlib import Path
import pandas as pd
import duckdb

from ..config import DATA_DIR, PARQUET_PATH, GEO_PATHS
from .geo import load_all_geo

# Path to your DuckDB file (adjust name/path if needed)
DB_PATH = DATA_DIR / "qa.duckdb"

# Columns the rest of the app expects
REQUIRED = {
    "region_level",
    "region_code",
    "year",
    "sex_code",
    "sex_level",
    "age_code",
    "age_level",
    "emp",
}


@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    """
    Load the main fact data for the dashboard.

    1. Prefer DuckDB (qa.duckdb, main.supply_per_region)
       - aggregate to reduce size
    2. If qa.duckdb does not exist, fall back to the old parquet file.
    """

    # ---------- 1) Use DuckDB if available ----------
    if DB_PATH.exists():
        con = duckdb.connect(str(DB_PATH))
        try:
            # We aggregate by year, nuts2, sex2, age10 so the result is much smaller.
            sql = """
                SELECT
                    -- NUTS2 only -> region_level = 4
                    4::INT                                     AS region_level,
                    UPPER(CAST(nuts2 AS VARCHAR))              AS region_code,
                    CAST(year AS INT)                          AS year,

                    CAST(sex2 AS VARCHAR)                      AS sex_code,
                    '2'::VARCHAR                               AS sex_level,
                    CAST(age10 AS VARCHAR)                     AS age_code,
                    '2'::VARCHAR                               AS age_level,

                    SUM(CAST(emp    AS DOUBLE))                AS emp,
                    SUM(CAST(une    AS DOUBLE))                AS une,
                    SUM(CAST(supply AS DOUBLE))                AS supply
                FROM "main"."supply_per_region"
                WHERE nuts2 IS NOT NULL
                  AND year  IS NOT NULL
                GROUP BY
                    region_level,
                    region_code,
                    year,
                    sex_code,
                    sex_level,
                    age_code,
                    age_level
            """
            df = con.sql(sql).df()
        finally:
            con.close()

        # Ensure required columns exist
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"DuckDB SELECT did not produce required columns: {sorted(missing)}"
            )

        # --------- Memory optimisations / cleanup ----------
        df["region_code"] = (
            df["region_code"].astype(str).str.strip().str.upper()
        )
        df["sex_code"] = df["sex_code"].astype(str).str.strip()
        df["age_code"] = df["age_code"].astype(str).str.strip()

        # downcast numeric columns to smaller types where safe
        df["region_level"] = df["region_level"].astype("int8")
        df["year"] = df["year"].astype("int16")

        for col in ["emp", "une", "supply"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        # strings -> category to save memory
        for col in ["region_code", "sex_code", "sex_level", "age_code", "age_level"]:
            df[col] = df[col].astype("category")

        return df

    # ---------- 2) Fallback: parquet (dev / backup) ----------
    df = pd.read_parquet(PARQUET_PATH)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet missing required columns: {sorted(missing)}"
        )
    return df


@lru_cache(maxsize=1)
def load_geo() -> dict[int, dict]:
    """
    Load all GeoJSON levels defined in GEO_PATHS.
    """
    return load_all_geo(GEO_PATHS)
