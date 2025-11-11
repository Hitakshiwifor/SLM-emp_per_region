from functools import lru_cache
import pandas as pd
# import duckdb as ddb  # enable later if you like
from ..config import PARQUET_PATH, GEO_PATHS
from .geo import load_all_geo

@lru_cache(maxsize=1)
def load_df() -> pd.DataFrame:
    # Parquet for now; swap to DuckDB easily later
    df = pd.read_parquet(PARQUET_PATH)
    need = {"region_level","region_code","year","sex_code","age_code","age_level","emp"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Parquet missing columns: {sorted(miss)}")
    return df

@lru_cache(maxsize=1)
def load_geo() -> dict[int, dict]:
    return load_all_geo(GEO_PATHS)
