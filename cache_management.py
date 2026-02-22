import os
import pickle
from pathlib import Path
import pandas as pd
import geopandas as gpd

# Note: pyarrow is equired for feather and parquet  --> pip install pyarrow

class CacheError(Exception):
    """Raised when a cache operation fails."""

# Configuration handled within the module
CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str, extension: str = "pkl") -> str:
    """
    Generate a safe cache file path with a specific extension.

    Args:
        key: Cache key identifier

    Returns:
        Path to cache file with .pkl extension
    """
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.{extension}")


def cache_get(key: str):
    """
    Retrieve a cached object, checking for feather files first.

    Args:
        key: Cache key identifier

    Returns:
        Cached object if found, None otherwise

    Raises:
        CacheError: If cache read operation fails
    """
    try:
        # 1. Try to load as a Feather (GeoDataFrame/DataFrame) first
        feather_path = _cache_path(key, "feather")
        if os.path.exists(feather_path):
            # Check if it's a GeoDataFrame (has geometry) or regular DataFrame
            df = pd.read_feather(feather_path)
            if "geometry" in df.columns:
                return gpd.read_feather(feather_path)
            return df

        # 2. Fallback to Pickle for other objects
        pickle_path = _cache_path(key, "pkl")
        if not os.path.exists(pickle_path):
            return None
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    """
    Store an object using Feather for DataFrames or Pickle for others.

    Args:
        key: Cache key identifier
        value: Object to cache (must be picklable)

    Raises:
        CacheError: If cache write operation fails
    """
    try:
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Use .to_feather() if the object is a (Geo)DataFrame
        if isinstance(value, (pd.DataFrame, gpd.GeoDataFrame)):
            path = _cache_path(key, "feather")
            value.to_feather(path)
        # 2. Use Pickle for everything else (like MultiDiGraph)
        else:
            path = _cache_path(key, "pkl")
            with open(path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)    
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e