# effis_loader.py
import zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

COUNTRY_CANDIDATES = ["iso2", "ISO2", "COUNTRY", "country", "Country", "CNTR", "cntr_code"]
DATE_CANDIDATES = ["FIREDATE", "firedate", "FIRE_DATE", "initialdate", "INITIALDATE", "DATE"]
AREA_CANDIDATES = ["AREA_HA", "area_ha", "Area_HA", "AREA", "area"]
ID_CANDIDATES = ["id", "ID", "Id", "OBJECTID", "fid"]

def _find_col(gdf, candidates):
    lower = {c.lower(): c for c in gdf.columns}
    return next((lower[c.lower()] for c in candidates if c.lower() in lower), None)

def _read_vector(path):
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            members = zf.namelist()
        vector = next((m for m in members
                       if m.lower().endswith((".geojson", ".json", ".shp", ".gpkg"))), None)
        if vector is None:
            raise ValueError(f"No GeoJSON/shapefile inside {path}: {members}")
        return gpd.read_file(f"zip://{p}!{vector}")
    return gpd.read_file(path)

def load_effis(path, countries=None, start_date=None, end_date=None, min_area_ha=0.0):
    gdf = _read_vector(path)
    gdf = (gdf.set_crs(4326) if gdf.crs is None else gdf).to_crs(4326)
    c_country, c_date = _find_col(gdf, COUNTRY_CANDIDATES), _find_col(gdf, DATE_CANDIDATES)
    c_area, c_id = _find_col(gdf, AREA_CANDIDATES), _find_col(gdf, ID_CANDIDATES)
    if c_date is None:
        raise ValueError(f"No fire-date column found in {list(gdf.columns)}")
    out = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
    out["country"] = gdf[c_country].astype(str).str.strip().str.upper() if c_country else "NA"
    dates = pd.to_datetime(gdf[c_date], errors="coerce", format="mixed")
    if dates.isna().mean() > 0.5:
        dates = pd.to_datetime(gdf[c_date], errors="coerce", dayfirst=True)
    out["fire_date"] = dates.dt.tz_localize(None) if getattr(dates.dt, "tz", None) is not None else dates
    out["area_ha"] = pd.to_numeric(gdf[c_area], errors="coerce") if c_area else np.nan
    out["fire_id"] = (gdf[c_id].astype(str) if c_id
                      else pd.Series(range(len(gdf)), index=gdf.index).astype(str))
    out["fire_id"] = "effis_" + out["fire_id"].str.replace(r"[^\w\-]", "_", regex=True)
    out = out[out.geometry.notna() & out["fire_date"].notna()]
    if countries:  out = out[out["country"].isin({c.upper() for c in countries})]
    if start_date: out = out[out["fire_date"] >= pd.Timestamp(start_date)]
    if end_date:   out = out[out["fire_date"] <= pd.Timestamp(end_date)]
    if min_area_ha: out = out[out["area_ha"].fillna(0) >= min_area_ha]
    return out.sort_values("area_ha", ascending=False).reset_index(drop=True)