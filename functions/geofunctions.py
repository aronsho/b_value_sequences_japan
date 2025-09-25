# ========= IMPORTS =========
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np


def load_japan_polygon(SHAPE_DIR) -> gpd.GeoDataFrame:
    """
    Build a single polygon (GeoDataFrame) for mainland Japan intersected
    with an approximate CSV polygon mask.
    """
    # Mainland Japan (exclude Okinawa), then dissolve to one geometry
    gdf = gpd.read_file(SHAPE_DIR / "jp.shp")
    gdf = gdf[gdf["name"] != "Okinawa"]
    # union_all for modern GeoPandas; fallback to unary_union if needed
    combined = gdf.geometry.union_all()
    combined_gdf = gpd.GeoDataFrame(geometry=[combined], crs=gdf.crs)

    # CSV polygon mask
    df_lonlat = pd.read_csv(SHAPE_DIR / "polygon_approx.csv")
    polygon_approx = Polygon(zip(df_lonlat["lon"], df_lonlat["lat"]))
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon_approx], crs=gdf.crs)

    # Intersection
    return gpd.overlay(combined_gdf, polygon_gdf, how="intersection")


def buffered_polygon_vertices_latlon(
    poly_gdf: gpd.GeoDataFrame,
    buffer_m: float,
    epsg_src: int,
    epsg_metric: int,
) -> np.ndarray:
    """
    Buffer a polygon (meters), return exterior vertices as [[lat, lon], ...].
    """
    # project to metric CRS, buffer, then back
    poly_metric = poly_gdf.to_crs(epsg=epsg_metric)
    buffered = poly_metric.buffer(buffer_m).to_crs(epsg=epsg_src)

    # Convert exterior coords to [[lat, lon], ...]
    exterior = np.asarray(list(buffered.geometry.iloc[0].exterior.coords))
    # exterior is [[lon, lat], ...] -> swap to [[lat, lon], ...]
    return exterior[:, [1, 0]]
