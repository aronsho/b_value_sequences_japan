# sbatch --array=0-7 --mem-per-cpu=8000 --wrap="python 1_prepare_catalogs.py"

# ========= IMPORTS =========
import time as time_module
import os
import numpy as np
from pathlib import Path
import pandas as pd
import itertools as it

from seismostats import Catalog
from seismostats.utils import CoordinateTransformer, cat_intersect_polygon

# own functions
from functions.transformation_functions import transform_and_rotate
from functions.geofunctions import (
    load_japan_polygon,
    buffered_polygon_vertices_latlon)

# ======== get slurm ID ========
job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", job_index, "type", type(job_index))
t = time_module.time()

# ======== SPECIFY PARAMETERS ===
DIMENSIONS = [2, 3]                      # dimensionality considered
BUFFER_MS = [30_000, 40_000, 50_000, 400_000]     # distance to coast in m

param_grid = it.product(
    DIMENSIONS,
    BUFFER_MS,
)
param_combinations = list(param_grid)
print(f"{len(param_combinations)} parameter combinations found.")
(DIMENSION, BUFFER_M) = param_combinations[job_index]
print(f"Parameters: DIMENSION={DIMENSION}, BUFFER_M={BUFFER_M}")

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

SHAPE_DIR = Path(variables["SHAPE_DIR"])
CAT_DIR = Path(variables["CAT_DIR"])
p1 = np.array(eval(variables["p1"]))
p2 = np.array(eval(variables["p2"]))
EPSG_GEOGRAPHIC = variables["EPSG_GEOGRAPHIC"]
EPSG_JAPAN_M = variables["EPSG_JAPAN_M"]
DEPTH_THRESHOLD = variables["DEPTH_THRESHOLD"]
TIME_START = variables["TIME_START"]

# ======== HELPERS ===============


def load_clean_catalog(csv_path: Path) -> Catalog:
    # get earthquake catalog
    df = pd.read_csv(csv_path, parse_dates=["time"])
    # convert time to datetime (mixed formats)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    # exclude rows with missing magnitude
    df = df[~df["magnitude"].isna()]
    # only include event_type='earthquake'
    df = df[df["event_type"] == "earthquake"]
    # drop rows with missing depth
    df = df[~df["depth"].isna()]
    # create catalog object
    cat = Catalog(df)
    # set binning width
    cat.bin_magnitudes(0.1, inplace=True)
    # only consider events from TIME_START onwards
    cat = cat[cat["time"] >= TIME_START]
    # only consider events with depth <= DEPTH_THRESHOLD km
    cat = cat[cat["depth"] <= DEPTH_THRESHOLD]
    return cat


def transform_catalog_3d(cat: Catalog, p1, p2) -> Catalog:
    """
    In-place: compute local (x,y,z) via custom transform_and_rotate around two
    reference points near Kyoto. No distortion here, as no projection is used.
    """
    lats = cat["latitude"].values
    lons = cat["longitude"].values
    depths = cat["depth"].values
    cart_coords, _ = transform_and_rotate(p1, p2, lats, lons, depths)
    cat["y"] = cart_coords[0, :]
    cat["x"] = cart_coords[1, :]
    cat["z"] = cart_coords[2, :]
    return cat


def transform_catalog_2d(cat: Catalog) -> Catalog:
    """
    In-place: compute local 2D (x,y) with CoordinateTransformer using
    reference at (lon=135.769601, lat=35.018970). This uses a projection
    that introduces some distortion. It is used for the 2D case only.
    """
    ct_ref = CoordinateTransformer(EPSG_GEOGRAPHIC)
    ref_easting, ref_northing, _ = ct_ref.to_local_coords(
        135.769601, 35.018970, 0)
    ct = CoordinateTransformer(EPSG_GEOGRAPHIC, ref_easting, ref_northing)
    x, y, z = ct.to_local_coords(
        cat["longitude"].values,
        cat["latitude"].values,
        cat["depth"].values,
    )
    cat["x"], cat["y"], cat["z"] = x * 100, y * 100, z
    return cat


def save_catalog(
        cat: Catalog,
        buffer_m: float,
        dim: int, out_dir: Path) -> Path:
    fname = f"df_japan_buffered_catalog_{int(buffer_m/1000)}km_{dim}D.csv"
    out_path = out_dir / fname
    cat.to_csv(out_path)
    return out_path


# ========= MAIN =========
if __name__ == "__main__":
    pass
    # 1) Catalog
    print('Load catalog...')
    cat = load_clean_catalog(CAT_DIR / "all_JMA.csv")

    # 2) Geo mask (Japan ∩ approx polygon), buffer, intersect
    print('Buffer catalog...')
    japan_poly = load_japan_polygon(SHAPE_DIR)
    verts_latlon = buffered_polygon_vertices_latlon(
        japan_poly,
        buffer_m=BUFFER_M,
        epsg_src=EPSG_GEOGRAPHIC,
        epsg_metric=EPSG_JAPAN_M,
    )
    cat = cat_intersect_polygon(cat, verts_latlon)

    # 3) Transform coordinates
    print('Transform catalog...')
    if DIMENSION == 2:
        cat = transform_catalog_2d(cat)
    elif DIMENSION == 3:
        cat = transform_catalog_3d(cat, p1, p2)
    else:
        raise ValueError("DIMENSION must be 2 or 3.")

    # 4) Save
    out_path = save_catalog(cat, BUFFER_M, DIMENSION, CAT_DIR)
    print(f"Saved: {out_path}")

print("time = ", time_module.time() - t)
print(
    'sbatch --array=0-7' +
    ' --mem-per-cpu=8000 --wrap="python 1_prepare_catalogs.py"')
