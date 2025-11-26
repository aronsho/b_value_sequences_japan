# sbatch
# --array=0-499 --time=480 --mem-per-cpu=128000 --wrap="python 2f_map_full.py"


# ========= IMPORTS =========
import os
import time as time_module
import numpy as np
import pandas as pd
from pathlib import Path
import itertools

from seismostats.analysis import estimate_mc_maxc, BPositiveBValueEstimator
from seismostats.utils import cat_intersect_polygon

from functions.main_functions import find_sequences, load_catalog
from functions.space_map import mac_space
from functions.transformation_functions import transform_and_rotate
from functions.geofunctions import (
    load_japan_polygon,
    buffered_polygon_vertices_latlon)

# ======== get slurm ID ========
job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", job_index, "type", type(job_index))
t = time_module.time()

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_TMP = Path("results/map/tmp_full")

N_REALIZATIONS = 500
N = 25600  # number of tiles
DELTA_XY = 0.2  # in deg
DELTA_Z = 0.2   # in deg
COMPUTE_GRID = True  # First time running, this has to be True!

if job_index > N_REALIZATIONS - 1:
    raise ValueError(
        "SLURM_ARRAY_TASK_ID too large for available realizations")
print(f"Parameters: N={N}, job_index={job_index}")

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

SHAPE_DIR = Path(variables["SHAPE_DIR"])
CAT_DIR = Path(variables["CAT_DIR"])

# transformation to local
p1 = np.array(eval(variables["p1"]))
p2 = np.array(eval(variables["p2"]))
EPSG_GEOGRAPHIC = variables["EPSG_GEOGRAPHIC"]
EPSG_JAPAN_M = variables["EPSG_JAPAN_M"]

# b-val estimation
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]
MIN_N_M = variables["MIN_N_M"]

# sequences
DIMENSION = variables["DIMENSION"]
BUFFER_M = variables["BUFFER_M"]
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
EXCLUDE_AFTERSHOCK_DAYS = variables["EXCLUDE_AFTERSHOCK_DAYS"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= HELPERS =========


def estimate_mc(magnitudes):
    """Maximum curvature as Mc method."""
    mc, _ = estimate_mc_maxc(magnitudes, DELTA_M, CORRECTION_FACTOR)
    return mc


def create_grid(filtered_df, buffer_m, delta_xy, delta_z):
    # make grid
    coords_latlon = [
        filtered_df.longitude.values,
        filtered_df.latitude.values,
        filtered_df.depth.values]
    limits_latlon = [
        [min(coords_latlon[0]), max(coords_latlon[0])],
        [min(coords_latlon[1]), max(coords_latlon[1])],
        [min(coords_latlon[2]), max(coords_latlon[2])]]
    lon_vec = np.arange(limits_latlon[0][0], limits_latlon[0][1], delta_xy)
    lat_vec = np.arange(limits_latlon[1][0], limits_latlon[1][1], delta_xy)
    depth_vec = np.arange(limits_latlon[2][1], limits_latlon[2][0], -delta_z)
    grid = np.array(list(itertools.product(lon_vec, lat_vec, depth_vec)))
    grid = grid.T
    df_grid = pd.DataFrame()
    df_grid['longitude'] = grid[0]
    df_grid['latitude'] = grid[1]
    df_grid['depth'] = grid[2]

    # only evaluate at points of interest
    japan_poly = load_japan_polygon(SHAPE_DIR)
    verts_latlon = buffered_polygon_vertices_latlon(
        japan_poly,
        buffer_m=buffer_m,
        epsg_src=EPSG_GEOGRAPHIC,
        epsg_metric=EPSG_JAPAN_M,
    )
    df_grid = cat_intersect_polygon(df_grid, verts_latlon)

    # transform coordinates to x, y, z
    cart_coords, _ = transform_and_rotate(
        p1, p2, df_grid.latitude, df_grid.longitude, df_grid.depth)
    df_grid['x'] = cart_coords[1, :]
    df_grid['y'] = cart_coords[0, :]
    df_grid['z'] = - cart_coords[2, :]
    return df_grid


# ========= MAIN =========
if __name__ == "__main__":
    # --- Load catalogs ---
    print('Loading catalogs...')
    fname_close = "df_japan_buffered_catalog_" + \
        str(int(BUFFER_M/1000))+"km_" + str(DIMENSION) + "D.csv"
    fname_far = "df_japan_buffered_catalog_400km_" + str(DIMENSION) + "D.csv"
    cat_close = load_catalog(fname_close, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)
    cat_far = load_catalog(fname_far, MC_FIXED -
                           CORRECTION_FACTOR, DELTA_M, CAT_DIR)

    # --- Find sequences ---
    print('Finding sequences...')
    seqs, main_idx, _ = find_sequences(
        cat_close,
        cat_far,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=pd.Timedelta(days=EXCLUDE_AFTERSHOCK_DAYS),
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ,
        post_include_aftershocks=True,
    )

    # --- Filter catalog ---
    df_large_close = cat_close.loc[main_idx]
    all_sequences = pd.concat(seqs + [df_large_close])
    filtered_df = cat_close.drop(all_sequences.index, errors="ignore")
    filtered_df.mc = min(filtered_df.magnitude)

    # evaluation coordinates
    print('Create grid...')
    if COMPUTE_GRID is True:
        df_grid = create_grid(filtered_df, BUFFER_M, DELTA_XY, DELTA_Z)
        # Save grid in DIR
        df_grid.to_csv(DIR / "df_grid.csv", index=False)
        grid = np.array([df_grid['x'], df_grid['y'], df_grid['z']])
    else:
        # Load grid from CSV
        df_grid = pd.read_csv(DIR / "df_grid.csv")
        grid = np.array([df_grid['x'], df_grid['y'], df_grid['z']])

    # coordinates and limits
    coords = [filtered_df.x.values,
              filtered_df.y.values,
              - filtered_df.z.values]
    limits = [
        [min(coords[0].min(), grid[0].min()),
         max(coords[0].max(), grid[0].max())],
        [min(coords[1].min(), grid[1].min()),
         max(coords[1].max(), grid[1].max())],
        [min(coords[2].min(), grid[2].min()),
         max(coords[2].max(), grid[2].max())]
    ]

    # --- Run mac_space for each grid size ---
    print('Estimating map...')
    b_avg, b_std, _, _, _ = mac_space(
        coords=coords,
        mags=filtered_df.magnitude,
        delta_m=DELTA_M,
        times=filtered_df.time,
        limits=limits,
        n_space=N,
        n_realizations=1,
        eval_coords=grid,
        min_num=MIN_N_M,
        method=BPositiveBValueEstimator,
        mc=filtered_df.mc,
        mc_method=estimate_mc,
        transform=True,
        voronoi_method="random",
        dmc=DMC,
    )

    # save b-value maps as a DataFrame
    b_df = pd.DataFrame({"b_val": b_avg})
    tmp_file = RESULT_TMP / f"b_values_N{N}_R{job_index+1}.csv"
    b_df.to_csv(tmp_file, index=False)
    print(f"N = {N}: Saved realization {job_index+1} to {tmp_file}")

print("time = ", time_module.time() - t)
print('sbatch --array=0-99 --time=480 ' +
      '--mem-per-cpu=128000 --wrap="python 2f_map_full.py"')
