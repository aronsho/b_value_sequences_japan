# sbatch --array=0-4 --mem-per-cpu=8000 --wrap="python 2d_map_low.py"

# ========= IMPORTS =========
import time as time_module
import os
from pathlib import Path

import numpy as np
import pandas as pd

from seismostats.analysis import estimate_mc_maxc, BPositiveBValueEstimator
from functions.main_functions import find_sequences, load_catalog
from functions.space_map import mac_space

# ======== get slurm ID ========
job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", job_index, "type", type(job_index))
t = time_module.time()

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_DIR = Path("results/map")

N_REALIZATIONS = 500
NS = np.array([200, 400, 800, 1600, 3200])  # number of tiles

# Map job_index to N
N = NS[job_index]
print(f"Parameters: spatial N={N}")
if job_index > len(NS) * N_REALIZATIONS - 1:
    raise ValueError("SLURM_ARRAY_TASK_ID is too large")

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

CAT_DIR = Path(variables["CAT_DIR"])

# transformation to local variables
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
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DIMENSION = variables["DIMENSION"]
BUFFER_M = variables["BUFFER_M"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
EXCLUDE_AFTERSHOCK_DAYS = variables["EXCLUDE_AFTERSHOCK_DAYS"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= HELPERS =========


def estimate_mc(magnitudes, delta_m):
    """Maximum curvature as Mc method."""
    mc, _ = estimate_mc_maxc(magnitudes, delta_m, CORRECTION_FACTOR)
    return mc


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
    # note that post_include_aftershocks means that the sequenecs are found as
    # usual, but aftershock events are added back to the sequences in the end.
    # The reason for this is that we want to remove the sequence including all
    # aftershocks from the map.
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
    # Here, the sequences are removed from the catalog, so that the map
    # we estimate is not influenced by the sequences.
    df_large_close = cat_close.loc[main_idx]
    all_sequences = pd.concat(seqs + [df_large_close])
    filtered_df = cat_close.drop(all_sequences.index, errors="ignore")
    filtered_df.mc = min(filtered_df.magnitude)

    # coordinates and limits
    grid = np.array(
        [df_large_close["x"], df_large_close["y"], -df_large_close["z"]])
    coords = [filtered_df.x.values,
              filtered_df.y.values, -filtered_df.z.values]
    limits = [
        [coords[0].min(), coords[0].max()],
        [coords[1].min(), coords[1].max()],
        [coords[2].min(), coords[2].max()],
    ]

    # global volume and length scales (once)
    volume = (
        (limits[0][1] - limits[0][0])
        * (limits[1][1] - limits[1][0])
        * (limits[2][1] - limits[2][0])
    )
    length_scale = (volume / N * 3 / (4 * np.pi)) ** (1 / 3)

    # --- Run mac_space for each grid size ---
    print('Estimating maps...')
    b_avg, b_std, mac_spatial, mu_mac_spatial, std_mac_spatial = mac_space(
        coords=coords,
        mags=filtered_df.magnitude,
        delta_m=filtered_df.delta_m,
        mc=filtered_df.mc,
        times=filtered_df.time,
        limits=limits,
        n_space=N,
        n_realizations=N_REALIZATIONS,
        eval_coords=grid,
        min_num=MIN_N_M,
        method=BPositiveBValueEstimator,
        mc_method=estimate_mc,
        transform=True,
        voronoi_method="random",
        dmc=DMC,
    )

    b_df = pd.DataFrame({"b_val": b_avg, "b_std": b_std})
    tmp_file = RESULT_DIR / f"b_values_N{N}.csv"
    b_df.to_csv(tmp_file, index=False)
    print(f"N = {N}: Saved realization {job_index} to {tmp_file}")

    df = pd.DataFrame({
        "n_space": [N],
        "mc": [filtered_df.mc],
        "volume": [volume],
        "length_scale": [length_scale],
        "mac_spatial": [mac_spatial],
        "mu_mac_spatial": [mu_mac_spatial],
        "std_mac_spatial": [std_mac_spatial],
    })

    out_file = RESULT_DIR / f"macs_volume_lengthscale_N{N}.csv"
    df.to_csv(out_file, index=False)

print("time = ", time_module.time() - t)
print('sbatch --array=0-4 --mem-per-cpu=8000 --wrap="python 2d_map_low.py"')
