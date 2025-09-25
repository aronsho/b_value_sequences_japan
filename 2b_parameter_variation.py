# sbatch
# --array=0-647 --mem-per-cpu=4000 --wrap="python 2b_parameter_variation.py"

# ========= IMPORTS =========
import os
import time as time_module
import itertools as it
from pathlib import Path
import pandas as pd


from functions.main_functions import (
    find_sequences, load_catalog, estimate_b_values)

# ======== get slurm ID ========
job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", job_index, "type", type(job_index))
t = time_module.time()

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_DIR = Path("results/parameter_variation")

# multiple values
MAGNITUDE_THRESHOLDS = [5.5, 6.0, 6.5]
B_METHODS = ["global", "local"]
RUPTURE_RELATIONS = ["surface", "subsurface"]
DAYS_AFTERS = [50, 100, 200]
DISTANCE_TO_COASTS = [30, 40, 50]
DIMENSIONS = [2, 3]
EXCLUDE_AFTERSHOCKS_DAYS = [0, 1, 2]

param_grid = it.product(
    MAGNITUDE_THRESHOLDS,
    B_METHODS,
    RUPTURE_RELATIONS,
    DAYS_AFTERS,
    DISTANCE_TO_COASTS,
    DIMENSIONS,
    EXCLUDE_AFTERSHOCKS_DAYS,
)
param_combinations = list(param_grid)
print(f"{len(param_combinations)} parameter combinations found.")
(MAGNITUDE_THRESHOLD,
 B_METHOD,
 RUPTURE_RELATION,
 DAYS_AFTER,
 DISTANCE_TO_COAST,
 DIMENSION,
 EXCLUDE_AFTERSHOCKS_DAY) = param_combinations[job_index]
print(f"Parameters: "
      f"MAGNITUDE_THRESHOLD={MAGNITUDE_THRESHOLD}, B_METHOD={B_METHOD}, "
      f"RUPTURE_RELATION={RUPTURE_RELATION}, DAYS_AFTER={DAYS_AFTER}, "
      f"DISTANCE_TO_COAST={DISTANCE_TO_COAST}, DIMENSION={DIMENSION}, "
      f"EXCLUDE_AFTERSHOCKS_DAY={EXCLUDE_AFTERSHOCKS_DAY}")

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

CAT_DIR = Path(variables["CAT_DIR"])

# b-val estimation, catalog in general
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]
MIN_N_M = variables["MIN_N_M"]

# for sequeneces
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
RADIUS_CLOSE = variables["RADIUS_CLOSE"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= MAIN =========


if __name__ == "__main__":
    # load catalogs
    print('Loading catalogs...')
    fname_close = (f"df_japan_buffered_catalog_{DISTANCE_TO_COAST}km_"
                   f"{DIMENSION}D.csv")
    fname_far = f"df_japan_buffered_catalog_400km_{DIMENSION}D.csv"
    cat_close = load_catalog(
        fname_close, MC_FIXED - CORRECTION_FACTOR, DELTA_M, CAT_DIR)
    cat_far = load_catalog(fname_far, MC_FIXED -
                           CORRECTION_FACTOR, DELTA_M, CAT_DIR)

    # find sequences
    print('Finding sequences...')
    seqs, main_idx, cat_close = find_sequences(
        cat_close, cat_far,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=pd.Timedelta(days=EXCLUDE_AFTERSHOCKS_DAY),
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ
    )
    print(f"  {len(seqs)} sequences found.")

    # estimate b-values
    print('Estimating b-values...')
    df_b = estimate_b_values(
        seqs,
        main_idx,
        cat_close,
        B_METHOD,
        delta_m=DELTA_M,
        dmc=DMC,
        correction_factor=CORRECTION_FACTOR,
        radius_close=RADIUS_CLOSE,
        n_check=MIN_N_M
    )

    # save
    save_name = (
        f"df_b_values_{MAGNITUDE_THRESHOLD}M_{B_METHOD}_{RUPTURE_RELATION}_"
        f"{DAYS_AFTER}days_{DISTANCE_TO_COAST}km_{DIMENSION}D_"
        f"{EXCLUDE_AFTERSHOCKS_DAY}days.csv"
    )
    out_path = RESULT_DIR / save_name
    df_b.to_csv(out_path)
    print(f"  Saved: {out_path}")

print("time = ", time_module.time() - t)
print('sbatch --array=0-647 --mem-per-cpu=4000 ' +
      '--wrap="python 2b_parameter_variation.py"')
