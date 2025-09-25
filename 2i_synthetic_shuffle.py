# sbatch --array=0-49 --mem-per-cpu=4000 --wrap=
# "python 2i_synthetic_shuffle.py"

# ========= IMPORTS =========
import os
import time as time_module
from pathlib import Path
import pandas as pd
import numpy as np

from functions.main_functions import (
    find_sequences, load_catalog, estimate_b_values)

# ======== get slurm ID ========
job_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print("running index:", job_index, "type", type(job_index))
t = time_module.time()

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_TMP = Path("results/synthetics/shuffle_tmp")
N_REALIZATIONS = 500

# each job_index should perform 10 realizations
realization_indices = range(job_index * 10, (job_index + 1) * 10)
if realization_indices[-1] >= N_REALIZATIONS:
    if job_index * 10 >= N_REALIZATIONS:
        raise ValueError("job_index too large, exceeds N_REALIZATIONS")
    realization_indices = range(job_index * 10, N_REALIZATIONS)

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

CAT_DIR = Path(variables["CAT_DIR"])

# b-val estimation
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]
MIN_N_M = variables["MIN_N_M"]
B_METHOD = variables["B_METHOD"]

# sequences
DIMENSION = variables["DIMENSION"]
DISTANCE_TO_COAST = variables["BUFFER_M"] // 1000  # in km
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
RADIUS_CLOSE = variables["RADIUS_CLOSE"]
EXCLUDE_AFTERSHOCK_DAYS = variables["EXCLUDE_AFTERSHOCK_DAYS"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= HELPERS ====


def shuffle_magnitudes_across_sequences(
        seqs: list[pd.DataFrame]) -> list[pd.DataFrame]:

    # 1) Pool magnitudes (preserve dtype)
    pooled = np.concatenate([s['magnitude'].to_numpy() for s in seqs])

    # 2) Shuffle once (global permutation)
    rng = np.random.default_rng()
    shuffled = rng.permutation(pooled)

    # 3) Re-split by original sequence lengths
    out = []
    start = 0
    for seq in seqs:
        n = len(seq)
        seq_new = seq.copy()
        # positional slice
        seq_new.loc[:, 'magnitude'] = shuffled[start:start+n]
        out.append(seq_new)
        start += n
    return out

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
        exclude_aftershocks=pd.Timedelta(days=EXCLUDE_AFTERSHOCK_DAYS),
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ
    )

    # shuffle magnitudes across sequences
    print("Shuffling magnitudes ...")
    for nn in realization_indices:
        seqs_new = shuffle_magnitudes_across_sequences(seqs)

        # estimate b-values
        print('Estimating b-values...')
        df_b = estimate_b_values(
            seqs_new,
            main_idx,
            cat_close,
            B_METHOD,
            delta_m=DELTA_M,
            dmc=DMC,
            correction_factor=0,
            radius_close=RADIUS_CLOSE,
            n_check=MIN_N_M
        )

        # save
        save_name = (
            f"df_b_values_{MAGNITUDE_THRESHOLD}M_{B_METHOD}_{RUPTURE_RELATION}"
            f"_{DAYS_AFTER}days_{DISTANCE_TO_COAST}km_{DIMENSION}D_"
            f"{EXCLUDE_AFTERSHOCK_DAYS}days_{nn}.csv"
        )
        out_path = RESULT_TMP / save_name
        df_b.to_csv(out_path)
        print(f"  Saved: {out_path}")

print("time = ", time_module.time() - t)
print('sbatch --array=0-49 --mem-per-cpu=4000 ' +
      '--wrap="python 2i_synthetic_shuffle.py"')
