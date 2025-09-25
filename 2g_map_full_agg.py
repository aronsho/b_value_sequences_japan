# python 2g_map_full_agg.py

# ========= IMPORTS =========
from pathlib import Path

import numpy as np
import pandas as pd

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_DIR = Path("results/map")
RESULT_TMP = Path("results/map/tmp_full")

N = 51200  # number of tiles

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

# ========= MAIN =========
if __name__ == "__main__":
    # aggregate results
    # Gather all realization files for this N
    files_for_N = list(RESULT_TMP.glob(f"b_values_N{N}_R*.csv"))
    if not files_for_N:
        raise ValueError(f"No b_values files found for N={N}")

    # Stack vectors and average
    stack = np.column_stack([pd.read_csv(f)["b_val"] for f in files_for_N])

    # Mask zeros (treat them as missing)
    stack_masked = np.where(stack == 0, np.nan, stack)

    # Compute mean/std ignoring NaNs
    b_mean = np.nanmean(stack_masked, axis=1)
    b_std = np.nanstd(stack_masked, axis=1, ddof=1)

    b_df = pd.DataFrame({
        "b_avg": b_mean,
        "b_std": b_std,
    })

    out_file = RESULT_DIR / f"b_values_N{N}_full.csv"
    b_df.to_csv(out_file, index=False)
    print(f"N = {N}: saved aggregated results to {out_file}")
