# python 2e_map_agg.py

# ========= IMPORTS =========
from pathlib import Path

import numpy as np
import pandas as pd

# ======== SPECIFY PARAMETERS ===
# single value
RESULT_DIR = Path("results/map")
RESULT_TMP = Path("results/map/tmp")

# mutliple values
NS = np.array([6400, 12800, 25600, 51200, 102400])  # number of tiles

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

# ========= MAIN =========
if __name__ == "__main__":
    # aggregate results
    for N in NS:
        # Gather all realization files for this N
        files_for_N = list(RESULT_TMP.glob(f"b_values_N{N}_R*.csv"))
        if not files_for_N:
            print(f"No b_values files found for N={N}, skipping.")
            continue

        # Stack vectors and average
        stack = np.column_stack([pd.read_csv(f)["b_val"] for f in files_for_N])
        b_mean = stack.mean(axis=1)
        b_std = stack.std(axis=1, ddof=1)

        b_df = pd.DataFrame({
            "b_avg": b_mean,
            "b_std": b_std,
        })

        out_file = RESULT_DIR / f"b_values_{N}.csv"
        b_df.to_csv(out_file, index=False)
        print(f"N = {N}: saved aggregated results to {out_file}")

        files_for_N = list(RESULT_TMP.glob(
            f"macs_volume_lengthscale_N{N}_R*.csv"))
        if not files_for_N:
            print(
                f"No macs_volume_lengthscale files found for N={N}, skipping.")
            continue
        dfs = [pd.read_csv(f) for f in files_for_N]
        combined = pd.concat(dfs, ignore_index=True)

        df = pd.DataFrame({
            "n_space": [combined["n_space"].mean()],
            "mc": [combined["mc"].mean()],
            "volume": [combined["volume"].mean()],
            "length_scale": [combined["length_scale"].mean()],
            "mac_spatial": [combined["mac_spatial"].mean()],
            "mu_mac_spatial": [combined["mu_mac_spatial"].mean()],
            "std_mac_spatial": [combined["std_mac_spatial"].mean()],
        })

        out_file = RESULT_DIR / f"macs_volume_lengthscale_N{N}.csv"
        df.to_csv(out_file, index=False)
        print(f"N = {N}: saved aggregated results to {out_file}")
