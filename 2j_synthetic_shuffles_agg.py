# python 2j_synthetic_shuffles_agg.py

# ======== IMPORTS ===

import pandas as pd
from pathlib import Path

from functions.main_functions import test_hypothesis

# ======== SPECIFY PARAMETERS ===
RESULTS_TMP = Path("results/synthetics/shuffle_tmp")
RESULTS_DIR = Path("results/synthetics")

# ======== LOAD PARAMETERS ======

DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

# sequences
DIMENSION = variables["DIMENSION"]
DIST_COAST = int(variables["BUFFER_M"] / 1000)
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DAYS_AFTER = variables["DAYS_AFTER"]
EXCLUDE_AFTERSHOCK_DAYS = variables["EXCLUDE_AFTERSHOCK_DAYS"]

# ======== IMPORT DATA & RUN HYPOTHESIS TESTS ======
results_list = []
results_rd_list = []

for ii in range(500):
    filename = (
        RESULTS_TMP/"df_b_values_"
        f"{MAGNITUDE_THRESHOLD}M_global_{RUPTURE_RELATION}_"
        f"{DAYS_AFTER}days_{DIST_COAST}km_{DIMENSION}D_"
        f"{EXCLUDE_AFTERSHOCK_DAYS}days_{ii}.csv"
    )
    df = pd.read_csv(filename, index_col=0)
    res = test_hypothesis(df)
    print(res)
    res["iteration"] = ii   # keep track of which run it came from
    results_list.append(res)

# Combine results into a single DataFrame
all_results = pd.DataFrame(results_list)

# Save to CSV
outpath = RESULTS_DIR / "all_results.csv"
all_results.to_csv(outpath, index=False)
