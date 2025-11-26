# python 2c_parameter_variation_agg.py

import pandas as pd
from pathlib import Path
import itertools as it
from functions.main_functions import test_hypothesis

# ======== SPECIFY PARAMETERS ========
RESULT_DIR = Path("results/parameter_variation")

PARAMS = {
    "magnitude_threshold": [5.5, 6.0, 6.5],
    "b_method": ["global", "local"],
    "rupture_relation": ["surface", "subsurface"],
    "days_after": [50, 100, 200],
    "distance_to_coast": [30, 40, 50],
    "dimension": [2, 3],
    "exclude_aftershocks_day": [0, 1, 2],
    "depth_threshold": [50, 100, 150],
}

# Generate parameter combinations as dicts
param_combinations = [
    dict(zip(PARAMS.keys(), values))
    for values in it.product(*PARAMS.values())
]

# Define result metrics we expect from test_hypothesis
METRICS = [
    "mean_diff_beforeafter", "p_beforeafter",
    "corr_beforeafter", "p_corr_beforeafter",
    "slope_beforeafter", "intercept_beforeafter",
    "n_beforeafter",
    "mean_diff_close_beforeafter", "p_close_beforeafter",
    "corr_close_beforeafter", "p_corr_close_beforeafter",
    "slope_close_beforeafter", "intercept_close_beforeafter",
    "n_close_beforeafter",
    "mean_diff_far_beforeafter", "p_far_beforeafter",
    "corr_far_beforeafter", "p_corr_far_beforeafter",
    "slope_far_beforeafter", "intercept_far_beforeafter",
    "n_far_beforeafter",
    "mean_diff_farclose", "p_farclose",
    "corr_farclose", "p_corr_farclose",
    "slope_farclose", "intercept_farclose",
    "n_farclose",
    "mean_diff_after_farclose", "p_after_farclose",
    "corr_after_farclose", "p_corr_after_farclose",
    "slope_after_farclose", "intercept_after_farclose",
    "n_after_farclose",
    "mean_diff_before_farclose", "p_before_farclose",
    "corr_before_farclose", "p_corr_before_farclose",
    "slope_before_farclose", "intercept_before_farclose",
    "n_before_farclose",
    "mean_diff_before_before", "p_before_before",
    "corr_before_before", "p_corr_before_before",
    "slope_before_before", "intercept_before_before",
    "n_before_before",
    "mean_diff_beforebefore_close", "p_beforebefore_close",
    "corr_beforebefore_close", "p_corr_beforebefore_close",
    "slope_beforebefore_close", "intercept_beforebefore_close",
    "n_beforebefore_close",
]

# Collect results here
all_results = []

# ======== MAIN LOOP ========
for params in param_combinations:
    filename = RESULT_DIR / (
        f"df_b_values_{params['magnitude_threshold']}M_{params['b_method']}_"
        f"{params['rupture_relation']}_{params['days_after']}days_"
        f"{params['distance_to_coast']}km_{params['dimension']}D_"
        f"{params['exclude_aftershocks_day']}days_"
        f"{params['depth_threshold']}depth.csv"
    )

    if not filename.exists():
        print(f"File {filename} does not exist, skipping...")
        continue

    df = pd.read_csv(filename, index_col=0)

    # Perform hypothesis test
    results = test_hypothesis(df)

    # Merge params and results into one dict
    record = {**params, **{metric: results[metric] for metric in METRICS}}
    all_results.append(record)

# Convert all results at once to DataFrame
df_results = pd.DataFrame(all_results, columns=list(PARAMS.keys()) + METRICS)
df_results.to_csv(RESULT_DIR / "aggregated_results.csv", index=False)
