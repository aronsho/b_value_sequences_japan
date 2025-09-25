# python 2c_parameter_variation_agg.py

import pandas as pd
from pathlib import Path
import itertools as it

from functions.main_functions import test_hypothesis

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

# ======== MAIN ========

df_results = pd.DataFrame(columns=["magnitude_threshold",
                                   "b_method",
                                   "rupture_relation",
                                   "days_after",
                                   "distance_to_coast",
                                   "dimension",
                                   "exclude_aftershocks_day",
                                   'mean_diff_beforeafter',
                                   'p_beforeafter',
                                   'mean_diff_close_beforeafter',
                                   'p_close_beforeafter',
                                   'mean_diff_far_beforeafter',
                                   'p_far_beforeafter',
                                   'mean_diff_farclose',
                                   'p_farclose',
                                   'mean_diff_after_farclose',
                                   'p_after_farclose',
                                   'mean_diff_before_farclose',
                                   'p_before_farclose',
                                   'mean_diff_before_before',
                                   'p_before_before',
                                   'mean_diff_beforebefore_close',
                                   'p_beforebefore_close'
                                   ])

for ii, params in enumerate(param_combinations):
    (magnitude_threshold,
     b_method,
     rupture_relation,
     days_after,
     distance_to_coast,
     dimension,
     exclude_aftershocks_day) = params
    filename = (
        RESULT_DIR /
        f"df_b_values_{magnitude_threshold}M_{b_method}_{rupture_relation}_"
        f"{days_after}days_{distance_to_coast}km_{dimension}D"
        f"_{exclude_aftershocks_day}days.csv"
    )
    if not filename.exists():
        print(f"File {filename} does not exist, skipping...")
        continue
    df = pd.read_csv(filename, index_col=0)

    # perform hypothesis test
    results = test_hypothesis(df)

    df_results.loc[ii] = [
        magnitude_threshold,
        b_method,
        rupture_relation,
        days_after,
        distance_to_coast,
        dimension,
        exclude_aftershocks_day,
        results['mean_diff_beforeafter'],
        results['p_beforeafter'],
        results['mean_diff_close_beforeafter'],
        results['p_close_beforeafter'],
        results['mean_diff_far_beforeafter'],
        results['p_far_beforeafter'],
        results['mean_diff_farclose'],
        results['p_farclose'],
        results['mean_diff_after_farclose'],
        results['p_after_farclose'],
        results['mean_diff_before_farclose'],
        results['p_before_farclose'],
        results['mean_diff_before_before'],
        results['p_before_before'],
        results['mean_diff_beforebefore_close'],
        results['p_beforebefore_close']
    ]

df_results.to_csv(RESULT_DIR / "aggregated_results.csv", index=False)
