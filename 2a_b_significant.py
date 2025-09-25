# python 2a_b_significant.py

# ========= IMPORTS =========
import numpy as np
import pandas as pd
from pathlib import Path

from seismostats.analysis import (
    b_significant_1D,
    estimate_mc_maxc,
    BPositiveBValueEstimator,
)
from functions.main_functions import find_sequences, load_catalog

# ======== SPECIFY PARAMETERS ===
RESULT_DIR = Path("results/b_significant")

# multiple values
EXCLUDE_AFTERSHOCK_DAYS = [0, 1, 2]  # no of days after mainshock to exclude
N_MS = np.arange(150, 250, 50)  # no of magnitudes used for b-value estimation

# ======== LOAD PARAMETERS ======
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

SHAPE_DIR = Path(variables["SHAPE_DIR"])
CAT_DIR = Path(variables["CAT_DIR"])

# b-val estimation
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]

# sequences
DIMENSION = variables["DIMENSION"]
BUFFER_M = variables["BUFFER_M"]
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# evaluation
P_THRESHOLD = variables["P_THRESHOLD"]

# ========= HELPERS =========


def get_histograms(
    seqs: list[pd.DataFrame],
    delta_m,
    correction_factor: float,
    dmc: float,
    main_idx: list[int],
    n_ms: np.ndarray,
    stai_cutoff: pd.Timedelta,
    sort_parameter: str,
) -> np.ndarray:
    """
    Compute p-values for each sequence, for different sample sizes (n_ms).
    """
    p = np.zeros((len(seqs), len(n_ms)))

    for ii, sequence in enumerate(seqs):
        # estimate Mc
        mc, _ = estimate_mc_maxc(
            sequence.magnitude,
            fmd_bin=0.1,
            correction_factor=correction_factor,
        )

        # sort sequence
        sequence = sequence.sort_values(sort_parameter)

        # main event
        main_event = cat_close.loc[main_idx[ii]]

        # remove aftershocks inside cutoff
        idx_after = sequence[sequence["time"] >
                             main_event["time"] + stai_cutoff].index
        idx_before = sequence[sequence["time"] < main_event["time"]].index
        sequence = sequence.loc[np.concatenate([idx_before, idx_after])]

        # cutoff below Mc
        mags = sequence.magnitude.values
        times = sequence.time.values
        idx = mags > mc - delta_m / 2
        mags, times = mags[idx], times[idx]

        # estimate p-values for different n_m
        for jj, n_m in enumerate(n_ms):
            if len(mags) < n_m * 10:
                p[ii, jj] = np.nan
                continue
            p_val, _, _, _ = b_significant_1D(
                mags, mc, delta_m, times, n_m, method=BPositiveBValueEstimator,
                dmc=dmc,
            )
            p[ii, jj] = p_val

    return p


def run_with_cutoff(stai_cutoff: pd.Timedelta) -> None:
    '''take care: uses global variables.'''
    seqs, main_idx, _ = find_sequences(
        cat_close,
        cat_400km,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=stai_cutoff,
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ,
    )

    # time-sorted histograms
    p = get_histograms(seqs,
                       DELTA_M,
                       CORRECTION_FACTOR,
                       DMC,
                       main_idx,
                       N_MS,
                       stai_cutoff,
                       sort_parameter="time")
    np.savetxt(
        RESULT_DIR / f"p_values_time_{stai_cutoff.days}dcutoff.csv",
        p,
        delimiter=",",
    )

    # space-sorted histograms
    p = get_histograms(seqs,
                       DELTA_M,
                       CORRECTION_FACTOR,
                       DMC,
                       main_idx,
                       N_MS,
                       stai_cutoff,
                       sort_parameter="distance_to_main")
    np.savetxt(
        RESULT_DIR / f"p_values_space_{stai_cutoff.days}dcutoff.csv",
        p,
        delimiter=",",
    )


# ========= MAIN =========
if __name__ == "__main__":
    # load catalogs
    fname_close = "df_japan_buffered_catalog_" + \
        str(int(BUFFER_M/1000))+"km_" + str(DIMENSION) + "D.csv"
    fname_far = "df_japan_buffered_catalog_400km_" + str(DIMENSION) + "D.csv"
    cat_close = load_catalog(fname_close, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)
    cat_400km = load_catalog(fname_far, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)

    # run for multiple aftershock cutoffs
    for exclude_aftershock_days in EXCLUDE_AFTERSHOCK_DAYS:
        run_with_cutoff(pd.Timedelta(days=exclude_aftershock_days))
