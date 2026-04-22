import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from seismostats import Catalog
from seismostats.analysis import (
    ClassicBValueEstimator,
    estimate_mc_maxc,
    BPositiveBValueEstimator)
from seismostats.analysis.b_significant import transform_n

from functions.general_functions import dist_to_ref


def load_catalog(
        fname: str, mc: float, delta_m: float, cat_dir: str) -> Catalog:
    """Load, filter, and return a Catalog from CSV."""
    df = pd.read_csv(cat_dir / fname, index_col=0)
    df["time"] = pd.to_datetime(df["time"], format="mixed")
    cat = Catalog(df)
    cat.delta_m = delta_m
    cat.mc = mc
    cat = cat[cat["magnitude"] >= mc]

    return cat


def rupture_length(magnitude: np.ndarray, relation: str) -> np.ndarray:
    """Return rupture length [km] using Wells & Coppersmith relations."""
    if relation == "surface":
        a, b = -3.22, 0.69
    elif relation == "subsurface":
        a, b = -2.44, 0.59
    else:
        raise ValueError(f"Unknown rupture relation: {relation}")
    return 10 ** (a + b * magnitude)


def distance_series(cat_like: pd.DataFrame,
                    main: pd.DataFrame,
                    dimension: int) -> np.ndarray:
    """Vectorized distance from every event in `cat_like` to `main` event."""
    if dimension == 3:
        return dist_to_ref(
            cat_like["x"], main["x"],
            cat_like["y"], main["y"],
            cat_like["z"], main["z"])
    if dimension == 2:
        # type: ignore[arg-type]
        return dist_to_ref(cat_like["x"], main["x"], cat_like["y"], main["y"])
    raise ValueError("DIMENSION must be 2 or 3.")


def find_all_sequences(cat_close: Catalog,
                       cat_far: Catalog,
                       magnitude_threshold: float,
                       relation: str,
                       days_after: pd.Timedelta,
                       days_before: pd.Timedelta,
                       exclude_aftershocks: pd.Timedelta,
                       dimension: int,
                       radius_far: float,
                       min_n_seq: int,
                       post_include_aftershocks: bool = False,
                       ) -> tuple[list[pd.DataFrame], list[int], Catalog]:
    """
    Identify large-event sequences in catalog.
    Returns (list of sequences, list of main event indices, updated cat_close).
    """

    # select large events
    large_far = cat_far[cat_far["magnitude"] >= magnitude_threshold].copy()
    mask_close = cat_close["magnitude"] >= magnitude_threshold
    large_close = cat_close[mask_close].copy()

    # add rupture lengths
    large_far["rupture_length"] = rupture_length(
        large_far["magnitude"].values, relation)
    rupt_len = rupture_length(large_close["magnitude"].values, relation)
    large_close["rupture_length"] = rupt_len
    cat_close.loc[mask_close, "rupture_length"] = rupt_len

    # find sequences
    sequences, main_indices = [], []
    for idx, main in large_close.iterrows():
        start = main["time"] - days_before
        stop = main["time"] + days_after

        # select nearby events
        dist_all = distance_series(cat_close, main, dimension)

        # mask for sequence
        mask = (dist_all <= radius_far * main["rupture_length"]) & (
            (cat_close["time"] > start) & (cat_close["time"] < stop)
        )
        seq = cat_close[mask].copy()
        seq["distance_to_main"] = dist_all[mask]

        # drop the main event and immediate aftershocks
        seq = seq.drop(idx)
        mask = (seq["time"] < main["time"]) | (
            seq["time"] > main["time"] + exclude_aftershocks)
        seq_loop = seq[mask]

        # only keep sequences with > min_n_seq events
        if len(seq_loop) > min_n_seq:
            if post_include_aftershocks:
                sequences.append(seq)
            else:
                sequences.append(seq_loop)
            main_indices.append(idx)

    return sequences, main_indices, cat_close


def find_sequences(cat_close: Catalog,
                   cat_far: Catalog,
                   magnitude_threshold: float,
                   relation: str,
                   days_after: pd.Timedelta,
                   days_before: pd.Timedelta,
                   exclude_aftershocks: pd.Timedelta,
                   dimension: int,
                   radius_far: float,
                   min_n_seq: int,
                   post_include_aftershocks: bool = False,
                   show_progress: bool = False,
                   ) -> tuple[list[pd.DataFrame], list[int], Catalog]:
    """
    Identify large-event sequences in catalog. Then, it removes events that
    overlap with other large events to avoid double-counting.
    """

    # select large events
    large_far = cat_far[cat_far["magnitude"] >= magnitude_threshold].copy()
    mask_close = cat_close["magnitude"] >= magnitude_threshold
    large_close = cat_close[mask_close].copy()

    # add rupture lengths
    large_far["rupture_length"] = rupture_length(
        large_far["magnitude"].values, relation)
    rupt_len = rupture_length(large_close["magnitude"].values, relation)
    large_close["rupture_length"] = rupt_len
    cat_close.loc[mask_close, "rupture_length"] = rupt_len

    # find sequences
    sequences, main_indices = [], []
    if show_progress:
        progress_bar = tqdm(total=len(large_close) *
                            len(large_far), desc="Progress")
    for idx, main in large_close.iterrows():
        # select nearby events
        dist_all = distance_series(cat_close, main, dimension)
        mask_main = (dist_all <= radius_far * main["rupture_length"]) & (
            cat_close["time"] > main["time"] - days_before) & (
            cat_close["time"] < main["time"] + days_after)

        # avoid overlap with other large events
        for jj in large_far.index:
            if jj == idx:
                continue
            if show_progress:
                progress_bar.update(1)
            other = large_far.loc[jj]
            dist = distance_series(other, main, dimension)

            # avoid overlap
            if dist < radius_far * (
                    main["rupture_length"] + other["rupture_length"]):
                dist_all_other = distance_series(cat_close, other, dimension)
                mask_other = (dist_all_other <= radius_far *
                              other["rupture_length"])
                if other["time"] > main["time"]:
                    mask_time = cat_close["time"] >= other["time"]
                elif other["time"] <= main["time"]:
                    mask_time = cat_close["time"] <= other["time"] + days_after
                mask_loop = mask_other & mask_time
                mask_main = mask_main & ~mask_loop

        seq = cat_close[mask_main].copy()
        seq["distance_to_main"] = dist_all[mask_main]

        # drop the main event and immediate aftershocks
        mask = (seq["time"] < main["time"]) | (
            seq["time"] > main["time"] + exclude_aftershocks)
        seq_loop = seq[mask]

        # only keep sequences with > min_n_seq events
        if len(seq_loop) > min_n_seq:
            if post_include_aftershocks:
                sequences.append(seq)
            else:
                sequences.append(seq_loop)
            main_indices.append(idx)

    return sequences, main_indices, cat_close


def estimate_b_values(
    sequences: list[pd.DataFrame],
    main_indices: list[int],
    cat: Catalog,
    b_method: str,
    delta_m: float,
    dmc: float,
    correction_factor: float,
    radius_close: float,
    n_check: int,
) -> pd.DataFrame:

    # Adjust required sample size
    n_check = n_check if b_method == "global" else 2 * n_check

    # Final column order
    subset_labels = [
        "sequence",
        "close_after", "close_before",
        "far_after", "far_before",
        "before", "after",
        "close", "far",
        "before1", "before2",
        "before1_close", "before2_close",
    ]
    metrics = ["b", "std", "p_l", "n"]
    columns = [f"{m}_{label}" for label in subset_labels for m in metrics]

    results = []

    def estimate(mags, times, estimator, mc, n_check):
        """Return (b, std, p_l, n) for a given subset."""
        if len(mags) < n_check:
            return np.nan, np.nan, np.nan, np.nan
        if b_method == "global":
            estimator.calculate(mags, mc=dmc, delta_m=delta_m)
        elif b_method == "local":
            estimator.calculate(
                mags, mc=mc, delta_m=delta_m, dmc=dmc, times=times
            )
        return (estimator.b_value,
                estimator.std,
                estimator.p_lilliefors(),
                estimator.n)

    # go through sequences
    for seq, idx in zip(sequences, main_indices):
        main = cat.loc[idx]

        # make sure that main event is not included
        seq = seq.drop(idx, errors="ignore")

        # Step 1. Estimate completeness magnitude (Mc)
        mc, _ = estimate_mc_maxc(
            seq.magnitude, fmd_bin=0.1, correction_factor=correction_factor
        )

        # Step 2. Prepare magnitudes, times, distances depending on method
        if b_method == "global":
            pre = BPositiveBValueEstimator()
            pre.calculate(seq.magnitude, mc=mc, delta_m=delta_m,
                          times=seq.time, dmc=dmc)
            mags, times = pre.magnitudes, pre.times
            distances = seq["distance_to_main"].values[pre.idx]
            estimator2 = ClassicBValueEstimator()
        elif b_method == "local":
            mask = seq["magnitude"].values >= mc
            mags = seq["magnitude"].values[mask]
            times = seq["time"].values[mask]
            distances = seq["distance_to_main"].values[mask]
            estimator2 = BPositiveBValueEstimator()

        # Step 3. Define masks for subsets
        close = distances <= (
            radius_close * main["rupture_length"]
        )
        far = ~close
        before = times < main["time"]
        after = ~before

        masks = {
            "sequence": np.ones(len(mags), dtype=bool),
            "close_after": close & after,
            "close_before": close & before,
            "far_after": far & after,
            "far_before": far & before,
            "before": before,
            "after": after,
            "close": close,
            "far": far,
        }

        # Step 4. Estimate values for each subset
        row = {}
        for label, mask in masks.items():
            b, s, p, n = estimate(mags[mask],
                                  times[mask],
                                  estimator2,
                                  mc,
                                  n_check)
            row[f"b_{label}"] = b
            row[f"std_{label}"] = s
            row[f"p_l_{label}"] = p
            row[f"n_{label}"] = n

        # Step 5. Split "before" events into two halves
        m_before, t_before = mags[before], times[before]
        half = len(m_before) // 2
        for part, sl in zip(
                ["1", "2"], [slice(None, half), slice(half, None)]):
            b, s, p, n = estimate(m_before[sl],
                                  t_before[sl],
                                  estimator2,
                                  mc,
                                  n_check)
            row[f"b_before{part}"] = b
            row[f"std_before{part}"] = s
            row[f"p_l_before{part}"] = p
            row[f"n_before{part}"] = n

        # Step 6. Split "close & before" events into two halves
        m_cb, t_cb = mags[close & before], times[close & before]
        half_cb = len(m_cb) // 2
        for part, sl in zip(
                ["1", "2"], [slice(None, half_cb), slice(half_cb, None)]):
            b, s, p, n = estimate(m_cb[sl],
                                  t_cb[sl],
                                  estimator2,
                                  mc,
                                  n_check)
            row[f"b_before{part}_close"] = b
            row[f"std_before{part}_close"] = s
            row[f"p_l_before{part}_close"] = p
            row[f"n_before{part}_close"] = n

        results.append(row)

    # Step 7. Build final DataFrame with fixed column order
    return pd.DataFrame(results, index=main_indices, columns=columns)


def test_hypothesis(df: pd.DataFrame) -> pd.DataFrame:
    # Define all tests: (name_prefix, series1, series2, p1, p2)
    tests = [
        ("beforeafter",
         df["b_after"].values, df["b_before"].values,
         df["n_after"].values, df["n_before"].values),
        ("close_beforeafter",
         df["b_close_after"].values, df["b_close_before"].values,
         df["n_close_after"].values, df["n_close_before"].values),
        ("far_beforeafter",
         df["b_far_after"].values, df["b_far_before"].values,
         df["n_far_after"].values, df["n_far_before"].values),
        ("farclose",
         df["b_far"].values, df["b_close"].values,
         df["n_far"].values, df["n_close"].values),
        ("after_farclose",
         df["b_far_after"].values, df["b_close_after"].values,
         df["n_far_after"].values, df["n_close_after"].values),
        ("before_farclose",
         df["b_far_before"].values, df["b_close_before"].values,
         df["n_far_before"].values, df["n_close_before"].values),
        ("before_before",
         df["b_before1"].values, df["b_before2"].values,
         df["n_before1"].values, df["n_before2"].values),
        ("beforebefore_close",
         df["b_before1_close"].values, df["b_before2_close"].values,
         df["n_before1_close"].values, df["n_before2_close"].values),
    ]

    results = {}
    for name, b1, b2, n1, n2 in tests:
        # transform b-values such that they can be compared even if n differs
        b_sequence = df["b_sequence"].values
        n_sequence = df["n_sequence"].values
        mask = (~np.isnan(b_sequence)) & (~np.isnan(b1)) & (~np.isnan(b2))
        b1_t = transform_n(b1[mask], b_sequence[mask],
                           n1[mask], n_sequence[mask])
        b2_t = transform_n(b2[mask], b_sequence[mask],
                           n2[mask], n_sequence[mask])

        # pairwise differences
        diff_t = b1_t - b2_t
        diff = b1[mask] - b2[mask]

        # perform t-test with transformed data
        if diff_t.size == 0:
            mean_diff, p_val, corr, p_corr = np.nan, np.nan, np.nan, np.nan
        else:
            _, p_val = stats.ttest_1samp(
                diff_t, popmean=0, alternative='greater')
            mean_diff = np.nanmean(diff)
            corr, p_corr = stats.pearsonr(b1[mask], b2[mask])
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                b1[mask], b2[mask])

        results[f"mean_diff_{name}"] = mean_diff
        results[f"p_{name}"] = p_val
        results[f"corr_{name}"] = corr
        results[f"p_corr_{name}"] = p_corr
        results[f"slope_{name}"] = slope
        results[f"intercept_{name}"] = intercept
        results[f"n_{name}"] = np.sum(mask)

    return results
