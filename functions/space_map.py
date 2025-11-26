import numpy as np
import inspect
import warnings
from typing import Callable

from seismostats.analysis.bvalue import (
    ClassicBValueEstimator, BValueEstimator)
from seismostats.analysis.avalue import AValueEstimator
from functions.general_functions import (
    transform_n, update_welford, finalize_welford)
from functions.space_functions import (
    mirror_voronoi,
    neighbors_vor,
    find_nearest_vor_node,
    find_points_in_tile,
    volumes_vor,
)
from seismostats.analysis.b_significant import (
    values_from_partitioning, est_morans_i)


def mac_space(
        coords: np.ndarray,
        mags: np.ndarray,
        delta_m: float,
        times: np.ndarray,
        limits: np.ndarray,
        n_space: int,
        n_realizations: int,
        eval_coords: np.ndarray | None = None,
        min_num: int = 10,
        method: BValueEstimator | AValueEstimator = ClassicBValueEstimator,
        mc: float | None = None,
        mc_method: Callable | None = None,
        transform: bool = True,
        scaling_factor: float = 1.0,
        voronoi_method: str = 'random',
        min_count: int = 1,
        ** kwargs,
):
    """
    This function estimates the mean autocorrelation (Moran's I) for the
    D-dimensional case (tested for 2 and 3 dimensions). Additionally, it
    provides the mean a- and b-values for each grid-point. The partitioning
    method is based on voronoi tessellation.

    Source:
        - Mirwald et al. (2024), How to b-significant when analyzing b-value
            variations

    Args:
        coords:         Coordinates of the earthquakes. It should have the
            structure [x1, ... , xD], where xi are vectors of the same length
            (N = number of events)
        mags:           Magnitudes of the earthquakes (length N)
        delta_m:        Magnitude bin width
        times:          Times of the earthquakes (length N)
        limits:         Limits of the area of interest. It should be a list
            with the minimum and maximum values of each variable.
            [[x1min, x1max], ..., [xDmin, xDmax]] (D = number of dimensions).
            The limits should be such that all the coordinates are within
            the limits.
        n_space:        Number of voronoi nodes
        n_realizations: Number of realizations of random voronoi tessellation
            for the estimation of the mean values
        eval_coords:    Coordinates of the grid-points where the mean a- and
            b-values are estimated. It should have the structure
            [x1, ..., xD], where xi are vectors length M (M = number of
            grid-points).
        min_num:        Minimum number of events to estimate a- and b-values in
            each tile
        method:         Method to estimate b-values. Options are "positive" and
            "classic"
        mc:             Completeness magnitude. If mc_method is also provided,
            max(mc, mc_method) is used.
        mc_method:      Method to estimate the completeness magnitude. needs
            to be a function that takes the magnitudes as input and returns
            the  mc
        transform:      If True, the b-values are transformed according to the
            number of events used to estimate them (such that their
            distribution is IID under the null hypothesis of unchanging
            b-value). IF a-values are estimated, transform is set to False.
        scaling_factor: Scaling factor for the a-value estimation
        voronoi_method: Method to partition the area. Options are 'random'
            (random area) and 'location' (based on density of events). Default
            is 'random'.
        min_count:     Minimum number of realizations required to estimate
            the mean and standard deviation.
        **kwargs:       additional keyword arguments for b-value(a-value)
            method

    """

    # 1. preparation
    # convert all to np.dnarrays
    mags = np.asarray(mags)
    times = np.asarray(times)
    coords = np.asarray(coords)
    limits = np.asarray(limits)
    eval_coords = np.asarray(
        eval_coords) if eval_coords is not None else coords

    # dimensions (2 and 3D possible)
    dim, n_events = coords.shape
    n_eval = len(eval_coords[0, :])
    x_is_a = issubclass(method, AValueEstimator)
    x_is_b = issubclass(method, BValueEstimator)

    # some data checks
    if len(mags) != n_events:
        raise ValueError("The number of magnitudes and coordinates do not "
                         "match")
    if len(times) != n_events:
        raise ValueError("The number of times and coordinates do not match")
    if len(limits) != dim:
        raise ValueError("The number of limits and dimensions do not match")
    if len(eval_coords[:, 0]) != dim:
        raise ValueError("The number of evaluation coordinates and dimensions "
                         "do not match")

    for ii in range(dim):
        if np.min(eval_coords[ii, :]) < limits[ii][0] or np.max(
                eval_coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The evaluation coordinates are outside the limits")
        if np.min(coords[ii, :]) < limits[ii][0] or np.max(
                coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The earthquake coordinates are outside the limits")

    # estimate overall mc
    if mc is not None:
        if mc_method is not None:
            warnings.warn(
                "Both mc and mc_method are provided."
                "In this case, max(mc, mc_method) is used for local estimates")
            mc_min = mc
        elif mc_method is None:
            mc_local = mc
    elif mc is None and mc_method is None:
        raise ValueError(
            "Either mc or mc_method should be provided")

    # estimate overall b-value for transformation if needed
    if x_is_a:
        transform = False
    elif x_is_b and transform:
        estimator = method()
        sig = inspect.signature(estimator.calculate)
        if 'times' in sig.parameters:
            b_all = estimator.calculate(
                mags, mc=mc, delta_m=delta_m, times=times, **kwargs)
        else:
            b_all = estimator.calculate(mags, mc=mc, delta_m=delta_m, **kwargs)

    # initiate the aggregates
    aggregate = (np.zeros(n_eval), np.zeros(n_eval), np.zeros(n_eval))
    aggregate_std = (np.zeros(n_eval), np.zeros(n_eval), np.zeros(n_eval))

    # initiate arrays for autocorrelation
    ac_spatial = np.zeros(n_realizations)
    n_p_spatial = np.zeros(n_realizations)
    n_spatial = np.zeros(n_realizations)

    # 2. loop over realizations
    for ii in range(n_realizations):
        if voronoi_method == 'random':
            # create random voronoi nodes
            coords_vor = np.random.rand(dim, n_space)
            for jj in range(dim):
                coords_vor[jj, :] = limits[jj][0] + (
                    limits[jj][1] - limits[jj][0]) * coords_vor[jj, :]
        if voronoi_method == 'location':
            # choose random coordinates of EQs as voronoi nodes
            idx = np.random.choice(n_events, n_space)
            coords_vor = coords[:, idx]
        vor = mirror_voronoi(coords_vor, limits)

        # create spatial neighbors matrix
        w_space = neighbors_vor(vor, n_space)

        # find magnitudes and times corresponding to the voronoi nodes
        nearest_events = find_nearest_vor_node(coords_vor, coords)
        tile_magnitudes = find_points_in_tile(
            coords_vor, coords, mags, nearest=nearest_events)
        tile_times = find_points_in_tile(
            coords_vor, coords, times, nearest=nearest_events)
        if mc_method is not None:
            mc_local = np.zeros(n_space)
            for mm, tiles in enumerate(tile_magnitudes):
                if len(tiles) > 0:
                    mc_local[mm] = max(mc_min, mc_method(tiles))
                else:
                    mc_local[mm] = np.nan

        # estimate a- or b-values
        if x_is_a:
            volume_space = volumes_vor(vor, n_space)
            for mm in range(dim):
                volume_space /= (limits[mm][1] - limits[mm][0])
            volume_space *= scaling_factor
            x_vec, std_vec, n_m = values_from_partitioning(
                tile_magnitudes,
                tile_times,
                mc_local,
                delta_m,
                method=method,
                list_scaling=volume_space,
                **kwargs)
            std_vec[n_m > 0] = 0  # as a std is not yet implemented
        if x_is_b:
            x_vec, std_vec, n_m = values_from_partitioning(
                tile_magnitudes,
                tile_times,
                mc_local,
                delta_m,
                method=method,
                **kwargs)
        x_vec[n_m < min_num] = np.nan

        # find the nearest voronoi node for each grid-point
        nearest = find_nearest_vor_node(coords_vor, eval_coords)
        x_loop = x_vec[nearest]
        std_loop = std_vec[nearest]

        # use welford algorithm to estimate the standard deviation
        aggregate = update_welford(aggregate, x_loop)
        aggregate_std = update_welford(aggregate_std, std_loop)

        # estimate Morans I (spatial autocorrelation)
        if transform:
            x_vec = transform_n(x_vec, b_all, n_m, max(n_m))
        ac_spatial[ii], n_spatial[ii], n_p_spatial[ii] = est_morans_i(
            x_vec, w_space)

    # 3. estimate the averages & estimate expected standard deviation of MAC
    x_average, var_sample = finalize_welford(aggregate, min_count=min_count)
    std_method, _ = finalize_welford(aggregate_std, min_count=min_count)
    x_std = np.maximum(std_method, np.sqrt(var_sample))

    mac_spatial = np.mean(ac_spatial)
    mean_n_p_spatial = np.mean(n_p_spatial)
    mean_n_spatial = np.mean(n_spatial)
    mu_mac_spatial = -1/mean_n_spatial
    std_mac_spatial = np.sqrt(1/mean_n_p_spatial)

    return (
        x_average, x_std,
        mac_spatial, mu_mac_spatial, std_mac_spatial,
    )
