import numpy as np
import inspect

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
        coords,
        mags,
        delta_m,
        mc,
        times,
        limits,
        n_space,
        n_realizations,
        eval_coords=None,
        min_num=10,
        method: BValueEstimator | AValueEstimator = ClassicBValueEstimator,
        mc_method=None,
        transform=True,
        scaling_factor=1,
        voronoi_method='random',
        **kwargs,
):
    """
    This function estimates the mean autocorrelation for the D-dimensional
    case (tested for 2 and 3 dimensions). Additionally, it provides the mean
    a- and b-values for each grid-point. The partitioning method is based on
    voronoi tesselation (random area).

    Args:
        coords:     Coordinates of the earthquakes. It should have the
                structure [x1, ... , xD], where xi are vectors of the same
                length (number of events)
        mags:       Magnitudes of the earthquakes
        delta_m:    Magnitude bin width
        mc:         Completeness magnitude
        times:      Times of the earthquakes
        limits:     Limits of the area of interest. It should be a list with
                the minimum and maximum values of each variable.
                [[x1min, x1max], ..., [xDmin, xDmax]]
                The limits should be such that all the coordinates are within
                the limits.
        n_space:    Number of voronoi nodes
        n_m_time:   Number of events in each time window
        n_realizations: Number of realizations of random voronoi tesselation
                for the estimation of the mean values
        n_skip_time:    Number of time steps to skip (if n_skip_time=1, all
            possible partitions in time are considered)
        eval_coords: Coordinates of the grid-points where the mean a- and
                b-values are estimated. It should have the structure
                [x1, ..., xD], where xi are vectors of the same length (number
                od points where the mean a- and b-values are estimated)
        min_num:    Minimum number of events to estimate a- and b-values in
                each tile
        method:   Method to estimate b-values. Options are "positive" and
                "classic"
        mc_method: Method to estimate the completeness magnitude. needs to be
                a function that takes the magnitudes and delta_m as input and
                returns the  mc (only!)
        transform:  If True, the b-values are transformed according to the
                number of events uesd to estimate them (such that their
                distribution is IID under the null hypothesis of unchanging
                b-value)
        voronoi_method: Method to partition the area. Options are 'random' and
                'location'
        **kwargs:   additional keyword arguments for b-value(a-value) method

    """

    # convert all to np.dnarrays
    mags = np.array(mags)
    times = np.array(times)
    coords = np.array(coords)
    limits = np.array(limits)
    if eval_coords is not None:
        eval_coords = np.array(eval_coords)
    else:
        # in this case, the values are estimated at the earthquake locations
        eval_coords = coords

    # 0. preparation (2 and 3D possible)
    dim = len(coords[:, 0])

    # 1. some data checks
    if len(mags) != len(coords[0, :]):
        raise ValueError("The number of magnitudes and coordinates do not "
                         "match")
    if len(times) != len(coords[0, :]):
        raise ValueError("The number of times and coordinates do not match")
    if len(limits) != dim:
        raise ValueError("The number of limits and dimensions do not match")
    if len(eval_coords[:, 0]) != dim:
        raise ValueError("The number of evaluation coordinates and dimensions "
                         "do not match")
    if min(mags) < mc - delta_m/2:
        raise ValueError("The completeness magnitude is larger than the "
                         "smallest magnitude. please filter")
    for ii in range(dim):
        if min(eval_coords[ii, :]) < limits[ii][0] or max(
                eval_coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The evaluation coordinates are outside the limits")
        if min(coords[ii, :]) < limits[ii][0] or max(
                coords[ii, :]) > limits[ii][1]:
            raise ValueError(
                "The earthquake coordinates are outside the limits")

    # estimate overall b-value
    if mc_method is not None:
        mc = mc_method(mags, delta_m)

    estimator = method()
    sig = inspect.signature(estimator.calculate)
    if 'times' in sig.parameters:
        b_all = estimator.calculate(
            mags, mc=mc, delta_m=delta_m, times=times, **kwargs)
    else:
        b_all = estimator.calculate(mags, mc=mc, delta_m=delta_m, **kwargs)

    # define all the matrices aggregates are for the standard deviation
    # estimation
    x_average = np.zeros(len(eval_coords[0, :]))
    aggregate = (np.zeros(len(eval_coords[0, :])), np.zeros(
        len(eval_coords[0, :])), np.zeros(len(eval_coords[0, :])))
    aggregate_std = (np.zeros(len(eval_coords[0, :])), np.zeros(
        len(eval_coords[0, :])), np.zeros(len(eval_coords[0, :])))

    ac_spatial = np.zeros(n_realizations)
    n_p_spatial = np.zeros(n_realizations)
    n_spatial = np.zeros(n_realizations)

    for ii in range(n_realizations):
        if voronoi_method == 'random':
            # create voronoi nodes (randomly distributed within the
            # limits)
            coords_vor = np.random.rand(dim, n_space)
            for jj in range(dim):
                coords_vor[jj, :] = limits[jj][0] + (
                    limits[jj][1] - limits[jj][0]) * coords_vor[jj, :]
        if voronoi_method == 'location':
            # choose random coordinates of the earthquakes as voronoi nodes
            idx = np.random.choice(len(coords[0, :]), n_space)
            coords_vor = coords[:, idx]
        vor = mirror_voronoi(coords_vor, limits)

        # create spatial neighbors matrix
        w_space = neighbors_vor(vor, n_space)
        # find maggnitudes and times corresponding to the voronoi nodes
        tile_magnitudes, tile_times = find_points_in_tile(
            coords_vor, coords, mags, times)
        if mc_method is not None:
            mc = np.zeros(len(tile_magnitudes))
            for kk, tiles in enumerate(tile_magnitudes):
                if len(tiles) > 2 * min_num:
                    mc[kk] = mc_method(tiles, delta_m)

        # estimate a- or b-values
        if issubclass(method, AValueEstimator):
            volume_space = volumes_vor(vor, n_space)
            for jj in range(dim):
                volume_space /= (limits[jj][1] - limits[jj][0])
            volume_space *= scaling_factor
            x_vec, _, n_m = values_from_partitioning(
                tile_magnitudes,
                tile_times,
                mc,
                delta_m,
                method=method,
                list_scaling=volume_space,
                **kwargs)
        if issubclass(method, BValueEstimator):
            x_vec, std_vec, n_m = values_from_partitioning(
                tile_magnitudes,
                tile_times,
                mc,
                delta_m,
                method=method,
                **kwargs)
        x_vec[n_m < min_num] = np.nan

        # 2.7 average the b-values or a-values
        # find the nearest voronoi node for each grid-point
        nearest = find_nearest_vor_node(coords_vor, eval_coords)
        x_loop = x_vec[nearest]
        std_loop = std_vec[nearest]

        # use welford algorithm to estimate the standard deviation
        aggregate = update_welford(aggregate, x_loop)
        aggregate_std = update_welford(aggregate_std, std_loop)

        # estimate Morans I (spatial autocorrelation)
        if transform:
            x_vec_t = transform_n(x_vec, b_all, n_m, max(n_m))
        ac_spatial[ii], n_spatial[ii], n_p_spatial[ii] = est_morans_i(
            x_vec_t, w_space)

    # 3. estimate the averages & estimate expected standard deviation of MAC
    x_average, var_sample = finalize_welford(aggregate)
    std_method, _ = finalize_welford(aggregate_std)
    # std is the maximum of the mean of the shi and bold standard deviations
    # and the std of the sample
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
