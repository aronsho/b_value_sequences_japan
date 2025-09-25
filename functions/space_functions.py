import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree


def mirror_voronoi(coords_vor, limits):
    """
    Mirror the Voronoi points so that the cells are confined within the limits.

    Args:
        coords_vor: array of shape (d, n) containing the Voronoi points,
                    where d is the number of dimensions and n is the number of
                    points.
        limits:     list of tuples [(low_0, high_0), ...,
                                    (low_{d-1}, high_{d-1})] for each dimension

    Returns:
        vor:        a Voronoi object computed on the mirrored points.
    """
    mirrored_sets = [coords_vor]

    for dim, (low, high) in enumerate(limits):
        # Upper mirror
        upper = coords_vor.copy()
        upper[dim] = 2 * high - coords_vor[dim]
        mirrored_sets.append(upper)
        # Lower mirror
        lower = coords_vor.copy()
        lower[dim] = 2 * low - coords_vor[dim]
        mirrored_sets.append(lower)

    all_points = np.hstack(mirrored_sets).T
    return Voronoi(all_points)


def neighbors_vor(vor, n, nan_idx=None):
    """
    Find neighbors of given Voronoi points. Indices larger than n (indicating
    mirrorred regions) or with -1 (indicating an open region) are ignored.

    Args:
        vor:    Voronoi object (which contains ridge_points)
        n:      Number of original Voronoi points
        nan_idx: (Optional) Not used in this version

    Returns:
        w:      An n x n matrix where an entry is 1 if the corresponding
                points are neighbors.
    """
    # Create an n x n matrix of zeros
    w = np.zeros((n, n))

    # Extract ridge points as a NumPy array
    ridge_points = vor.ridge_points

    # Build a mask to filter out invalid pairs:
    # - both indices must not be -1
    # - both indices must be less than n (i.e., within the original points)
    mask = (
        (ridge_points[:, 0] != -1) &
        (ridge_points[:, 1] != -1) &
        (ridge_points[:, 0] < n) &
        (ridge_points[:, 1] < n)
    )

    # Filter the ridge points using the mask
    valid_ridges = ridge_points[mask]

    if valid_ridges.size > 0:
        # For each valid ridge, determine the row and column indices:
        # row = max(index1, index2) and col = min(index1, index2)
        ii = valid_ridges[:, 0]
        jj = valid_ridges[:, 1]
        rows = np.maximum(ii, jj)
        cols = np.minimum(ii, jj)

        # Set the corresponding elements in w to 1 using advanced indexing
        w[rows, cols] = 1

    return w


def neighbors_order(values: np.ndarray) -> np.ndarray:
    """
    Define a nearest neighbor matrix such that values that are next to each
    other in the sorted order are neighbors.

    For example, values = [1, 2, 3, np.nan, 0] will result in
        w = [[0, 1, 0, 0, 1],
             [1, 0, 1, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0]]

    Note that the last value is a neighbor of the first value, as they are
    next to each other in the sorted order.
    """
    values = np.asarray(values)
    n = len(values)

    # Get indices for non-NaN values
    valid = ~np.isnan(values)
    idx_valid = np.arange(n)[valid]
    order = idx_valid[np.argsort(values[valid])]

    # Create neighbor connections
    w = np.zeros((n, n))
    if len(order) > 1:
        w[order[1:], order[:-1]] = 1
        w[order[:-1], order[1:]] = 1

    return w


def find_nearest_vor_node(coords_vor, coords):
    """
    Find the nearest voronoi node for each coordinate in coords.
    coords_vor and coords should have the same number of dimensions D.

    Args:
        coords_vor: coordinates of the voronoi nodes,  [x1, ...,  xD],
                where xi are vectors of the same length(number of events)
        coords: coordinates of the EQs[x1, ...,  xD], where xi are vectors
                of the same length(number of voronoi cells)

    Returns:
        nearest: indices of the nearest voronoi node for each coordinate in
        coords
    """
    tree = cKDTree(coords_vor.T)
    _, nearest = tree.query(coords.T)
    return nearest


def find_points_in_tile(coords_vor, coords, values, *args):
    """
    Find the magnitudes and additional values of the earthquakes in each tile.

    Args:
        coords_vor: coordinates of the Voronoi nodes, [x1, ..., xD], where xi
                are vectors of the same length (number of events)
        coords:     coordinates of the EQs [x1, ..., xD], where xi are vectors
                of the same length (number of Voronoi cells)
        values:     primary values corresponding to the coordinates in coords
        *args:      additional vectors of values corresponding to the
                coordinates in coords

    Returns:
        If only a single value vector is given (i.e. no extra vectors in args),
        returns a list (tile_values) containing the values for each tile.
        If additional vectors are provided, returns a tuple where the first
        element is tile_values, and each subsequent element is the
        corresponding list of values for that extra vector.
    """
    nearest = find_nearest_vor_node(coords_vor, coords)
    n_tiles = coords_vor.shape[1]
    tile_values = [values[nearest == i] for i in range(n_tiles)]
    tile_extra_values = tuple([vec[nearest == i]
                              for i in range(n_tiles)] for vec in args)

    if args:
        return (tile_values,) + tile_extra_values
    else:
        return tile_values


def volumes_vor(vor,  n):
    """
    Estimate area of given voronoi points. indices larger than n or or
    present in nan_idx are not considered."""
    vol = np.zeros(n)
    for ii, reg_num in enumerate(vor.point_region[:n]):
        indices = vor.regions[reg_num]
        if -1 in indices:  # some regoins are open, then the area is infinite
            vol[ii] = np.inf
        else:
            vol[ii] = ConvexHull(vor.vertices[indices]).volume
    return vol


def volumes_vor_pyvoro(cells):
    """
    Get the volume of each Voronoi cell from PyVoro output.

    Args:
        cells: List of PyVoro cell dicts (output of pyvoro.compute_voronoi)

    Returns:
        volumes: np.array of shape (n_cells,), cell volumes
    """
    return np.array([cell['volume'] for cell in cells])
