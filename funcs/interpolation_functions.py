import warnings

import numba as nb
import numpy as np

#: A tabulated column arrives as the array pandas already holds, which numba
#: types as read-only.  A kernel compiled for a read-only argument also accepts
#: a writable one, so one compiled entry point serves either.
_READONLY_F8 = nb.types.Array(nb.float64, 1, "C", readonly=True)


def _njit(signature, **options):
    """Compile a kernel, keeping the machine code on disk between runs.

    The cache is keyed on this source file, so editing a kernel below compiles
    it again instead of loading the code of the previous version.

    A cache that cannot be used is not a reason not to run: where the source
    has no location on disk, or the directory beside it cannot be written, the
    kernel is compiled in memory as it was before, and the run continues.
    """

    def decorate(function):
        try:
            return nb.njit(signature, cache=True, **options)(function)
        except Exception as reason:
            warnings.warn(
                f"{function.__name__} is compiled without an on-disk cache "
                f"and will be compiled again on the next run: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            return nb.njit(signature, cache=False, **options)(function)

    return decorate


@_njit('(float64[::1], float64)', inline='always')
def _searchsorted_opt(arr, val):
    """
    Perform a binary search to find the index where `val` should be inserted to maintain order.

    Parameters
    ----------
    arr : np.ndarray
        Array in which to search.
    val : float
        Value to search for.

    Returns
    -------
    int
        Index where `val` should be inserted.
    """
    i = 0
    while i < arr.size and val > arr[i]:
        i += 1
    return i

@_njit('(float64[:,::1], float64[::1], float64[::1], float64[:,::1])', parallel=True)
def _bilinear_interpolation(rand_points, grid_x, grid_y, distr):
    """
    Perform bilinear interpolation on a set of random points.

    Parameters
    ----------
    rand_points : np.ndarray
        Array of points where interpolation is performed.
    grid_x : np.ndarray
        X-coordinates of the grid.
    grid_y : np.ndarray
        Y-coordinates of the grid.
    distr : np.ndarray
        Distribution values on the grid.

    Returns
    -------
    np.ndarray
        Interpolated values at the random points.

    A point outside the grid is evaluated from the nearest cell, so the
    interpolant is continued in a straight line past the outermost node.  That
    continuation carries no tabulated information, and the caller is
    responsible for checking its own sampling interval against `grid_x` and
    `grid_y` before calling.
    """
    results = np.empty(len(rand_points))
    len_y = grid_y.shape[0]

    for i in nb.prange(len(rand_points)):
        x, y = rand_points[i]

        # Find the indices of the grid points surrounding the point
        idx_x1 = _searchsorted_opt(grid_x, x) - 1
        idx_x2 = idx_x1 + 1
        idx_y1 = _searchsorted_opt(grid_y, y) - 1
        idx_y2 = idx_y1 + 1

        # Ensure the indices are within the bounds of the grid
        idx_x1 = max(0, min(idx_x1, len(grid_x) - 2))
        idx_x2 = max(1, min(idx_x2, len(grid_x) - 1))
        idx_y1 = max(0, min(idx_y1, len_y - 2))
        idx_y2 = max(1, min(idx_y2, len_y - 1))

        # Get the coordinates of the grid points
        x1, x2 = grid_x[idx_x1], grid_x[idx_x2]
        y1, y2 = grid_y[idx_y1], grid_y[idx_y2]

        # Get the values at the corners of the cell
        z11 = distr[idx_x1, idx_y1]
        z21 = distr[idx_x2, idx_y1]
        z12 = distr[idx_x1, idx_y2]
        z22 = distr[idx_x2, idx_y2]

        # Calculate the interpolation weights
        xd = (x - x1) / (x2 - x1)
        yd = (y - y1) / (y2 - y1)

        # Perform the interpolation
        c0 = z11 * (1 - xd) + z21 * xd
        c1 = z12 * (1 - xd) + z22 * xd

        result = c0 * (1 - yd) + c1 * yd

        results[i] = result

    return results

@_njit('(float64[:,::1], float64[::1], float64[::1], float64[::1], float64[:,:,::1], float64[::1])', parallel=True)
def _trilinear_interpolation(rand_points, grid_x, grid_y, grid_z, distr, max_energy):
    """
    Perform trilinear interpolation on a set of random points.

    Parameters
    ----------
    rand_points : np.ndarray
        Array of points where interpolation is performed.
    grid_x : np.ndarray
        X-coordinates of the grid.
    grid_y : np.ndarray
        Y-coordinates of the grid.
    grid_z : np.ndarray
        Z-coordinates of the grid.
    distr : np.ndarray
        Distribution values on the grid.
    max_energy : np.ndarray
        Maximum energy values at the random points.

    Returns
    -------
    np.ndarray
        Interpolated values at the random points.
    """
    results = np.zeros(len(rand_points))
    len_y, len_z = grid_y.shape[0], grid_z.shape[0]

    for i in nb.prange(len(rand_points)):
        x, y, z = rand_points[i]

        # Conditional to consider angle dependence on energy
        if z > max_energy[i]:
            continue

        # Outside the tabulated energy support the production density is
        # undefined, so the point contributes nothing.  The index clamping
        # below would otherwise continue the first cell in a straight line
        # past the start of the table, where a spectrum that rises with energy
        # returns a negative density.
        if z < grid_z[0] or z > grid_z[len_z - 1]:
            continue

        idx_x1 = _searchsorted_opt(grid_x, x) - 1
        idx_x2 = idx_x1 + 1
        idx_y1 = _searchsorted_opt(grid_y, y) - 1
        idx_y2 = idx_y1 + 1
        idx_z1 = _searchsorted_opt(grid_z, z) - 1
        idx_z2 = idx_z1 + 1

        idx_x1 = max(0, min(idx_x1, len(grid_x) - 2))
        idx_x2 = max(1, min(idx_x2, len(grid_x) - 1))
        idx_y1 = max(0, min(idx_y1, len_y - 2))
        idx_y2 = max(1, min(idx_y2, len_y - 1))
        idx_z1 = max(0, min(idx_z1, len_z - 2))
        idx_z2 = max(1, min(idx_z2, len_z - 1))

        x1, x2 = grid_x[idx_x1], grid_x[idx_x2]
        y1, y2 = grid_y[idx_y1], grid_y[idx_y2]
        z1, z2 = grid_z[idx_z1], grid_z[idx_z2]

        z111 = distr[idx_x1, idx_y1, idx_z1]
        z211 = distr[idx_x2, idx_y1, idx_z1]
        z121 = distr[idx_x1, idx_y2, idx_z1]
        z221 = distr[idx_x2, idx_y2, idx_z1]
        z112 = distr[idx_x1, idx_y1, idx_z2]
        z212 = distr[idx_x2, idx_y1, idx_z2]
        z122 = distr[idx_x1, idx_y2, idx_z2]
        z222 = distr[idx_x2, idx_y2, idx_z2]

        xd = (x - x1) / (x2 - x1)
        yd = (y - y1) / (y2 - y1)
        zd = (z - z1) / (z2 - z1)

        c00 = z111 * (1 - xd) + z211 * xd
        c01 = z112 * (1 - xd) + z212 * xd
        c10 = z121 * (1 - xd) + z221 * xd
        c11 = z122 * (1 - xd) + z222 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        result = c0 * (1 - zd) + c1 * zd

        # The mass and angle of a sampled point lie inside their grids by
        # construction and the energy has just been checked, so all eight
        # corner values enter with weights in [0, 1] and only round-off can
        # drive the sum below zero.  A tabulated density is never negative.
        if result < 0.0:
            result = 0.0

        results[i] = result

    return results


@_njit(
    (nb.float64[:, :, ::1], nb.float64[::1], nb.float64[::1], nb.float64[::1],
     _READONLY_F8, _READONLY_F8, _READONLY_F8, _READONLY_F8, nb.int64),
    parallel=True,
)
def _scatter_3D(grid, grid_x, grid_y, grid_z, mass, angle, energy, value, rows):
    """Write each tabulated row into the cell of the dense grid it names."""
    for i in nb.prange(rows):
        ix = _searchsorted_opt(grid_x, mass[i])
        iy = _searchsorted_opt(grid_y, angle[i])
        iz = _searchsorted_opt(grid_z, energy[i])
        grid[ix, iy, iz] = value[i]
    return grid


def _fill_distr_3D(grid_x, grid_y, grid_z, Distr):
    """
    Fill a 3D distribution grid with values from the original distribution.

    Parameters
    ----------
    grid_x : np.ndarray
        X-coordinates of the grid.
    grid_y : np.ndarray
        Y-coordinates of the grid.
    grid_z : np.ndarray
        Z-coordinates of the grid.

    Returns
    -------
    np.ndarray
        Filled 3D distribution grid.
    """
    distr_grid = np.zeros((len(grid_x), len(grid_y), len(grid_z)))
    # The kernel is compiled for one argument type, so the columns are handed
    # over in that type.  For a table of tabulated doubles this is the array
    # pandas already holds and nothing is copied.
    mass__, angle__, energy__, value__ = (
        np.ascontiguousarray(Distr[0], dtype=np.float64),
        np.ascontiguousarray(Distr[1], dtype=np.float64),
        np.ascontiguousarray(Distr[2], dtype=np.float64),
        np.ascontiguousarray(Distr[3], dtype=np.float64),
    )
    Distr_size = len(Distr)
    return _scatter_3D(
        distr_grid, grid_x, grid_y, grid_z,
        mass__, angle__, energy__, value__, Distr_size,
    )


@_njit(
    (nb.float64[:, ::1], nb.float64[::1], nb.float64[::1],
     _READONLY_F8, _READONLY_F8, _READONLY_F8, nb.int64),
    parallel=True,
)
def _scatter_2D(grid, grid_m, grid_a, mass, angle, energy, rows):
    """Write each tabulated row into the cell of the dense grid it names."""
    for i in nb.prange(rows):
        ix = _searchsorted_opt(grid_m, mass[i])
        iy = _searchsorted_opt(grid_a, angle[i])
        grid[ix, iy] = energy[i]
    return grid


def _fill_distr_2D(grid_m, grid_a, Energy_distr):
    """
    Fill a 2D distribution grid with values from the original energy distribution.

    Parameters
    ----------
    grid_m : np.ndarray
        Mass grid values.
    grid_a : np.ndarray
        Angle grid values.

    Returns
    -------
    np.ndarray
        Filled 2D distribution grid.
    """
    distr_grid = np.zeros((len(grid_m), len(grid_a)))
    mass__, angle__, energy__ = (
        np.ascontiguousarray(Energy_distr[0], dtype=np.float64),
        np.ascontiguousarray(Energy_distr[1], dtype=np.float64),
        np.ascontiguousarray(Energy_distr[2], dtype=np.float64),
    )
    Distr_size = len(Energy_distr)
    return _scatter_2D(
        distr_grid, grid_m, grid_a, mass__, angle__, energy__, Distr_size,
    )
