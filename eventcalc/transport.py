"""Shared uniform-dipole transport in GeV, metres and tesla metres."""
import numpy as np
from numba import njit

@njit(cache=True)
def dipole_displacement(momentum, charge, z_min, z_max, field_integral, aperture_z):
    """ECAL shift relative to a straight trajectory, plus forward-crossing flag.

    A uniform B_y changes px by -0.299792458*q*integral(B_y dz).
    Integrating the circular orbit through the magnet avoids the small-angle
    kick failure for soft tracks. Material and fringe fields are omitted.
    Input momenta are at the decay vertex; the returned shift is independent
    of that vertex when it is upstream of the magnet. Thus the lifetime
    integral remains analytic. The neutral/zero-field limit is exact.
    """
    px, py, pz = momentum
    if pz <= 0: return 0., 0., False
    if charge == 0 or field_integral == 0: return 0., 0., True
    length = z_max-z_min
    curvature = 0.299792458*charge*field_integral/length
    kick = curvature*length
    px_out = px-kick
    transverse2 = px*px+pz*pz
    if px_out*px_out >= transverse2: return 0., 0., False
    pz_out = np.sqrt(transverse2-px_out*px_out)
    # Rationalized difference avoids cancellation as the field tends to zero.
    dx = length*(px+px_out)/(pz+pz_out)
    # Difference of angles via atan2 is stable for small kicks.
    angle = np.arctan2(px*pz_out-px_out*pz, px*px_out+pz*pz_out)
    dy = py*angle/curvature
    drift = aperture_z-z_max
    return (dx-length*px/pz + drift*(px_out/pz_out-px/pz),
            dy-length*py/pz + drift*(py/pz_out-py/pz), True)


def ray_box_intervals(origins, momenta, lower, upper):
    """Forward distances along rays inside a finite axis-aligned cuboid."""
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if lower.shape!=(3,) or upper.shape!=(3,) or not np.all(np.isfinite([lower,upper])) or np.any(upper<=lower):
        raise ValueError('Box bounds must be finite ordered three-vectors')
    o, p = np.broadcast_arrays(np.asarray(origins, float), np.asarray(momenta, float))
    if o.ndim != 2 or o.shape[1] != 3 or not np.all(np.isfinite([o, p])):
        raise ValueError('Expected finite (N,3) origins and momenta')
    norm = np.linalg.norm(p, axis=1)
    u = np.divide(p, norm[:, None], out=np.zeros_like(p), where=norm[:, None] > 0)
    lo, hi = np.zeros(len(o)), np.full(len(o), np.inf)
    valid = norm > 0
    for axis in range(3):
        moving = u[:, axis] != 0
        a = np.divide(lower[axis] - o[:, axis], u[:, axis], out=np.zeros(len(o)), where=moving)
        b = np.divide(upper[axis] - o[:, axis], u[:, axis], out=np.zeros(len(o)), where=moving)
        lo = np.maximum(lo, np.where(moving, np.minimum(a, b), -np.inf))
        hi = np.minimum(hi, np.where(moving, np.maximum(a, b), np.inf))
        valid &= moving | ((o[:, axis] >= lower[axis]) & (o[:, axis] <= upper[axis]))
    valid &= hi > lo
    return np.where(valid, lo, 0.), np.where(valid, hi, 0.)
