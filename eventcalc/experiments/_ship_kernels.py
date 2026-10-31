"""SHiP-specific volume and daughter-selection interval kernels."""
import numpy as np
from numba import njit
from eventcalc.transport import dipole_displacement

@njit(cache=True)
def _volume(origin, direction, g):
    lo, hi = 0., np.inf
    if abs(direction[2]) < 1e-15:
        if origin[2] < g[0] or origin[2] > g[1]: return 0., 0.
    else:
        z1, z2 = (g[0]-origin[2])/direction[2], (g[1]-origin[2])/direction[2]
        lo, hi = max(lo, min(z1, z2)), min(hi, max(z1, z2))
    for dim, win, wout in ((0, g[2], g[3]), (1, g[4], g[5])):
        slope = (wout-win)/(2*(g[1]-g[0]))
        offset = win/2-slope*g[0]
        for sign in (-1., 1.):
            a = sign*direction[dim]-slope*direction[2]
            b = offset+slope*origin[2]-sign*origin[dim]
            if abs(a) < 1e-15:
                if b < 0: return 0., 0.
            elif a > 0: hi = min(hi, b/a)
            else: lo = max(lo, b/a)
    return (lo, hi) if hi > lo else (0., 0.)


@njit(cache=True)
def _ip(lo, hi, origin, direction, momentum, target_z, radius):
    if np.isinf(radius): return lo, hi
    if momentum[2] <= 0: return lo, lo
    a = origin[:2] - (origin[2]-target_z)*momentum[:2]/momentum[2]
    b = direction[:2] - direction[2]*momentum[:2]/momentum[2]
    aa, ab, cc = np.dot(b, b), np.dot(a, b), np.dot(a, a)-radius*radius
    if aa < 1e-30: return (lo, hi) if cc <= 0 else (lo, lo)
    disc = ab*ab-aa*cc
    if disc < 0: return lo, lo
    root = np.sqrt(disc)
    return max(lo, (-ab-root)/aa), min(hi, (-ab+root)/aa)


@njit(cache=True)
def _prepare(p, d, visible, weights, origins, mass, g, cuts, charges):
    n, nk, ns = len(p), d.shape[1], len(cuts)
    maximum = n*ns*max(1, 2*nk)
    lower = np.empty(maximum); upper = np.empty(maximum)
    weight = np.empty(maximum); selected = np.empty(maximum, np.int64); event_index = np.empty(maximum, np.int64)
    size = 0
    for i in range(n):
        momentum = np.sqrt(np.sum(p[i, :3]**2))
        if weights[i] == 0: continue
        if momentum <= 0:
            raise ValueError('Stationary mothers require a time-dependent source calculation')
        u = p[i, :3]/momentum
        lo, hi = _volume(origins[i], u, g)
        if hi <= lo: continue
        inverse_boost = mass/momentum
        for s in range(ns):
            count, all_visible, energy, radius = cuts[s]
            if count == 0:
                lower[size], upper[size], weight[size], selected[size] = lo*inverse_boost, hi*inverse_boost, weights[i], s
                event_index[size] = i
                size += 1
                continue
            left = np.empty(nk); right = np.empty(nk); mom = np.empty((nk, 3))
            nh, nv = 0, 0
            for k in range(nk):
                if not visible[i, k]: continue
                nv += 1
                if d[i, k, 3] <= energy or d[i, k, 2] <= 0: continue
                shift_x, shift_y, forward = dipole_displacement(d[i, k, :3], charges[i, k], g[10], g[11], g[12], g[6])
                if not forward: continue
                a, b = lo, hi
                for dim, width in ((0, g[7]), (1, g[8])):
                    slope = u[dim]-u[2]*d[i, k, dim]/d[i, k, 2]
                    offset = origins[i, dim]+(g[6]-origins[i, 2])*d[i, k, dim]/d[i, k, 2] + (shift_x if dim == 0 else shift_y)
                    if abs(slope) < 1e-15:
                        if abs(offset) >= width/2: b = a
                    else:
                        x = (-width/2-offset)/slope; y = (width/2-offset)/slope
                        a, b = max(a, min(x, y)), min(b, max(x, y))
                if b > a:
                    left[nh], right[nh], mom[nh] = a, b, d[i, k, :3]
                    nh += 1
            if nh < count or (all_visible and nh < nv): continue
            endpoints = np.sort(np.concatenate((left[:nh], right[:nh])))
            for k in range(len(endpoints)-1):
                a, b = endpoints[k], endpoints[k+1]
                if b <= a: continue
                mid = (a+b)/2; hits = 0; psum = np.zeros(3)
                for j in range(nh):
                    if left[j] < mid < right[j]:
                        hits += 1; psum += mom[j]
                if hits < count or (all_visible and hits != nv): continue
                a, b = _ip(a, b, origins[i], u, psum, g[9], radius)
                if b <= a: continue
                lower[size], upper[size], weight[size], selected[size] = a*inverse_boost, b*inverse_boost, weights[i], s
                event_index[size] = i
                size += 1
    return lower[:size].copy(), upper[:size].copy(), weight[:size].copy(), selected[:size].copy(), event_index[:size].copy()
