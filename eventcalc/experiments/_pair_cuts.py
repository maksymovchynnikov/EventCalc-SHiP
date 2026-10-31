from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PairCuts:
    """Exactly two visible charged daughters. Thresholds are independent options.

    No hidden efficiency, transverse-mass, PID or neutral-veto approximation.
    Charge and visible arrays must be supplied, so invisible daughters survive
    in the full event record but do not count as detected tracks.
    """
    track_p_min: float = 0.
    pair_energy_min: float = 0.
    opposite_charge: bool = True
    transverse_mass_max: float | None = None

    def __post_init__(self):
        if not np.all(np.isfinite([self.track_p_min, self.pair_energy_min])) or min(self.track_p_min, self.pair_energy_min) < 0:
            raise ValueError('Thresholds must be finite and nonnegative')
        if self.transverse_mass_max is not None and (not np.isfinite(self.transverse_mass_max) or self.transverse_mass_max <= 0):
            raise ValueError('Transverse mass bound must be positive')

    def select(self, p4, charge, visible, *, transverse_axis=None):
        # Last axis: px,py,pz,E. Padded daughters must have visible=False.
        p4, charge, visible = np.asarray(p4, float), np.asarray(charge, float), np.asarray(visible, bool)
        if p4.ndim != 3 or p4.shape[2] != 4 or charge.shape != p4.shape[:2] or visible.shape != charge.shape:
            raise ValueError('Expected p4 (N,D,4), charge/visible (N,D)')
        if not np.all(np.isfinite(p4)) or not np.all(np.isfinite(charge)) or np.any(p4[..., 3] < 0):
            raise ValueError('Invalid daughter record')
        tracks = visible & (charge != 0)
        pair = (tracks.sum(axis=1) == 2) & (visible.sum(axis=1) == 2)
        track_pass = np.all(~tracks | (np.linalg.norm(p4[..., :3], axis=2) > self.track_p_min), axis=1)
        energy_pass = np.sum(np.where(tracks, p4[..., 3], 0.), axis=1) > self.pair_energy_min
        opposite = (np.sum(np.where(tracks, charge, 0.), axis=1) == 0) if self.opposite_charge else True
        mt_pass = True
        if self.transverse_mass_max is not None:
            if transverse_axis is None:
                raise ValueError('Transverse-mass selection requires an explicit reference axis')
            summed = np.sum(np.where(tracks[..., None], p4, 0.), axis=1)
            mt_pass = transverse_mass(summed, transverse_axis) < self.transverse_mass_max
        return pair & track_pass & energy_pass & opposite & mt_pass


def transverse_mass(pair_p4, axis):
    """sqrt(M_pair^2 + pT_pair^2) + pT_pair, with caller-defined axis.

    The source's HNL cut must not silently become a validated DP/iDM cut.
    """
    a = np.asarray(pair_p4, float)
    n = np.broadcast_to(np.asarray(axis, float), a[..., :3].shape)
    if a.ndim != 2 or a.shape[1] != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(n)):
        raise ValueError('Expected finite pair four-vectors and axes')
    length = np.linalg.norm(n, axis=1)
    if np.any(length == 0): raise ValueError('Reference axis must be nonzero')
    n = n / length[:, None]
    p = a[:, :3]
    m2 = a[:, 3]**2 - np.sum(p*p, axis=1)
    if np.any(a[:, 3] < 0) or np.any(m2 < -1e-10 * np.maximum(a[:, 3]**2, 1)):
        raise ValueError('Pair four-vector must be future timelike or null')
    pt = np.linalg.norm(p - np.sum(p*n, axis=1)[:, None]*n, axis=1)
    return np.sqrt(np.maximum(m2, 0) + pt**2) + pt
