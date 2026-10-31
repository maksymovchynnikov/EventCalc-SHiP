"""SHiP geometry and selections; legacy defaults preserved."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Geometry:
    z_min: float = 32.
    z_max: float = 82.
    x_in: float = 1.
    x_out: float = 4.
    y_in: float = 2.7
    y_out: float = 6.2
    aperture_z: float = 97.32
    aperture_x: float = 4.32
    aperture_y: float = 6.48
    target_z: float = 0.
    magnet_z_min: float = 86.
    magnet_z_max: float = 91.
    field_integral: float = 0.77  # central integral B_y dz in tesla metres; SHiP-ECN3

    @classmethod
    def ship(cls):
        from funcs import ship_setup as s
        return cls(z_min=s.z_min, z_max=s.z_max, x_in=s.Delta_x_in,
                   x_out=s.Delta_x_out, y_in=s.Delta_y_in, y_out=s.Delta_y_out)

    def array(self):
        a = np.array(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(a)) or self.z_max <= self.z_min or self.aperture_z < self.z_max or np.any(a[2:6] <= 0) or np.any(a[7:9] <= 0):
            raise ValueError('Invalid decay volume or downstream aperture')
        if self.field_integral != 0 and not (self.z_max <= self.magnet_z_min < self.magnet_z_max <= self.aperture_z):
            raise ValueError('The dipole must lie after the decay volume and before the aperture')
        return a


@dataclass(frozen=True)
class Selection:
    name: str
    min_particles: int = 2
    all_visible: bool = False
    energy_min: float = 1.
    ip_max: float | None = None


def ship_selections():
    return (Selection('decay_in_volume', 0, energy_min=0.),
            Selection('at_least_two_energy'),
            Selection('at_least_two_energy_ip', ip_max=2.5),
            Selection('all_visible_energy', all_visible=True),
            Selection('all_visible_energy_ip', all_visible=True, ip_max=2.5))


def _selection_array(selections):
    if not selections or len({s.name for s in selections}) != len(selections):
        raise ValueError('Selections need unique names')
    rows = []
    for s in selections:
        if not s.name or isinstance(s.min_particles, bool) or int(s.min_particles) != s.min_particles or s.min_particles < 0:
            raise ValueError('Invalid particle count')
        if not np.isfinite(s.energy_min) or s.energy_min < 0 or (s.ip_max is not None and (not np.isfinite(s.ip_max) or s.ip_max < 0)):
            raise ValueError('Invalid energy or impact-parameter cut')
        if s.min_particles == 0 and (s.all_visible or s.ip_max is not None):
            raise ValueError('A zero-particle selection is decay-volume only')
        rows.append([s.min_particles, s.all_visible, s.energy_min, np.inf if s.ip_max is None else s.ip_max])
    return np.asarray(rows, dtype=float)


def _prepare_ship(batch, *, geometry=None, selections=None):
    """Prepare one bounded batch; weights already include production and BR.

Weights are integral contributions per incident particle, NOT a normalized
histogram and NOT decay probabilities. They may be signed. This function
does not infer which daughters are detectable: supply a visibility mask.
"""
    from eventcalc.sensitivity import _prepare, PreparedAcceptance
    geometry = geometry or Geometry.ship()
    selections = tuple(ship_selections() if selections is None else selections)
    arrays, mass = batch.arrays()
    g = geometry.array()
    if batch.charges is None:
        if geometry.field_integral != 0:
            raise ValueError('Magnetic transport requires explicit daughter charges; set field_integral=0 for straight trajectories')
        charges = np.zeros(arrays[1].shape[:2])
    else:
        charges = np.asarray(batch.charges, dtype=float)
        if charges.shape != arrays[1].shape[:2] or not np.all(np.isfinite(charges)):
            raise ValueError('charges must be finite with shape [event, daughter]')
    values = _prepare(*arrays, mass, g, _selection_array(selections), np.ascontiguousarray(charges))
    return PreparedAcceptance(*values[:4], selections, geometry, event_index=values[4])



class ShipResponse:
    def __init__(self, geometry=None, selections=None):
        self.geometry = geometry or Geometry.ship()
        self.selections = tuple(ship_selections() if selections is None else selections)
        self.geometry.array()
        _selection_array(self.selections)

    def prepare(self, batch):
        return _prepare_ship(batch, geometry=self.geometry, selections=self.selections)
