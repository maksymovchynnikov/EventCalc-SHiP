"""Full collision source for ordinary EventCalc event generation.

A source row's momentum, position and signed weight are indivisible. The
source is folded once per mass and seed, then reused for all lifetimes.
"""
from dataclasses import dataclass
import numpy as np
from .model import ProductionModel


def uses_full_source(config):
    return (config.model == "Inelastic-DM" or
            config.model == "Dark-photons" and
            not (config.dp_production_mode or "combined").startswith("legacy-"))


def trajectory_intervals(momenta, origins, experiment):
    """Path lengths inside the truncated pyramid, from the collision position.

    Clip each ray against all six half-spaces. This also handles particles
    travelling backwards and particles that enter through a side face.
    """
    p = np.linalg.norm(momenta[:, :3], axis=1)
    direction = np.divide(momenta[:, :3], p[:, None],
                          out=np.zeros_like(origins), where=p[:, None] > 0)
    low = np.zeros(len(p)); high = np.full(len(p), np.inf)
    faces = [(np.array([0., 0., -1.]), -experiment.z_min_m),
             (np.array([0., 0., 1.]), experiment.z_max_m)]
    for axis, win, wout in ((0, experiment.x_width_in_m, experiment.x_width_out_m),
                            (1, experiment.y_width_in_m, experiment.y_width_out_m)):
        slope = (wout-win)/(2*(experiment.z_max_m-experiment.z_min_m))
        intercept = win/2-slope*experiment.z_min_m
        for sign in (-1, 1):
            normal = np.array([0., 0., -slope]); normal[axis] = sign
            faces.append((normal, intercept))
    for normal, bound in faces:
        speed = direction @ normal
        distance = bound-origins @ normal
        crossing = np.divide(distance, speed, out=np.zeros_like(p), where=speed != 0)
        low = np.where(speed < 0, np.maximum(low, crossing), low)
        high = np.where(speed > 0, np.minimum(high, crossing), high)
        high = np.where((speed == 0) & (distance < 0), -1., high)
    valid = (high > low) & (p > 0) & np.isfinite(high)
    return direction, np.where(valid, low, 0.), np.where(valid, high, 0.), valid


@dataclass(frozen=True)
class SourceSample:
    mother: np.ndarray
    momentum: np.ndarray
    inside_volume: np.ndarray
    origins: np.ndarray
    categories: np.ndarray
    mechanisms: np.ndarray


class CollisionSource:
    """A finite weighted integration sample, not an unweighted flux table."""
    def __init__(self, batches, experiment, metadata=None):
        batches = list(batches)
        if not batches:
            raise ValueError("No production at this mass in the selected source")
        self.mass = float(batches[0].mass)
        self.momenta = np.concatenate([b.momenta for b in batches])
        self.origins = np.concatenate([b.origins for b in batches])
        self.weights = np.concatenate([b.weights for b in batches])
        self.categories = np.concatenate([np.full(len(b.weights), b.category) for b in batches])
        self.mechanisms = np.concatenate([np.full(len(b.weights), b.channel) for b in batches])
        if (not all(np.isfinite(a).all() for a in (self.momenta, self.origins, self.weights))
                or any(not np.isclose(b.mass, self.mass) for b in batches)):
            raise ValueError("Invalid production source")
        self.total_yield = float(self.weights.sum())
        if self.total_yield <= 0:
            raise ValueError("The signed production integral must be positive; increase production_power")
        self.direction, self.low, self.high, self.valid = trajectory_intervals(
            self.momenta, self.origins, experiment)
        self.metadata = dict(metadata or {})

    @classmethod
    def build(cls, config, mass, experiment, provider=None):
        provider = provider or ProductionModel()
        mode = config.dp_production_mode or "combined"
        categories = ("primary", "cascade") if mode == "combined" else (mode,)
        model = "idm" if config.model == "Inelastic-DM" else "dp"
        batches = provider.batches(mass, model=model, delta=config.idm_delta,
                                   power=config.production_power,
                                   seed=1 if config.seed is None else config.seed,
                                   categories=categories)
        metadata = dict(provider.metadata)
        metadata["runtime_integration"] = dict(model=model, mediator_mass_GeV=float(mass),
            delta=config.idm_delta, power=config.production_power,
            seed=1 if config.seed is None else config.seed, categories=list(categories))
        return cls(batches, experiment, metadata)

    def sampler(self, c_tau, mode):
        return CollisionSampler(self, c_tau, mode)


class CollisionSampler:
    """Sample |source weight| and retain its sign in the decay weight.

    Fiducial mode samples only rays intersecting the volume. The normalization
    ``integral`` restores the omitted source and all signs. It is a proposal
    normalization, not a polar-angle efficiency. Attempted mode includes all
    source rays and gives misses zero weight. Neither mode uses an energy floor.
    """
    def __init__(self, source, c_tau, mode):
        self.source = source
        self.c_tau = float(c_tau)
        self.indices = (np.flatnonzero(source.valid) if mode == "fiducial"
                        else np.arange(len(source.weights)))
        weights = np.abs(source.weights[self.indices])
        self.cdf = np.cumsum(weights)
        self.absolute_yield = float(weights.sum())
        self.integral = self.absolute_yield/source.total_yield
        self.cell_count = len(self.indices)
        if self.absolute_yield == 0:
            raise ValueError("No source rays reach the decay volume; increase production_power")

    def sample(self, count, parent_rng, vertex_rng, pdg):
        source = self.source
        pick = np.searchsorted(self.cdf, parent_rng.random(count)*self.cdf[-1], side="right")
        ix = self.indices[np.minimum(pick, len(self.indices)-1)]
        momentum = source.momenta[ix]
        p = np.linalg.norm(momentum[:, :3], axis=1)
        rate = np.divide(source.mass, self.c_tau*p, out=np.zeros(count), where=p > 0)
        length = source.high[ix]-source.low[ix]
        interval = -np.expm1(-rate*length)
        probability = np.exp(-rate*source.low[ix])*interval
        u = vertex_rng.random(count)
        displacement = np.divide(-np.log1p(-u*interval), rate,
                                 out=np.zeros(count), where=rate > 0)
        distance = source.low[ix]+displacement
        vertex = source.origins[ix]+distance[:, None]*source.direction[ix]
        signed_probability = np.sign(source.weights[ix])*probability
        mother = np.column_stack((momentum, np.full(count, source.mass),
                                  np.full(count, pdg), signed_probability, vertex))
        return SourceSample(mother, momentum, source.valid[ix], source.origins[ix],
                            source.categories[ix], source.mechanisms[ix])
