"""Small analytic toy source for testing the scan command, not a physics model."""
import numpy as np
from eventcalc.sensitivity import DecayBatch


def batches(*, replica, events=8192, chunk_size=1024, mass=1., seed=9182):
    rng = np.random.default_rng(np.random.SeedSequence([seed, replica]))
    for offset in range(0, events, chunk_size):
        n = min(chunk_size, events-offset)
        energy = rng.uniform(20., 200., n)
        mother = np.zeros((n, 4))
        mother[:, 2] = np.sqrt(energy**2-mass**2)
        mother[:, 3] = energy
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        daughters = np.zeros((n, 2, 4))
        daughters[:, 0, :3] = direction*mass/2
        daughters[:, 1, :3] = -direction*mass/2
        daughters[:, :, 3] = mass/2
        yield DecayBatch(mother, daughters, np.ones((n, 2), bool),
                         np.full(n, 1./events), mass, frame='rest', charges=np.tile([-1.,1.],(n,1)))
