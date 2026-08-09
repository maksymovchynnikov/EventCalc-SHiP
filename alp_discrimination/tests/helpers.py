import numpy as np

from alp_discrimination.physics.spectra import WeightedSpectrum


def spectrum(energies, weights, *, model="toy", source="inclusive", mass=1.0):
    weights = np.asarray(weights, float)
    return WeightedSpectrum(
        model_id=model, source=source, mass_gev=mass, ctau_m=10.0,
        selection_name="mother_level", energies_gev=np.asarray(energies, float),
        absolute_event_weights=weights, expected_events=float(weights.sum()), seed=1,
        generated_samples=len(weights), accepted_samples=len(weights), exposure_pot=1.0,
        visible_br=1.0, yield_per_pot_per_coupling_squared=1.0,
        unit_coupling_ctau_m=10.0, coupling_squared_gev_inv2=1.0,
        n_llp_total=1.0, epsilon_polar=1.0, epsilon_azimuthal=1.0,
        mean_decay_probability=1.0, source_expected_events={source: float(weights.sum())},
    )
