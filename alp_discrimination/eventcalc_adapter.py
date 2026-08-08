"""The sole boundary between the discrimination workflows and EventCalc generation."""

from __future__ import annotations

import numpy as np

from .cache import CacheStore
from .config import (
    AnalysisConfig,
    SamplingSettings,
    ctau_source_seed,
    lower_ctau_m,
    spectrum_source_seed,
)
from .eventcalc_proposals import (
    ADAPTER_VERSION,
    KinematicProposal,
    MotherSample,
    ProposalGenerator,
    _input_fingerprints,
    _proposal_identity,
    _runtime_versions,
    _sanitize_interpolation,
    _validate_proposal_arrays,
    generate_mother_sample,
    legacy_numpy_seed,
)
from .models import ModelDefinition, ProductionSource, get_model
from .selections import (
    DEFAULT_ECAL,
    DEFAULT_ECAL_SEED_OFFSET,
    ECALGeometry,
    MotherLevelSelection,
    Selection,
    SelectionContext,
    selection_for_name,
)
from .spectra import WeightedSpectrum, combine_absolute_source_spectra


def _selection_from_config(config: AnalysisConfig) -> Selection:
    geometry_settings = getattr(config, "ecal_geometry", None)
    geometry = DEFAULT_ECAL
    if geometry_settings is not None:
        geometry = ECALGeometry(
            z_m=geometry_settings.z_m,
            width_x_m=geometry_settings.width_x_m,
            height_y_m=geometry_settings.height_y_m,
            centre_x_m=geometry_settings.centre_x_m,
            centre_y_m=geometry_settings.centre_y_m,
        )
    seed_policy = getattr(config, "seed_policy", None)
    seed_offset = (
        seed_policy.ecal_seed_offset
        if seed_policy is not None
        else DEFAULT_ECAL_SEED_OFFSET
    )
    return selection_for_name(
        config.selection_name,
        geometry=geometry,
        ecal_seed_offset=seed_offset,
    )


def _validate_spectrum_arrays(arrays: dict[str, np.ndarray], metadata: dict) -> None:
    spectrum = WeightedSpectrum.from_cache(arrays, metadata)
    identity = metadata["identity"]
    expected = {
        "model_id": identity["model"], "source": identity["source"],
        "mass_gev": identity["mass_gev"], "ctau_m": identity["ctau_m"],
        "selection_name": identity["selection_name"], "exposure_pot": identity["exposure_pot"],
    }
    if any(getattr(spectrum, name) != value for name, value in expected.items()):
        raise ValueError("spectrum metadata disagrees with its cache identity")
    if spectrum.expected_events <= 0.0:
        raise ValueError("cached spectra must have positive total weight")


class EventCalcAdapter:
    def __init__(
        self, config: AnalysisConfig, *, cache: CacheStore | None = None,
        force: bool = False, selection: Selection | None = None,
    ):
        self.config = config
        self.cache = cache or CacheStore(config.name)
        if self.cache.profile != config.name:
            raise ValueError("cache and analysis configuration profiles must agree")
        self.force = force
        self.selection = selection or _selection_from_config(config)
        if self.selection.name != config.selection_name:
            raise ValueError("selection strategy and configuration identifier disagree")
        self._proposal_generator = ProposalGenerator(config, self.cache)
        self._proposals = self._proposal_generator.proposals

    def _sampling(self, stage: str) -> SamplingSettings:
        return self._proposal_generator.sampling(stage)

    def prepare_kinematic_proposal(
        self, model: ModelDefinition, source: ProductionSource, mass_gev: float,
        seed: int, stage: str, *, proposal_ctau_m: float | None = None,
    ) -> KinematicProposal:
        return self._proposal_generator.prepare(
            model,
            source,
            mass_gev,
            seed,
            stage,
            force=self.force,
            proposal_ctau_m=proposal_ctau_m,
        )

    def evaluate_spectrum(
        self, proposal: KinematicProposal, ctau_m: float, true_sample_seed: int,
        coupling_squared_gev_inv2: float | None = None, *, cache_result: bool = True,
    ) -> WeightedSpectrum:
        if proposal.energy_support_mode == "lifetime_specific_truncated_Ea":
            if not np.isclose(
                ctau_m,
                proposal.proposal_ctau_m,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "lifetime-specific proposal may only be evaluated at its "
                    "preparation lifetime"
                )
        elif (
            proposal.sanitation_policy == "strict_core"
            and ctau_m < proposal.proposal_ctau_m
        ):
            raise ValueError(
                "full-support cached proposal is valid only at or above its "
                "preparation lifetime"
            )
        coupling_squared = (
            proposal.unit_coupling_ctau_m / ctau_m
            if coupling_squared_gev_inv2 is None else float(coupling_squared_gev_inv2)
        )
        if coupling_squared <= 0.0 or not np.isclose(
            ctau_m, proposal.unit_coupling_ctau_m / coupling_squared, rtol=1e-12, atol=0.0
        ):
            raise ValueError("lifetime and coupling-squared normalization disagree")
        selection_context = SelectionContext(
            source_seed=proposal.proposal_seed,
            true_sample_seed=true_sample_seed,
        )
        identity = {
            "adapter_version": ADAPTER_VERSION, "profile": self.config.name,
            "model": proposal.model_id, "source": proposal.source,
            "mass_gev": proposal.mass_gev, "ctau_m": ctau_m,
            "coupling_squared_gev_inv2": coupling_squared,
            "exposure_pot": self.config.exposure_pot, "proposal_cache_key": proposal.cache_key,
            "proposal_seed": proposal.proposal_seed, "true_sample_seed": true_sample_seed,
            "interpolation_points": proposal.interpolation_points,
            "resample_size": proposal.resample_size, "selection_name": self.selection.name,
            "selection": self.selection.cache_identity(selection_context),
            "sanitation_policy": proposal.sanitation_policy,
            "inputs": proposal.input_fingerprints, "runtime": _runtime_versions(),
        }
        key = None
        if cache_result:
            _, _, key = self.cache.paths("weighted_spectrum", identity)
            if not self.force:
                loaded = self.cache.load(
                    "weighted_spectrum", identity, _validate_spectrum_arrays
                )
                if loaded:
                    return WeightedSpectrum.from_cache(*loaded)
            else:
                print(f"CACHE FORCED   [weighted_spectrum] {key[:12]}")
        mothers = generate_mother_sample(proposal, ctau_m, true_sample_seed)
        selection_mask = np.asarray(
            self.selection.mask(mothers, selection_context),
            dtype=bool,
        )
        if selection_mask.shape != (len(mothers),):
            raise ValueError("selection mask has the wrong shape")
        energies = mothers.energy_gev[selection_mask]
        preselection_probabilities = mothers.decay_probability
        if not len(energies):
            raise RuntimeError("no accepted mother samples remain after selection")
        n_llp_total = (
            self.config.exposure_pot
            * proposal.yield_per_pot_per_coupling_squared
            * coupling_squared
        )
        scale = (
            n_llp_total * proposal.epsilon_polar * proposal.visible_br
            / proposal.resample_size
        )
        preselection_weights = scale * preselection_probabilities
        weights = np.asarray(preselection_weights[selection_mask], dtype=float)
        preselection_expected_events = float(np.sum(preselection_weights))
        expected_events = float(np.sum(weights))
        spectrum = WeightedSpectrum(
            model_id=proposal.model_id, source=proposal.source, mass_gev=proposal.mass_gev,
            ctau_m=ctau_m, selection_name=self.selection.name, energies_gev=energies,
            absolute_event_weights=weights, expected_events=expected_events,
            seed=proposal.proposal_seed, generated_samples=proposal.resample_size,
            accepted_samples=len(energies), exposure_pot=self.config.exposure_pot,
            visible_br=proposal.visible_br,
            yield_per_pot_per_coupling_squared=proposal.yield_per_pot_per_coupling_squared,
            unit_coupling_ctau_m=proposal.unit_coupling_ctau_m,
            coupling_squared_gev_inv2=coupling_squared, n_llp_total=n_llp_total,
            epsilon_polar=proposal.epsilon_polar,
            epsilon_azimuthal=len(mothers) / proposal.resample_size,
            mean_decay_probability=float(preselection_probabilities.mean()),
            preselection_expected_events=preselection_expected_events,
            preselection_samples=len(mothers),
            selection_efficiency_weighted=(
                expected_events / preselection_expected_events
                if preselection_expected_events > 0.0
                else 0.0
            ),
            source_expected_events={proposal.source: expected_events},
        )
        if spectrum.expected_events == 0.0:
            if cache_result:
                print(f"CACHE SKIPPED  [weighted_spectrum] {key[:12]}: zero total weight")
            return spectrum
        if not cache_result:
            return spectrum
        metadata = self.cache.save(
            "weighted_spectrum", identity, spectrum.arrays(), spectrum.metadata()
        )
        spectrum.cache_key = metadata.get("cache_key", key)
        return spectrum

    def evaluate_model(
        self, model_id: str, mass_gev: float, ctau_m: float,
        model_seed: int, stage: str, *, proposal_ctau_m: float | None = None,
    ) -> WeightedSpectrum:
        model = get_model(model_id)
        sources = {}
        for source_index, source in enumerate(model.sources):
            seed_policy = getattr(self.config, "seed_policy", None)
            if seed_policy is not None:
                seed_offset = self.config.templates.seed_offset if stage == "spectrum" else 0
                seed_policy.mass_index_from_model_seed(
                    model_seed, model_id, seed_offset=seed_offset,
                )
                seed = seed_policy.source_proposal_seed_from_model_seed(
                    model_seed, source_index,
                )
                true_sample_seed = seed_policy.true_sample_seed_from_model_seed(
                    model_seed, source_index,
                )
            else:
                seed = (
                    ctau_source_seed(model_seed, source_index) if stage == "ctau"
                    else spectrum_source_seed(model_seed, source_index)
                )
                true_sample_seed = seed + 1
            proposal = self.prepare_kinematic_proposal(
                model,
                source,
                mass_gev,
                seed,
                stage,
                proposal_ctau_m=proposal_ctau_m,
            )
            sources[source.identifier] = self.evaluate_spectrum(
                proposal, ctau_m, true_sample_seed,
            )
        if len(sources) == 1:
            return next(iter(sources.values()))
        combined = combine_absolute_source_spectra(model.identifier, sources)
        identity = {
            "adapter_version": ADAPTER_VERSION, "profile": self.config.name,
            "model": model.identifier, "source": "combined", "mass_gev": mass_gev,
            "ctau_m": ctau_m, "exposure_pot": self.config.exposure_pot,
            "selection_name": self.selection.name,
            "component_cache_keys": {
                name: value.cache_key for name, value in sources.items()
            },
        }
        _, _, key = self.cache.paths("weighted_spectrum", identity)
        if not self.force:
            loaded = self.cache.load(
                "weighted_spectrum", identity, _validate_spectrum_arrays
            )
            if loaded:
                return WeightedSpectrum.from_cache(*loaded)
        else:
            print(f"CACHE FORCED   [weighted_spectrum] {key[:12]} (combined)")
        metadata = self.cache.save(
            "weighted_spectrum", identity, combined.arrays(), combined.metadata()
        )
        combined.cache_key = metadata.get("cache_key", key)
        return combined


__all__ = [
    "ADAPTER_VERSION",
    "EventCalcAdapter",
    "KinematicProposal",
    "MotherLevelSelection",
    "MotherSample",
    "Selection",
    "SelectionContext",
    "generate_mother_sample",
    "legacy_numpy_seed",
]
