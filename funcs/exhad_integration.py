"""The names the historical exHad callers import, over the one adapter.

Everything here forwards to :mod:`funcs.exhadDecays`, which is the tree's only
exHad adapter, or to :func:`funcs.decayProducts.simulateDecays_rest_frame`,
which is the single decay entry point both drivers call.  This module holds no
physics and no routing of its own.
"""

from funcs.exhadDecays import (MODELS, configure_llp, matched_process_labels,
                               metadata, simulate_decays_weighted, _generator)

__all__ = ['MODELS', 'configure_llp', 'matched_process_labels', 'metadata',
           'simulate_decays_weighted', 'import_fermion_alp', 'simulate_decays',
           'legacy_output']


def import_fermion_alp(llp):
    """Load the fermion-coupled ALP tables through the LLP's own loader."""
    llp.import_ALP_fermion()


def simulate_decays(llp, decay_module, mass, pdgs, branching, size, matrix_elements,
                    selected, visible, *, seed=1, weighted=False,
                    weight_floor_fraction=0.):
    """Decay one block of mothers, with or without ALP importance sampling.

    Unweighted, this is the merged decay entry point with the selected LLP's
    exHad coordinates attached, and it returns ``(events, sizes, labels)``.
    Weighted, it is the ALP importance-sampling path, and it returns
    ``(events, sizes, weights)``.
    """
    if weighted:
        return simulate_decays_weighted(
            llp, decay_module, mass, pdgs, branching, size, matrix_elements,
            selected, visible, seed=seed,
            weight_floor_fraction=weight_floor_fraction)
    if weight_floor_fraction != 0.:
        raise ValueError('weight_floor_fraction requires weighted sampling')
    return decay_module.simulateDecays_rest_frame(
        mass, pdgs, branching, size, matrix_elements, selected, visible,
        llp_name=llp.LLP_name, particle_path=llp.particle_path,
        exhad_variant=llp.scalar_lifetime, seed=seed,
        return_process_labels=True,
        decay_channels=getattr(llp, 'decayChannels', None))


def legacy_output(llp, mothers, daughters, sizes, selected):
    """Label a finished block for export, leaving every event where it is.

    Rows the matched generator produced are named one by one, so a reader keys
    each block on a distinct process name.  The event rows, the per-row counts
    and the selection are returned exactly as they were sampled.
    """
    labels = matched_process_labels(
        llp.decayChannels, selected, getattr(llp, '_exhad_pooled_indices', ()))
    return mothers, daughters, labels, sizes, selected
