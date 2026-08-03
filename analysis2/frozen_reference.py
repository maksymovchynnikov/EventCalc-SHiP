"""Immutable numerical reference values for the final frozen-reference calculation.

These values are intentionally small, selected regression targets rather than
generated distribution tables.  They were read from the tracked frozen scan,
template banks, distance maps, and five-seed profiled-likelihood outputs.
"""

from __future__ import annotations

from dataclasses import dataclass


MASSES_GEV = (0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.05)
MODEL_IDS = ("alp_photon_combined", "alp_su2l")
PROFILE_SEEDS = (73_241, 83_244, 93_247, 103_250, 113_253)


@dataclass(frozen=True)
class FrozenEndpoint:
    """Independent production and scan-diagnostic lifetime endpoints."""

    mass_gev: float
    model_id: str
    raw_log_log_lower_m: float
    raw_log_log_upper_m: float
    bisection_lower_m: float
    bisection_upper_m: float
    padded_grid_lower_m: float
    padded_grid_upper_m: float


ENDPOINTS = (
    FrozenEndpoint(0.30, "alp_photon_combined", 3.0, 275.42415737677209,
                   3.0, 275.42860634586657, 3.0, 272.94570697105343),
    FrozenEndpoint(0.30, "alp_su2l", 3.0, 547.13050878566207,
                   3.0, 547.12281133738236, 3.0, 541.46325910356222),
    FrozenEndpoint(0.40, "alp_photon_combined", 4.0, 143.58596346641741,
                   4.0, 143.58828780184922, 4.0, 142.56137736137705),
    FrozenEndpoint(0.40, "alp_su2l", 4.0, 401.57819660382211,
                   4.0, 401.57671463608460, 4.0, 397.89337196303796),
    FrozenEndpoint(0.50, "alp_photon_combined", 5.0, 85.078136106898995,
                   5.0, 85.07845199736937, 5.0, 84.597254910266557),
    FrozenEndpoint(0.50, "alp_su2l", 5.0, 315.33957789736502,
                   5.0, 315.34253201071414, 5.0, 312.73671139603272),
    FrozenEndpoint(0.60, "alp_photon_combined", 6.0, 54.619466206020370,
                   6.0, 54.620301830514741, 6.0, 54.378729816526509),
    FrozenEndpoint(0.60, "alp_su2l", 6.0, 258.10394567216883,
                   6.0, 258.10364973684585, 6.0, 256.16946244017112),
    FrozenEndpoint(0.75, "alp_photon_combined", 7.5, 30.849801427725691,
                   7.5, 30.849761048645433, 7.5, 30.762667450733804),
    FrozenEndpoint(0.75, "alp_su2l", 7.5, 200.60423665684473,
                   7.5, 200.60204188519404, 7.5, 199.29001652804294),
    FrozenEndpoint(0.90, "alp_photon_combined", 9.0, 18.963660286668198,
                   9.0, 18.963762353205809, 9.0, 18.935414113492897),
    FrozenEndpoint(0.90, "alp_su2l", 9.0, 162.90399392624602,
                   9.0, 162.90298097056296, 9.0, 161.96320180086889),
    FrozenEndpoint(1.00, "alp_photon_combined", 10.0, 14.080490604635353,
                   10.0, 14.080658221099586, 10.0, 14.070857070231174),
    FrozenEndpoint(1.00, "alp_su2l", 10.0, 144.33224073415502,
                   10.0, 144.33404250767614, 10.0, 143.56369492339741),
    FrozenEndpoint(1.05, "alp_photon_combined", 10.500000000000002,
                   12.108093868064573, 10.500000000000002,
                   12.107944020699280, 10.5, 12.104643579961705),
    FrozenEndpoint(1.05, "alp_su2l", 10.500000000000002,
                   136.39897461198709, 10.500000000000002,
                   136.40086182944941, 10.5, 135.70125429330454),
)


@dataclass(frozen=True)
class FrozenProbability:
    mass_gev: float
    model_prefix: str
    lifetime_index: int
    bin_index: int
    probability: float


SELECTED_PROBABILITIES = (
    FrozenProbability(0.30, "photon", 0, 0, 0.81163714169565959),
    FrozenProbability(0.30, "su2", 19, 17, 7.4878229047430379e-05),
    FrozenProbability(0.40, "photon", 10, 9, 0.0051796440927751354),
    FrozenProbability(0.40, "su2", 10, 9, 0.13033640489833864),
    FrozenProbability(0.50, "photon", 0, 0, 0.61813327357633852),
    FrozenProbability(0.50, "su2", 19, 18, 0.00038003071770614089),
    FrozenProbability(0.60, "photon", 10, 9, 0.011810034345363830),
    FrozenProbability(0.60, "su2", 10, 9, 0.12494753959944153),
    FrozenProbability(0.75, "photon", 0, 0, 0.43935982967490245),
    FrozenProbability(0.75, "su2", 19, 19, 0.00024404508350675490),
    FrozenProbability(0.90, "photon", 10, 10, 0.020814649109409153),
    FrozenProbability(0.90, "su2", 10, 10, 0.11606264937860378),
    FrozenProbability(1.00, "photon", 0, 0, 0.26941036057070494),
    FrozenProbability(1.00, "su2", 19, 20, 0.00017439099599723484),
    FrozenProbability(1.05, "photon", 10, 10, 0.033418876268652017),
    FrozenProbability(1.05, "su2", 19, 20, 0.00019421306870480727),
)


NUMBER_OF_ENERGY_BINS = (18, 18, 19, 19, 20, 20, 21, 21)
MINIMUM_TOTAL_VARIATION = (
    0.8545186856265761,
    0.8131159877322028,
    0.7686619348570054,
    0.7273221203601271,
    0.6636272279065970,
    0.6027599650324473,
    0.5594527267048203,
    0.5358399636443427,
)
PERSISTENT_EVENT_THRESHOLDS = (2, 2, 2, 2, 3, 3, 4, 4)

# Rows follow MASSES_GEV; columns follow PROFILE_SEEDS.  Each entry is the
# per-seed worst-case correct fraction at that mass's persistent threshold.
FIVE_SEED_ACCURACY_AT_THRESHOLD = (
    (0.97766, 0.97699, 0.97700, 0.97748, 0.97773),
    (0.96213, 0.96026, 0.96011, 0.96049, 0.95974),
    (0.94502, 0.94443, 0.94482, 0.94517, 0.94572),
    (0.92596, 0.92698, 0.92641, 0.92692, 0.92517),
    (0.94509, 0.94602, 0.94590, 0.94531, 0.94592),
    (0.91654, 0.91585, 0.91933, 0.91797, 0.91748),
    (0.93110, 0.93148, 0.93099, 0.93061, 0.93090),
    (0.91882, 0.91848, 0.91912, 0.91759, 0.91811),
)
