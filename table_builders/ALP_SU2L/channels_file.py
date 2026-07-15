# X_s channel list and B0/B+ convention

from .constants import (
    M_B_PLUS,
    BPLUS_TO_XA_CHANNELS,
)

from .config import (B0_TO_BPLUS_BR_FACTOR)

# Internal particle bookkeeping for EventCalc-style two-body decay products.
# This ALP code is a dummy/internal code, not an official PDG code.
PDG_ALP = 12345678
PDG_DUMMY_RECOIL = 999999

CHARGE_ALP = 0
STABILITY_ALP = 1

DEFAULT_RECOIL_CHARGE = 1
DEFAULT_RECOIL_STABILITY = 1


def get_allowed_channels(alp_mass, channels=BPLUS_TO_XA_CHANNELS):
    """
    Keep only kinematically allowed B+ -> X + a channels.
    """
    allowed_channels = []

    for channel in channels:
        if M_B_PLUS > alp_mass + channel["mass"]:
            allowed_channels.append(channel)

    return allowed_channels


def effective_B_fragmentation_factor(f_b_to_Bplus, f_b_to_B0, include_B0=True):
    """
    Effective fragmentation factor when the BR table is defined for B+.

    If include_B0=True, we approximate
        BR(B0 -> X a) = 0.93 * BR(B+ -> X a).
    """
    f_eff = f_b_to_Bplus

    if include_B0:
        f_eff += B0_TO_BPLUS_BR_FACTOR * f_b_to_B0

    return f_eff
