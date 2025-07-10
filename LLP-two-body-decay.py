#!/usr/bin/env python3
# DP-two-body-decay.py – prompt 2-body sampler with *full* angular range (π/2)

import math, sys, numpy as np
from funcs import initLLP, kinematics, boost, PDG
from funcs.LLP_selection import (
    particle_selection, mixing_pattern, uncertainty,
    prompt_masses_and_c_taus, resampleSize
)
from funcs.TwoBodyDecay import decay_products

# ------------------- LLP menu prompts --------------------------------
LLP = initLLP.LLP(
    mass=None,
    particle_selection=particle_selection,
    mixing_pattern=mixing_pattern,
    uncertainty=uncertainty,
)
masses, c_taus_list = prompt_masses_and_c_taus()

# ------------- daughters: PDG fixed, masses via prompt ----------------
pdg1, pdg2 = 1111, 2222
try:
    m1, m2 = map(float, input(
        "\nEnter daughter masses m1 m2 in GeV "
        "(default 0.4 0.3 – press <Enter>): ").split())
except ValueError:
    m1, m2 = 0.4, 0.3
print(f"Using daughters {pdg1}/{pdg2} with masses {m1}/{m2} GeV")

PDG.particle_db[pdg1][0] = m1
PDG.particle_db[pdg2][0] = m2
charge1, charge2       = PDG.get_charge(pdg1),     PDG.get_charge(pdg2)
stability1, stability2 = PDG.get_stability(pdg1), PDG.get_stability(pdg2)


print(f"\n→ Sampling {resampleSize} prompt decays per mass "
      f"into ({pdg1},{pdg2}) over θ ∈ [0, π/2]")

# ------------------------ main loop ----------------------------------
for iM, (M, c_taus) in enumerate(zip(masses, c_taus_list), 1):
    if not (LLP.m_min_tabulated < M < LLP.m_max_tabulated):
        print(f"Mass {M} GeV outside table. Skipping.")
        continue

    LLP.set_mass(M)
    LLP.compute_mass_dependent_properties()
    c_tau_val = c_taus[0] if isinstance(c_taus, (list, tuple, np.ndarray)) else c_taus
    LLP.set_c_tau(c_tau_val)

    # full-sphere sampling (π/2 limit)
    kin = kinematics.Grids(
        LLP.Distr, LLP.Energy_distr,
        resampleSize, LLP.mass, LLP.c_tau_input,
        theta_max_sim=math.pi/2
    )
    kin.interpolate(False)
    kin.resample(resampleSize, False)

    theta  = kin.get_theta()
    energy = kin.get_energy()
    phi    = np.random.uniform(-np.pi, np.pi, resampleSize)
    pmod   = np.sqrt(energy**2 - M**2)

    px = pmod * np.cos(phi) * np.sin(theta)
    py = pmod * np.sin(phi) * np.sin(theta)
    pz = pmod * np.cos(theta)
    momentum = np.column_stack((px, py, pz, energy))     # (N,4)

    # decay and boost --------------------------------------------------
    unboosted = decay_products(
        M, resampleSize, m1, m2,
        pdg1, pdg2,
        charge1, charge2, stability1, stability2
    )
    boosted = boost.tab_boosted_decay_products(M, momentum, unboosted)

    # mother θ–E
    theta_mom = theta
    Em        = energy

    # first daughter
    px1, py1, pz1, E1 = boosted[:,0], boosted[:,1], boosted[:,2], boosted[:,3]
    p1mod  = np.sqrt(px1**2 + py1**2 + pz1**2)
    theta1 = np.arccos(np.divide(pz1, p1mod, out=np.zeros_like(p1mod), where=p1mod>0))

    tag = f"{M:g}"
    np.savetxt(f"dp_kinematics_{tag}.txt",
               np.column_stack((theta_mom, Em)), fmt="%.8e")
    np.savetxt(f"particle1_kinematics_{tag}.txt",
               np.column_stack((theta1, E1)),   fmt="%.8e")

    print(f"✅  {iM}/{len(masses)} wrote {resampleSize} events for mass {M} GeV")

print("\nFinished")
