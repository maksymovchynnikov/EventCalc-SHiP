import sys

# Path to the library containing pythia8.so
sys.path.insert(0, '/home/name/Downloads/pythia8312/lib')

# Import Pythia8 and other required modules
import pythia8

# Define a dictionary with particle PDG codes
particle_db = {
    # PDG codes only
    11: None,    # Electron
    -11: None,   # Positron
    13: None,    # Muon
    -13: None,   # Anti-muon
    15: None,    # Tau
    -15: None,   # Anti-tau
    22: None,    # Photon
    211: None,   # Charged pion (π+)
    -211: None,  # Charged pion (π-)
    111: None,   # Neutral pion (π0)
    321: None,   # Charged kaon (K+)
    -321: None,  # Charged kaon (K-)
    130: None,   # Neutral kaon (K0_L) - Long-lived
    310: None,   # Neutral kaon (K0_S) - Short-lived
    2: None,     # Up quark
    -2: None,    # Anti-up quark
    1: None,     # Down quark
    -1: None,    # Anti-down quark
    3: None,     # Strange quark
    -3: None,    # Anti-strange quark
    4: None,     # Charm quark
    -4: None,    # Anti-charm quark
    5: None,     # Bottom quark
    -5: None,    # Anti-bottom quark
    6: None,     # Top quark
    -6: None,    # Anti-top quark
    21: None,    # Gluon
    113: None,   # Rho meson (ρ0)
    213: None,   # Rho meson (ρ+)
    -213: None,  # Rho meson (ρ-)
    223: None,   # Omega meson (ω)
    331: None,   # η′(958) meson
    221: None,   # η meson
    333: None,   # φ(1020) meson
    20113: None, # a1(1260) meson
    # Neutrinos
    12: None,    # Electron neutrino (νe)
    -12: None,   # Electron anti-neutrino (ν̅e)
    14: None,    # Muon neutrino (νμ)
    -14: None,   # Muon anti-neutrino (ν̅μ)
    16: None,    # Tau neutrino (ντ)
    -16: None,   # Tau anti-neutrino (ν̅τ)
}

# Initialize Pythia
pythia = pythia8.Pythia()

# Initialize Pythia (required before accessing ParticleData)
pythia.init()

# Define the output file path
output_file = 'particle_masses.txt'

try:
    with open(output_file, 'w') as f:
        # Write the header
        f.write(f"{'PDG':>6} {'Mass (GeV)':>12}\n")
        
        # Iterate over PDG codes and retrieve mass from Pythia
        for pdg in particle_db.keys():
            if pythia.particleData.isParticle(pdg):
                mass = pythia.particleData.m0(pdg)  # Mass in GeV
                f.write(f"{pdg:>6} {mass:>12.6f}\n")
            else:
                f.write(f"{pdg:>6} {'N/A':>12} (Not recognized by Pythia)\n")
    
    print(f"Particle masses have been successfully exported to '{output_file}'.")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")

# Finalize Pythia (optional, good practice)
pythia.stat()

