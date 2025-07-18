#inelasticDMCalcs.py
import csv
from fractions import Fraction
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

from funcs import PDG as pdg

CHI_PDG = 1000022
CHI_PR_PDG = 1000033
CHI_NAME = "chi"
CHI_PR_NAME = "chiPr"

def get_model_properties(model_num):
    """ takes input of model number and returns dictionary with relative_mass_split, dm_dark_photon_mass_ratio, dark_coupling"""
    
    this_dir = Path(__file__).resolve().parent        # .../funcs

    # 2. Go one level up (project root) then to Distributions/Inelastic-DM
    csv_path = (
        this_dir.parent                 # project_root
        / "Distributions"
        / "Inelastic-DM"
        / "models.csv"
    )

    # 3. Read and locate the requested model
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["model_num"]) == model_num:
                rel_split = float(row["relative_mass_split"])

                ratio_raw = row["dm_dark_photon_mass_ratio"]
                # Accept "1/3" or "0.3333"
                try:
                    dm_dp_ratio = float(ratio_raw)
                except ValueError:
                    dm_dp_ratio = float(Fraction(ratio_raw))

                dark_coupling = float(row["dark_coupling"])

                return {
                    "relative_mass_split": rel_split,
                    "dm_dark_photon_mass_ratio": dm_dp_ratio,
                    "dark_coupling": dark_coupling,
                }

    raise KeyError(f"Model {model_num} not found in {csv_path}")

def get_lifetime(m_chiPr, epsilonSqr, model_num):
    """ returns the mean lifetime of m_chiPr for the given mass and model """

    this_dir = Path(__file__).resolve().parent 
    
    tsv_path = (
        this_dir.parent                 # project_root
        / "Distributions"
        / "Inelastic-DM"
        / f"ctau-IDM_Model {model_num}.txt"
    )

    df = pd.read_csv(tsv_path, sep="\t", header=None,
                     names=["mass", "ctau_times_epsilonSqr"])

    if df.empty:
        raise ValueError(f"{tsv_path} is empty")
    
    idx = (df["mass"] - m_chiPr).abs().idxmin()
    ctau_epsilonSqr = df.at[idx, "ctau_times_epsilonSqr"]

    ctau = ctau_epsilonSqr / epsilonSqr
    return float(ctau)

def get_m_chi(m_chiPr, delta):
    """ returns the mass of m_chi in terms of m_chiPr and delta """
    return m_chiPr/(delta + 1)

def get_m_A(m_chiPr, delta, ratio_mChi_mA):
    """ returns the mass of the dark photon in terms of m_chiPr, chi_to_A_mass and delta"""
    return get_m_chi(m_chiPr, delta)/ratio_mChi_mA

def compute_masses(m_chiPr, model):
    """ computes the masses of m_chi and m_A based on m_chiPr and model number provided """
    model_properties = get_model_properties(model)

    m_chi = get_m_chi(m_chiPr, model_properties["relative_mass_split"])
    m_A = get_m_A(m_chiPr, model_properties["relative_mass_split"], model_properties["dm_dark_photon_mass_ratio"])

    return m_chi, m_A

def create_temp_particles(m_chiPr, model):
    """ creates temporary pdg particles that will be used in the decay of SDP """

    m_chi = compute_masses(m_chiPr, model)[0] # m_chi is the first mass in this list

    pdg.particle_db[CHI_PDG] = [m_chi, 0, 1]
    pdg.particle_db[CHI_PR_PDG] = [m_chiPr, 0, 1]

def get_inelastic_dm_ctau(mass, model, couplingSqr, particle_path="./Distributions/Inelastic-DM"):
        """
        Quickly get cτ for Inelastic-DM at a given mass and couplingSqr.
        """
        # Load ctau file
        ctau_path = os.path.join(particle_path, f"ctau-iDM_Model{model}.txt")
        ctau_data = pd.read_csv(ctau_path, sep="\t", header=None)
        
        mass_ctau = ctau_data.iloc[:,0].to_numpy()
        ctau_values = ctau_data.iloc[:,1].to_numpy()
        
        interp = RegularGridInterpolator(
            (mass_ctau,),
            ctau_values,
            bounds_error=False,
            fill_value=None
        )
        c_tau_unscaled = interp([mass])[0]
        return c_tau_unscaled / couplingSqr

def calc_mChiPr_mA_ratio(model, m_chiPr):
    m_A = compute_masses(m_chiPr, model)[1]
    return m_chiPr/m_A

def register_stable_particle(pythia, pdgid, name):
    if pdgid not in pdg.particle_db.keys():
        # no pdgid in the data base so cannot create particle
        return 
    
    try: 
        particle_name       = name            # or build a name yourself
        mass       = pdg.get_mass(pdgid)      # in GeV
        spinType   = 2                        # Majorana/Dirac fermion
        charge3    = int(round(3 * pdg.get_charge(pdgid)))  # Q × 3
        colType    = 0                        # colour-singlet
        width      = 0.0                      # ⇔ absolutely stable
        tau0       = 0                        # lifetime (mm); 0 is fine with width=0

        # fixed-format “:all = …” line (see Pythia manual §4.1) :contentReference[oaicite:0]{index=0}
        pythia.readString(
            f"{pdgid}:all = {particle_name} {name} {spinType} {charge3} {colType} "
            f"{mass} {width} {mass} {mass} {tau0}"
        )
        # make the intention crystal-clear
        pythia.readString(f"{pdgid}:isResonance = false")   # no BW smearing / FSR
        pythia.readString(f"{pdgid}:mayDecay   = off")
    except:
        # no idm particle, so not needed to be created
        return