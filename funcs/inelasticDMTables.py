from __future__ import annotations
from pathlib import Path
from scipy.stats import gaussian_kde 
from . import PDG, inelasticDMCalcs, ship_setup

import numpy as np
import pandas as pd

COM_ENERGY = 400 # CoM energy of ShIP
MAX_THETA_SHIP  = ship_setup.theta_max_dec_vol
MAX_THETA_IDM   = 5 * MAX_THETA_SHIP

CHI_PR_PDG = inelasticDMCalcs.CHI_PR_PDG         # default PDG of ChiPr (1000033)

PX, PY, PZ, E, M, PDG_ID = range(6)    # column indices

def _flatten_to_6d(arr: np.ndarray) -> np.ndarray:
    """Return data (nev, ndecay, 6)."""
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 6:
        return arr
    if arr.ndim == 2 and arr.shape[1] % 6 == 0:
        nev, cols = arr.shape
        return arr.reshape(nev, cols // 6, 6)
    raise ValueError("boosted_products must be (nev, ndecay, 6) or (nev, n*6)")

def extract_chipr_tables(boostedProducts: np.ndarray,
                   *,
                   pdg_chipr: int = CHI_PR_PDG,
                   nbins_theta: int = 200,        
                   nbins_energy: int = 80,       
                   outdir: str | Path = "Distributions/Inelastic-DM"
                   ):
    """
    Build chiPr theta-E double-differential pdf and eMax(theta) tables with uniform
    binning:
        theta: [0 , theta_max_data]  in 100 equal-width bins
        E: [max(mchiPr, E_min_data) , min(E_max_data, 400 GeV)] in 100 bins
    """
    # -- reshape & select chiPr four-momenta ------------------------------------
    bp6 = _flatten_to_6d(boostedProducts)
    sel = bp6[..., PDG_ID] == float(pdg_chipr)
    px, py, pz, energy = (bp6[..., k][sel] for k in (PX, PY, PZ, E))
    mass = PDG.get_mass(pdg_chipr)

    # enforce global energy bounds
    keep   = (energy >= mass) & (energy <= COM_ENERGY)
    px, py, pz, energy = (v[keep] for v in (px, py, pz, energy))

    # ---------- theta and E arrays ---------------------------------------------
    p_mag  = np.sqrt(px**2 + py**2 + pz**2)
    theta  = np.arccos(np.divide(pz, p_mag, out=np.zeros_like(p_mag),
                                 where=p_mag != 0))


    # fixed, uniform bin edges
    t_edges = np.linspace(0.0, theta.max() + 1e-12, nbins_theta + 1)
    e_min   = max(mass, energy.min())
    e_max   = min(COM_ENERGY, energy.max())
    e_edges = np.linspace(e_min, e_max, nbins_energy + 1)

    theta_hi = t_edges[1:]                     
    th_grid, E_grid = np.meshgrid(theta_hi, e_edges[1:], indexing="ij")

    data = np.vstack([theta, energy])           # shape (2, N)
    kde  = gaussian_kde(data)                   # Gaussian kernel, Scott’s factor

    pts  = np.vstack([th_grid.ravel(),
                      E_grid.ravel()])          # evaluation points

    pdf_raw = kde(pts).reshape(th_grid.shape)   # shape (ntheta, nE)

    # ------------------------------------------------------------------
    #  normalise so that  ∫∫ pdf dtheta dE = 1
    # ------------------------------------------------------------------
    dtheta = np.diff(t_edges)[:, None]          
    dE = np.diff(e_edges)[None, :]             
    norm = (pdf_raw * dtheta * dE).sum()
    pdf   = pdf_raw / norm

    # ------------------------------------------------------------------
    # --------  build E_max(theta) table  ------------------------------
    # ------------------------------------------------------------------
    idx_theta = np.digitize(theta, t_edges) - 1
    emax      = np.full(nbins_theta, fill_value=mass)
    for i in range(nbins_theta):
        seltheta = idx_theta == i
        if seltheta.any():
            emax[i] = energy[seltheta].max()

    # ------------------------------------------------------------------
    #  dump to ASCII  (same column order, so downstream code stays intact)
    # ------------------------------------------------------------------
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_emax = pd.DataFrame({"mass_GeV":   mass,
                            "theta_hi_rad": theta_hi,
                            "Emax_GeV":    emax})

    df_dd = pd.DataFrame({"mass_GeV":     mass,
                          "theta_hi_rad": th_grid.ravel(),
                          "energy_GeV":   E_grid.ravel(),
                          "d2f_dtheta_dE":    pdf.ravel()})

    for df, fname in ((df_emax, "Emax-iDM.txt"),
                      (df_dd,   "DoubleDistr-iDM.txt")):
        path = out_dir / fname
        if path.exists():
            old = pd.read_csv(path, sep="\t", header=None)
            old.columns = df.columns
            df = pd.concat([old, df], ignore_index=True)

        df.to_csv(path, sep="\t", header=False, index=False,
                  float_format="%.8e")

    return {"emax": df_emax, "double": df_dd}

def _cleanup_idm_tables():
    """
    Delete the DoubleDistr-iDM.txt and Emax-iDM.txt files that were
    created during the run (if they exist).
    """
    distro_dir = Path(__file__).parent.parent / "Distributions" / "Inelastic-DM"
    for fname in ("DoubleDistr-iDM.txt", "Emax-iDM.txt"):
        fpath = distro_dir / fname
        try:
            fpath.unlink(missing_ok=True)  
            # if fpath.exists(): fpath.unlink()
        except Exception as exc:              # keep going even if removal fails
            print(f"[WARN] Could not remove {fpath}: {exc}")