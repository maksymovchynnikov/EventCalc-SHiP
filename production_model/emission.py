import numpy as np
MP,MPI,ALPHA=.938272,.13957,1/137.035999

def mixing_weights(mass, scales):
    """Original analytic mixing weights, including leptonic-width calibration."""
    m = np.asarray(mass)
    return np.array([
        .000952990551143179 / ((m*m-.6006250000000001)**2
                              + .15106001765793114*m*m*(m*m-.0729)**3),
        .0001097656080235845 / (.000039137536+(m*m-.6115240000000001)**2),
        .0006340762869295578 / (.000018860591928610893+(m*m-1.039288497025)**2)
    ]).T * np.asarray(scales)
