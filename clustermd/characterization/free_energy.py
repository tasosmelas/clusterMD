'Calculate descriptive two-dimensional free-energy landscapes from projected occupancy.\n\nThe histogram uses 150 bins per axis, a pseudocount of 1e-10 and a\n9 kcal/mol display cap; this is distinct from cluster-population energies.'
import numpy as np
from clustermd.constants import MAX_FREE_ENERGY_KCAL

def calculate_free_energy(pc1, pc2, k_B_T, xl12, yl12):
    """Estimate a 2D occupancy-derived relative free-energy surface.

Histogram binning, pseudocount and clipping are descriptive visualization
choices; they are not a multidimensional thermodynamic reconstruction."""
    hist, xedges, yedges = np.histogram2d(pc1, pc2, bins=150, range=[xl12, yl12])
    hist = hist.T  # Explicit transposition for correct Cartesian rendering
    
    free_energy = -k_B_T * np.log((hist + 1e-10) / np.max(hist))
    free_energy[free_energy > MAX_FREE_ENERGY_KCAL] = MAX_FREE_ENERGY_KCAL 
    
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
    
    return X, Y, free_energy
