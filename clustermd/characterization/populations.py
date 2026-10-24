"""Calculate cluster populations and population-derived quantities."""

import numpy as np
from clustermd.constants import R_KCAL

def calculate_populations_and_energies(labels, temp):
    """Calculate assigned-only cluster populations and -RT ln(n/n_max).

Unassigned frames do not enter the energy normalization. Physical
interpretation assumes the occupancies represent appropriate sampling."""
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    states, counts = np.unique(valid_labels, return_counts=True)
    population_map = {state: count for state, count in zip(states, counts)}
    
    k_B_T = R_KCAL * temp
    max_count = np.max(counts) if len(counts) > 0 else 1
    free_energies = {state: -k_B_T * np.log(count / max_count) for state, count in zip(states, counts)}
    
    return states, counts, population_map, free_energies, k_B_T

