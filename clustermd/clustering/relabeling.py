"""Order cluster identifiers by descending assigned population."""

import numpy as np

def sort_cluster_populations(labels):
    """Relabel assigned clusters by decreasing population, preserving -1.

Equal-sized clusters retain their ascending original-label order."""
    unique_clusters = np.unique(labels[labels != -1])
    cluster_counts = {cid: np.sum(labels == cid) for cid in unique_clusters}
    sorted_old_cids = sorted(cluster_counts, key=cluster_counts.get, reverse=True)
    
    id_mapping = {-1: -1}
    for new_id, old_id in enumerate(sorted_old_cids): id_mapping[old_id] = new_id
    
    reordered_labels = np.copy(labels)
    for old_id, new_id in id_mapping.items():
        if old_id != -1: reordered_labels[labels == old_id] = new_id
        
    return reordered_labels, np.unique(reordered_labels[reordered_labels != -1])

