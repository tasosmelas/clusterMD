"""Select observed samples nearest arithmetic centroids in retained coordinates."""
import numpy as np
from clustermd.models import RepresentativeSelection

def select_representative(cluster_coords, cluster_frames, sample_indices=None):
    """Return the observed frame nearest the arithmetic projection-space centroid.

cluster_frames are source-frame IDs; sample_indices are analyzed-row
ordinals when supplied; the cluster-local position is also retained."""
    centroid = np.mean(cluster_coords, axis=0)
    distances = np.linalg.norm(cluster_coords - centroid, axis=1)
    nearest_centroid_idx = np.argmin(distances)
    nearest_centroid_frame = cluster_frames[nearest_centroid_idx]
    sample_index = (sample_indices[nearest_centroid_idx]
                    if sample_indices is not None else nearest_centroid_idx)
    return RepresentativeSelection(int(sample_index), int(nearest_centroid_frame),
                                   int(nearest_centroid_idx), centroid)

def select_representatives(projection, labels):
    """Select one observed centroid-nearest frame for each assigned cluster."""
    return {cid: select_representative(projection.coordinates[labels == cid],
                projection.frame_indices[labels == cid], projection.sample_indices[labels == cid])
            for cid in np.unique(labels[labels != -1])}
