"""Export projected coordinates and assigned cluster labels."""

import numpy as np

def export_cluster_coordinates(frames, labels, data_nd, out_file, n_dimensions):
    """Exports the assigned coordinate matrices to a .dat file."""
    out_data = np.column_stack((frames, labels, data_nd))
    fmt_string = f"%10d %4d " + ' '.join(['%11.7f'] * n_dimensions)
    np.savetxt(out_file, out_data, fmt=fmt_string)

import logging
import os
import time

from clustermd.constants import Colors


def export_assignment_output(frames, labels, data_nd, results_dir, n_dimensions):
    """Write the clustering assignment matrix and return its path and elapsed time."""
    t0 = time.time()
    out_filename = os.path.join(results_dir, f"hdbscan.{n_dimensions}-D.clusters.dat")
    logging.info(
        f"\n{Colors.LIGHT_BLUE}[*]{Colors.RESET} "
        f"Exporting assigned coordinate matrices to {out_filename}..."
    )
    export_cluster_coordinates(frames, labels, data_nd, out_filename, n_dimensions)
    return out_filename, time.time() - t0

