"""Run HDBSCAN clustering on multidimensional projections."""

from dataclasses import dataclass

import time
import math
import logging
import numpy as np
import os
from clustermd.constants import Colors
from clustermd.models import ClusteringResult
from clustermd.clustering.relabeling import sort_cluster_populations

def resolve_hdbscan_parameters(total_frames: int, config) -> tuple[int, int]:
    """Select HDBSCAN parameters from manual settings or a frame-count heuristic.

The automatic choice is a practical starting point, not an optimization
or evidence of scientific parameter stability."""
    if config.automatic_parameters:
        scale_factor = math.sqrt(total_frames / 1_000_000.0)
        dynamic_size = max(100, int(500 * scale_factor))
        dynamic_samples = max(10, int(dynamic_size / 10.0))
        logging.info(f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Auto-scaling activated: min_cluster_size={dynamic_size}")
    else:
        dynamic_size = config.min_cluster_size
        dynamic_samples = config.min_samples if config.min_samples is not None else max(2, int(dynamic_size / 10.0))
        
    return dynamic_size, dynamic_samples

def execute_clustering_algorithm(data_nd, size, samples, method):
    """Executes the core HDBSCAN clustering algorithm."""
    import fast_hdbscan
    t0 = time.time()
    clusterer = fast_hdbscan.HDBSCAN(min_cluster_size=size, min_samples=samples, cluster_selection_method=method)
    labels = clusterer.fit_predict(data_nd)
    runtime = time.time() - t0
    return labels, runtime


@dataclass(frozen=True)
class ClusteringStageResult:
    """Resolved HDBSCAN settings and relabeled clustering outputs."""
    clustering: ClusteringResult
    labels: np.ndarray
    unique_clusters: np.ndarray
    min_cluster_size: int
    min_samples: int
    assigned_frames: int
    noise_frames: int


def run_clustering_stage(
    *,
    clustering_coordinates,
    total_frames,
    spec,
    data_file,
    analysis_step,
    n_dimensions,
    engine=None,
):
    """Resolve parameters, run HDBSCAN, relabel clusters, and report populations."""
    if engine is None:
        engine = execute_clustering_algorithm

    dynamic_size, dynamic_samples = resolve_hdbscan_parameters(
        total_frames, spec.clustering
    )

    auto_status_console = (
        "Active"
        if spec.clustering.automatic_parameters
        else "Inactive (manual parameters)"
    )

    logging.info("\nANALYSIS SETUP")
    for label, value in (
        ("Projection", os.path.basename(data_file)),
        ("Frames", f"{total_frames:,}"),
        ("Dimensions / stride", f"{n_dimensions} / {analysis_step}"),
        ("Temperature", f"{spec.reporting.temperature:g} K"),
        ("HDBSCAN", f"{spec.clustering.cluster_selection_method} | size {dynamic_size} | samples {dynamic_samples}"),
        ("Parameters", "Automatic" if spec.clustering.automatic_parameters else "Manual"),
    ):
        logging.info(f"  {label:<22} {value}")

    logging.info(
        f"\n{Colors.LIGHT_BLUE}[*]{Colors.RESET} Running HDBSCAN clustering..."
    )
    labels, hdbscan_runtime = engine(
        clustering_coordinates,
        dynamic_size,
        dynamic_samples,
        spec.clustering.cluster_selection_method,
    )

    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != total_frames or labels.dtype.kind not in 'iu' or np.any(labels < -1):
        from clustermd.models import PipelineExecutionError
        raise PipelineExecutionError('HDBSCAN returned invalid labels or a label count different from analyzed observations.')
    initial_cluster_count = len(np.unique(labels[labels != -1]))
    logging.info(
        f"{Colors.BLUE}[+]{Colors.RESET} Identified {initial_cluster_count} clusters "
        f"in {hdbscan_runtime:.2f} seconds."
    )

    logging.info(
        f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Sorting cluster assignments by "
        "descending population size..."
    )
    raw_labels = labels
    labels, unique_clusters = sort_cluster_populations(labels)

    clustering = ClusteringResult(
        labels,
        raw_labels,
        {cid: int(np.sum(labels == cid)) for cid in unique_clusters},
        int(np.sum(labels == -1)),
        {
            "min_cluster_size": dynamic_size,
            "min_samples": dynamic_samples,
            "cluster_selection_method": spec.clustering.cluster_selection_method,
        },
        {"hdbscan": hdbscan_runtime},
    )

    noise_frames = np.sum(labels == -1)
    assigned_frames = total_frames - noise_frames

    # Presentation only: determine the first column width from displayed labels.
    cluster_width = max(len("Cluster"), len("Unassigned"),
                        *(len(str(int(cid))) for cid in unique_clusters)) + 2
    console_rows = [(str(int(cid)), int(np.sum(labels == cid))) for cid in unique_clusters]
    console_rows.append(("Unassigned", int(noise_frames)))
    heading = f"    {'Cluster':<{cluster_width}}{'Frames':>12}{'Population':>14}"
    logging.info("\n    CLUSTER POPULATIONS")
    logging.info(heading)
    logging.info("    " + "-" * (len(heading) - 4))
    for label, count in console_rows:
        logging.info(f"    {label:<{cluster_width}}{count:>12,d}{(count / total_frames) * 100:>13.2f}%")

    return ClusteringStageResult(
        clustering=clustering,
        labels=labels,
        unique_clusters=unique_clusters,
        min_cluster_size=dynamic_size,
        min_samples=dynamic_samples,
        assigned_frames=int(assigned_frames),
        noise_frames=int(noise_frames),
    )

