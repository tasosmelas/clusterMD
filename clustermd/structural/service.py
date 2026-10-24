"""Structural-stage orchestration over backend-independent models."""

import logging
import time

from clustermd.constants import Colors
from clustermd.models import ProjectionDataset, StructuralRequest
from clustermd.characterization.representatives import select_representatives
from clustermd.structural.registry import resolve_structural_backend


def run_structural_stage(
    *,
    projection,
    clustering,
    clustering_coordinates,
    labels,
    spec,
    target_analysis_dir,
    results_dir,
    skip_structural,
    representatives=None,
):
    """Prepare representatives, execute the structural backend, and own its bookkeeping."""
    has_complete_structural_pair = bool(
        spec.trajectory_file and spec.topology_file
    )
    effective_skip_structural = bool(
        skip_structural or not has_complete_structural_pair
    )

    if not skip_structural and not has_complete_structural_pair:
        missing = []
        if not spec.trajectory_file:
            missing.append("trajectory")
        if not spec.topology_file:
            missing.append("topology")
        logging.warning(
            f"{Colors.YELLOW}[WARN]{Colors.RESET} Structural processing skipped: "
            f"no complete {spec.structural.backend} structural input pair was resolved "
            f"(missing {', '.join(missing)}). Clustering and characterization remain valid."
        )

    if representatives is None:
        # Compatibility for direct service callers. Workflow supplies characterization results.
        retained_projection = ProjectionDataset(
            coordinates=clustering_coordinates,
            sample_indices=projection.sample_indices,
            frame_indices=projection.frame_indices,
        )
        representatives = select_representatives(retained_projection, labels)

    request = StructuralRequest(
        projection=projection, clustering=clustering, representatives=representatives,
        config=spec.structural, target_analysis_dir=target_analysis_dir, results_dir=results_dir,
        skip_processing=effective_skip_structural, trajectory_file=spec.trajectory_file,
        topology_file=spec.topology_file,
        trajectory_discovery=spec.trajectory_discovery, topology_discovery=spec.topology_discovery,
    )

    backend = resolve_structural_backend(spec)

    extract_elapsed = 0.0
    if not effective_skip_structural:
        logging.info(
            f"\n{Colors.LIGHT_BLUE}[*]{Colors.RESET} "
            f"Extracting cluster-associated structural ensembles via {spec.structural.backend}..."
        )
        t0_extract = time.time()
        structures = backend.process(request)
        extract_elapsed = time.time() - t0_extract
    else:
        structures = backend.process(request)

    return structures, representatives, extract_elapsed
