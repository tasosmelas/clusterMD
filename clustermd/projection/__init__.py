"""Projection-stage orchestration for normalized clustering input."""

from dataclasses import dataclass
import logging
import time

from clustermd.constants import Colors
from clustermd.models import ProjectionDataset, ProjectionValidationError
from clustermd.projection.registry import resolve_projection_backend


@dataclass(frozen=True)
class PreparedProjection:
    """Projection data plus the retained coordinate view used for clustering."""
    projection: ProjectionDataset
    parse_stats: dict
    frames: object
    clustering_coordinates: object
    available_dimensions: int
    requested_dimensions: int
    analyzed_dimensions: int
    total_frames: int
    load_runtime: float
    generation_runtime: float = 0.0


def prepare_projection_for_clustering(spec):
    """Load, validate, and retain the requested clustering dimensions."""
    from clustermd.validation import validate_analysis_spec
    validate_analysis_spec(spec)
    generation_runtime = 0.0
    if spec.projection.generate:
        from clustermd.projection.generation import generate_projection
        generation_start = time.perf_counter()
        generate_projection(spec)
        generation_runtime = time.perf_counter() - generation_start
    t0 = time.perf_counter()
    try:
        projection = resolve_projection_backend(spec).create_projection(spec)
        if spec.projection.generate:
            from clustermd.projection.generation import validate_generated_dataset
            validate_generated_dataset(spec, projection)
    except Exception as exc:
        if spec.projection.generate and spec.projection.provenance:
            from clustermd.projection.generation import _save_provenance
            spec.projection.provenance.update(status='validation_failed', error=str(exc))
            _save_provenance(spec)
        raise
    import numpy as np
    coordinates = np.asarray(projection.coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 1 or not np.isfinite(coordinates).all():
        raise ProjectionValidationError('Projection coordinates must be a finite matrix with at least one component.')
    if len(projection.sample_indices) != len(coordinates) or len(np.unique(projection.sample_indices)) != len(coordinates):
        raise ProjectionValidationError('Projection observation identities must be unique and match coordinate rows.')
    samples = np.asarray(projection.sample_indices)
    frames = np.asarray(projection.frame_indices)
    if samples.ndim != 1 or samples.dtype.kind not in 'iu' or np.any(samples < 0):
        raise ProjectionValidationError('Projection observation identities must be non-negative integer ordinals.')
    if frames.ndim != 1 or len(frames) != len(coordinates) or frames.dtype.kind not in 'iu' or np.any(frames < 0):
        raise ProjectionValidationError('Projection frame identities must be non-negative integer IDs matching coordinate rows.')
    parse_stats = projection.metadata["parse_stats"]

    if len(projection.coordinates) < 2:
        raise ProjectionValidationError(
            f"Insufficient valid frames sampled ({len(projection.coordinates)}). "
            "Minimum 2 required."
        )

    if spec.clustering.analysis_step > 1:
        logging.info(
            f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Applied frame stride of "
            f"{spec.clustering.analysis_step} during ingestion."
        )

    frames = projection.frame_indices
    clustering_coordinates = projection.coordinates
    available_dims = clustering_coordinates.shape[1]

    requested_dim = spec.clustering.clustering_dimensions
    effective_dim = requested_dim
    if effective_dim > 0:
        if effective_dim > available_dims:
            logging.warning(
                f"{Colors.YELLOW}[WARN]{Colors.RESET} Requested --dim "
                f"{requested_dim} exceeds available dimensions ({available_dims}). "
                f"Capping to {available_dims}."
            )
            effective_dim = available_dims

        clustering_coordinates = clustering_coordinates[:, :effective_dim]
        logging.info(
            f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Constraining multidimensional "
            f"analysis to the primary {effective_dim} dimensions."
        )

    n_dimensions = clustering_coordinates.shape[1]
    total_frames = clustering_coordinates.shape[0]
    load_runtime = time.perf_counter() - t0

    logging.info(
        f"{Colors.BLUE}[+]{Colors.RESET} Data ingested successfully: "
        f"{total_frames:,} frames ({n_dimensions}D)."
    )

    return PreparedProjection(
        projection=projection,
        parse_stats=parse_stats,
        frames=frames,
        clustering_coordinates=clustering_coordinates,
        available_dimensions=available_dims,
        requested_dimensions=requested_dim,
        analyzed_dimensions=n_dimensions,
        total_frames=total_frames,
        load_runtime=load_runtime,
        generation_runtime=generation_runtime,
    )
