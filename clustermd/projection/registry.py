"""Central projection provider resolution."""

from clustermd.models import PipelineExecutionError
from clustermd.projection.readers import (
    PrecomputedProjectionBackend,
)
from clustermd.projection.gromacs import (
    GromacsProjectionBackend,
)


PROJECTION_BACKENDS = {
    "precomputed": PrecomputedProjectionBackend,
    "gromacs": GromacsProjectionBackend,
}


def resolve_projection_backend(spec):
    """Instantiate the projection backend selected by normalized configuration."""

    backend_name = spec.projection.backend.strip().lower()

    try:
        backend_cls = PROJECTION_BACKENDS[backend_name]
    except KeyError as exc:
        available = ", ".join(
            sorted(PROJECTION_BACKENDS)
        )
        raise PipelineExecutionError(
            f"Unknown projection backend "
            f"'{spec.projection.backend}'. "
            f"Available backends: {available}."
        ) from exc

    return backend_cls()
