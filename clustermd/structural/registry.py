"""Central structural provider resolution."""

from clustermd.models import PipelineExecutionError
from clustermd.structural.carma import CarmaStructuralBackend
from clustermd.structural.gromacs import GromacsStructuralBackend


STRUCTURAL_BACKENDS = {
    "carma": CarmaStructuralBackend,
    "gromacs": GromacsStructuralBackend,
}


def resolve_structural_backend(spec):
    """Instantiate the structural backend selected by normalized configuration."""
    backend_name = spec.structural.backend.strip().lower()

    try:
        backend_cls = STRUCTURAL_BACKENDS[backend_name]
    except KeyError as exc:
        available = ", ".join(sorted(STRUCTURAL_BACKENDS))
        raise PipelineExecutionError(
            f"Unknown structural backend '{spec.structural.backend}'. "
            f"Available backends: {available}."
        ) from exc

    return backend_cls()