"""Projection backend protocol."""

from typing import Protocol

from clustermd.models import AnalysisSpec, ProjectionDataset


class ProjectionBackend(Protocol):
    """Interface implemented by projection providers."""

    def create_projection(self, spec: AnalysisSpec) -> ProjectionDataset: ...
