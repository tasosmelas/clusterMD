"""Structural backend protocol."""

from typing import Protocol

from clustermd.models import StructuralRequest, StructuralResult


class StructuralBackend(Protocol):
    """Interface implemented by structural-processing providers."""

    def process(self, request: StructuralRequest) -> StructuralResult: ...
