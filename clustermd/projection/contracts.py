"""Native projection adapter contracts, independent of orchestration."""
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class GeneratedProjection:
    projection_file: Path
    artifacts: list[Path] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
