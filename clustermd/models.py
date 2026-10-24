"""Backend-independent configuration, observation identities, and result models."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np

class ClusterMDError(Exception):
    """Base exception for clusterMD."""
    pass

class InputDiscoveryError(ClusterMDError):
    """Raised when required input files cannot be located."""
    pass

class ProjectionValidationError(ClusterMDError):
    """Raised when projection input validation fails."""
    pass

class PipelineExecutionError(ClusterMDError):
    """Raised when the pipeline cannot continue."""
    pass

@dataclass
class ProjectionConfig:
    projection_method: str | None = None
    projection_dimensions: int | None = None
    projection_stride: int | None = None
    selection: str | None = None
    backend: str = "precomputed"
    generate: bool = False
    segment: str | None = None
    atom_preset: str | None = None
    index_file: str | None = None
    torsion_file: str | None = None
    provenance: dict = field(default_factory=dict)
    generation_backend: str | None = None
    residues: str | None = None
    frame_start: int | None = None
    frame_stop: int | None = None
    torsion_scope: str | None = None
    carma_executable: str | None = None

@dataclass
class ClusteringConfig:
    clustering_dimensions: int = 0
    analysis_step: int = 1
    min_cluster_size: int = 500
    min_samples: int | None = 50
    cluster_selection_method: str = "eom"
    automatic_parameters: bool = False

@dataclass
class StructuralConfig:
    selection: str | None = None
    rmsd_threshold: float = 0.0
    backend: str = "carma"
    @property
    def segid(self): return self.selection
    @property
    def rmsd(self): return self.rmsd_threshold

@dataclass
class ReportingConfig:
    temperature: float = 298.0
    top_changes: int = 15
    diagnostic_pair: list[int] | None = None
    max_events: int = 3

@dataclass
class AnalysisSpec:
    projection_file: str | None = None
    trajectory_file: str | None = None
    topology_file: str | None = None
    trajectory_discovery: str | None = None
    topology_discovery: str | None = None
    results_dir: str | None = None
    target_analysis_dir: str = "."
    command_line: str = ""
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    cluster_only: bool = False

    @classmethod
    def from_args(cls, args, **context):
        """Normalize CLI/user input after centralized discovery has resolved providers."""
        normalized_context = dict(context)

        # Resolved values supplied by app/discovery take precedence. CLI fields
        # remain fallbacks for direct programmatic callers.
        projection_file = normalized_context.get(
            "projection_file",
            getattr(args, "file", getattr(args, "projection", None)),
        )
        trajectory_file = normalized_context.get(
            "trajectory_file",
            getattr(args, "traj", None),
        )
        topology_file = normalized_context.get(
            "topology_file",
            getattr(args, "topo", getattr(args, "topology", None)),
        )

        if projection_file is not None:
            normalized_context["projection_file"] = projection_file
        if trajectory_file is not None:
            normalized_context["trajectory_file"] = trajectory_file
        if topology_file is not None:
            normalized_context["topology_file"] = topology_file

        projection_backend = normalized_context.pop(
            "projection_backend",
            "precomputed",
        )
        structural_backend = normalized_context.pop(
            "structural_backend",
            "carma",
        )

        generate_projection = normalized_context.pop("generate_projection", False)
        generic_selection = getattr(args, "selection", None)
        structural_selection = (
            generic_selection
            if generic_selection is not None
            else (getattr(args, "segid", None) if structural_backend == "carma" or not generate_projection else None)
        )

        return cls(
            **normalized_context,
            projection=ProjectionConfig(
                backend=projection_backend,
                projection_method=getattr(args, "proj", None) or ("dpca" if generate_projection else None),
                projection_dimensions=getattr(args, "pdim", 5),
                projection_stride=getattr(args, "pstep", 1),
                generate=generate_projection,
                generation_backend=("gromacs" if projection_backend == "gromacs" else "carma") if generate_projection else None,
                segment=getattr(args, "segid", None),
                atom_preset=getattr(args, "atom", None),
                index_file=getattr(args, "index", None),
                torsion_file=getattr(args, "torsions", None),
                residues=getattr(args, "residues", None),
                torsion_scope=getattr(args, "torsion_scope", None),
                carma_executable=getattr(args, "carma_executable", None),
                frame_start=getattr(args, "frame_start", None),
                frame_stop=getattr(args, "frame_stop", None),
            ),
            clustering=ClusteringConfig(
                getattr(args, "dim", 0),
                getattr(args, "step", 1),
                getattr(args, "size", 500),
                getattr(args, "samples", 50),
                getattr(args, "method", "eom"),
                getattr(args, "auto", False),
            ),
            structural=StructuralConfig(
                selection=structural_selection,
                rmsd_threshold=getattr(args, "rmsd", 0.0),
                backend=structural_backend,
            ),
            reporting=ReportingConfig(
                getattr(args, "temp", 298.0),
                getattr(args, "top", 15),
                getattr(args, "transitions", None),
                getattr(args, "events", 3),
            ),
            cluster_only=getattr(args, "cluster_only", False),
        )

    # Transitional views preserve exact existing report/diagnostic formatting.
    # Read-only properties. Mutation is architecturally forbidden.
    @property
    def dim(self): return self.clustering.clustering_dimensions
    @property
    def step(self): return self.clustering.analysis_step
    @property
    def size(self): return self.clustering.min_cluster_size
    @property
    def samples(self): return self.clustering.min_samples
    @property
    def method(self): return self.clustering.cluster_selection_method
    @property
    def auto(self): return self.clustering.automatic_parameters
    @property
    def segid(self): return self.structural.selection
    @property
    def rmsd(self): return self.structural.rmsd_threshold
    @property
    def temp(self): return self.reporting.temperature
    @property
    def top(self): return self.reporting.top_changes
    @property
    def transitions(self): return self.reporting.diagnostic_pair
    @property
    def events(self): return self.reporting.max_events

@dataclass
class ProjectionDataset:
    coordinates: np.ndarray
    sample_indices: np.ndarray
    frame_indices: np.ndarray | None
    times: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    feature_type: str = "unknown"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ClusteringResult:
    labels: np.ndarray
    raw_labels: np.ndarray
    cluster_populations: dict
    unassigned_count: int
    parameters: dict
    timings: dict

@dataclass
class RepresentativeSelection:
    sample_index: int
    frame_index: int
    cluster_local_index: int
    centroid: np.ndarray

@dataclass
class CharacterizationResult:
    representatives: dict = field(default_factory=dict)
    populations: dict = field(default_factory=dict)
    free_energies: dict = field(default_factory=dict)
    label_changes: dict = field(default_factory=dict)

@dataclass
class StructuralRequest:
    projection: ProjectionDataset
    clustering: ClusteringResult
    representatives: dict
    config: StructuralConfig
    target_analysis_dir: str
    results_dir: str
    timings: dict = field(default_factory=dict)
    rmsds: dict = field(default_factory=dict)
    representative_frames: dict = field(default_factory=dict)
    retained_counts: dict = field(default_factory=dict)
    skip_processing: bool = False
    trajectory_file: str | None = None
    topology_file: str | None = None
    trajectory_discovery: str | None = None
    topology_discovery: str | None = None

@dataclass
class StructuralResult:
    info: dict
    rmsds: dict
    representative_frames: dict
    retained_counts: dict
    timings: dict

@dataclass
class ExecutionReportContext:
    """Normalized inputs required to finalize the execution report."""
    results_dir: str
    pipeline_start: float
    process: Any
    total_frames: int
    spec: AnalysisSpec
    n_dimensions: int
    available_dimensions: int
    requested_dimensions: int
    min_cluster_size: int
    min_samples: int | None
    target_analysis_dir: str
    data_file: str
    timing_stats: dict
    unique_clusters: np.ndarray
    labels: np.ndarray
    parse_stats: dict
    structures: StructuralResult
    command_line: str
    skip_label_changes: bool
    skip_structural: bool
    fatal_frame_error: bool
    plot_status: dict

@dataclass
class AnalysisResult:
    projection: ProjectionDataset
    clustering: ClusteringResult
    characterization: CharacterizationResult
    structural: StructuralResult | None
    results_dir: str
    assignment_file: str
    timings: dict
