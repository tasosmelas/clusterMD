"""CLI, normalized configuration and downstream identity validation."""
import math
from numbers import Integral, Real
def validate_cli_arguments(parser, args):
    if getattr(args, "pdim", 5) < 1 or getattr(args, "pstep", 1) < 1:
        parser.error("--pdim and --pstep must be positive integers.")
    if getattr(args, "proj", None) and args.file:
        parser.error("--file and --proj select different projection modes; supply one.")
    if getattr(args, "proj", None) and (not args.traj or not args.topo):
        parser.error("--proj requires explicit --traj and --topo.")
    if getattr(args, "proj", None) == "dpca" and getattr(args, "atom", None):
        parser.error("--atom selects Cartesian PCA atoms; dPCA uses torsions.")
    from clustermd.projection.selection import validate_generation_options
    from clustermd.models import ProjectionValidationError
    try:
        validate_generation_options(getattr(args, 'residues', None),
            getattr(args, 'frame_start', None), getattr(args, 'frame_stop', None),
            bool(getattr(args, 'proj', None)), getattr(args, 'torsions', None), getattr(args, 'torsion_scope', None))
        if getattr(args, 'torsion_scope', None) is not None and args.proj != 'dpca':
            raise ProjectionValidationError('--torsion-scope requires --proj dpca.')
    except ProjectionValidationError as exc:
        parser.error(str(exc))
    if getattr(args, 'carma_executable', None) is not None:
        if not getattr(args, 'proj', None) or not args.carma_executable.strip():
            parser.error('--carma-executable requires explicit --proj and a nonempty executable path/name.')
    # Numerical/public CLI validation.
    if args.size <= 0:
        parser.error("--size must be a positive integer.")
    if args.samples is not None and args.samples <= 0:
        parser.error("--samples must be a positive integer.")
    if args.step <= 0:
        parser.error("--step must be a positive integer.")
    if not math.isfinite(args.temp) or args.temp <= 0:
        parser.error("--temp must be strictly positive (K).")
    if not math.isfinite(args.rmsd) or args.rmsd < 0:
        parser.error("--rmsd must be non-negative.")
    if args.dim < 0:
        parser.error("--dim must be non-negative (0 for all dimensions).")
    if args.top < 0:
        parser.error(
            "--top must be non-negative (0 disables displayed connectivity edges)."
        )

    if args.transitions is not None and len(args.transitions) not in [0, 2]:
        parser.error(
            "--transitions requires either 0 arguments (overview) "
            "or 2 cluster IDs (inspect a specific pair)."
        )

    generic_selection = getattr(args, "selection", None)
    requested_segid = getattr(args, "segid", None)

    if generic_selection is not None and requested_segid is not None and not (getattr(args, "proj", None) or (args.traj and args.topo and not args.file)):
        parser.error("--selection and --segid cannot be used together.")

    # File existence, compatibility, discovery, and backend selection are
    # intentionally centralized in clustermd.discovery.resolve_analysis_inputs().


from dataclasses import dataclass
import logging

from clustermd.constants import Colors


@dataclass(frozen=True)
class FrameValidationState:
    """Downstream gating derived from sampled frame identifiers."""
    has_duplicates: bool
    non_monotonic: bool
    fatal_frame_error: bool
    skip_label_changes: bool
    skip_structural: bool


def assess_frame_validation(parse_stats):
    """Interpret sampled-frame validation metadata without changing clustering validity."""
    has_duplicates = bool(parse_stats['sampled_duplicates'])
    non_monotonic = not bool(parse_stats['sampled_monotonic'])
    return FrameValidationState(
        has_duplicates=has_duplicates,
        non_monotonic=non_monotonic,
        fatal_frame_error=has_duplicates or non_monotonic,
        skip_label_changes=has_duplicates or non_monotonic,
        skip_structural=has_duplicates,
    )


def log_frame_validation_warnings(state):
    """Report frame-mapping warnings before cluster-only processing."""
    if state.has_duplicates:
        logging.warning(f"\n{Colors.YELLOW}[WARN]{Colors.RESET} Duplicate sampled frame identifiers detected.")
        logging.warning("       Clustering assignments are valid in coordinate space,")
        logging.warning("       but one-to-one structural frame mapping is ambiguous.")
    if state.non_monotonic:
        prefix = "" if state.has_duplicates else "\n"
        logging.warning(f"{prefix}{Colors.YELLOW}[WARN]{Colors.RESET} Sampled frame identifiers are not strictly increasing.")
        logging.warning("       Clustering assignments are valid in coordinate space,")
        logging.warning("       but chronological interpretation of this assignment file is not valid.")


def log_frame_validation_failures(state):
    """Report frame-mapping failures during full analysis."""
    if state.has_duplicates:
        logging.error(f"\n{Colors.RED}[-]{Colors.RESET} Critical error: Duplicate sampled frame identifiers detected.")
    if state.non_monotonic:
        prefix = "" if state.has_duplicates else "\n"
        logging.error(f"{prefix}{Colors.RED}[-]{Colors.RESET} Critical error: Sampled frame identifiers are not strictly increasing.")
    if state.skip_label_changes and state.skip_structural:
        logging.warning(f"{Colors.YELLOW}[*]{Colors.RESET} Bypassing observed label-change analysis, connectivity visualizations, and structural extraction.")
    elif state.skip_label_changes:
        logging.warning(f"{Colors.YELLOW}[*]{Colors.RESET} Bypassing observed label-change analysis and connectivity visualizations.")



def validate_analysis_spec(spec):
    """Reject invalid API configuration independently of CLI parsing."""
    from clustermd.models import ProjectionValidationError
    c, p, s, r = spec.clustering, spec.projection, spec.structural, spec.reporting
    def integer(value, name, minimum):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ProjectionValidationError(f"{name} must be an integer >= {minimum}.")
    integer(c.analysis_step, 'analysis_step', 1)
    integer(c.clustering_dimensions, 'clustering_dimensions', 0)
    integer(c.min_cluster_size, 'min_cluster_size', 1)
    if c.min_samples is not None:
        integer(c.min_samples, 'min_samples', 1)
    if c.cluster_selection_method not in {'eom', 'leaf'}:
        raise ProjectionValidationError('cluster_selection_method must be eom or leaf.')
    if not isinstance(r.temperature, Real) or not math.isfinite(r.temperature) or r.temperature <= 0:
        raise ProjectionValidationError('temperature must be finite and strictly positive (K).')
    if not isinstance(s.rmsd_threshold, Real) or not math.isfinite(s.rmsd_threshold) or s.rmsd_threshold < 0:
        raise ProjectionValidationError('rmsd_threshold must be finite and non-negative (Angstrom).')
    integer(r.top_changes, 'top_changes', 0)
    from clustermd.projection.selection import validate_generation_options
    validate_generation_options(p.residues, p.frame_start, p.frame_stop, p.generate, p.torsion_file, p.torsion_scope)
    if p.torsion_scope is not None and p.projection_method != "dpca":
        raise ProjectionValidationError("--torsion-scope requires --proj dpca.")
    if p.carma_executable is not None:
        if not p.generate or not isinstance(p.carma_executable, str) or not p.carma_executable.strip():
            raise ProjectionValidationError('--carma-executable requires projection generation and a nonempty path/name.')
        if (p.generation_backend or p.backend) == 'gromacs':
            raise ProjectionValidationError('--carma-executable applies only to CARMA projection generation.')
    if p.generate:
        if p.projection_method not in {'pca', 'dpca'}:
            raise ProjectionValidationError('Generated projection method must be pca or dpca.')
        integer(p.projection_dimensions, 'projection_dimensions', 1)
        integer(p.projection_stride, 'projection_stride', 1)
        if p.atom_preset not in {None, 'ca', 'backbone', 'heavy'}:
            raise ProjectionValidationError('atom_preset must be ca, backbone or heavy.')
