"""Public CLI definitions and conversion to a normalized analysis specification."""
import argparse
import os
import sys
import shlex

from clustermd.models import AnalysisSpec
from clustermd.validation import validate_cli_arguments


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Automated HDBSCAN clustering, observed change mapping, and "
            "structural extraction pipeline."
        )
    )

    parser.add_argument('--size', type=int, default=500,
                        help='HDBSCAN min_cluster_size (Default: 500)')
    parser.add_argument('--samples', type=int, default=50,
                        help='HDBSCAN min_samples (Default: 50)')
    parser.add_argument('--auto', action='store_true',
                        help='Set min_cluster_size and min_samples using the frame-count heuristic')
    parser.add_argument('--cluster-only', action='store_true',
                        help='Run ingestion and HDBSCAN clustering, export assignments, then exit before downstream analysis.')
    parser.add_argument('--method', type=str, default='eom', choices=['eom', 'leaf'],
                        help='HDBSCAN cluster-selection method (Default: eom)')
    parser.add_argument('--dim', type=int, default=0,
                        help='Number of projected dimensions to analyze; 0 uses all available dimensions')
    parser.add_argument('--rmsd', type=float, default=0.0,
                        help='Optional RMSD threshold for downstream structural filtering in Angstrom; 0 disables filtering')
    parser.add_argument('--temp', type=float, default=298.0,
                        help='Temperature in Kelvin used for population-derived free-energy estimates (Default: 298.0)')
    parser.add_argument('--step', type=int, default=1,
                        help='Frame stride applied during projection-file ingestion (Default: 1)')
    parser.add_argument('--segid', default=None,
                        help='Projection segment (CARMA) or index group (GROMACS); also selects the CARMA structural segment unless overridden by --selection')
    upstream = parser.add_argument_group('Projection generation options')
    upstream.add_argument('--proj', type=str.lower, choices=['pca', 'dpca'], default=None,
                          help='Generate a fresh projection (default dPCA if no existing projection is found)')
    upstream.add_argument('--carma-executable', help='CARMA executable for projection generation only; requires --proj. Default: carma on PATH. Records executable identity; structural CARMA remains separate.')
    upstream.add_argument('--pdim', type=int, default=5, help='Components to generate (default: 5)')
    upstream.add_argument('--pstep', type=int, default=1, help='Trajectory frame stride for generation (default: 1)')
    upstream.add_argument('--torsion-scope', choices=['internal', 'owned'], help='dPCA residue boundary rule: internal requires all four atoms in selected residues; owned includes neighboring atoms. CARMA supports internal only.')
    upstream.add_argument('--residues', help='Projection residue IDs, e.g. 5-20,28; requires --proj. CARMA dPCA uses internal angles; GROMACS defaults to residue-owned angles.')
    upstream.add_argument('--frame-start', type=int, help='First original trajectory frame (one-based, inclusive; generation only)')
    upstream.add_argument('--frame-stop', type=int, help='Last original trajectory frame (one-based, inclusive; generation only)')
    upstream.add_argument('--atom', type=lambda value: value.lower(), choices=['ca', 'backbone', 'heavy'],
                          default=None, help='Cartesian PCA atoms (default: CA); not applicable to dPCA')
    upstream.add_argument('--index', default=None, help='Optional GROMACS atom index groups for --segid')
    upstream.add_argument('--torsions', default=None,
                          help='Optional GROMACS dPCA NDX containing one group of atom quadruplets')
    parser.add_argument('--top', type=int, default=15,
                        help='Number of observed changes to display in the diagnostic table and plots (Default: 15. 0 disables displayed edges)')
    parser.add_argument('--transitions', dest='transitions', nargs='*', type=int,
                        help='Pass 0 args to view the most frequent observed changes, or 2 cluster IDs to inspect a specific cluster pair.')
    parser.add_argument('--events', type=int, default=3,
                        help='Maximum chronological change events to print in diagnostic mode (Default: 3)')

    # Optional input overrides. Backend selection is intentionally NOT public:
    # it is inferred centrally from the resolved file combination.
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help=(
            "Optional existing projection override. If omitted, clusterMD searches "
            "for a compatible CARMA fluctuations DAT or GROMACS projection XVG."
        ),
    )
    parser.add_argument(
        '--traj',
        type=str,
        default=None,
        help=(
            "Optional trajectory override. If omitted, clusterMD searches for a "
            "compatible DCD, XTC or TRR."
        ),
    )
    parser.add_argument(
        '--topo',
        type=str,
        default=None,
        help=(
            "Optional topology/run-input override. If omitted, clusterMD searches "
            "for a compatible PSF, TPR or GRO."
        ),
    )
    parser.add_argument(
        '--selection',
        type=str,
        default=None,
        help=(
            "Structural selection for representative extraction: "
            "CARMA segment or GROMACS atom group (e.g. Backbone)."
        ),
    )

    # Transitional aliases accepted but intentionally hidden from public help.
    parser.add_argument('--projection', dest='file', help=argparse.SUPPRESS)
    parser.add_argument('--topology', dest='topo', help=argparse.SUPPRESS)

    return parser


def parse_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_cli_arguments(parser, args)
    return args


def determine_execution_mode(args, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    clustering_flags = [
        '--cluster-only', '--size', '--samples', '--auto', '--method', '--dim', '--rmsd',
        '--step', '--segid', '--file', '--projection', '--traj',
        '--topo', '--topology', '--selection', '--proj', '--carma-executable', '--pdim', '--pstep', '--atom', '--index', '--torsions', '--residues', '--torsion-scope', '--frame-start', '--frame-stop',
    ]
    is_full_pipeline = True

    if args.transitions is not None:
        option_names = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
        passed_clustering_flags = option_names.intersection(clustering_flags)
        if not passed_clustering_flags:
            is_full_pipeline = False

    return "analysis" if is_full_pipeline else "diagnostic"


def command_line_from_argv(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return " ".join(
        [shlex.quote(os.path.basename(sys.argv[0]))]
        + [shlex.quote(arg) for arg in argv]
    )


def build_analysis_spec(args, **context):
    return AnalysisSpec.from_args(args, **context)
