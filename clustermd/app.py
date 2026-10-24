"""CLI execution, input resolution, output-directory creation, and logging lifecycle."""
import os
import sys
import datetime
import logging
from types import SimpleNamespace

from clustermd.cli import (
    parse_cli,
    determine_execution_mode,
    command_line_from_argv,
    build_analysis_spec,
)
from clustermd.constants import Colors
from clustermd.reporting.formatting import console_header
from clustermd.discovery import resolve_analysis_inputs, discover_diagnostic_file
from clustermd.diagnostics import run_diagnostic_validation
from clustermd.models import ClusterMDError
from clustermd.utils.logging_utils import PlainFileFormatter
from clustermd.workflow import run_analysis


def main(argv=None):
    args = parse_cli(argv)
    mode = determine_execution_mode(args, argv)
    command_line = command_line_from_argv(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    file_handler = None
    try:
        if mode == "analysis":
            logging.info(
                console_header()
            )

            logging.info(
                f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Resolving analysis inputs..."
            )

            resolved = resolve_analysis_inputs(
                file_override=args.file,
                traj_override=args.traj,
                topo_override=args.topo,
                cluster_only=args.cluster_only,
                projection_method=args.proj,
            )

            data_file = resolved.projection_file
            target_analysis_dir = os.path.dirname(data_file) if data_file else os.getcwd()

            logging.info(
                f"  Projection: {data_file or 'will be generated'}"
            )
            logging.info(
                f"  Backend: "
                f"{resolved.source_family}"
            )

            if resolved.trajectory_file:
                logging.info(
                    f"  Trajectory: "
                    f"{resolved.trajectory_file}"
                )
            else:
                logging.info(
                    "  Trajectory: not resolved"
                )

            if resolved.topology_file:
                logging.info(
                    f"  Topology: "
                    f"{resolved.topology_file}"
                )
            else:
                logging.info(
                    "  Topology: not resolved"
                )

            logging.info(
                f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Analysis directory: "
                f"{target_analysis_dir}"
            )

            timestamp = datetime.datetime.now().strftime(
                "%d.%m.%Y_%H.%M.%S.%f"
            )[:-3]
            results_dir = f"clusterMD_results_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)

            file_handler = logging.FileHandler(
                os.path.join(results_dir, "clustering_execution.log")
            )
            file_handler.name = "clusterMD_file"
            file_handler.setFormatter(PlainFileFormatter('%(message)s'))
            logging.getLogger().addHandler(file_handler)

            logging.info(f"Command line: {command_line}")
            logging.info(
                f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Output directory: "
                f"{results_dir}"
            )

            run_analysis(
                build_analysis_spec(
                    args,
                    projection_file=resolved.projection_file,
                    generate_projection=resolved.generate_projection,
                    trajectory_file=resolved.trajectory_file,
                    topology_file=resolved.topology_file,
                    trajectory_discovery=resolved.trajectory_discovery,
                    topology_discovery=resolved.topology_discovery,
                    projection_backend=resolved.projection_backend,
                    structural_backend=resolved.structural_backend,
                    results_dir=results_dir,
                    target_analysis_dir=target_analysis_dir,
                    command_line=command_line,
                )
            )

        else:
            logging.info(console_header(title="Post-Run Observed-Change Inspection"))
            dat_file = discover_diagnostic_file()

            logging.info(
                f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Target data located: "
                f"{os.path.abspath(dat_file)}"
            )

            spec = build_analysis_spec(args)
            diagnostic_args = SimpleNamespace(
                transitions=spec.reporting.diagnostic_pair,
                top=spec.reporting.top_changes,
                events=spec.reporting.max_events,
            )
            run_diagnostic_validation(diagnostic_args, dat_file)

    except ClusterMDError as e:
        logging.error(
            f"{Colors.RED}[-]{Colors.RESET} Critical error: {e}"
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logging.error(
            f"\n{Colors.RED}[-]{Colors.RESET} Process interrupted by user. "
            "Shutting down gracefully."
        )
        sys.exit(130)

    finally:
        if file_handler is not None:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

    return 0
