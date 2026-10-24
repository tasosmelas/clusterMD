"""Shared GROMACS execution and verified time-to-frame mapping."""
from pathlib import Path
import logging
import shlex
import re
import tempfile
import numpy as np
from clustermd.models import PipelineExecutionError
from clustermd.utils.logging_utils import get_clusterMD_file_handler


def log_gromacs_detail(message, *args):
    """Send backend detail only to the designated execution-log handler."""
    handler = get_clusterMD_file_handler()
    if handler is not None:
        record = logging.getLogger().makeRecord(__name__, logging.INFO, __file__, 0, message, args, None)
        handler.handle(record)
        handler.flush()
    else:
        logging.getLogger(__name__).debug(message, *args)


def run_gmx(args, *, selections=None, cwd=None):
    stdin = ''.join(f'{s}\n' for s in selections) if selections else None
    log_gromacs_detail("GROMACS command: %s; stdin groups: %s", shlex.join(['gmx', *args]), selections or [])
    from clustermd.utils.native import run_native
    return run_native(['gmx', *args], stdin=stdin, cwd=cwd,
                      log_path=Path(cwd) / 'native_commands.log' if cwd else None)


def validate_xtc_gro_atom_counts(trajectory, topology):
    """Reject atom-count mismatches for XTC trajectories paired with GRO structures.

    Other format combinations retain the existing GROMACS behavior; this check
    does not establish atom identities or atom ordering.
    """
    trajectory = Path(trajectory)
    topology = Path(topology)
    if trajectory.suffix.lower() != ".xtc" or topology.suffix.lower() != ".gro":
        return

    result = run_gmx(["check", "-f", str(trajectory.resolve())])
    matches = re.findall(r"(?m)^\s*#\s*Atoms\s+(\d+)\s*$", result.stdout)
    if len(matches) != 1:
        raise PipelineExecutionError(
            f"Could not determine XTC atom count from gmx check: {trajectory}"
        )
    trajectory_atoms = int(matches[0])

    try:
        with topology.open() as handle:
            next(handle)  # GRO title
            topology_atoms = int(next(handle).strip())
    except (OSError, StopIteration, ValueError) as exc:
        raise PipelineExecutionError(
            f"Could not read declared GRO atom count: {topology}"
        ) from exc

    if trajectory_atoms != topology_atoms:
        raise PipelineExecutionError(
            "GROMACS structural input atom-count mismatch: "
            f"trajectory {trajectory.name} has {trajectory_atoms} atoms; "
            f"topology {topology.name} has {topology_atoms} atoms."
        )


def read_trajectory_times(trajectory, topology):
    # One COM per frame keeps this scan small; coordinates are discarded.
    with tempfile.TemporaryDirectory(prefix='clustermd-times-') as directory:
        path = Path(directory) / 'times.xvg'
        run_gmx(['traj', '-s', str(Path(topology).resolve()),
                 '-f', str(Path(trajectory).resolve()), '-ox', str(path),
                 '-com', '-nopbc', '-fp', '-tu', 'ps', '-xvg', 'none'],
                selections=['System'])
        times = []
        with path.open() as handle:
            for line in handle:
                if line.strip() and not line.lstrip().startswith(('#', '@')):
                    try:
                        times.append(float(line.split()[0]))
                    except ValueError as exc:
                        raise PipelineExecutionError('Malformed GROMACS trajectory time axis.') from exc
    return validate_times(times)


# Re-export shared frame-mapping helpers.
from clustermd.frame_mapping import (
    validate_times, map_times_to_frames, check_output_times, write_frame_selector,
)
