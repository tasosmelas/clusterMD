"""GROMACS structural backend.

clusterMD owns cluster membership and representative selection.
GROMACS is used only as a structural trajectory provider/processor.
"""

from __future__ import annotations

import logging
import sys
import shutil
import tempfile
from contextlib import contextmanager
import time
from pathlib import Path

import numpy as np
from clustermd.utils.gromacs import (
    run_gmx as _run_gmx, read_trajectory_times, map_times_to_frames,
    check_output_times, write_frame_selector, log_gromacs_detail,
    validate_xtc_gro_atom_counts,
)

from clustermd.models import (
    PipelineExecutionError,
    StructuralRequest,
    StructuralResult,
)


def _superposition_sample_indices(frame_count, limit=500):
    """Select evenly spaced retained-frame ordinals up to the sample limit."""
    if frame_count < 1 or limit < 1:
        raise PipelineExecutionError("Superposition frame count and limit must be positive.")
    return np.linspace(0, frame_count - 1, min(frame_count, limit), dtype=np.int64)


@contextmanager
def _cluster_progress(cluster_id, position, total):
    """Replace the in-progress terminal line when a cluster finishes."""
    prefix = f"    -> Cluster {cluster_id:4d} ({position:4d}/{total}) : "
    initial = prefix + "Processing structural ensemble..."
    sys.stdout.write("\r" + initial)
    sys.stdout.flush()
    log_gromacs_detail(initial)
    completed = False

    def finish(message):
        nonlocal completed
        final = prefix + message
        padding = " " * max(0, len(initial) - len(final))
        sys.stdout.write("\r" + final + padding + "\n")
        sys.stdout.flush()
        log_gromacs_detail(final)
        completed = True

    try:
        yield finish
    finally:
        if not completed:
            # Leave warnings/errors on a fresh terminal line on failure.
            sys.stdout.write("\n")
            sys.stdout.flush()


def _read_coordinate_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read gmx traj -ox output.

    Returns
    -------
    times
        Shape (n_frames,)
    coordinates
        Shape (n_frames, n_atoms, 3), in nm.
    """

    rows = []

    with path.open() as fh:
        for line in fh:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith(("#", "@")):
                continue

            try:
                values = [float(value) for value in stripped.split()]
            except ValueError as exc:
                raise PipelineExecutionError(
                    f"Malformed coordinate XVG row in {path}: {stripped}"
                ) from exc

            rows.append(values)

    if not rows:
        raise PipelineExecutionError(
            f"No coordinate data found in {path}."
        )

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise PipelineExecutionError(
            f"Inconsistent coordinate row widths in {path}."
        )

    data = np.asarray(rows, dtype=float)

    if not np.isfinite(data).all():
        raise PipelineExecutionError(
            f"Non-finite coordinate values found in {path}."
        )

    times = data[:, 0]
    flat_coordinates = data[:, 1:]

    if flat_coordinates.shape[1] == 0 or flat_coordinates.shape[1] % 3 != 0:
        raise PipelineExecutionError(
            f"Coordinate column count "
            f"{flat_coordinates.shape[1]} is not divisible by three."
        )

    n_atoms = flat_coordinates.shape[1] // 3
    coordinates = flat_coordinates.reshape(len(data), n_atoms, 3)

    return times, coordinates


def _write_average_pdb(
    template_pdb: Path,
    output_pdb: Path,
    average_coordinates_nm: np.ndarray,
) -> None:
    """Write arithmetic mean Cartesian coordinates using template PDB records."""

    coordinates = np.asarray(average_coordinates_nm, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise PipelineExecutionError(
            "Average coordinates must have shape (n_atoms, 3)."
        )

    with template_pdb.open() as fh:
        lines = fh.readlines()

    atom_lines = [
        index
        for index, line in enumerate(lines)
        if line.startswith(("ATOM  ", "HETATM"))
    ]

    if len(atom_lines) != len(coordinates):
        raise PipelineExecutionError(
            f"Template PDB contains {len(atom_lines)} atoms, "
            f"but the fitted trajectory contains {len(coordinates)}."
        )

    # GROMACS coordinate XVG: nm
    # PDB coordinate columns: Angstrom
    coordinates_angstrom = coordinates * 10.0

    for atom_index, line_index in enumerate(atom_lines):
        x, y, z = coordinates_angstrom[atom_index]
        line = lines[line_index]

        lines[line_index] = (
            line[:30]
            + f"{x:8.3f}{y:8.3f}{z:8.3f}"
            + line[54:]
        )

    with output_pdb.open("w") as fh:
        fh.writelines(lines)


def _read_rmsd_xvg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read native RMSD output as physical times (ps) and RMSDs (nm)."""
    times = []
    rmsds = []

    with path.open() as fh:
        for line in fh:
            stripped = line.strip()

            if not stripped or stripped.startswith(("#", "@")):
                continue

            fields = stripped.split()

            if len(fields) < 2:
                raise PipelineExecutionError(
                    f"Malformed RMSD XVG row in {path}: {stripped}"
                )

            try:
                time_value = float(fields[0])
                rmsd_value = float(fields[1])
            except ValueError as exc:
                raise PipelineExecutionError(
                    f"Non-numeric RMSD XVG row in {path}: {stripped}"
                ) from exc

            times.append(time_value)
            rmsds.append(rmsd_value)

    if not rmsds:
        raise PipelineExecutionError(
            f"No RMSD values found in {path}."
        )

    if not np.isfinite(times).all() or not np.isfinite(rmsds).all() or np.any(np.asarray(rmsds) < 0):
        raise PipelineExecutionError(f"Non-finite or negative RMSD data in {path}.")
    return (np.asarray(times, dtype=float), np.asarray(rmsds, dtype=float))


class GromacsStructuralBackend:
    """Structural processing using GROMACS trajectory tools."""

    def process(self, request: StructuralRequest) -> StructuralResult:
        return self._process(request, prepare_whole=True)

    def _process(self, request: StructuralRequest, *, prepare_whole) -> StructuralResult:
        from dataclasses import replace
        request = replace(request, timings={'sort': 0.0, 'rmsd': 0.0, 'super': 0.0},
                          rmsds={}, representative_frames={}, retained_counts={})
        if request.skip_processing:
            return StructuralResult(
                {},
                request.rmsds,
                request.representative_frames,
                request.retained_counts,
                request.timings,
            )

        if not request.trajectory_file:
            raise PipelineExecutionError(
                "GROMACS structural processing requires trajectory_file."
            )

        if not request.topology_file:
            raise PipelineExecutionError(
                "GROMACS structural processing requires topology_file."
            )

        trajectory = Path(request.trajectory_file).resolve()
        topology = Path(request.topology_file).resolve()

        if not trajectory.is_file():
            raise PipelineExecutionError(
                f"GROMACS trajectory not found: {trajectory}"
            )

        if not topology.is_file():
            raise PipelineExecutionError(
                f"GROMACS topology/run-input file not found: {topology}"
            )

        if request.projection.frame_indices is None:
            raise PipelineExecutionError(
                "GROMACS structural processing requires trajectory frame indices."
            )

        if request.projection.times is None:
            raise PipelineExecutionError(
                "GROMACS structural processing requires physical trajectory times."
            )

        # Experimental default for the current PCA experiment.
        # Later this should become explicit backend-neutral selection provenance.
        selection = request.config.selection or "Backbone"

        labels = request.clustering.labels
        frame_indices = np.asarray(request.projection.frame_indices)
        times = np.asarray(request.projection.times)

        if len(labels) != len(frame_indices):
            raise PipelineExecutionError(
                "Cluster labels and structural frame indices have different lengths."
            )

        if len(times) != len(frame_indices):
            raise PipelineExecutionError(
                "Projection times and structural frame indices have different lengths."
            )

        validate_xtc_gro_atom_counts(trajectory, topology)

        trajectory_times = request.projection.metadata.get("trajectory_times")
        if trajectory_times is None:
            trajectory_times = read_trajectory_times(trajectory, topology)
        mapped_frames = map_times_to_frames(times, trajectory_times)
        if not request.projection.metadata.get("frame_mapping_verified") or not np.array_equal(frame_indices, mapped_frames):
            raise PipelineExecutionError("Structural processing requires verified trajectory frame mapping from the projection provider.")
        if not np.isfinite(request.config.rmsd_threshold) or request.config.rmsd_threshold < 0:
            raise PipelineExecutionError("RMSD threshold must be finite and non-negative.")

        if prepare_whole and topology.suffix.lower() == '.tpr' and np.any(labels != -1):
            with tempfile.TemporaryDirectory(prefix='clustermd-structural-whole-') as directory:
                repaired = Path(directory) / 'whole_input.trr'
                _run_gmx(['trjconv', '-s', str(topology), '-f', str(trajectory),
                          '-o', str(repaired), '-pbc', 'mol'], selections=['System'], cwd=Path(directory))
                repaired_times = read_trajectory_times(repaired, topology)
                check_output_times(repaired_times, trajectory_times, 'structural whole-molecule preparation')
                result = self._process(replace(request, trajectory_file=str(repaired)), prepare_whole=False)
                for info in result.info.values():
                    info['pbc_policy'] = 'TPR whole-molecule repair before structural extraction; all source frames preserved'
                    info['source_trajectory'] = str(trajectory)
                return result

        output_root = Path(request.results_dir).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        structural_info = {}

        cluster_ids = np.unique(labels[labels != -1])

        for cluster_position, cluster_id in enumerate(cluster_ids, start=1):
            cluster_id = int(cluster_id)
            with _cluster_progress(cluster_id, cluster_position, len(cluster_ids)) as finish_progress, \
                    tempfile.TemporaryDirectory(prefix=f"clustermd-cluster-{cluster_id:04d}-") as work_dir:
                cluster_mask = labels == cluster_id
                cluster_frames = frame_indices[cluster_mask]
                cluster_times = times[cluster_mask]

                representative = request.representatives.get(cluster_id)

                if representative is None:
                    raise PipelineExecutionError(
                        f"No clusterMD representative is available for cluster {cluster_id}."
                    )

                positions = np.flatnonzero(request.projection.sample_indices == representative.sample_index)
                if len(positions) != 1:
                    raise PipelineExecutionError(f"Representative identity for cluster {cluster_id} is ambiguous or missing.")
                representative_sample = int(positions[0])
                if labels[representative_sample] != cluster_id or int(representative.frame_index) != int(frame_indices[representative_sample]):
                    raise PipelineExecutionError(f"Representative for cluster {cluster_id} is not a mapped cluster member.")
                representative_time = float(times[representative_sample])

                cluster_dir = Path(work_dir)

                def _run_cluster_gmx(args, *, selections):
                    return _run_gmx(args, selections=selections, cwd=cluster_dir)

                cluster_xtc = cluster_dir / "cluster.xtc"
                representative_pdb = cluster_dir / f"cluster_{cluster_id:04d}_representative.pdb"
                fitted_xtc = cluster_dir / "cluster_fitted.xtc"
                coordinate_xvg = cluster_dir / "fitted_coordinates.xvg"
                average_pdb = cluster_dir / f"cluster_{cluster_id:04d}_average.pdb"
                rmsd_xvg = cluster_dir / "rmsd_to_representative.xvg"

                t0 = time.time()

                frame_selector = cluster_dir / "frame_selector.xvg"
                representative_selector = cluster_dir / "representative_selector.xvg"
                write_frame_selector(frame_selector, trajectory_times, cluster_frames)
                write_frame_selector(representative_selector, trajectory_times, frame_indices[[representative_sample]])

                # Exact arbitrary cluster membership, checked against output timestamps.
                _run_cluster_gmx(
                    [
                        "trjconv",
                        "-s", str(topology),
                        "-f", str(trajectory),
                        "-drop", str(frame_selector), "-dropunder", "0.5",
                        "-o", str(cluster_xtc),
                    ],
                    selections=[selection],
                )

                # Exact representative already selected by clusterMD.
                _run_cluster_gmx(
                    [
                        "trjconv",
                        "-s", str(topology),
                        "-f", str(trajectory),
                        "-o", str(representative_pdb),
                        "-drop", str(representative_selector), "-dropunder", "0.5",
                    ],
                    selections=[selection],
                )

                request.timings["sort"] += time.time() - t0

                t0 = time.time()

                # Fit every cluster member onto the fixed clusterMD representative.
                _run_cluster_gmx(
                    [
                        "trjconv",
                        "-s", str(representative_pdb),
                        "-f", str(cluster_xtc),
                        "-o", str(fitted_xtc),
                        "-fit", "rot+trans",
                    ],
                    selections=["System", "System"],
                )

                # Verify that the aligned ensemble still has exact membership and
                # physical timestamps before filtering or exporting any files.
                _run_cluster_gmx(
                    ["traj", "-s", str(representative_pdb), "-f", str(fitted_xtc),
                     "-ox", str(coordinate_xvg), "-x", "-y", "-z",
                     "-nocom", "-nopbc", "-fp", "-tu", "ps"],
                    selections=["System"],
                )
                coordinate_times, coordinates = _read_coordinate_xvg(coordinate_xvg)
                if len(coordinates) != len(cluster_frames):
                    raise PipelineExecutionError(
                        f"GROMACS returned {len(coordinates)} fitted structures "
                        f"for cluster {cluster_id}, but {len(cluster_frames)} were assigned.")
                check_output_times(coordinate_times, cluster_times, "Fitted coordinates")

                # Frames are already aligned to the fixed representative.
                # Do not construct an arithmetic-average PDB; the representative
                # is the sole RMSD reference throughout the entire run.
                request.timings["super"] += time.time() - t0

                t0 = time.time()

                # The trajectory is already aligned to clusterMD's representative.
                _run_cluster_gmx(
                    [
                        "rms",
                        "-s", str(representative_pdb),
                        "-f", str(fitted_xtc),
                        "-o", str(rmsd_xvg),
                        "-fit", "none", "-nomw",
                    ],
                    selections=["System"],
                )

                rmsd_times, rmsds = _read_rmsd_xvg(rmsd_xvg)

                if len(rmsds) != len(cluster_frames):
                    raise PipelineExecutionError(
                        f"GROMACS returned {len(rmsds)} RMSD values for "
                        f"cluster {cluster_id}, but clusterMD assigned "
                        f"{len(cluster_frames)} observations."
                    )

                check_output_times(rmsd_times, cluster_times, "RMSD")
                request.timings["rmsd"] += time.time() - t0

                # gmx rms reports nm; clusterMD accepts and reports RMSD thresholds in Angstrom.
                pre_filter_rmsds_nm = np.asarray(rmsds, dtype=float)
                final_rmsds_nm = pre_filter_rmsds_nm
                retained_count = int(len(cluster_frames))
                retained_trajectory = fitted_xtc
                retained_times = cluster_times
                retained_frame_indices = cluster_frames

                rmsd_threshold_angstrom = float(request.config.rmsd_threshold)

                if rmsd_threshold_angstrom > 0.0:
                    rmsds_angstrom = pre_filter_rmsds_nm * 10.0
                    core_mask = rmsds_angstrom <= rmsd_threshold_angstrom
                    candidate_retained_count = int(np.sum(core_mask))

                    if candidate_retained_count > 0:
                        retained_count = candidate_retained_count
                        core_cluster_frames = cluster_frames[core_mask]
                        core_cluster_times = cluster_times[core_mask]

                        # Subset the aligned full ensemble directly. The filtered
                        # structures remain in the same representative frame.
                        core_fitted_xtc = cluster_dir / "cluster_core_fitted.xtc"
                        core_selector = cluster_dir / "core_frame_selector.xvg"
                        write_frame_selector(core_selector, cluster_times, np.flatnonzero(core_mask))
                        t0 = time.time()
                        _run_cluster_gmx(
                            ["trjconv", "-s", str(representative_pdb),
                             "-f", str(fitted_xtc),
                             "-drop", str(core_selector), "-dropunder", "0.5",
                             "-o", str(core_fitted_xtc)],
                            selections=["System"],
                        )
                        request.timings["sort"] += time.time() - t0
                        # The same reference-relative RMSDs remain valid for the
                        # core; they must not be recomputed against a new mean.
                        final_rmsds_nm = pre_filter_rmsds_nm[core_mask]
                        retained_trajectory = core_fitted_xtc
                        retained_times = core_cluster_times
                        retained_frame_indices = core_cluster_frames

                        log_gromacs_detail(
                            "GROMACS cluster %d: retained %d/%d structures after "
                            "RMSD filtering at %.4f A",
                            cluster_id,
                            retained_count,
                            len(cluster_frames),
                            rmsd_threshold_angstrom,
                        )
                    else:
                        # If the threshold would remove
                        # every structure, retain the complete structural ensemble.
                        retained_count = int(len(cluster_frames))
                        logging.warning(
                            "\nNo structures satisfy the RMSD threshold %.4f A for "
                            "cluster %d; retaining the complete structural ensemble.",
                            rmsd_threshold_angstrom,
                            cluster_id,
                        )

                superposition_pdb = cluster_dir / f"cluster_{cluster_id:04d}_superposition.pdb"
                superposition_indices = _superposition_sample_indices(retained_count)
                superposition_selector = cluster_dir / "superposition_selector.xvg"
                write_frame_selector(superposition_selector, retained_times, superposition_indices)
                t0 = time.time()
                _run_cluster_gmx(
                    ["trjconv", "-s", str(representative_pdb), "-f", str(retained_trajectory),
                     "-drop", str(superposition_selector), "-dropunder", "0.5",
                     "-o", str(superposition_pdb)], selections=["System"],
                )
                if not superposition_pdb.is_file():
                    raise PipelineExecutionError("GROMACS superposition PDB was not created.")
                with superposition_pdb.open() as handle:
                    model_count = sum(line.startswith("MODEL ") for line in handle)
                model_count = model_count or 1  # Single-model PDBs may omit MODEL records.
                if model_count != len(superposition_indices):
                    raise PipelineExecutionError(
                        f"GROMACS superposition contains {model_count} models; expected {len(superposition_indices)}."
                    )
                log_gromacs_detail("Cluster %d superposition: %d/%d retained frames, evenly spaced, cap 500.",
                                   cluster_id, model_count, retained_count)
                request.timings["super"] += time.time() - t0

                max_rmsd_angstrom = float(np.max(final_rmsds_nm) * 10.0)

                request.rmsds[cluster_id] = float(np.max(pre_filter_rmsds_nm) * 10.0)
                request.representative_frames[cluster_id] = int(
                    representative.frame_index
                )
                request.retained_counts[cluster_id] = retained_count

                # Export both full and filtered trajectories independently.
                final_structures = {
                    "representative_structure": representative_pdb,
                    "superposition_structure": superposition_pdb,
                    "fitted_trajectory": fitted_xtc,
                }
                if retained_trajectory != fitted_xtc:
                    final_structures["core_fitted_trajectory"] = retained_trajectory
                for source in final_structures.values():
                    if not source.is_file():
                        raise PipelineExecutionError(f"GROMACS final structure is missing: {source.name}")
                saved_structures = {}
                for key, source in final_structures.items():
                    if key == "fitted_trajectory":
                        filename = f"cluster_{cluster_id:04d}_fitted.xtc"
                    elif key == "core_fitted_trajectory":
                        filename = f"cluster_{cluster_id:04d}_core_fitted.xtc"
                    else:
                        filename = source.name
                    destination = output_root / filename
                    shutil.copy2(source, destination)
                    saved_structures[key] = str(destination)

                structural_info[cluster_id] = {
                    "cluster_directory": str(output_root),
                    **saved_structures,
                    "intermediates_retained": False,
                    "superposition_frame_count": int(len(superposition_indices)),
                    "superposition_frame_limit": 500,
                    "superposition_sampling": "evenly_spaced_retained_frame_ordinals_including_endpoints",
                    "superposition_retained_ordinals": superposition_indices.tolist(),
                    "rmsd_reference": "centroid_nearest_representative",
                    "rmsd_values_nm": final_rmsds_nm.tolist(),
                    "pre_filter_rmsd_values_nm": pre_filter_rmsds_nm.tolist(),
                    "mean_full_rmsd_angstrom": float(np.mean(pre_filter_rmsds_nm) * 10),
                    "max_full_rmsd_angstrom": float(np.max(pre_filter_rmsds_nm) * 10),
                    "mean_core_rmsd_angstrom": float(np.mean(final_rmsds_nm) * 10),
                    "max_core_rmsd_angstrom": (max_rmsd_angstrom if rmsd_threshold_angstrom > 0.0 and retained_trajectory != fitted_xtc else None),
                    "frame_count": int(len(cluster_frames)),
                    "retained_count": retained_count,
                    "retained_frame_indices": retained_frame_indices.tolist(),
                    "representative_frame_index": int(
                        representative.frame_index
                    ),
                    "representative_time": representative_time,
                    "selection": selection,
                    "frame_mapping_verified": True,
                    "frame_indices": cluster_frames.tolist(),
                    "frame_index_semantics": "zero_based_trajectory_frame_ordinal",
                    "rmsd_mass_weighted": False,
                    "trajectory_discovery": getattr(request, "trajectory_discovery", None),
                    "topology_discovery": getattr(request, "topology_discovery", None),
                    "rmsd_threshold_angstrom": rmsd_threshold_angstrom,
                }

                log_gromacs_detail(
                    "GROMACS cluster %d: %d assigned structures, %d retained, "
                    "representative frame %d (%.3f ps), max RMSD %.4f A",
                    cluster_id,
                    len(cluster_frames),
                    retained_count,
                    int(representative.frame_index),
                    representative_time,
                    max_rmsd_angstrom,
                )

                if rmsd_threshold_angstrom > 0.0:
                    finish_progress(f"Retained {retained_count}/{len(cluster_frames)} frames after RMSD filtering.")
                else:
                    finish_progress(f"Extracted {len(cluster_frames)} frames.")

        return StructuralResult(
            structural_info,
            request.rmsds,
            request.representative_frames,
            request.retained_counts,
            request.timings,
        )
