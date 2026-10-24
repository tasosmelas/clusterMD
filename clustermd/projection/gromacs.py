'Read GROMACS XVG component series and preserve projection-time metadata.'

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

from clustermd.utils.gromacs import read_trajectory_times, map_times_to_frames

from clustermd.models import (
    ProjectionDataset,
    ProjectionValidationError,
)


def read_gromacs_projection(
    file_path,
    analysis_step=1,
):
    """
    Read a GROMACS `gmx anaeig -proj` XVG projection.

    GROMACS writes one time/projection series per eigenvector.
    Series may be separated by blank lines (`-xvg none`) or `&`
    in normal XVG output.

    Returns a normalized ProjectionDataset with:
      - rows = trajectory observations
      - columns = principal components
      - times = GROMACS trajectory times in ps
      - frame_indices = projection-row ordinals until verified against a trajectory
    """

    path = Path(file_path)

    if not path.is_file():
        raise ProjectionValidationError(
            f"GROMACS projection file not found: {path}"
        )

    if analysis_step < 1:
        raise ProjectionValidationError(
            "analysis_step must be a positive integer."
        )

    blocks = []
    current = []

    total_lines = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_lines += 1
            line = raw_line.strip()

            # `-xvg none` separates eigenvector series with blank lines.
            if not line:
                if current:
                    blocks.append(
                        np.asarray(current, dtype=float)
                    )
                    current = []
                continue

            # Standard XVG metadata.
            if line.startswith(("#", "@")):
                continue

            # Standard XVG dataset separator.
            if line.startswith("&"):
                if current:
                    blocks.append(
                        np.asarray(current, dtype=float)
                    )
                    current = []
                continue

            fields = line.split()

            if len(fields) != 2:
                raise ProjectionValidationError(
                    f"{path}: line {line_number}: expected "
                    f"2 columns (time, projection), found "
                    f"{len(fields)}."
                )

            try:
                time_value = float(fields[0])
                projection_value = float(fields[1])
            except ValueError as exc:
                raise ProjectionValidationError(
                    f"{path}: line {line_number}: "
                    "non-numeric projection data."
                ) from exc

            if not (
                np.isfinite(time_value)
                and np.isfinite(projection_value)
            ):
                raise ProjectionValidationError(
                    f"{path}: line {line_number}: "
                    "NaN or infinite value detected."
                )

            current.append(
                (time_value, projection_value)
            )

    if current:
        blocks.append(
            np.asarray(current, dtype=float)
        )

    if not blocks:
        raise ProjectionValidationError(
            f"{path}: no GROMACS projection data found."
        )

    reference_times = blocks[0][:, 0]

    if reference_times.size < 2:
        raise ProjectionValidationError(
            f"{path}: projection contains fewer than 2 samples."
        )

    if not np.all(np.diff(reference_times) > 0):
        raise ProjectionValidationError(
            f"{path}: trajectory times are not strictly increasing."
        )

    for pc_index, block in enumerate(blocks, start=1):

        if block.ndim != 2 or block.shape[1] != 2:
            raise ProjectionValidationError(
                f"{path}: PC{pc_index} does not have "
                "the expected two-column layout."
            )

        if len(block) != len(reference_times):
            raise ProjectionValidationError(
                f"{path}: PC{pc_index} contains "
                f"{len(block)} samples; expected "
                f"{len(reference_times)}."
            )

        if not np.allclose(
            block[:, 0],
            reference_times,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ProjectionValidationError(
                f"{path}: PC{pc_index} time axis "
                "does not match PC1."
            )

    coordinates = np.column_stack(
        [block[:, 1] for block in blocks]
    )

    # Observation identity before clusterMD analysis subsampling.
    trajectory_frame_indices = np.arange(
        len(reference_times),
        dtype=np.int64,
    )

    # --step remains clusterMD analysis subsampling.
    selected = np.arange(
        0,
        len(reference_times),
        analysis_step,
        dtype=np.int64,
    )

    coordinates = coordinates[selected]
    times = reference_times[selected]
    frame_indices = trajectory_frame_indices[selected]

    sample_indices = np.arange(
        len(selected),
        dtype=np.int64,
    )

    parse_stats = {
        "total_lines": total_lines,
        "selected_by_stride": len(selected),
        "valid_sampled": len(selected),
        "malformed_sampled": 0,
        "first_sampled_frame": (
            int(frame_indices[0])
            if len(frame_indices)
            else None
        ),
        "last_sampled_frame": (
            int(frame_indices[-1])
            if len(frame_indices)
            else None
        ),
        "sampled_monotonic": True,
        "sampled_duplicates": False,
    }

    feature_names = tuple(
        f"PC{i}"
        for i in range(1, coordinates.shape[1] + 1)
    )

    return ProjectionDataset(
        coordinates=coordinates,
        sample_indices=sample_indices,
        frame_indices=frame_indices,
        times=times,
        feature_names=feature_names,
        feature_type=("dPCA" if "dpca" in path.name.lower() else "PCA" if "pca" in path.name.lower() else "unknown"),
        source=os.path.abspath(path),
        metadata={
            "parse_stats": parse_stats,
            "analysis_step": analysis_step,
            "projection_stride": None,
            "available_dimensions": coordinates.shape[1],
            "backend": "gromacs",
            "format": "gromacs_anaeig_xvg",
            "feature_type_evidence": "filename inference; native-generated method is checked separately",
            "time_unit": "ps",
            "frame_index_semantics": (
                "zero_based_projection_row_ordinal_unverified"
            ),
        },
    )


class GromacsProjectionBackend:
    """Loads a GROMACS PCA projection into ProjectionDataset."""

    def create_projection(self, spec):

        if spec.projection_file is None:
            raise ProjectionValidationError(
                "GROMACS projection backend requires "
                "a projection file."
            )

        projection = read_gromacs_projection(
            spec.projection_file,
            analysis_step=spec.clustering.analysis_step,
        )
        if spec.trajectory_file and spec.topology_file and (not spec.cluster_only or spec.projection.generate):
            trajectory_times = read_trajectory_times(spec.trajectory_file, spec.topology_file)
            projection.frame_indices = map_times_to_frames(projection.times, trajectory_times)
            projection.metadata.update(
                trajectory_times=trajectory_times,
                frame_mapping_verified=True,
                frame_index_semantics="zero_based_trajectory_frame_ordinal",
            )
            stats = projection.metadata["parse_stats"]
            stats["first_sampled_frame"] = int(projection.frame_indices[0])
            stats["last_sampled_frame"] = int(projection.frame_indices[-1])
        semantics = projection.metadata["frame_index_semantics"]
        projection.metadata["parse_stats"]["frame_index_semantics"] = semantics
        if spec.results_dir:
            output_dir = Path(spec.results_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(output_dir / "gromacs_sample_mapping.tsv",
                       np.column_stack((projection.sample_indices, projection.frame_indices, projection.times)),
                       fmt=['%d', '%d', '%.17g'], delimiter='\t',
                       header='sample_index\tframe_or_projection_row_index\ttime_ps')
            provenance = {"source": projection.source, "time_unit": "ps",
                          "frame_index_semantics": semantics,
                          "frame_mapping_verified": projection.metadata.get("frame_mapping_verified", False),
                          "analysis_step": spec.clustering.analysis_step,
                          "trajectory_file": spec.trajectory_file, "topology_file": spec.topology_file}
            (output_dir / "gromacs_projection_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return projection
