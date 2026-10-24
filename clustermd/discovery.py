"""Centralized input discovery and backend resolution for clusterMD."""

from dataclasses import dataclass
from pathlib import Path
import logging
import os
import glob
import re

from clustermd.constants import Colors
from clustermd.models import InputDiscoveryError


CARMA_TRAJECTORY_SUFFIXES = {".dcd"}
CARMA_TOPOLOGY_SUFFIXES = {".psf"}

GROMACS_TRAJECTORY_SUFFIXES = {".xtc", ".trr"}
GROMACS_TOPOLOGY_SUFFIXES = {".tpr", ".gro"}


@dataclass(frozen=True)
class ResolvedAnalysisInputs:
    """Resolved files and provider choices for one analysis run."""

    projection_file: str | None
    trajectory_file: str | None
    topology_file: str | None
    projection_backend: str
    structural_backend: str
    source_family: str
    trajectory_discovery: str | None = None
    topology_discovery: str | None = None
    generate_projection: bool = False


def _existing_path(value, label):
    if value is None:
        return None

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise InputDiscoveryError(f"{label} not found: {path}")
    return path


def _suffix(path):
    return path.suffix.lower()


def _family_from_projection(path):
    suffix = _suffix(path)
    if suffix == ".xvg":
        return "gromacs"
    if suffix == ".dat":
        return "carma"

    raise InputDiscoveryError(
        f"Unsupported projection file type '{suffix}' for {path}. "
        "Current existing-projection mode accepts CARMA *.dat or GROMACS *.xvg files."
    )


def _family_from_trajectory(path):
    suffix = _suffix(path)
    if suffix in CARMA_TRAJECTORY_SUFFIXES:
        return "carma"
    if suffix in GROMACS_TRAJECTORY_SUFFIXES:
        return "gromacs"
    raise InputDiscoveryError(
        f"Unsupported trajectory type '{suffix}' for {path}. "
        "Expected DCD for CARMA or XTC/TRR for GROMACS."
    )


def _family_from_topology(path):
    suffix = _suffix(path)
    if suffix in CARMA_TOPOLOGY_SUFFIXES:
        return "carma"
    if suffix in GROMACS_TOPOLOGY_SUFFIXES:
        return "gromacs"
    raise InputDiscoveryError(
        f"Unsupported topology type '{suffix}' for {path}. "
        "Expected PSF for CARMA or TPR/GRO for GROMACS."
    )


def _ordered_unique_dirs(*paths):
    dirs = []

    for value in paths:
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        directory = path if path.is_dir() else path.parent
        for candidate in (directory, directory.parent):
            if candidate not in dirs:
                dirs.append(candidate)

    cwd = Path.cwd().resolve()
    if cwd not in dirs:
        dirs.insert(0, cwd)
    if cwd.parent not in dirs:
        dirs.append(cwd.parent)

    return dirs


def _projection_candidates(family, search_dirs):
    candidates = []

    if family == "carma":
        patterns = (
            "*fluctuations.dat",
            "projection/*fluctuations.dat",
            "carma_results*/*fluctuations.dat",
        )
    elif family == "gromacs":
        # Deliberately do not accept arbitrary *.xvg because directories may
        # also contain eigenvalues, RMSD curves, etc.
        patterns = (
            "proj*.xvg",
            "projection*.xvg",
            "*projection*.xvg",
            "projection/proj*.xvg",
            "projection/*projection*.xvg",
            "gromacs_results*/proj*.xvg",
            "gromacs_results*/*projection*.xvg",
        )
    else:
        raise ValueError(f"Unknown input family: {family}")

    for directory in search_dirs:
        for pattern in patterns:
            for name in glob.glob(str(directory / pattern)):
                match = Path(name)
                if match.is_file():
                    resolved = match.resolve()
                    if resolved not in candidates:
                        candidates.append(resolved)
            # Preserve first-nonempty glob precedence: local inputs win
            # over results subdirectories and parent-directory fallbacks.
            if candidates:
                return candidates

    return candidates


def _select_projection(candidates, family):
    if not candidates:
        return None

    selected = max(candidates, key=lambda p: p.stat().st_mtime)

    if len(candidates) > 1:
        logging.warning(
            f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple {family} projection files found:"
        )
        for candidate in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
            marker = " [selected]" if candidate == selected else ""
            logging.warning(f"       - {candidate}{marker}")

    return selected


def _matching_files(search_dirs, suffixes):
    files = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for name in sorted(glob.glob(str(directory / "*"))):
            child = Path(name)
            if (child.is_file() and child.suffix.lower() in suffixes
                    and not _generated_structural_output(child)):
                candidate = child.absolute()
                if candidate not in files:
                    files.append(candidate)
    return files


def _generated_structural_output(path):
    """Exclude clusterMD ensemble outputs from implicit molecular discovery.

    Match the output namespace precisely, rather than excluding legitimate
    input names containing words such as 'cluster' or 'fitted'. Explicit CLI
    paths bypass candidate discovery and can still select these files.
    """
    stem = path.stem.lower()
    return stem == 'unassigned_noise' or bool(re.fullmatch(
        r'cluster_\d+(?:_(?:core|fitted|core_fitted))?', stem))


def _discovery_source(path, *, explicit, search_dirs, kind):
    """Describe how one structural input was obtained for provenance/reporting."""
    if path is None:
        return None

    if explicit:
        return f"explicit --{kind}"

    path = Path(path)
    if path.is_symlink():
        return "symlink in analysis directory"

    cwd = Path.cwd().resolve()
    try:
        parent = path.parent.resolve()
    except OSError:
        parent = path.parent

    if parent == cwd:
        return "discovered in current directory"

    # Preserve the useful CARMA wording when files live beside/above carma_results_*.
    if any("carma_results_" in str(directory) for directory in search_dirs):
        for directory in search_dirs:
            try:
                if parent == Path(directory).resolve():
                    if "carma_results_" in str(directory):
                        return "file in carma results directory"
                    return "parent-directory fallback"
            except OSError:
                pass

    return f"glob discovery in {path.parent.absolute()}"


def _pair_score(trajectory, topology):
    """Rank candidate structural pairs by compatibility before recency.

    A tuple is used deliberately: filesystem mtimes must never outweigh a
    same-stem molecular pair such as md.xtc + md.tpr/gro.
    """
    same_parent = int(trajectory.parent == topology.parent)
    same_stem = int(trajectory.stem == topology.stem)

    trajectory_name = trajectory.stem.lower()
    generated_markers = (
        "fitted",
        "cluster",
        "core",
        "super",
        "average",
        "representative",
        "aligned",
    )
    looks_generated = int(any(marker in trajectory_name for marker in generated_markers))

    # Prefer a TPR over GRO when otherwise equivalent.
    tpr_preference = int(topology.suffix.lower() == ".tpr")

    recency = max(trajectory.stat().st_mtime, topology.stat().st_mtime)

    return (
        same_stem,
        1 - looks_generated,
        same_parent,
        tpr_preference,
        recency,
    )


def _discover_structural_pair(family, search_dirs, traj_override=None, topo_override=None):
    traj = _existing_path(traj_override, "Trajectory file")
    topo = _existing_path(topo_override, "Topology file")

    if traj is not None:
        traj_family = _family_from_trajectory(traj)
        if traj_family != family:
            raise InputDiscoveryError(
                f"Trajectory {traj.name} implies {traj_family}, but the resolved "
                f"analysis family is {family}."
            )

    if topo is not None:
        topo_family = _family_from_topology(topo)
        if topo_family != family:
            raise InputDiscoveryError(
                f"Topology {topo.name} implies {topo_family}, but the resolved "
                f"analysis family is {family}."
            )

    if family == "carma":
        traj_suffixes = CARMA_TRAJECTORY_SUFFIXES
        topo_suffixes = CARMA_TOPOLOGY_SUFFIXES
    else:
        traj_suffixes = GROMACS_TRAJECTORY_SUFFIXES
        topo_suffixes = GROMACS_TOPOLOGY_SUFFIXES

    trajectories = [traj] if traj is not None else _matching_files(search_dirs, traj_suffixes)
    topologies = [topo] if topo is not None else _matching_files(search_dirs, topo_suffixes)

    if traj is not None and topo is not None:
        return traj, topo

    if not trajectories or not topologies:
        return (traj or (trajectories[0] if trajectories else None),
                topo or (topologies[0] if topologies else None))

    pairs = [(t, p) for t in trajectories for p in topologies]
    pairs.sort(key=lambda pair: _pair_score(*pair), reverse=True)
    chosen_traj, chosen_topo = pairs[0]

    if traj is None and len(trajectories) > 1:
        logging.warning(
            f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple compatible trajectory "
            f"files found for {family}; selected {chosen_traj.name}."
        )
    if topo is None and len(topologies) > 1:
        logging.warning(
            f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple compatible topology "
            f"files found for {family}; selected {chosen_topo.name}."
        )

    for kind, candidates, chosen in (("trajectory", trajectories, chosen_traj),
                                     ("topology", topologies, chosen_topo)):
        if len(candidates) > 1:
            for candidate in candidates:
                logging.warning("       %s candidate: %s%s", kind, candidate,
                                " [selected]" if candidate == chosen else "")
    return chosen_traj, chosen_topo


def _family_from_overrides(file_override=None, traj_override=None, topo_override=None):
    families = []

    if file_override is not None:
        families.append(_family_from_projection(_existing_path(file_override, "Projection file")))
    if traj_override is not None:
        families.append(_family_from_trajectory(_existing_path(traj_override, "Trajectory file")))
    if topo_override is not None:
        families.append(_family_from_topology(_existing_path(topo_override, "Topology file")))

    unique = set(families)

    if len(unique) > 1:
        raise InputDiscoveryError(
            "Explicit input overrides belong to different backend families. "
            "Use DCD+PSF+CARMA projection data together, or XTC/TRR+TPR/GRO+GROMACS projection data together."
        )

    return next(iter(unique)) if unique else None


def _discover_family(search_dirs):
    # Family detection should only inspect availability. Actual projection
    # selection happens once later, so warnings are emitted only once.
    carma_projection = bool(_projection_candidates("carma", search_dirs))
    gromacs_projection = bool(_projection_candidates("gromacs", search_dirs))

    carma_traj = _matching_files(search_dirs, CARMA_TRAJECTORY_SUFFIXES)
    carma_topo = _matching_files(search_dirs, CARMA_TOPOLOGY_SUFFIXES)
    gmx_traj = _matching_files(search_dirs, GROMACS_TRAJECTORY_SUFFIXES)
    gmx_topo = _matching_files(search_dirs, GROMACS_TOPOLOGY_SUFFIXES)

    carma_complete = bool(carma_traj and carma_topo)
    gromacs_complete = bool(gmx_traj and gmx_topo)

    if carma_projection and not gromacs_projection:
        return "carma"
    if gromacs_projection and not carma_projection:
        return "gromacs"

    if carma_complete and not gromacs_complete:
        return "carma"
    if gromacs_complete and not carma_complete:
        return "gromacs"

    if carma_projection and gromacs_projection:
        raise InputDiscoveryError(
            "Both CARMA and GROMACS projection files were discovered. "
            "Use --file to select the intended projection."
        )

    if carma_complete and gromacs_complete:
        raise InputDiscoveryError(
            "Both CARMA (DCD+PSF) and GROMACS (XTC/TRR+TPR/GRO) input sets were discovered. "
            "Use --traj/--topo or --file to disambiguate the intended analysis."
        )

    return None


def resolve_analysis_inputs(
    file_override=None,
    traj_override=None,
    topo_override=None,
    cluster_only=False,
    projection_method=None,
):
    """Resolve projection, molecular inputs, and backends in one place.

    Explicit CLI files override searching. Without overrides, clusterMD searches
    the current working directory and its parent, then, once one file is known,
    its containing directory and parent. Own projection/ subdirectories are also
    considered; generated-result layouts extend molecular discovery to the directory
    above the results folder. Backend selection is inferred from file families.
    """

    explicit_family = _family_from_overrides(
        file_override=file_override,
        traj_override=traj_override,
        topo_override=topo_override,
    )

    explicit_file = _existing_path(file_override, "Projection file")
    explicit_traj = _existing_path(traj_override, "Trajectory file")
    explicit_topo = _existing_path(topo_override, "Topology file")

    search_dirs = _ordered_unique_dirs(explicit_file, explicit_traj, explicit_topo)
    family = explicit_family or _discover_family(search_dirs)

    if family is None:
        raise InputDiscoveryError(
            "Unable to determine an analysis backend from the available files. "
            "Expected CARMA '*fluctuations.dat' and optionally DCD+PSF, or a "
            "GROMACS projection XVG with XTC/TRR+TPR/GRO. "
            "Use --file, --traj and/or --topo to override discovery."
        )

    if projection_method is not None and explicit_file is None:
        projection_file = None
    elif explicit_file is not None:
        projection_file = explicit_file
    else:
        projection_file = _select_projection(
            _projection_candidates(family, search_dirs),
            family.upper(),
        )

    generate = projection_file is None
    if generate and (explicit_traj is None or explicit_topo is None):
        raise InputDiscoveryError(
            "No compatible projection found. Generation requires explicit --traj and --topo; "
            "optionally choose --proj pca or dpca (default: dpca)."
        )
    if not generate and _family_from_projection(projection_file) != family:
        raise InputDiscoveryError("Projection and molecular inputs belong to different backends.")

    # Once the projection is known, include its directory and parent in the
    # structural search. This preserves the common CARMA layout where the
    # fluctuations file lives inside carma_results*/ while DCD/PSF live above it.
    search_dirs = _ordered_unique_dirs(
        projection_file,
        explicit_traj,
        explicit_topo,
    )

    # Generated projection directories can be nested below the analysis directory.
    # Search its molecular-input directory too; never recurse through old runs.
    if projection_file is not None:
        projection_dir = Path(projection_file).parent
        if (projection_dir.name == "projection"
                and projection_dir.parent.name.startswith("clusterMD_results_")):
            source_dir = projection_dir.parent.parent
            if source_dir not in search_dirs:
                search_dirs.append(source_dir)

    trajectory_file, topology_file = _discover_structural_pair(
        family,
        search_dirs,
        traj_override=explicit_traj,
        topo_override=explicit_topo,
    )

    trajectory_discovery = _discovery_source(
        trajectory_file,
        explicit=explicit_traj is not None,
        search_dirs=search_dirs,
        kind="traj",
    )
    topology_discovery = _discovery_source(
        topology_file,
        explicit=explicit_topo is not None,
        search_dirs=search_dirs,
        kind="topo",
    )

    # Structural processing is optional when an existing projection is available.
    # Discovery attempts to locate a compatible trajectory/topology pair; if the
    # pair is incomplete, downstream structural orchestration can skip cleanly.
    if (trajectory_file is None) != (topology_file is None):
        logging.warning(
            f"{Colors.YELLOW}[WARN]{Colors.RESET} Only one structural input was "
            f"resolved for the {family} backend. Structural processing will be skipped "
            "unless the missing partner is supplied with --traj/--topo."
        )

    projection_backend = "gromacs" if family == "gromacs" else "precomputed"
    structural_backend = family

    return ResolvedAnalysisInputs(
        projection_file=str(projection_file) if projection_file is not None else None,
        generate_projection=generate,
        trajectory_file=str(trajectory_file) if trajectory_file is not None else None,
        topology_file=str(topology_file) if topology_file is not None else None,
        projection_backend=projection_backend,
        structural_backend=structural_backend,
        source_family=family,
        trajectory_discovery=trajectory_discovery,
        topology_discovery=topology_discovery,
    )


def discover_projection_file():
    """Find an existing projection using the shared input-discovery workflow."""
    resolved = resolve_analysis_inputs(cluster_only=True)
    return resolved.projection_file


def discover_diagnostic_file():
    dat_files = glob.glob("hdbscan.*-D.clusters.dat")

    if not dat_files:
        print()
        raise InputDiscoveryError(
            "'hdbscan.*-D.clusters.dat' not found in the current directory.\n"
            f"{Colors.YELLOW}[*]{Colors.RESET} You are running in diagnostic mode without full pipeline flags.\n"
            f"{Colors.YELLOW}[*]{Colors.RESET} Please navigate INSIDE a 'clusterMD_results_*' folder to run this command."
        )

    if len(dat_files) > 1:
        logging.warning(
            f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple cluster dat files found. "
            "Selecting the most recently modified."
        )
    return max(dat_files, key=os.path.getmtime)
