"""Readable execution report and console completion summaries.

Scientific calculations and fatal-validation decisions remain in their source stages.
"""

import glob
import logging
import os
import time
import shlex

import numpy as np

from clustermd.constants import Colors
from clustermd.models import PipelineExecutionError, ExecutionReportContext
from clustermd.reporting.formatting import report_header, section, field, table
from clustermd.utils.environment import get_software_env, get_gromacs_env
from clustermd.utils.formatting import format_duration


def _variance_from_local_eigenvalues(context):
    """Report associated local carma eigenvalues without certifying their origin."""
    if context.spec.projection.backend.strip().lower() == "gromacs":
        return "Not assessed"
    projection = os.path.basename(context.data_file)
    suffix = ".fluctuations.dat"
    if not projection.endswith(suffix):
        return "Not available (projection association unknown)"
    prefix = projection[:-len(suffix)]
    candidate = os.path.join(
        os.path.dirname(os.path.abspath(context.data_file)) if context.spec.projection.generate
        else context.target_analysis_dir, prefix + ".eigenvalues.dat")
    if not os.path.isfile(candidate):
        return "Not available (no matching eigenvalue file)"
    try:
        eigenvalues = np.atleast_1d(np.loadtxt(candidate))
        if eigenvalues.ndim != 1 or len(eigenvalues) < context.n_dimensions:
            return "Not available (invalid eigenvalue count)"
        if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0):
            return "Not available (invalid eigenvalues)"
        total = float(np.sum(eigenvalues))
        if total <= 0:
            return "Not available (zero total variance)"
        value = float(np.sum(eigenvalues[:context.n_dimensions])) / total * 100
        return (f"{value:.2f}% (generated CARMA eigenvalues)"
                if context.spec.projection.generate else
                f"{value:.2f}% (associated CARMA eigenvalues; link unverified)")
    except (OSError, ValueError, TypeError):
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Could not read matching CARMA eigenvalues.")
        return "Not available (eigenvalue file unreadable)"


def _count_display(value):
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)


def _structural_status(context, fatal_struct_error):
    """Assess extraction using the same evaluator as the JSON provenance record."""
    from clustermd.reporting.structural_status import structural_status
    return structural_status(
        context.spec, context.structures.info, context.unique_clusters,
        skip_structural=context.skip_structural,
        fatal_struct_error=fatal_struct_error,
    )


def _observed_changes(context):
    """Use the established label-change definition, without altering clustering."""
    if context.skip_label_changes:
        return None
    from clustermd.characterization.label_changes import map_observed_label_changes
    return map_observed_label_changes(context.labels)


def _cluster_rows(context):
    structures = context.structures
    threshold = context.spec.structural.rmsd_threshold
    filtered = threshold > 0
    rows = []
    backend = context.spec.structural.backend.strip().lower()
    for item in context.unique_clusters:
        cid = int(item)
        count = int(np.sum(context.labels == cid))
        retained = structures.retained_counts.get(cid)
        representative = structures.representative_frames.get(cid)
        full = structures.rmsds.get(cid)
        full_label = f"{float(full):.2f}" if full is not None and float(full) >= 0 else "—"
        rep_label = str(representative) if representative is not None else "—"
        base = [str(cid), f"{count:,}", f"{count / context.total_frames * 100:.2f}%"]
        if filtered:
            base += [f"{int(retained):,}" if retained is not None else "—",
                     f"{int(retained) / count * 100:.2f}%" if retained is not None and count else "—"]
        base += [rep_label, full_label]
        if filtered:
            if backend == "gromacs":
                item_info = structures.info.get(cid, {})
                core = item_info.get("max_core_rmsd_angstrom") if isinstance(item_info, dict) else None
            else:
                core = structures.info.get("core_max_rmsd_angstrom", {}).get(cid)
            base.append(f"{float(core):.2f}" if core is not None else "—")
        rows.append(base)
    return rows


def write_execution_report(context: ExecutionReportContext, fatal_struct_error=False):
    """Write a single results-first, backend-aware TXT report."""
    spec, structures = context.spec, context.structures
    backend = spec.structural.backend.strip().lower()
    env = get_software_env()
    total = context.total_frames
    noise = int(np.sum(context.labels == -1))
    assigned = total - noise
    failed = context.fatal_frame_error or fatal_struct_error
    changes = _observed_changes(context)
    structural_status = _structural_status(context, fatal_struct_error)
    threshold = spec.structural.rmsd_threshold
    filtered = threshold > 0
    lines = [report_header(env["clusterMD_ver"]) + "\n"]
    add = lines.append

    add(section("RESULTS"))
    add(field("Outcome", "INCOMPLETE" if failed else "Completed"))
    add(field("Frames analyzed", f"{total:,}"))
    add(field("Clusters", len(context.unique_clusters)))
    add(field("Assigned", f"{assigned:,} ({assigned / total * 100:.2f}%)"))
    add(field("Unassigned", f"{noise:,} ({noise / total * 100:.2f}%)"))
    add(field("Structural extraction", structural_status))

    add(section("INPUTS"))
    add(field("Projection", os.path.basename(context.data_file)))
    add(field("Projection path", os.path.abspath(context.data_file)))
    add(field("Projection source", "Generated in this run" if spec.projection.generate else "Existing file reused"))
    add(field("Projection generator", (spec.projection.generation_backend or spec.projection.backend).upper() if spec.projection.generate else "Not invoked"))
    if spec.projection.generate:
        add(field("Projection method", spec.projection.projection_method.upper() if spec.projection.projection_method == "pca" else "dPCA"))
    add(field("Trajectory", os.path.basename(spec.trajectory_file) if spec.trajectory_file else "Not supplied"))
    add(field("Trajectory path", os.path.abspath(spec.trajectory_file)) if spec.trajectory_file else "")
    add(field("Topology", os.path.basename(spec.topology_file) if spec.topology_file else "Not supplied"))
    add(field("Topology path", os.path.abspath(spec.topology_file)) if spec.topology_file else "")
    add(field("Trajectory source", spec.trajectory_discovery or "Not recorded"))
    add(field("Topology source", spec.topology_discovery or "Not recorded"))
    add(field("Structural backend", backend.upper()))
    add(field("Structural selection", spec.structural.selection or ("Backbone (default)" if backend == "gromacs" else "CA C N O (default)")))
    add("  Command:\n    " + context.command_line + "\n")

    add(section("PROJECTION"))
    ps = context.parse_stats
    add(field("Available dimensions", context.available_dimensions))
    add(field("Dimensions analyzed", context.n_dimensions))
    add(field("Analysis stride", spec.clustering.analysis_step))
    add(field("Input lines", _count_display(ps.get("total_lines", "Not recorded"))))
    add(field("Rows selected", _count_display(ps.get("selected_by_stride", "Not recorded"))))
    add(field("Valid rows", _count_display(ps.get("valid_sampled", "Not recorded"))))
    add(field("Malformed rows skipped", _count_display(ps.get("malformed_sampled", "Not recorded"))))
    add(field("First frame ID", _count_display(ps.get("first_sampled_frame", "Not recorded"))))
    add(field("Last frame ID", _count_display(ps.get("last_sampled_frame", "Not recorded"))))
    add(field("Monotonic frame IDs", "Yes" if ps.get("sampled_monotonic") else "No"))
    add(field("Duplicate frame IDs", "Yes" if ps.get("sampled_duplicates") else "No"))
    if backend == "gromacs":
        add(field("Frame index semantics", ps.get("frame_index_semantics", "Unverified")))
        add(field("Time/frame mapping", "See gromacs_sample_mapping.tsv and projection metadata"))
    else:
        add(field("Cumulative variance", _variance_from_local_eigenvalues(context)))

    add(section("HDBSCAN"))
    add(field("Cluster selection", spec.clustering.cluster_selection_method))
    add(field("Parameter selection", "Automatic" if spec.clustering.automatic_parameters else "Manual"))
    add(field("min_cluster_size", context.min_cluster_size))
    add(field("min_samples", context.min_samples))

    add(section("CLUSTERS"))
    add(table(["Cluster", "Frames", "Population"],
              [(str(int(cid)), f"{int(np.sum(context.labels == cid)):,}",
                f"{int(np.sum(context.labels == cid))/total*100:.2f}%") for cid in context.unique_clusters] +
              [("Unassigned", f"{noise:,}", f"{noise/total*100:.2f}%")], right_aligned={1, 2}))

    add(section("OBSERVED LABEL CHANGES"))
    if changes is None:
        add("  Not performed (chronological frame validation failed).\n")
    elif not changes:
        add("  No changes between assigned cluster labels observed.\n")
    else:
        pairs = sorted(changes.items(), key=lambda item: (-item[1], item[0]))
        command_tokens = shlex.split(context.command_line)
        explicit_top = any(t == "--top" or t.startswith("--top=") for t in command_tokens)
        limit = spec.reporting.top_changes if explicit_top else 30
        shown = pairs[:limit] if limit > 0 else []
        add(table(["Cluster pair", "Count"],
                  [(f"{int(a)} <--> {int(b)}", str(count)) for (a, b), count in shown], {1})) if shown else add("  No pairs displayed (--top 0).\n")
        if len(shown) < len(pairs):
            add(field("Displayed pairs", f"{len(shown)} of {len(pairs)}"))
            add(field("Full results", "observed_label_changes_report.txt"))
        add(field("Total changes", sum(changes.values())))
        add(field("Unique cluster pairs", len(changes)))

    add(section("STRUCTURAL EXTRACTION"))
    add(field("Status", structural_status))
    # A skipped/failed stage must not produce pages of empty representative/RMSD rows.
    if structural_status == "Completed" or structural_status.startswith("Partial ("):
        add(field("RMSD filtering", f"Enabled (cutoff {threshold:g} Å)" if filtered else "Disabled"))
        add(field("Representative", "Original centroid-nearest observed frame"))
        headers = ["Cluster"]
        if filtered:
            headers += ["Assigned", "Retained", "Retained %"]
        headers += ["Rep. frame", "Max RMSD (Å)"]
        if filtered:
            headers += ["Core RMSD (Å)"]
        structural_rows = _cluster_rows(context)
        if filtered:
            structural_rows = [[r[0], r[1], r[3], r[4], r[5], r[6], r[7]] for r in structural_rows]
        else:
            structural_rows = [[r[0], r[3], r[4]] for r in structural_rows]
        add(table(headers, structural_rows, set(range(1, len(headers)))))

    add(section("FIGURES"))
    for key, label in (("landscape", "Conformational landscape"),
                       ("fes_connectivity", "FES and connectivity"),
                       ("pcoa_connectivity", "PCoA and connectivity")):
        add(field(label, (context.plot_status or {}).get(key, "Not recorded")))
    add(field("Temperature", f"{spec.reporting.temperature:g} K"))

    add(section("PERFORMANCE"))
    timing = context.timing_stats
    for key, label in (("generation", "Projection generation"),
                       ("load", "Data ingestion"), ("hdbscan", "HDBSCAN clustering"),
                       ("export", "Assignment export"), ("plots", "Figure generation"),
                       ("extract", "Structural extraction")):
        add(field(label, f"{timing.get(key, 0):.2f} s"))
    duration = format_duration(time.time() - context.pipeline_start)
    add(field("Total runtime", duration[:-1] + " s" if duration.endswith("s") and duration[:-1].replace(".", "", 1).isdigit() else duration))
    add(field("Memory (RSS at report)", f"{context.process.memory_info().rss / (1024 ** 3):.2f} GB"))

    add(section("SOFTWARE"))
    for label, value in (("clusterMD", env["clusterMD_ver"]), ("Python", env["python_ver"]),
                         ("fast_hdbscan", env["fh_ver"]), ("NumPy", env["numpy_ver"]),
                         ("SciPy", env["scipy_ver"])):
        add(field(label, value))
    if backend == "gromacs":
        gmx = get_gromacs_env()
        add(field("GROMACS executable", gmx["gromacs_path"]))
        add(field("GROMACS version", gmx["gromacs_ver"]))
    else:
        add(field("CARMA executable", env["carma_path"]))
        add(field("CARMA version", env["carma_ver"]))

    add(section("NOTES"))
    add("  - Cluster IDs are ordered by descending population; -1 denotes unassigned frames (HDBSCAN noise).\n")
    add("  - Observed changes count consecutive assigned labels after omitting unassigned frames;\n"
        "    they do not establish direct molecular transitions or kinetic rates.\n")
    add("  - Population-derived free energies assume suitable equilibrium sampling;\n"
        "    they are not kinetic barriers.\n")
    add("  - Representatives are selected before optional RMSD filtering.\n")
    add("  - Maximum RMSD refers to all originally assigned frames; core maximum\n"
        "    refers only to retained frames when filtering is enabled.\n")
    if backend == "carma" and spec.trajectory_file and spec.topology_file:
        add("  [WARN] Projection-to-DCD frame correspondence relies on supplied frame IDs\n"
            "         and cannot be independently established by this run.\n")
    if context.skip_label_changes:
        add("  [WARN] Temporal label-change mapping was skipped due to frame validation.\n")
    if context.skip_structural:
        add("  [WARN] Structural extraction was skipped due to frame validation.\n")
    if failed:
        add("  [WARN] One or more requested operations failed; inspect the execution log.\n")
    add("  - Runtime completion does not establish independent scientific equivalence.\n")
    add(field("Execution log", "clustering_execution.log"))

    filename = os.path.join(context.results_dir, "clusterMD_analysis_report.txt")
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("".join(lines))
    return filename, structural_status



def write_cluster_only_report(*, spec, results_dir, data_file, parse_stats,
                              available_dimensions, n_dimensions, total_frames,
                              min_cluster_size, min_samples, unique_clusters,
                              labels, timing_stats, pipeline_start, process,
                              assignment_file, frame_validation):
    """Report completed clustering stages without invoking characterization.

    No label changes, representatives, structural extraction or figures are
    computed here. This is deliberately separate from the full-run writer.
    """
    from clustermd.reporting.projection_status import numerical_diagnostic

    env = get_software_env()
    backend = spec.structural.backend.strip().lower()
    noise = int(np.sum(labels == -1))
    assigned = total_frames - noise
    diagnostic = numerical_diagnostic(spec)
    lines = [report_header(env["clusterMD_ver"]) + "\n"]
    add = lines.append

    add(section("RESULTS"))
    add(field("Mode", "Cluster-only"))
    add(field("Outcome", "Clustering completed"))
    add(field("Frames analyzed", f"{total_frames:,}"))
    add(field("Clusters", len(unique_clusters)))
    add(field("Assigned", f"{assigned:,} ({assigned / total_frames * 100:.2f}%)"))
    add(field("Unassigned", f"{noise:,} ({noise / total_frames * 100:.2f}%)"))
    add(field("Assignment file", os.path.relpath(assignment_file, results_dir)))
    add(field("Characterization", "Not performed (--cluster-only)"))
    add(field("Structural extraction", "Not performed (--cluster-only)"))
    add(field("Figures", "Not generated (--cluster-only)"))

    add(section("INPUTS"))
    add(field("Projection", os.path.basename(data_file)))
    add(field("Projection path", os.path.abspath(data_file)))
    add(field("Projection source", "Generated in this run" if spec.projection.generate else "Existing file reused"))
    add(field("Projection generator", (spec.projection.generation_backend or spec.projection.backend).upper()
              if spec.projection.generate else "Not invoked"))
    if spec.projection.generate:
        add(field("Projection method", "dPCA" if spec.projection.projection_method == "dpca" else "PCA"))
    add(field("Trajectory", os.path.basename(spec.trajectory_file) if spec.trajectory_file else "Not supplied"))
    add(field("Topology", os.path.basename(spec.topology_file) if spec.topology_file else "Not supplied"))
    add("  Command:\n    " + spec.command_line + "\n")

    add(section("PROJECTION"))
    add(field("Available dimensions", available_dimensions))
    add(field("Dimensions analyzed", n_dimensions))
    add(field("Analysis stride", spec.clustering.analysis_step))
    for label, key in (("Input lines", "total_lines"), ("Rows selected", "selected_by_stride"),
                       ("Valid rows", "valid_sampled"), ("Malformed rows skipped", "malformed_sampled"),
                       ("First frame ID", "first_sampled_frame"), ("Last frame ID", "last_sampled_frame")):
        add(field(label, _count_display(parse_stats.get(key, "Not recorded"))))
    add(field("Monotonic frame IDs", "Yes" if parse_stats.get("sampled_monotonic") else "No"))
    add(field("Duplicate frame IDs", "Yes" if parse_stats.get("sampled_duplicates") else "No"))
    if backend == "gromacs":
        add(field("Frame index semantics", parse_stats.get("frame_index_semantics", "Unverified")))
        add(field("Time/frame mapping", "See gromacs_sample_mapping.tsv and projection metadata"))
    else:
        # Equivalent to full-report association rules but without a structural context.
        from types import SimpleNamespace
        variance_context = SimpleNamespace(spec=spec, data_file=data_file,
                                           target_analysis_dir=spec.target_analysis_dir,
                                           n_dimensions=n_dimensions)
        add(field("Cumulative variance", _variance_from_local_eigenvalues(variance_context)))

    add(section("HDBSCAN"))
    add(field("Cluster selection", spec.clustering.cluster_selection_method))
    add(field("Parameter selection", "Automatic" if spec.clustering.automatic_parameters else "Manual"))
    add(field("min_cluster_size", min_cluster_size))
    add(field("min_samples", min_samples))

    add(section("CLUSTERS"))
    rows = [(str(int(cid)), f"{int(np.sum(labels == cid)):,}",
             f"{int(np.sum(labels == cid)) / total_frames * 100:.2f}%") for cid in unique_clusters]
    rows.append(("Unassigned", f"{noise:,}", f"{noise / total_frames * 100:.2f}%"))
    add(table(["Cluster", "Frames", "Population"], rows, {1, 2}))

    add(section("PERFORMANCE"))
    for key, label in (("generation", "Projection generation"), ("load", "Data ingestion"),
                       ("hdbscan", "HDBSCAN clustering"), ("export", "Assignment export")):
        add(field(label, f"{timing_stats.get(key, 0):.2f} s"))
    add(field("Total runtime", f"{time.time() - pipeline_start:.2f} s"))
    add(field("Memory (RSS at report)", f"{process.memory_info().rss / (1024 ** 3):.2f} GB"))

    add(section("SOFTWARE"))
    for label, value in (("clusterMD", env["clusterMD_ver"]), ("Python", env["python_ver"]),
                         ("fast_hdbscan", env["fh_ver"]), ("NumPy", env["numpy_ver"]),
                         ("SciPy", env["scipy_ver"])):
        add(field(label, value))

    add(section("VALIDATION"))
    add(field("Clustering execution", "Completed"))
    add(field("Projection diagnostic", diagnostic['status'].upper().replace('_', ' ')))
    if diagnostic.get("scope") or diagnostic.get("reason"):
        add(field("Diagnostic scope", diagnostic.get("scope", diagnostic.get("reason"))))
    if 'atol' in diagnostic:
        add(field("Numerical tolerances", f"atol={diagnostic['atol']}, rtol={diagnostic.get('rtol')}, ddof={diagnostic.get('variance_ddof', 0)}"))
        add(field("Diagnostic details", "projection/numerical_diagnostics.json"))
    add(field("Scientific equivalence", "Requires independent native/reference comparisons"))
    if frame_validation.fatal_frame_error or frame_validation.skip_label_changes or frame_validation.skip_structural:
        add("  [WARN] Frame validation raised conditions requiring review; see execution log.\n")
    add(section("NOTES"))
    add("  - Only projection ingestion, HDBSCAN clustering and assignment export were performed.\n")
    add("  - Observed label changes, figures and structural ensembles were not calculated.\n")
    add("  - Cluster IDs are ordered by descending population; -1 denotes unassigned frames (HDBSCAN noise).\n")
    add(field("Execution log", "clustering_execution.log"))
    filename = os.path.join(results_dir, "clusterMD_analysis_report.txt")
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("".join(lines))
    return filename

def finalize_execution_report(context: ExecutionReportContext):
    """Write the execution report before propagating fatal validation errors."""
    logging.info(f"\n{Colors.LIGHT_BLUE}[*]{Colors.RESET} Writing analysis report...")
    info = context.structures.info
    fatal_struct_error = False
    if info.get("dcd") and info.get("dcd") != "N/A":
        if info.get("compatible") is False or info.get("bounds_valid") is False:
            fatal_struct_error = True
        elif info.get("failed_count", 0) > 0:
            fatal_struct_error = True
    overall_fatal_error = context.fatal_frame_error or fatal_struct_error
    filename, structural_status = write_execution_report(context, fatal_struct_error)

    from clustermd.reporting.projection_status import append_projection_status
    append_projection_status(context, overall_fatal_error)

    logging.info(f"{Colors.BLUE}[+]{Colors.RESET} Analysis report saved to: {filename}")
    logging.info("\nANALYSIS INCOMPLETE" if overall_fatal_error else "\nANALYSIS COMPLETE")
    noise = int(np.sum(context.labels == -1))
    assigned = context.total_frames - noise
    for label, value in (("Frames analyzed", f"{context.total_frames:,}"),
                         ("Clusters", len(context.unique_clusters)),
                         ("Assigned", f"{assigned:,} ({assigned/context.total_frames*100:.2f}%)"),
                         ("Unassigned", f"{noise:,} ({noise/context.total_frames*100:.2f}%)"),
                         ("Structural Extraction", structural_status),
                         ("Results", context.results_dir + "/")):
        logging.info(field(label, value).rstrip("\n"))

    if overall_fatal_error:
        raise PipelineExecutionError(
            "Pipeline encountered fatal validation errors. See the execution report for details."
        )
