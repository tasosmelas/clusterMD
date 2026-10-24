"""Coordinate projection, clustering, characterization, and reporting."""

from clustermd.models import CharacterizationResult, AnalysisResult, ExecutionReportContext, ProjectionDataset
from clustermd.characterization.representatives import select_representatives
from clustermd.validation import validate_analysis_spec
from clustermd.projection import prepare_projection_for_clustering
from clustermd.characterization import characterize_clusters
from clustermd.structural.service import run_structural_stage
from clustermd.visualization.pipeline import render_outputs

import logging
import os
import psutil
import time
from clustermd.constants import Colors
from clustermd.clustering.hdbscan import execute_clustering_algorithm, run_clustering_stage
from clustermd.reporting.coordinates import export_assignment_output
from clustermd.reporting.execution import finalize_execution_report
from clustermd.validation import assess_frame_validation, log_frame_validation_warnings, log_frame_validation_failures

def run_analysis(spec):
    """Execute analysis and durably report its outcome, preserving raised errors."""
    from clustermd.reporting.run_record import write_run_record
    try:
        result = _execute_analysis(spec)
    except BaseException as exc:
        try:
            write_run_record(spec, error=exc)
        except Exception:
            logging.exception("Failed to write run outcome; original analysis error is preserved.")
        raise
    write_run_record(spec, result=result)
    return result


def _execute_analysis(spec):
    """Executes the complete clustering, observed-label-change, visualization, and structural-extraction sequence."""
    import matplotlib
    matplotlib.use('Agg') # Safe headless execution

    validate_analysis_spec(spec)
    data_file = spec.projection_file
    results_dir = spec.results_dir
    target_analysis_dir = spec.target_analysis_dir
    command_line = spec.command_line
    if not results_dir:
        raise ValueError("AnalysisSpec.results_dir must name an existing output directory.")
    pipeline_start = time.time()
    process = psutil.Process(os.getpid())
    timing_stats = {'load': 0, 'hdbscan': 0, 'export': 0, 'plots': 0, 'extract': 0}

    # 1. PROJECTION PREPARATION
    prepared = prepare_projection_for_clustering(spec)
    projection = prepared.projection
    data_file = spec.projection_file
    parse_stats = prepared.parse_stats
    frames = prepared.frames
    clustering_coordinates = prepared.clustering_coordinates
    available_dims = prepared.available_dimensions
    requested_dim = prepared.requested_dimensions
    n_dimensions = prepared.analyzed_dimensions
    total_frames = prepared.total_frames
    timing_stats['load'] = prepared.load_runtime
    timing_stats['generation'] = prepared.generation_runtime

    # 2. CLUSTERING
    clustering_stage = run_clustering_stage(
        clustering_coordinates=clustering_coordinates,
        total_frames=total_frames,
        spec=spec,
        data_file=data_file,
        analysis_step=spec.clustering.analysis_step,
        n_dimensions=n_dimensions,
        engine=execute_clustering_algorithm,
    )
    clustering = clustering_stage.clustering
    labels = clustering_stage.labels
    unique_clusters = clustering_stage.unique_clusters
    dynamic_size = clustering_stage.min_cluster_size
    dynamic_samples = clustering_stage.min_samples
    assigned_frames = clustering_stage.assigned_frames
    noise_frames = clustering_stage.noise_frames
    timing_stats['hdbscan'] = clustering.timings['hdbscan']

    out_filename, timing_stats['export'] = export_assignment_output(
        frames, labels, clustering_coordinates, results_dir, n_dimensions
    )

    frame_validation = assess_frame_validation(parse_stats)
    fatal_frame_error = frame_validation.fatal_frame_error
    skip_label_changes = frame_validation.skip_label_changes
    skip_structural = frame_validation.skip_structural
    log_frame_validation_warnings(frame_validation)

    if spec.cluster_only:
        from clustermd.reporting.execution import write_cluster_only_report
        report_filename = write_cluster_only_report(
            spec=spec, results_dir=results_dir, data_file=data_file,
            parse_stats=parse_stats, available_dimensions=available_dims,
            n_dimensions=n_dimensions, total_frames=total_frames,
            min_cluster_size=dynamic_size, min_samples=dynamic_samples,
            unique_clusters=unique_clusters, labels=labels,
            timing_stats=timing_stats, pipeline_start=pipeline_start,
            process=process, assignment_file=out_filename,
            frame_validation=frame_validation,
        )
        if spec.projection.generate:
            from clustermd.projection.generation import append_generation_report
            append_generation_report(spec)
        elapsed = time.time() - pipeline_start
        logging.info(f"\n{Colors.BLUE}[+]{Colors.RESET} Cluster-only mode complete.")
        logging.info(f"    Frames analyzed      : {total_frames:,}")
        logging.info(f"    Clusters identified  : {len(unique_clusters)}")
        logging.info(f"    Assigned frames      : {assigned_frames:,} ({assigned_frames / total_frames * 100:.2f}%)")
        logging.info(f"    Unassigned frames    : {noise_frames:,} ({noise_frames / total_frames * 100:.2f}%)")
        logging.info(f"    Projection generation : {timing_stats['generation']:.2f} s")
        logging.info(f"    Ingestion/validation  : {timing_stats['load']:.2f} s")
        logging.info(f"    HDBSCAN runtime      : {timing_stats['hdbscan']:.2f} s")
        logging.info(f"    Assignment file      : {out_filename}")
        logging.info(f"    Analysis report      : {report_filename}")
        logging.info(f"    Total runtime        : {elapsed:.2f} s")
        from clustermd.reporting.projection_status import log_numerical_status
        log_numerical_status(spec)
        return AnalysisResult(projection, clustering, CharacterizationResult(), None,
                              results_dir, out_filename, timing_stats)

    log_frame_validation_failures(frame_validation)

    # 3. CHARACTERIZATION
    characterization_data = characterize_clusters(
        labels=labels,
        frames=frames,
        coordinates=clustering_coordinates,
        temperature=spec.reporting.temperature,
        top_changes=spec.reporting.top_changes,
        diagnostic_pair=spec.reporting.diagnostic_pair,
        max_events=spec.reporting.max_events,
        results_dir=results_dir,
        skip_label_changes=skip_label_changes,
    )

    states = characterization_data.states
    population_map = characterization_data.population_map
    free_energies = characterization_data.free_energies
    k_B_T = characterization_data.k_B_T
    label_changes = characterization_data.label_changes
    top_edges = characterization_data.top_edges
    num_states = len(states)

    # 4. VISUALIZATION
    plot_status, timing_stats['plots'] = render_outputs(
        clustering_coordinates, n_dimensions, unique_clusters, labels, k_B_T, num_states,
        states, population_map, label_changes, top_edges, results_dir, skip_label_changes)

    representative_view = ProjectionDataset(clustering_coordinates, projection.sample_indices, projection.frame_indices)
    representatives = select_representatives(representative_view, labels)

    # 5. STRUCTURAL PROCESSING
    structures, representatives, timing_stats['extract'] = run_structural_stage(
        projection=projection,
        clustering=clustering,
        clustering_coordinates=clustering_coordinates,
        labels=labels,
        spec=spec,
        target_analysis_dir=target_analysis_dir,
        results_dir=results_dir,
        skip_structural=skip_structural,
        representatives=representatives,
    )

    # 6. FINALIZATION
    report_context = ExecutionReportContext(
        results_dir=results_dir,
        pipeline_start=pipeline_start,
        process=process,
        total_frames=total_frames,
        spec=spec,
        n_dimensions=n_dimensions,
        available_dimensions=available_dims,
        requested_dimensions=requested_dim,
        min_cluster_size=dynamic_size,
        min_samples=dynamic_samples,
        target_analysis_dir=target_analysis_dir,
        data_file=data_file,
        timing_stats=timing_stats,
        unique_clusters=unique_clusters,
        labels=labels,
        parse_stats=parse_stats,
        structures=structures,
        command_line=command_line,
        skip_label_changes=skip_label_changes,
        skip_structural=skip_structural,
        fatal_frame_error=fatal_frame_error,
        plot_status=plot_status,
    )
    finalize_execution_report(report_context)
    if spec.projection.generate:
        from clustermd.projection.generation import append_generation_report
        append_generation_report(spec)

    characterization = CharacterizationResult(representatives, population_map,
                                               free_energies, label_changes)
    return AnalysisResult(projection, clustering, characterization, structures,
                          results_dir, out_filename, timing_stats)