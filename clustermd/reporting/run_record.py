"""Shared run provenance and outcome, independent of backend-specific text rendering."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import numpy as np


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Unsupported run-record value: {type(value).__name__}')


def _file_identity(value, *, digest=False):
    if not value:
        return None
    path = Path(value).resolve()
    identity = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        stat = path.stat()
        identity.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if digest:
            with path.open('rb') as handle:
                identity['sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(handle)
    return identity


def _digest(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def write_run_record(spec, *, result=None, error=None):
    """Write machine-readable execution provenance and status metadata."""
    if not spec.results_dir or not Path(spec.results_dir).is_dir():
        return
    from clustermd.reporting.projection_status import numerical_diagnostic, executable_roles, FILTER_POLICIES
    config = asdict(spec)
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return repr(value)
        return value
    config = clean(config)
    # Frame arrays/native command traces already live in generation sidecars.
    generation = config['projection'].pop('provenance')
    generation_summary = {key: value for key, value in generation.items()
        if key not in {'projected_frame_ordinals','projected_frame_ids','projected_times_ps','commands'}}
    record = {
        'schema_version': 1, 'status': 'failed' if error is not None else 'completed',
        'configuration': config, 'environment': _runtime_environment(spec),
        'source_sha256': _source_identity(),
        'inputs': {'projection': _file_identity(spec.projection_file, digest=True),
                   'trajectory': _file_identity(spec.trajectory_file),
                   'topology': _file_identity(spec.topology_file)},
        'generation': generation_summary or None,
        'projection_numerical_diagnostic': numerical_diagnostic(spec),
        'executable_roles': executable_roles(spec),
        'validation_scope': 'runtime configuration, coordinate and identity checks; scientific acceptance requires separate reference comparisons',
        'policies': {
            'representative': 'first observed minimum distance to arithmetic centroid in analyzed coordinates',
            'observed_changes': 'undirected changes after omitting noise; not kinetic rates',
            'structural_filtering': 'downstream only; cluster assignments and preselected representatives unchanged',
            'rmsd_threshold_unit': 'Angstrom',
            **FILTER_POLICIES,
            'zero_retained_fallback': 'retain complete structural ensemble',
            'structural_pbc': ('TPR whole-molecule repair before structural extraction; source frames/times preserved'
                if spec.topology_file and str(spec.topology_file).lower().endswith('.tpr') and spec.structural.backend == 'gromacs'
                else 'input molecules must already be whole; topology does not provide automatic structural reconstruction'),
        },
    }
    if error is not None:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
    if result is not None:
        projection, clustering = result.projection, result.clustering
        record['projection'] = {
            'shape': list(projection.coordinates.shape), 'feature_type': projection.feature_type,
            'frame_index_semantics': projection.metadata.get('frame_index_semantics', 'one_based_source_frame_identifier'),
            'frame_mapping_verified': bool(projection.metadata.get('frame_mapping_verified', False)),
            'parse_stats': projection.metadata.get('parse_stats'),
        }
        record['clustering'] = {'populations': {str(k):int(v) for k,v in clustering.cluster_populations.items()},
            'unassigned_count': clustering.unassigned_count, 'parameters': clustering.parameters,
            'assignment_file': result.assignment_file}
        record['representatives'] = {str(k): {'sample_index': int(v.sample_index), 'frame_index': int(v.frame_index)}
            for k,v in result.characterization.representatives.items()}
        structures = result.structural
        from clustermd.reporting.structural_status import structural_status, status_code
        structural_details = structural_status(
            spec, structures.info if structures else {},
            clustering.cluster_populations.keys(),
        ) if not spec.cluster_only else "Not performed (cluster-only)"
        record['structural'] = {
            'backend': spec.structural.backend,
            'status': status_code(structural_details),
            'status_detail': structural_details,
            'retained_counts': {str(k):int(v) for k,v in structures.retained_counts.items()} if structures else {},
            'rmsds_angstrom': {str(k):float(v) for k,v in structures.rmsds.items()} if structures else {},
            'retained_frame_indices': {str(k):v['retained_frame_indices'] for k,v in structures.info.items()
                if isinstance(v,dict) and 'retained_frame_indices' in v} if structures else {},
        }
        record['timings_seconds'] = result.timings
    target = Path(spec.results_dir) / 'run_record.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, default=_json_value, allow_nan=False) + '\n')
    temporary.replace(target)


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime_environment(spec):
    import importlib.metadata
    import platform
    from clustermd import __version__
    environment = {'clusterMD': __version__, 'python': platform.python_version()}
    for name in ['numpy','scipy','fast-hdbscan','matplotlib','psutil']:
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment[name] = 'not installed'
    # Ordinary cluster-only runs do not invoke a native tool for reporting.
    if not spec.cluster_only:
        from clustermd.utils.environment import get_software_env, get_gromacs_env
        if spec.structural.backend == 'gromacs':
            environment.update(get_gromacs_env())
        elif spec.structural.backend == 'carma':
            detected = get_software_env()
            environment.update(carma_path=detected['carma_path'], carma_version=detected['carma_ver'])
    return environment
