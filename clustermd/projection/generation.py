'Coordinate native PCA/dPCA projection generation, validation and provenance.\n\nNative generators produce projection coordinates; clustering is downstream.'
from pathlib import Path
import json
import logging
import shutil
import tempfile
import time
import hashlib
import re
import numpy as np
from clustermd.models import ProjectionValidationError
from clustermd.constants import Colors


from clustermd.projection.contracts import GeneratedProjection


def generate_projection(spec):
    """Generate a projection with the selected native backend and record provenance."""
    from clustermd.projection.generators.carma import generate_carma
    from clustermd.projection.generators.gromacs import generate_gromacs
    generators = {'carma': generate_carma, 'gromacs': generate_gromacs}
    config = spec.projection
    from clustermd.projection.selection import validate_generation_options
    validate_generation_options(config.residues, config.frame_start, config.frame_stop,
                                config.generate, config.torsion_file, config.torsion_scope)
    if config.torsion_scope is not None and config.projection_method != "dpca":
        raise ProjectionValidationError("--torsion-scope requires --proj dpca.")
    backend = config.generation_backend or ("gromacs" if config.backend == "gromacs" else "carma")
    if config.carma_executable is not None and (backend != 'carma' or not config.generate
            or not isinstance(config.carma_executable, str) or not config.carma_executable.strip()):
        raise ProjectionValidationError('--carma-executable applies only to CARMA projection generation and needs a nonempty path/name.')
    expected_reader = {"carma": "precomputed", "gromacs": "gromacs"}.get(backend)
    if expected_reader is None or config.backend != expected_reader:
        raise ProjectionValidationError("Native generation backend and projection reader are incompatible.")
    config.generation_backend = backend
    if not spec.trajectory_file or not spec.topology_file:
        raise ProjectionValidationError('Projection generation requires trajectory and topology.')
    if config.projection_method == 'dpca' and config.atom_preset:
        raise ProjectionValidationError('--atom applies to Cartesian PCA, not dPCA.')
    if spec.dim > config.projection_dimensions:
        raise ProjectionValidationError('--dim cannot exceed --pdim for a generated projection.')
    if config.projection_dimensions < 1 or config.projection_stride < 1:
        raise ProjectionValidationError('Projection dimensions and stride must be positive.')
    output = Path(spec.results_dir).resolve() / 'projection'
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ProjectionValidationError(f'Generation directory is not empty: {output}')
    logging.info(f'{Colors.LIGHT_BLUE}[*]{Colors.RESET} Generating %s projection via %s (%s components, stride %s)...',
                 config.projection_method.upper(), backend,
                 config.projection_dimensions, config.projection_stride)
    started = time.perf_counter()
    input_checksums = {name: _sha256(path) for name, path in
                       [('trajectory', spec.trajectory_file), ('topology', spec.topology_file),
                        ('index', config.index_file), ('torsions', config.torsion_file)] if path}
    with tempfile.TemporaryDirectory(prefix='clustermd-projection-') as work:
        try:
            generated = generators[backend](spec, Path(work))
        except Exception as exc:
            for path in Path(work).iterdir():
                if path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.log', '.ndx', '.pdb', '.gro', '.json'}:
                    shutil.copy2(path, output / path.name)
            config.provenance = {
                'status': 'failed', 'backend': backend,
                'method': config.projection_method, 'error_type': type(exc).__name__,
                'error': str(exc), 'trajectory': spec.trajectory_file,
                'topology': spec.topology_file,
                'requested_residues': config.residues,
                'frame_start': config.frame_start, 'frame_stop': config.frame_stop,
            }
            _save_provenance(spec)
            raise
        if not generated.projection_file.is_file():
            raise ProjectionValidationError('Native backend did not produce a projection.')
        for path in [generated.projection_file, *generated.artifacts]:
            if path.is_file():
                shutil.copy2(path, output / path.name)
        native_text = (output / 'native_generation.log').read_text() if (output / 'native_generation.log').exists() else ''
        native_text = re.sub(r'\x1b\[[0-9;]*m', '', native_text)
        version_lines = [line.strip() for line in native_text.splitlines()
                         if re.search(r'carma\s+v\.?\s*\d|GROMACS.*\d{4}\.\d', line, re.I)]
        spec.projection_file = str(output / generated.projection_file.name)
    config.provenance = {
        'status': 'generated_pending_validation', 'backend': backend,
        'method': config.projection_method, 'dimensions': config.projection_dimensions,
        'projection_stride': config.projection_stride, 'analysis_stride': spec.step,
        'trajectory': spec.trajectory_file, 'topology': spec.topology_file,
        'projection_file': spec.projection_file, 'runtime_seconds': time.perf_counter() - started,
        'input_sha256': input_checksums,
        'native_version_lines': list(dict.fromkeys(version_lines)),
        **generated.provenance,
    }
    if backend == 'gromacs':
        native_path = shutil.which('gmx')
        config.provenance['native_executable'] = {
            'requested': 'gmx', 'resolved_path': native_path,
            'sha256': _sha256(native_path) if native_path else None,
            'explicit_override': False,
        }
    from clustermd.projection.diagnostics import projection_diagnostics
    eigenvalues = output / ('eigenval.xvg' if backend == 'gromacs' else f'carma.{"dPCA" if config.projection_method == "dpca" else "PCA"}.eigenvalues.dat')
    diagnostic = projection_diagnostics(spec.projection_file, eigenvalues, backend)
    (output / 'numerical_diagnostics.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
    config.provenance['numerical_diagnostics'] = diagnostic
    if diagnostic['status'] == 'warning':
        logging.warning(f'{Colors.YELLOW}[WARN]{Colors.RESET} Native projected variances differ from native eigenvalues; see projection/numerical_diagnostics.json. Native coordinates and tolerances unchanged.')
    _save_provenance(spec)
    if backend == 'gromacs':
        frames = np.asarray(config.provenance['projected_frame_ordinals'])
        np.savetxt(output / 'projection_sample_mapping.tsv',
                   np.column_stack((np.arange(len(frames)), frames,
                                    config.provenance['projected_times_ps'])),
                   fmt=['%d', '%d', '%.17g'], delimiter='\t',
                   header='projection_row_index\tsource_frame_ordinal_zero_based\ttime_ps')
    else:
        frames = np.asarray(config.provenance['projected_frame_ids'])
        np.savetxt(output / 'projection_sample_mapping.tsv',
                   np.column_stack((np.arange(len(frames)), frames)), fmt='%d', delimiter='\t',
                   header='projection_row_index\tsource_frame_id_one_based')
    config.provenance['runtime_seconds'] = time.perf_counter() - started
    config.provenance['timing_scope'] = 'generation, input hashing, native artifacts and numerical diagnostic; excludes ingestion'
    _save_provenance(spec)
    logging.info('%s[+]%s Generated projection: %s', Colors.BLUE, Colors.RESET, spec.projection_file)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save_provenance(spec):
    (Path(spec.results_dir) / 'projection' / 'projection_generation.json').write_text(
        json.dumps(spec.projection.provenance, indent=2) + '\n')


def validate_generated_dataset(spec, projection):
    """Check generated coordinates and source-frame identity before clustering."""
    info = spec.projection.provenance
    if projection.coordinates.shape[1] != spec.projection.projection_dimensions:
        raise ProjectionValidationError('Generated projection dimension count differs from --pdim.')
    if (spec.projection.generation_backend or spec.projection.backend) == 'gromacs':
        expected_times = np.asarray(info['projected_times_ps'])[::spec.step]
        from clustermd.utils.gromacs import check_output_times
        check_output_times(projection.times, expected_times, 'Generated projection')
        expected_frames = np.asarray(info['projected_frame_ordinals'])[::spec.step]
        # Cluster-only also needs verified upstream identities, unlike an external XVG.
        projection.frame_indices = expected_frames.astype(np.int64)
        projection.metadata.update(frame_mapping_verified=True,
            frame_index_semantics='zero_based_trajectory_frame_ordinal')
    else:
        expected_frames = np.asarray(info['projected_frame_ids'])[::spec.step]
        if not np.array_equal(projection.frame_indices, expected_frames):
            raise ProjectionValidationError('Generated CARMA frame IDs do not match source stride.')
        projection.metadata['frame_index_semantics'] = 'one_based_trajectory_frame_ordinal'
    stats = projection.metadata['parse_stats']
    stats.update(first_sampled_frame=int(projection.frame_indices[0]),
                 last_sampled_frame=int(projection.frame_indices[-1]),
                 frame_index_semantics=projection.metadata['frame_index_semantics'])
    projection.feature_type = 'dPCA' if spec.projection.projection_method == 'dpca' else 'PCA'
    projection.metadata.update(projection_stride=spec.projection.projection_stride,
                               generation_provenance=info)
    info['status'] = 'validated'
    info['validation_scope'] = 'dimensions, finite coordinates, source frame identities and timestamps'
    info['scientific_equivalence'] = 'not established by runtime validation'
    _save_provenance(spec)
    np.savetxt(Path(spec.results_dir) / 'projection' / 'analysis_sample_mapping.tsv',
               np.column_stack((projection.sample_indices, projection.frame_indices)),
               fmt='%d', delimiter='\t', header='sample_index\tsource_frame_id')


def append_generation_report(spec):
    """Append a concise native-projection summary; full metadata remains in JSON."""
    from clustermd.reporting.formatting import section, field

    info = spec.projection.provenance
    method = info.get('method', spec.projection.projection_method)
    backend = info.get('backend', spec.projection.generation_backend)
    with (Path(spec.results_dir) / 'clusterMD_analysis_report.txt').open('a', encoding='utf-8') as report:
        report.write(section('PROJECTION GENERATION'))
        for label, value in (
            ('Backend', str(backend).upper()),
            ('Method', 'dPCA' if method == 'dpca' else 'PCA' if method == 'pca' else method),
            ('Components', info.get('dimensions')),
            ('Projection stride', info.get('projection_stride')),
            ('Analysis stride', info.get('analysis_stride')),
            ('Generation status', info.get('status')),
        ):
            if value is not None:
                report.write(field(label, value))
        selection = info.get('selection')
        if isinstance(selection, dict):
            if method == 'pca':
                selection_label = ' / '.join(str(v) for v in
                    (selection.get('group'), selection.get('atom_preset')) if v)
            else:
                selection_label = str(selection.get('segment') or selection.get('group') or '')
            if selection_label:
                report.write(field('Projection selection', selection_label))
            features = selection.get('resolved_feature_count')
            if features is not None:
                report.write(field('Features', features))
        for key, label in (('fit_policy', 'Fit policy'),
                           ('pbc_policy', 'PBC policy'),
                           ('covariance_weighting', 'Covariance weighting')):
            value = info.get(key)
            if isinstance(value, str) and value:
                report.write(field(label, value))
        report.write(field('Detailed metadata', 'projection/'))
        report.write('  Native commands, residue selection, executable identity and numerical\n'
                     '  diagnostics are recorded in projection/ metadata files.\n')
