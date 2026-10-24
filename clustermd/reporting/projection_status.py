"""Explicit validation scopes, native executable roles and filter references."""
from pathlib import Path
import logging
import shutil


FILTER_POLICIES = {
    'rmsd_cutoff_reference': 'original centroid-nearest representative structure',
    'filtered_alignment_reference': 'original representative; no realignment to the first retained frame',
    'reported_maximum_rmsd_reference': 'original representative; maximum over all assigned frames before optional RMSD filtering',
    'rmsd_cutoff_interpretation': 'retain frames whose representative-relative RMSD is <= the threshold in Angstrom; zero disables filtering',
}


def numerical_diagnostic(spec):
    """Do not infer numerical validation from unverified existing-projection metadata."""
    if spec.projection.generate:
        return spec.projection.provenance.get('numerical_diagnostics') or {
            'status': 'unavailable', 'reason': 'no generated numerical diagnostic'}
    return {'status': 'not_assessed',
            'reason': 'existing projection reused; no numerical diagnostic performed in this run'}


def executable_roles(spec):
    generation = spec.projection.provenance
    return {
        'projection_generation': (generation.get('native_executable') if spec.projection.generate
                                  else {'status': 'not_invoked_existing_projection'}),
        'structural_processing': {
            'requested': 'gmx' if spec.structural.backend == 'gromacs' else 'carma',
            'resolved_path': shutil.which('gmx' if spec.structural.backend == 'gromacs' else 'carma'),
            'identity_scope': 'structural executable resolved from PATH; not a projection override',
            'cluster_only': spec.cluster_only,
        },
    }


def log_numerical_status(spec):
    diagnostic = numerical_diagnostic(spec)
    message = '  Projection numerical diagnostic: %s (separate from runtime validation)'
    if diagnostic['status'] == 'warning':
        logging.warning(message, 'WARNING; see projection/numerical_diagnostics.json')
    else:
        logging.info(message, diagnostic['status'].upper())


def append_projection_status(context, runtime_failed=False):
    """Append clear validation notes without dumping Python dictionaries."""
    from clustermd.reporting.formatting import section, field

    spec = context.spec
    diagnostic = numerical_diagnostic(spec)
    roles = executable_roles(spec)
    with (Path(context.results_dir) / 'clusterMD_analysis_report.txt').open('a', encoding='utf-8') as report:
        report.write(section('VALIDATION'))
        report.write(field('Runtime validation', 'FAILED' if runtime_failed else 'PASSED'))
        report.write(field('Projection diagnostic', diagnostic['status'].upper().replace('_', ' ')))
        if diagnostic.get('scope') or diagnostic.get('reason'):
            report.write(field('Diagnostic scope', diagnostic.get('scope', diagnostic.get('reason'))))
        if 'atol' in diagnostic:
            report.write(field('Numerical tolerances',
                               f"atol={diagnostic['atol']}, rtol={diagnostic.get('rtol')}, "
                               f"ddof={diagnostic.get('variance_ddof', 0)}"))
            report.write(field('Diagnostic details', 'projection/numerical_diagnostics.json'))
        if diagnostic['status'] == 'warning':
            report.write('  [WARN] Native projected variances differ from native eigenvalues.\n')
            report.write('         Runtime validation does not certify numerical equivalence.\n')
        report.write(field('Scientific equivalence', 'Requires independent native/reference comparisons'))
        generation = roles['projection_generation']
        if spec.projection.generate:
            if isinstance(generation, dict):
                generator = generation.get('path') or generation.get('resolved_path') or generation.get('executable')
                report.write(field('Projection executable', generator or 'See projection metadata'))
            else:
                report.write(field('Projection executable', generation or 'See projection metadata'))
        else:
            report.write(field('Projection executable', 'Not invoked (existing projection)'))
        structural = roles['structural_processing']
        report.write(field('Structural executable', structural.get('resolved_path') or 'Not found on PATH'))
    log_numerical_status(spec)
