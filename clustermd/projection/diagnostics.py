'Check generated projection variances against associated native eigenvalues.\n\nThe diagnostic checks numerical consistency, not physical state validity.'
from pathlib import Path
import numpy as np
from clustermd.models import ProjectionValidationError


def projection_diagnostics(projection_file, eigenvalue_file, backend):
    """Compare native eigenvalues with generated coordinate variances.

Uses population variance (ddof=0), atol=1e-5 and rtol=1e-4;
missing eigenvalue evidence is reported as unavailable."""
    report = {'scope': 'native projected population variances versus native eigenvalues',
              'atol': 1e-5, 'rtol': 1e-4, 'variance_ddof': 0,
              'changes_native_output': False}
    if eigenvalue_file is None or not Path(eigenvalue_file).is_file():
        return {**report, 'status': 'unavailable', 'reason': 'native eigenvalue file absent'}
    if backend == 'gromacs':
        from clustermd.projection.gromacs import read_gromacs_projection
        coordinates = read_gromacs_projection(projection_file).coordinates
    else:
        coordinates = np.loadtxt(projection_file, ndmin=2)[:, 1:]
    with Path(eigenvalue_file).open() as handle:
        rows = [line for line in handle if line.strip() and not line.lstrip().startswith(('#', '@'))]
    try:
        eigenvalues = np.loadtxt(rows, ndmin=2)
    except ValueError as exc:
        raise ProjectionValidationError('Malformed native eigenvalue output.') from exc
    eigenvalues = eigenvalues[:, -1][:coordinates.shape[1]]
    if len(eigenvalues) != coordinates.shape[1] or not np.isfinite(eigenvalues).all():
        raise ProjectionValidationError('Native eigenvalues are incomplete or nonfinite.')
    variances = np.var(coordinates, axis=0, ddof=0, dtype=np.float64)
    errors = np.abs(variances-eigenvalues)
    matches = errors <= report['atol'] + report['rtol'] * np.abs(eigenvalues)
    return {**report, 'status': 'passed' if matches.all() else 'warning',
            'sample_count': len(coordinates), 'eigenvalues': eigenvalues.tolist(),
            'projected_variances': variances.tolist(),
            'absolute_errors': errors.tolist(), 'component_matches': matches.tolist(),
            'variance_consistency_passed': bool(matches.all()),
            'interpretation': 'Separate diagnostic; native reproduction requires an external comparison. '
                + ('CARMA 2.3 single-precision accumulation can produce sample-count-dependent discrepancies.'
                   if backend == 'carma' else 'This check does not independently validate the GROMACS eigensolver.')}
