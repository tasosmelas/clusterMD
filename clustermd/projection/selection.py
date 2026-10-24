"""Generation-only selections: topology residue IDs and inclusive source windows."""
import re
from numbers import Integral
import numpy as np
from clustermd.models import ProjectionValidationError


def parse_residues(text):
    """Integer IDs only; reject ambiguous insertion codes rather than ignoring them."""
    if text is None:
        return None
    if not isinstance(text, str) or not text.strip():
        raise ProjectionValidationError('--residues cannot be empty.')
    result = set()
    for token in text.split(','):
        match = re.fullmatch(r'\s*(-?\d+)(?:\s*-\s*(-?\d+))?\s*', token)
        if not match:
            raise ProjectionValidationError('--residues requires integer IDs/ranges, e.g. 5-20,28.')
        first = int(match[1]); last = int(match[2]) if match[2] else first
        if last < first:
            raise ProjectionValidationError('--residues ranges must be ascending.')
        if last - first > 100000:
            raise ProjectionValidationError('--residues range is excessively large.')
        result.update(range(first, last + 1))
    return result


def validate_generation_options(residues, start, stop, generate, torsions=None, torsion_scope=None):
    if any(value is not None for value in (residues, start, stop)) and not generate:
        raise ProjectionValidationError('--residues and frame windows require explicit --proj; they do not modify existing projections.')
    if torsion_scope is not None and (torsion_scope not in {'internal', 'owned'} or not generate or residues is None or torsions is not None):
        raise ProjectionValidationError('--torsion-scope requires dPCA generation with --residues and no --torsions.')
    for value, name in ((start, '--frame-start'), (stop, '--frame-stop')):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
            raise ProjectionValidationError(f'{name} must be a positive one-based frame number.')
    if stop is not None and stop < (start or 1):
        raise ProjectionValidationError('--frame-stop must be >= --frame-start.')
    parse_residues(residues)
    if residues is not None and torsions is not None:
        raise ProjectionValidationError('--residues and --torsions are alternative feature definitions; supply one.')


def window_ordinals(count, config):
    first = config.frame_start or 1
    last = config.frame_stop if config.frame_stop is not None else count
    if first > count or last > count:
        raise ProjectionValidationError(f'Frame window {first}-{last} exceeds trajectory length {count}.')
    indices = np.arange(first - 1, last, config.projection_stride, dtype=np.int64)
    if len(indices) < 2:
        raise ProjectionValidationError('Frame window and --pstep leave fewer than two trajectory frames.')
    if config.projection_dimensions > len(indices) - 1:
        raise ProjectionValidationError('--pdim exceeds the sample covariance rank (sampled frames minus one).')
    return indices


def selected_residue_keys(atoms, allowed, requested):
    """Require each requested ID to identify exactly one residue in the base group."""
    if requested is None:
        return None
    candidates = {}
    for index in sorted(allowed):
        key = atoms[index - 1]['residue']
        raw = key[2].strip()
        if not re.fullmatch(r'-?\d+', raw):
            # Integer selections must not include insertion-coded residues,
            # but unrelated insertion codes elsewhere must not block them.
            # Explicit insertion-code tokens are rejected by parse_residues.
            continue
        candidates.setdefault(int(raw), set()).add(key)
    missing = requested - candidates.keys()
    if missing:
        raise ProjectionValidationError(f'Requested integer residues absent from projection group: {sorted(missing)}. '
            'Integer IDs do not include insertion-coded residues; use an explicit index/torsion selection for those.')
    if any(len(candidates[i]) != 1 for i in requested):
        raise ProjectionValidationError('Residue IDs are ambiguous across chains/residues; select one chain with --segid/--index.')
    keys = {next(iter(candidates[i])) for i in requested}
    if len({key[:2] for key in keys}) != 1:
        raise ProjectionValidationError('--residues currently requires a single chain/segment.')
    return keys


def torsion_owner(quad, atoms):
    names = tuple(atoms[i - 1]['name'] for i in quad)
    if names == ('N', 'CA', 'C', 'N'):
        return 'psi', atoms[quad[1] - 1]['residue']
    if names == ('C', 'N', 'CA', 'C'):
        return 'phi', atoms[quad[2] - 1]['residue']
    raise ProjectionValidationError('Expected an ordered backbone phi/psi quadruplet.')


def write_selection_audit(path, atoms, indices=None, torsions=None, selected_keys=None):
    with path.open('w') as out:
        if indices is not None:
            out.write('feature_index\tsource_atom_index\tresidue_identity\tatom_name\n')
            for feature, index in enumerate(indices):
                atom = atoms[index - 1]
                out.write(f'{feature}\t{index}\t{atom["residue"]!r}\t{atom["name"]}\n')
        else:
            out.write('feature_index\tangle\towner_residue\tatom1\tatom2\tatom3\tatom4\n')
            for feature, quad in enumerate(torsions):
                try:
                    angle, owner = torsion_owner(quad, atoms)
                except ProjectionValidationError:
                    angle, owner = 'explicit', 'user-defined'
                out.write(f'{feature}\t{angle}\t{owner!r}\t' + '\t'.join(map(str, quad)) + '\n')
    if torsions is None or selected_keys is None:
        return []
    present = {torsion_owner(quad, atoms) for quad in torsions}
    return [{'angle': angle, 'residue_identity': list(key),
             'reason': 'no valid torsion in selected group/connectivity'}
            for key in sorted(selected_keys) for angle in ('phi', 'psi')
            if (angle, key) not in present]
