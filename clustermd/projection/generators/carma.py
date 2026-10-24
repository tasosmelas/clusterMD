"""CARMA PCA/dPCA generation with original one-based frame identities."""
from pathlib import Path
import shlex
import os
import hashlib
import re
import struct
import shutil
import subprocess
import numpy as np
from clustermd.models import PipelineExecutionError, ProjectionValidationError
from clustermd.projection.contracts import GeneratedProjection
from clustermd.projection.selection import (parse_residues, selected_residue_keys,
    window_ordinals, write_selection_audit)
from clustermd.utils.gromacs import log_gromacs_detail


def psf_atoms(psf):
    lines = Path(psf).read_text().splitlines(keepends=True)
    for pos, line in enumerate(lines):
        if '!NATOM' in line:
            count = int(line.split()[0])
            atoms = []
            for ordinal, row in enumerate(lines[pos + 1:pos + 1 + count], 1):
                values = row.split()
                if len(values) < 8 or int(values[0]) != ordinal:
                    raise ProjectionValidationError('PSF atom rows must be complete and consecutively numbered.')
                atoms.append({'name': values[4], 'residue': ('PSF', values[1], values[2], values[3]),
                              'line_index': pos + ordinal})
            if len(atoms) != count:
                raise ProjectionValidationError('PSF NATOM section is truncated.')
            return atoms, lines
    raise ProjectionValidationError('PSF has no NATOM section.')


def protein_segments(psf, method='dpca'):
    atoms, _ = psf_atoms(psf)
    segments = {}
    for atom in atoms:
        segments.setdefault(atom['residue'][1], set()).add(atom['name'])
    required = {'N', 'CA', 'C'} if method == 'dpca' else {'CA'}
    return [seg for seg, names in segments.items() if required <= names]


def dcd_frame_count(path):
    with Path(path).open('rb') as handle:
        header = handle.read(92)
    if len(header) != 92 or header[4:8] != b'CORD':
        raise ProjectionValidationError('Cannot read a standard coordinate DCD header for the frame window.')
    endian = '<' if struct.unpack('<i', header[:4])[0] == 84 else '>'
    if struct.unpack(endian + 'i', header[:4])[0] != 84 or struct.unpack(endian + 'i', header[88:92])[0] != 84:
        raise ProjectionValidationError('Unsupported DCD header record.')
    count, start, stride, stop = struct.unpack(endian + '4i', header[8:24])
    if count == 0:
        count = stop - start + stride  # CARMA's native NAMD header fallback
    if count < 1:
        raise ProjectionValidationError('DCD header contains no frames.')
    return count


def mask_psf_residues(path, atoms, lines, allowed):
    """Change only excluded segment fields; retain every atom and all connectivity."""
    segments = {atom['residue'][1].upper() for atom in atoms}
    marker = next((c for c in 'ZYXWVUTSRQPONMLKJIHGFEDCBA' if c not in segments), None)
    if marker is None:
        raise ProjectionValidationError('No unused temporary segment identifier for residue selection.')
    for index, atom in enumerate(atoms, 1):
        if index not in allowed:
            row = lines[atom['line_index']]
            field = list(re.finditer(r'\S+', row))[1]
            lines[atom['line_index']] = row[:field.start()] + marker.ljust(field.end()-field.start()) + row[field.end():]
    path.write_text(''.join(lines))


def native_backbone_torsions(atoms, allowed):
    residues = []
    for i in sorted(allowed):
        atom = atoms[i-1]
        if atom['name'] not in {'N', 'CA', 'C'}:
            continue
        key = atom['residue']
        if not residues or residues[-1][0] != key:
            residues.append((key, {}))
        if atom['name'] in residues[-1][1]:
            raise ProjectionValidationError('Duplicate CARMA backbone atom in a residue.')
        residues[-1][1][atom['name']] = i
    if any(set(names) != {'N', 'CA', 'C'} for _, names in residues):
        raise ProjectionValidationError('CARMA dPCA needs a complete N/CA/C triplet in every selected residue.')
    if len({key for key, _ in residues}) != len(residues):
        raise ProjectionValidationError('CARMA dPCA requires contiguous atom rows for each residue.')
    torsions = []
    for (key, left), (nextkey, right) in zip(residues, residues[1:]):
        if not key[2].lstrip('-').isdigit() or not nextkey[2].lstrip('-').isdigit():
            raise ProjectionValidationError('CARMA dPCA requires integer residue IDs.')
        if key[1] == nextkey[1] and int(nextkey[2]) == int(key[2])+1 and set(left) == set(right) == {'N', 'CA', 'C'}:
            torsions.extend([(left['N'], left['CA'], left['C'], right['N']),
                             (left['C'], right['N'], right['CA'], right['C'])])
    return torsions


def atom_flags(preset):
    names = {'ca': ['CA'], 'backbone': ['CA', 'C', 'N', 'O'], 'heavy': ['HEAVY']}[preset]
    return [value for name in names for value in ('-atmid', name)]


def generate_carma(spec, work):
    config = spec.projection
    if config.index_file or config.torsion_file:
        raise ProjectionValidationError('--index and --torsions apply to GROMACS generation.')
    if config.torsion_scope == 'owned':
        raise ProjectionValidationError('CARMA dPCA does not expose individual residue-owned torsion selection. Use --torsion-scope internal, or GROMACS for owned angles.')
    atoms, psf_lines = psf_atoms(spec.topology_file)
    segments = protein_segments(spec.topology_file, config.projection_method)
    segment = config.segment
    if segment and segment not in segments:
        raise ProjectionValidationError(f'Protein segment {segment!r} not found; available: {segments}')
    if not segment:
        if len(segments) == 1:
            segment = segments[0]
        elif config.projection_method == 'dpca' or config.atom_preset == 'heavy':
            raise ProjectionValidationError(f'CARMA projection needs --segid; protein segments: {segments}')
    if config.residues is not None and not segment:
        raise ProjectionValidationError('CARMA --residues requires one unambiguous --segid.')
    allowed = {i for i, atom in enumerate(atoms, 1) if not segment or atom['residue'][1] == segment}
    residue_keys = selected_residue_keys(atoms, allowed, parse_residues(config.residues))
    if residue_keys is not None:
        allowed = {i for i in allowed if atoms[i-1]['residue'] in residue_keys}
    audit = work / 'selection_audit.tsv'
    if config.projection_method == 'pca':
        preset = config.atom_preset or 'ca'
        indices = [i for i in sorted(allowed) if
                   (atoms[i-1]['name'].upper() == 'CA' if preset == 'ca' else
                    atoms[i-1]['name'].upper() in {'N', 'CA', 'C', 'O'} if preset == 'backbone' else
                    not atoms[i-1]['name'].upper().startswith('H'))]
        if not indices:
            raise ProjectionValidationError('Cartesian projection selection contains no atoms.')
        if residue_keys is not None and (len(indices) < 3 or config.projection_dimensions > 3*len(indices)-6):
            raise ProjectionValidationError('Selected Cartesian atoms cannot support --pdim after rigid-body removal.')
        write_selection_audit(audit, atoms, indices=indices)
        feature_count = len(indices)
    else:
        torsions = native_backbone_torsions(atoms, allowed)
        if not torsions or config.projection_dimensions > 2 * len(torsions):
            raise ProjectionValidationError('Selected CARMA residues contain too few native backbone torsions for --pdim.')
        omitted_angles = write_selection_audit(audit, atoms, torsions=torsions, selected_keys=residue_keys)
        feature_count = len(torsions)
    custom_window = config.frame_start is not None or config.frame_stop is not None
    count = dcd_frame_count(spec.trajectory_file) if custom_window else None
    source_ids = window_ordinals(count, config) + 1 if custom_window else None
    window_flags = ['-first', str(config.frame_start or 1), '-last', str(config.frame_stop or count)] if custom_window else []
    # Native filename-derived outputs must stay inside the sandbox.
    for name, source in [('input.dcd', spec.trajectory_file), ('input.psf', spec.topology_file)]:
        (work / name).symlink_to(Path(source).resolve())
    if residue_keys is not None:
        (work / 'input.psf').unlink()
        mask_psf_residues(work / 'input.psf', atoms, psf_lines, allowed)
    requested_executable = config.carma_executable or 'carma'
    resolved_executable = shutil.which(os.path.expanduser(requested_executable))
    if resolved_executable is not None:
        resolved_executable = str(Path(resolved_executable).resolve())
    if config.carma_executable is not None and resolved_executable is None:
        raise ProjectionValidationError(f'CARMA projection executable was not found or is not executable: {requested_executable}')
    executable_sha256 = None
    if resolved_executable is not None:
        digest = hashlib.sha256()
        with Path(resolved_executable).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        executable_sha256 = digest.hexdigest()
    commands = []
    native_log = work / 'native_generation.log'
    def run(args):
        command = [resolved_executable if config.carma_executable is not None else 'carma', *args]
        commands.append({'argv': command, 'cwd': 'temporary projection workspace'})
        log_gromacs_detail('CARMA projection command: %s', shlex.join(command))
        try:
            result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
        except FileNotFoundError as exc:
            raise PipelineExecutionError(f'CARMA executable {requested_executable!r} was not found.') from exc
        with native_log.open('a') as out:
            out.write('$ ' + shlex.join(command) + '\n' + result.stdout + '\n')
        log_gromacs_detail('%s', result.stdout)
        if result.returncode:
            raise PipelineExecutionError(f'CARMA projection failed: {shlex.join(command)}\n{result.stdout}')
    segflags = ['-segid', segment] if segment else []
    dimensions = str(config.projection_dimensions)
    # One landscape component avoids CARMA's unrelated native clustering stage.
    projargs = [dimensions, '1', str(spec.temp)]
    if config.projection_method == 'pca':
        flags = atom_flags(config.atom_preset or 'ca') + segflags
        # Preserve all atoms/PSF correspondence; covariance reads only the fitted selection.
        run(['-v', '-w', '-fit', '-noCAs', *window_flags,
             *(['-ref', str(config.frame_start or 1)] if custom_window else []),
             *flags, 'input.dcd', 'input.psf'])
        run(['-v', '-w', '-cov', '-eigen', '-proj', *projargs,
             '-step', str(config.projection_stride), *flags, 'carma.fitted.dcd', 'input.psf'])
        projection = work / 'carma.PCA.fluctuations.dat'
    else:
        run(['-v', '-w', *window_flags, *segflags, '-dPCA', *projargs,
             '-step', str(config.projection_stride), 'input.dcd', 'input.psf'])
        projection = work / 'carma.dPCA.fluctuations.dat'
    if not projection.is_file() and config.projection_method == 'pca' and (work / 'carma.fluctuations.dat').is_file():
        (work / 'carma.fluctuations.dat').rename(projection)
    if not projection.is_file():
        raise ProjectionValidationError('CARMA did not write its fluctuations file.')
    try:
        rows = np.loadtxt(projection, ndmin=2)
    except ValueError as exc:
        raise ProjectionValidationError('CARMA generated projection is malformed.') from exc
    if rows.shape[1] != config.projection_dimensions + 1 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ProjectionValidationError('CARMA generated projection has invalid shape or values.')
    expected = source_ids if custom_window else np.arange(len(rows), dtype=np.int64) * config.projection_stride + 1
    native_expected = (np.arange(len(source_ids), dtype=np.int64) * config.projection_stride + 1
                       if custom_window and config.projection_method == 'pca' else expected)
    if not np.array_equal(rows[:, 0], native_expected):
        raise ProjectionValidationError('CARMA generated frame numbers differ from original trajectory stride.')
    if custom_window and config.projection_method == 'pca':
        # Fitted DCD is compacted: restore original one-based IDs without rounding coordinates.
        shutil.copy2(projection, work / 'carma.PCA.native_row_ids.dat')
        original = projection.read_text().splitlines(keepends=True)
        if len(original) != len(expected):
            raise ProjectionValidationError('Unexpected CARMA projection line layout during frame remapping.')
        projection.write_text(''.join(re.sub(r'^(\s*)\S+', lambda m, frame=int(frame): m[1] + str(frame), line, count=1)
                                      for frame, line in zip(expected, original)))
    # Use inferred segment for structural extraction as well; atom choice remains independent.
    if spec.structural.selection is None and segment:
        spec.structural.selection = segment
    artifacts = [native_log, audit, *([work / 'input.psf'] if residue_keys is not None else []),
                 *work.glob('carma.*eigen*.dat'), *work.glob('*.varcov.dat'), *work.glob('carma.PCA.native_row_ids.dat')]
    return GeneratedProjection(projection, artifacts, {
        'native_executable': {'requested': requested_executable, 'resolved_path': resolved_executable,
                              'sha256': executable_sha256, 'explicit_override': config.carma_executable is not None},
        'selection': {'segment': segment, 'atom_preset': config.atom_preset or 'ca' if config.projection_method == 'pca' else None,
                      'requested_residues': config.residues,
                      'resolved_residues': [list(k) for k in sorted({atoms[i-1]['residue'] for i in allowed})],
                      'resolved_feature_count': feature_count,
                      'torsion_scope': 'internal' if config.projection_method == 'dpca' else None,
                      'omitted_angles': omitted_angles if config.projection_method == 'dpca' else [],
                      'residue_policy': 'all torsion atoms inside selected residues; CARMA native consecutive-residue triplets' if config.projection_method == 'dpca' else 'selected Cartesian atoms'},
        'torsions': 'backbone phi/psi (CARMA native)' if config.projection_method == 'dpca' else None,
        'fit_policy': ('CARMA native fit to first selected window frame' if custom_window else 'CARMA native fit to first frame') if config.projection_method == 'pca' else 'none (internal coordinates)',
        'pbc_policy': 'input DCD must already have whole molecules; no automatic reconstruction',
        'covariance_weighting': 'equal weights (no -mass)', 'commands': commands,
        'frame_window': {'start_one_based': config.frame_start or 1,
                         'stop_one_based': (config.frame_stop or count) if custom_window else None},
        'projected_frame_ids': expected.tolist(),
    })
