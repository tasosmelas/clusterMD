"""Native GROMACS PCA and phi/psi dPCA; source ordinal stride, not time stride."""
from pathlib import Path
import logging
import math
import numpy as np
from clustermd.models import ProjectionValidationError
from clustermd.constants import Colors
from clustermd.projection.contracts import GeneratedProjection
from clustermd.projection.selection import (parse_residues, window_ordinals,
    selected_residue_keys, torsion_owner, write_selection_audit)
from clustermd.utils.gromacs import run_gmx, read_trajectory_times


def read_index(path, *, native_blocks=False):
    """Read strict user groups, or retain every ordinal native dihedral block.

    mk_angndx labels describe rounded parameters and need not be unique.
    Native blocks are keyed by (ordinal, label), never overwritten or merged.
    """
    groups = {}
    group = None
    for raw in Path(path).read_text().splitlines():
        line = raw.split(';', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            label = line[1:-1].strip()
            group = (len(groups), label) if native_blocks else label
            if group in groups:
                raise ProjectionValidationError(f'Duplicate index group: {group}')
            groups[group] = []
        elif group is not None:
            try:
                groups[group].extend(int(value) for value in line.split())
            except ValueError as exc:
                raise ProjectionValidationError('Malformed GROMACS index file.') from exc
    return groups


def write_index(path, name, indices, width=15):
    with Path(path).open('w') as handle:
        handle.write(f'[ {name} ]\n')
        for start in range(0, len(indices), width):
            handle.write(' '.join(map(str, indices[start:start + width])) + '\n')


def pdb_atoms(path):
    atoms = []
    boundary = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith('TER'):
            boundary += 1
        if line.startswith(('ATOM  ', 'HETATM')):
            atoms.append({'name': line[12:16].strip(),
                          'residue': (boundary, line[21:22], line[22:27], line[17:20].strip()),
                          'element': line[76:78].strip().upper(),
                          'xyz': np.array([float(line[a:b]) for a, b in [(30,38),(38,46),(46,54)]])})
    if not atoms:
        raise ProjectionValidationError('Reference PDB contains no atoms.')
    return atoms


def backbone_torsions(atoms, allowed):
    """GRO fallback: adjacent complete residues with a plausible peptide C-N bond.

    The initial whole-protein geometry defines connectivity; gaps/chain boundaries
    are never bridged. TPR generation instead uses actual native dihedrals.
    """
    residues = []
    for index, atom in enumerate(atoms, 1):
        key = atom['residue']
        if not residues or residues[-1][0] != key:
            residues.append((key, {}))
        if index in allowed and atom['name'] in {'N', 'CA', 'C'}:
            if atom['name'] in residues[-1][1]:
                raise ProjectionValidationError('Duplicate backbone atoms under an ambiguous residue identifier. '
                    'GRO/PDB residue numbering does not establish unique connectivity; use a matching TPR '
                    'or --torsions with explicit atom quadruplets.')
            residues[-1][1][atom['name']] = index
    torsions = []
    for (key, left), (nextkey, right) in zip(residues, residues[1:]):
        if key[:2] != nextkey[:2] or set(left) != {'N', 'CA', 'C'} or set(right) != {'N', 'CA', 'C'}:
            continue
        distance = np.linalg.norm(atoms[left['C']-1]['xyz'] - atoms[right['N']-1]['xyz'])
        if 0.9 <= distance <= 1.8:  # Angstrom; peptide bond ~1.33
            torsions.extend([(left['N'],left['CA'],left['C'],right['N']),
                             (left['C'],right['N'],right['CA'],right['C'])])
    return torsions


def topology_torsions(groups, atoms, allowed):
    torsions = []
    seen = set()
    for indices in groups.values():
        if len(indices) % 4:
            raise ProjectionValidationError('Native dihedral index is not a list of quadruplets.')
        for pos in range(0, len(indices), 4):
            quad = tuple(indices[pos:pos+4])
            if not set(quad) <= allowed:
                continue
            names = tuple(atoms[i-1]['name'] for i in quad)
            if names not in {('N','CA','C','N'), ('C','N','CA','C')}:
                quad = quad[::-1]
                names = names[::-1]
            if names not in {('N','CA','C','N'), ('C','N','CA','C')}:
                continue
            keys = [atoms[i-1]['residue'] for i in quad]
            valid = (keys[0] == keys[1] == keys[2] != keys[3] if names[0] == 'N'
                     else keys[0] != keys[1] == keys[2] == keys[3])
            if valid and quad not in seen:
                seen.add(quad)
                torsions.append(quad)
    return torsions


def generate_gromacs(spec, work):
    config = spec.projection
    topology = str(Path(spec.topology_file).resolve())
    trajectory = str(Path(spec.trajectory_file).resolve())
    fulltimes = read_trajectory_times(trajectory, topology)
    ordinals = window_ordinals(len(fulltimes), config)
    times = fulltimes[ordinals]
    if config.projection_dimensions > len(times) - 1:
        raise ProjectionValidationError('--pdim exceeds the sample covariance rank (sampled frames minus one).')
    if len(times) < 2:
        raise ProjectionValidationError('--pstep leaves fewer than two trajectory frames.')
    commands = []
    native_log = work / 'native_generation.log'
    def run(args, selections=None):
        args = [str(value) for value in args]
        commands.append({'argv': ['gmx', *args], 'stdin_groups': selections or []})
        with native_log.open('a') as out:
            out.write('$ gmx ' + ' '.join(args) + '\nstdin groups: ' + repr(selections or []) + '\n')
        try:
            result = run_gmx(args, selections=selections, cwd=work)
        except Exception as exc:
            with native_log.open('a') as out:
                out.write(str(exc) + '\n')
            raise
        with native_log.open('a') as out:
            out.write(result.stdout + '\n')
    # Frame ordinals are selected with -skip; -dt would mean physical time.
    sampled = work / 'sampled.trr'
    has_bonds = Path(topology).suffix.lower() == '.tpr'
    pbc = 'mol' if has_bonds else 'no'
    if not has_bonds:
        logging.warning(f'{Colors.YELLOW}[WARN]{Colors.RESET} GRO has no bond topology: projection assumes whole input molecules. Use TPR for automatic PBC reconstruction.')
    custom_window = config.frame_start is not None or config.frame_stop is not None
    if custom_window:
        if not np.isfinite(fulltimes).all() or np.any(np.diff(fulltimes) <= 0):
            raise ProjectionValidationError('Frame windows require unique increasing GROMACS timestamps.')
        windowed = work / 'windowed.trr'
        stop = config.frame_stop or len(fulltimes)
        run(['trjconv', '-s', topology, '-f', trajectory, '-o', windowed,
             '-b', f'{fulltimes[(config.frame_start or 1)-1]:.17g}',
             '-e', f'{fulltimes[stop-1]:.17g}', '-pbc', pbc], ['System'])
        run(['trjconv', '-s', topology, '-f', windowed, '-o', sampled,
             '-skip', config.projection_stride, '-pbc', 'no'], ['System'])
        from clustermd.utils.gromacs import check_output_times
        check_output_times(read_trajectory_times(str(sampled), topology), times,
                           'Windowed projection trajectory')
    else:
        run(['trjconv', '-s', topology, '-f', trajectory, '-o', sampled,
             '-skip', config.projection_stride, '-pbc', pbc], ['System'])
    reference = work / 'reference.pdb'
    run(['trjconv', '-s', topology, '-f', sampled, '-o', reference,
         '-b', f'{times[0]:.17g}', '-e', f'{times[0]:.17g}'], ['System'])
    atoms = pdb_atoms(reference)
    if config.index_file:
        index = Path(config.index_file).expanduser().resolve()
        if not index.is_file():
            raise ProjectionValidationError(f'Index file not found: {index}')
    else:
        index = work / 'default_groups.ndx'
        run(['make_ndx', '-f', topology, '-o', index], ['q'])
    groups = read_index(index)
    group_name = config.segment or 'Protein'
    if group_name not in groups:
        raise ProjectionValidationError(f'GROMACS group {group_name!r} not found. Use --segid with an exact group name, optionally --index. Available: {list(groups)}')
    allowed = set(groups[group_name])
    if not allowed or min(allowed) < 1 or max(allowed) > len(atoms):
        raise ProjectionValidationError('Projection group is empty or has out-of-range atoms.')
    residue_keys = selected_residue_keys(atoms, allowed, parse_residues(config.residues))
    residue_atoms = {i for i in allowed if residue_keys is None or atoms[i-1]['residue'] in residue_keys}
    eigenval = work / 'eigenval.xvg'
    eigenvec = work / 'eigenvec.trr'
    average = work / 'projection_average.pdb'
    torsion_description = None
    audit = work / 'selection_audit.tsv'
    artifacts = [reference, native_log, audit]
    omitted_angles = []
    resolved_features = None
    if config.projection_method == 'pca':
        preset = config.atom_preset or 'ca'
        def selected(atom):
            name = atom['name'].lstrip('0123456789')
            if preset == 'ca':
                return name == 'CA'
            if preset == 'backbone':
                return name in {'N', 'CA', 'C', 'O'}
            element = atom['element']
            return element not in {'H', 'D'} if element else not name.startswith(('H', 'D'))
        indices = [i for i in sorted(residue_atoms) if selected(atoms[i-1])]
        write_selection_audit(audit, atoms, indices=indices)
        resolved_features = len(indices)
        if len(indices) < 3 or config.projection_dimensions > 3*len(indices)-6:
            raise ProjectionValidationError('Selected Cartesian atoms cannot support --pdim after rigid-body removal.')
        analysis_index = work / 'projection_atoms.ndx'
        write_index(analysis_index, 'Projection', indices)
        artifacts.append(analysis_index)
        run(['covar', '-s', reference, '-f', sampled, '-n', analysis_index,
             '-o', eigenval, '-v', eigenvec, '-av', average, '-l', work/'covar.log',
             '-fit', '-nomwa', '-nopbc', '-last', config.projection_dimensions], ['Projection', 'Projection'])
        projection_ref, projection_traj = reference, sampled
        anaeig_index = ['-n', analysis_index]
        anaeig_groups = ['Projection', 'Projection']
        fit = 'equal-weight fit to first sampled frame; covariance about arithmetic mean'
    else:
        if config.torsion_file:
            source = Path(config.torsion_file).expanduser().resolve()
            groups_t = read_index(source)
            if len(groups_t) != 1:
                raise ProjectionValidationError('--torsions must contain exactly one dihedral group.')
            values = next(iter(groups_t.values()))
            if not values or len(values) % 4 or min(values) < 1 or max(values) > len(atoms):
                raise ProjectionValidationError('Torsion index needs valid atom quadruplets.')
            torsions = [tuple(values[p:p+4]) for p in range(0, len(values),4)]
            if any(len(set(q)) != 4 or not set(q) <= allowed for q in torsions):
                raise ProjectionValidationError('Torsions must use four distinct atoms inside the selected group.')
            torsion_description = 'explicit atom quadruplets (--torsions)'
        elif has_bonds:
            native_index = work / 'native_dihedrals.ndx'
            run(['mk_angndx', '-s', topology, '-n', native_index, '-type', 'dihedral'])
            torsions = topology_torsions(read_index(native_index, native_blocks=True), atoms, allowed)
            torsion_description = 'backbone phi/psi from TPR native dihedral definitions'
        else:
            torsions = backbone_torsions(atoms, allowed)
            torsion_description = 'backbone phi/psi inferred from first-frame peptide geometry (GRO lacks bonds)'
            logging.warning(f'{Colors.YELLOW}[WARN]{Colors.RESET} GRO dPCA infers peptide connectivity from the first frame; inspect projection/torsions.ndx or supply --torsions.')
        if residue_keys is not None:
            torsions = [q for q in torsions if (set(q) <= residue_atoms if config.torsion_scope == 'internal'
                        else torsion_owner(q, atoms)[1] in residue_keys)]
        omitted_angles = write_selection_audit(audit, atoms, torsions=torsions, selected_keys=residue_keys)
        resolved_features = len(torsions)
        if omitted_angles:
            logging.warning(f'{Colors.YELLOW}[WARN]{Colors.RESET} %d requested phi/psi angles unavailable; see selection audit/provenance.', len(omitted_angles))
        if not torsions or config.projection_dimensions > 2*len(torsions):
            raise ProjectionValidationError('Selected backbone torsions cannot support --pdim. Check segments/connectivity or supply --torsions.')
        torsion_index = work / 'torsions.ndx'
        write_index(torsion_index, 'Dihedrals', [i for q in torsions for i in q], width=4)
        angular = work / 'dihedral_features.trr'
        run(['angle', '-f', sampled, '-n', torsion_index, '-type', 'dihedral',
             '-or', angular, '-od', work/'angle_distribution.xvg'], ['Dihedrals'])
        dummy = work / 'dihedral_reference.gro'
        count = math.ceil(2*len(torsions)/3)
        with dummy.open('w') as out:
            out.write(f'dPCA cos/sin feature reference\n{count}\n')
            for i in range(1,count+1):
                out.write(f'{1:5d}{"DUM":<5}{"C":>5}{i%100000:5d}{0:8.3f}{0:8.3f}{0:8.3f}\n')
            out.write('   0.00000   0.00000   0.00000\n')
        run(['covar', '-s', dummy, '-f', angular, '-o', eigenval, '-v', eigenvec,
             '-av', average, '-l', work/'covar.log', '-nofit', '-nomwa', '-nopbc',
             '-last', config.projection_dimensions], ['System'])
        artifacts.extend([torsion_index, dummy])
        projection_ref, projection_traj = dummy, angular
        anaeig_index, anaeig_groups = [], ['System']
        fit = 'none (cos/sin internal coordinates; dummy reference)'
    projection = work / f'projection_{config.projection_method}.xvg'
    run(['anaeig', '-s', projection_ref, '-f', projection_traj, '-v', eigenvec,
         *anaeig_index, '-first', '1', '-last', config.projection_dimensions,
         '-proj', projection, '-tu', 'ps', '-xvg', 'none'], anaeig_groups)
    artifacts.extend([eigenval, eigenvec, average, work/'covar.log'])
    return GeneratedProjection(projection, artifacts, {
        'selection': {'group': group_name, 'atom_preset': config.atom_preset or 'ca' if config.projection_method == 'pca' else None,
                      'requested_residues': config.residues,
                      'resolved_residues': [list(k) for k in sorted({atoms[i-1]['residue'] for i in residue_atoms})],
                      'resolved_feature_count': resolved_features, 'omitted_angles': omitted_angles,
                      'torsion_scope': config.torsion_scope or 'owned' if config.projection_method == 'dpca' else None,
                      'residue_policy': 'all torsion atoms inside selected residues' if config.torsion_scope == 'internal' else 'angles owned by selected residues; neighboring atoms inside base group may be outside residue selection'},
        'torsions': torsion_description, 'fit_policy': fit,
        'pbc_policy': 'native trjconv -pbc mol' if has_bonds else 'whole input required; no reconstruction',
        'covariance_weighting': 'equal weights (-nomwa)', 'commands': commands,
        'frame_window': {'start_one_based': config.frame_start or 1,
                         'stop_one_based': config.frame_stop or len(fulltimes)},
        'projected_frame_ordinals': ordinals.tolist(), 'projected_times_ps': times.tolist(),
    })
