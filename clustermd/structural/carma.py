"""Extract and characterize cluster-associated structures using carma."""

from clustermd.models import StructuralResult

import glob
import logging
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from clustermd.constants import ANSI_ESCAPE
from clustermd.constants import Colors
from clustermd.constants import FALLBACK_ATOM_FLAGS
from clustermd.utils.logging_utils import get_clusterMD_file_handler

def parse_atom_extraction_flags(target_dir):
    """Infer carma atom-selection flags from the analysis log, using backbone fallback."""
    log_files = glob.glob(os.path.join(target_dir, 'last_carma_run.log'))
    if not log_files: return FALLBACK_ATOM_FLAGS, "fallback default backbone selection"
    try:
        with open(log_files[0], 'r') as f:
            for line in f:
                if "atoms are declared in the PSF file" in line:
                    if "non-hydrogen" in line: return ["-atmid", "HEAVY"], "parsed from last_carma_run.log (heavy atoms)"
                    parts = line.split()
                    atoms_idx = parts.index("atoms")
                    atom_names = parts[1:atoms_idx]
                    flags = []
                    for atom in atom_names: flags.extend(["-atomid", atom])
                    return flags, "parsed from last_carma_run.log"
    except (FileNotFoundError, ValueError, IndexError) as e:
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Error parsing atom IDs ({e}). Reverting to default backbone selection.")
    return FALLBACK_ATOM_FLAGS, "fallback default backbone selection (parsing failed)"


def execute_carma_sandbox(cmd_list, run_dir='.'):
    """Run a carma command in an isolated directory and capture its output.

Return stdout, stderr and status; callers remain responsible for checking
return codes and verifying the required native output artifacts."""
    result = subprocess.run(cmd_list, capture_output=True, text=True, cwd=run_dir)
    
    file_handler = get_clusterMD_file_handler()
    if file_handler:
        if result.stdout:
            for line in result.stdout.splitlines(): 
                clean_line = ANSI_ESCAPE.sub('', line)
                file_handler.stream.write(clean_line + "\n")
        if result.stderr:
            for line in result.stderr.splitlines(): 
                clean_line = ANSI_ESCAPE.sub('', line)
                file_handler.stream.write("carma ERROR: " + clean_line + "\n")
        
        if result.returncode != 0:
            file_handler.stream.write(f"carma execution returned non-zero exit code ({result.returncode}) for command: {' '.join(cmd_list)}\n")
        
        file_handler.flush()
        
    return result.stdout, result.stderr, result.returncode


def _process_single_cluster(cid, index, total_clusters, labels, frames, data_nd, abs_original_dcd, abs_original_psf, abs_results_dir, extraction_flags, config, carma_stats, cluster_rmsds, cluster_reps, cluster_struct_counts, fh, representative=None, core_max_rmsds=None):
    """Processes one cluster for structural extraction, alignment, filtering, and representative-frame recovery."""
    rmsd_threshold = getattr(config, "rmsd_threshold", getattr(config, "rmsd", 0.0))
    total_str = f"({index+1:4d}/{total_clusters:4d})"
    progress_prefix = f"    -> Cluster {cid:4d} {total_str} : "
    progress_initial = progress_prefix + "Processing structural ensemble..."
    sys.stdout.write("\r" + progress_initial)
    sys.stdout.flush()
    if fh:
        fh.stream.write(f"    -> Processing structural ensemble: Cluster {cid:4d}...\n")
        fh.flush()
    
    cluster_mask = (labels == cid)
    cluster_frames = frames[cluster_mask]
    initial_frame_count = len(cluster_frames)
    cluster_struct_counts[cid] = initial_frame_count
    
    if representative is None:
        sys.stdout.write("\n")
        raise ValueError("Representative selection must be performed by clusterMD before structural processing.")
        
    nearest_centroid_frame = representative.frame_index
    nearest_centroid_idx_carma = representative.cluster_local_index + 1

    with tempfile.TemporaryDirectory(dir=abs_results_dir) as tmp_dir:
        temp_frames_dat = os.path.join(tmp_dir, "temp_frames.dat")
        np.savetxt(temp_frames_dat, cluster_frames, fmt='%d')
        
        t_sub = time.time()
        _, _, rc_sort = execute_carma_sandbox(["carma", "-v", "-sort", "temp_frames.dat", abs_original_dcd], run_dir=tmp_dir)
        carma_stats['sort'] += (time.time() - t_sub)
        
        temp_full_dcd = os.path.join(tmp_dir, "carma.reordered.dcd")
        if rc_sort == 0 and os.path.exists(os.path.join(tmp_dir, "carma.reordered.dcd")):
            os.rename(os.path.join(tmp_dir, "carma.reordered.dcd"), temp_full_dcd)
        else:
            sys.stdout.write(f"\r    -> Cluster {cid:4d} {total_str} : [FAILED] Trajectory slicing error.          \n")
            cluster_reps[cid] = "FAILED"
            return False
        
        if fh:
            fh.stream.write(f"      -> Phase 1: Aligning cluster ensemble to representative frame...\n")
        t_sub = time.time()
        _, _, rc_cov = execute_carma_sandbox(["carma", "-v", "-w", "-cov"] + extraction_flags + [temp_full_dcd, abs_original_psf], run_dir=tmp_dir)
        _, _, rc_fit = execute_carma_sandbox(["carma", "-v", "-w", "-fit", "-reference", str(nearest_centroid_idx_carma)] + extraction_flags + [temp_full_dcd, abs_original_psf], run_dir=tmp_dir)
        carma_stats['rmsd'] += (time.time() - t_sub)
        
        if rc_fit != 0:
            sys.stdout.write(f"\r    -> Cluster {cid:4d} {total_str} : [FAILED] Structural alignment error.        \n")
            cluster_reps[cid] = "FAILED"
            return False
            
        temp_fitted_dcd = os.path.join(tmp_dir, "carma.fitted.dcd")
        if os.path.exists(os.path.join(tmp_dir, "carma.fitted.dcd")):
            os.rename(os.path.join(tmp_dir, "carma.fitted.dcd"), temp_fitted_dcd)
        else:
            temp_fitted_dcd = temp_full_dcd 
        
        pseudo_psf = os.path.join(tmp_dir, "carma.selected_atoms.psf")
        safe_psf = os.path.join(tmp_dir, "cluster.psf")
        if os.path.exists(pseudo_psf):
            os.rename(pseudo_psf, safe_psf)
        else:
            safe_psf = abs_original_psf

        # The CARMA fitting stage has already calculated RMSD to the fixed representative.
        if fh:
            fh.stream.write("      -> Phase 2: Generating superposition output...\n")
        t_sub = time.time()
        _, _, rc_super = execute_carma_sandbox(["carma", "-v", "-cov", "-dot", "-super"] + extraction_flags + [temp_fitted_dcd, safe_psf], run_dir=tmp_dir)
        carma_stats['super'] += (time.time() - t_sub)
        
        if rc_super != 0:
            sys.stdout.write(f"\r    -> Cluster {cid:4d} {total_str} : [FAILED] Superposition processing error.    \n")
            cluster_reps[cid] = "FAILED"
            return False
            
        super_pdb = os.path.join(tmp_dir, "carma.superposition.pdb")
        if os.path.exists(super_pdb):
            os.rename(super_pdb, os.path.join(abs_results_dir, f"cluster_{cid:04d}_superposition.pdb"))
        
        # carma.fit-rms.dat is generated by the preceding -fit -reference operation.
        # Unlike carma.rms-average.dat, its RMSDs are measured against clusterMD's
        # centroid-nearest observed representative, in Angstrom (CARMA native units).
        fit_rms_path = os.path.join(tmp_dir, "carma.fit-rms.dat")
        cluster_rmsds[cid] = -1.0

        # The full fitted ensemble is a permanent output independently of filtering.
        full_dcd_path = os.path.join(abs_results_dir, f"cluster_{cid:04d}.dcd")
        shutil.copy2(temp_fitted_dcd, full_dcd_path)

        if not os.path.isfile(fit_rms_path):
            logging.warning("No representative-relative CARMA fit RMSD for cluster %s; "
                            "preserving full cluster without applying filtering.", cid)
        else:
            try:
                fit_rms = np.loadtxt(fit_rms_path, ndmin=2)
                rmsd_values = np.asarray(fit_rms[:, 1], dtype=float)
                if len(rmsd_values) != initial_frame_count or not np.all(np.isfinite(rmsd_values)) or np.any(rmsd_values < 0):
                    raise ValueError("CARMA fit RMSD count or values do not match assigned frames")
                cluster_rmsds[cid] = float(np.max(rmsd_values))
                if rmsd_threshold > 0.0:
                    core_mask = rmsd_values <= rmsd_threshold
                    retained_count = int(np.count_nonzero(core_mask))
                    if retained_count:
                        # Subset the already fitted full DCD; no re-fitting to the
                        # first retained core frame, which would change the reference.
                        core_ordinals = np.flatnonzero(core_mask) + 1
                        np.savetxt(temp_frames_dat, core_ordinals, fmt="%d")
                        t_sub = time.time()
                        _, _, rc_filter = execute_carma_sandbox(
                            ["carma", "-v", "-sort", "temp_frames.dat", temp_fitted_dcd],
                            run_dir=tmp_dir,
                        )
                        carma_stats["sort"] += time.time() - t_sub
                        core_intermediate = os.path.join(tmp_dir, "carma.reordered.dcd")
                        if rc_filter or not os.path.isfile(core_intermediate):
                            logging.warning("CARMA core extraction failed for cluster %s; "
                                            "preserving the complete ensemble.", cid)
                        else:
                            shutil.move(core_intermediate, os.path.join(
                                abs_results_dir, f"cluster_{cid:04d}_core.dcd"))
                            cluster_struct_counts[cid] = retained_count
                            if core_max_rmsds is not None:
                                core_max_rmsds[int(cid)] = float(np.max(rmsd_values[core_mask]))
                            if fh:
                                fh.stream.write(
                                    f"      -> RMSD to representative: retained "
                                    f"{retained_count}/{initial_frame_count} frames "
                                    f"at {rmsd_threshold} A.\n")
                    else:
                        logging.warning("No frames passed representative RMSD threshold "
                                        "for cluster %s; full trajectory preserved.", cid)
            except (ValueError, IndexError, OSError) as exc:
                logging.warning("Cannot read CARMA representative RMSD for cluster %s: %s; "
                                "preserving full trajectory.", cid, exc)

        _, _, rc_rep = execute_carma_sandbox(["carma", "-v", "-first", str(nearest_centroid_frame), "-last", str(nearest_centroid_frame), "-pdb"] + extraction_flags + [abs_original_dcd, abs_original_psf], run_dir=tmp_dir)
        extracted_pdbs = [f for f in glob.glob(os.path.join(tmp_dir, "*.pdb")) if "superposition" not in f and "average" not in f]
        
        if rc_rep == 0 and extracted_pdbs:
            os.rename(extracted_pdbs[0], os.path.join(abs_results_dir, f"cluster_{cid:04d}_representative.pdb"))
            cluster_reps[cid] = nearest_centroid_frame
        else:
            sys.stdout.write(f"\r    -> Cluster {cid:4d} {total_str} : [FAILED] Representative structure error.      \n")
            cluster_reps[cid] = "FAILED"
            return False
            
        retained = cluster_struct_counts.get(cid, initial_frame_count)
        progress_final = progress_prefix + f"Extracted {retained}/{initial_frame_count} frames."
        sys.stdout.write("\r" + progress_final + " " * max(0, len(progress_initial) - len(progress_final)) + "\n")
        sys.stdout.flush()
        if fh:
            fh.flush()
        return True


def execute_structural_extraction(unique_clusters, labels, frames, data_nd, target_analysis_dir, results_dir, config, carma_stats, cluster_rmsds, cluster_reps, cluster_struct_counts, skip_processing=False, representatives=None, trajectory_file=None, topology_file=None, trajectory_discovery=None, topology_discovery=None):
    """Main loop managing trajectory slicing, alignment, and representative-structure extraction via carma."""
    selection = getattr(config, "selection", getattr(config, "segid", None))
    search_link_dir = target_analysis_dir
    if 'carma_results_' not in target_analysis_dir:
        carma_dirs = glob.glob(os.path.join(target_analysis_dir, 'carma_results_*'))
        if carma_dirs: search_link_dir = max(carma_dirs, key=os.path.getmtime)
            
    atom_flags, source_str = parse_atom_extraction_flags(search_link_dir)
    segid_flag = ["-segid", selection] if selection else []
    extraction_flags = atom_flags + segid_flag
    
    dcd_links, psf_links = [], []
    for f in os.listdir(search_link_dir):
        full_path = os.path.join(search_link_dir, f)
        if os.path.islink(full_path) and os.path.exists(full_path):
            if f.endswith('.dcd'): dcd_links.append(full_path)
            elif f.endswith('.psf'): psf_links.append(full_path)

    dcd_links = sorted(dcd_links)
    psf_links = sorted(psf_links)

    base_dir = os.path.dirname(search_link_dir) if 'carma_results_' in search_link_dir else search_link_dir
    if not base_dir: base_dir = '.'
    
    fallback_dcds = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.dcd') and not f.startswith(('carma', 'cluster', 'transitions', 'dPCA'))])
    fallback_psfs = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.psf') and not f.startswith(('carma', 'cluster', 'transitions', 'dPCA'))])

    if not trajectory_file and len(dcd_links) > 1:
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple primary trajectory candidates detected:")
        for cand in dcd_links: logging.warning(f"       {os.path.basename(cand)}")
        logging.warning(f"       Selected: {os.path.basename(dcd_links[0])}")
    elif not trajectory_file and not dcd_links and len(fallback_dcds) > 1:
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple fallback trajectory candidates detected:")
        for cand in fallback_dcds: logging.warning(f"       {os.path.basename(cand)}")
        logging.warning(f"       Selected: {os.path.basename(fallback_dcds[0])}")
        
    if not topology_file and len(psf_links) > 1:
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple primary topology candidates detected:")
        for cand in psf_links: logging.warning(f"       {os.path.basename(cand)}")
        logging.warning(f"       Selected: {os.path.basename(psf_links[0])}")
    elif not topology_file and not psf_links and len(fallback_psfs) > 1:
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Multiple fallback topology candidates detected:")
        for cand in fallback_psfs: logging.warning(f"       {os.path.basename(cand)}")
        logging.warning(f"       Selected: {os.path.basename(fallback_psfs[0])}")

    # Centralized discovery takes precedence. Backend-local discovery
    # remains a fallback when explicit inputs are unavailable.
    if trajectory_file:
        original_dcd = trajectory_file
        traj_discovery = trajectory_discovery or "resolved by centralized discovery"
    else:
        original_dcd = dcd_links[0] if dcd_links else (fallback_dcds[0] if fallback_dcds else None)
        traj_discovery = (
            "symlink in carma results directory"
            if dcd_links
            else "parent-directory fallback"
        )

    if topology_file:
        original_psf = topology_file
        top_discovery = topology_discovery or "resolved by centralized discovery"
    else:
        original_psf = psf_links[0] if psf_links else (fallback_psfs[0] if fallback_psfs else None)
        top_discovery = (
            "symlink in carma results directory"
            if psf_links
            else "parent-directory fallback"
        )

    abs_original_dcd = os.path.abspath(original_dcd) if original_dcd and os.path.exists(original_dcd) else None
    abs_original_psf = os.path.abspath(original_psf) if original_psf and os.path.exists(original_psf) else None
    abs_results_dir = os.path.abspath(results_dir)

    has_struct = bool(abs_original_dcd and abs_original_psf)

    struct_info = {
        'dcd': abs_original_dcd if abs_original_dcd else "N/A",
        'psf': abs_original_psf if abs_original_psf else "N/A",
        'traj_discovery': traj_discovery if abs_original_dcd else "N/A",
        'top_discovery': top_discovery if abs_original_psf else "N/A",
        'flags': " ".join(atom_flags).replace('-atomid', '').replace('-atmid', '').strip() if has_struct else "N/A",
        'source': source_str if has_struct else "not evaluated (no structural trajectory/topology)",
        'compatible': True,
        'bounds_valid': True,
        'preflight_error': None,
        'success_count': 0,
        'failed_count': 0
    }

    if skip_processing:
        return struct_info

    if not shutil.which("carma"):
        logging.warning(f"\n{Colors.YELLOW}[WARN]{Colors.RESET} 'carma' binary not found in system $PATH. Bypassing structural extraction.")
        return struct_info

    if not has_struct:
        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Missing fundamental trajectory (.dcd) or topology (.psf) files in {base_dir}. Bypassing extraction phase.")
        return struct_info

    # Inputs are already displayed during analysis setup.
    
    # PREFLIGHT ATOM SELECTION VERIFICATION
    requested_atoms = set()
    for i in range(len(extraction_flags) - 1):
        if extraction_flags[i] == '-atomid':
            requested_atoms.add(extraction_flags[i+1])
            
    if abs_original_psf:
        psf_atoms_present = set()
        psf_segments_present = set()
        try:
            with open(abs_original_psf, 'r') as f:
                in_atoms_section = False
                for line in f:
                    if '!NATOM' in line:
                        in_atoms_section = True
                        continue
                    if in_atoms_section:
                        line_strip = line.strip()
                        if not line_strip or '!NBOND' in line_strip:
                            break
                        parts = line.split()
                        if len(parts) >= 5 and parts[0].isdigit():
                            psf_segments_present.add(parts[1])
                            psf_atoms_present.add(parts[4])
        except Exception:
            pass # Fallback to carma preflight if parse fails
            
        if selection and psf_segments_present and selection not in psf_segments_present:
            logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Requested segment '{selection}' was not found in the topology.")
            struct_info['compatible'] = False
            struct_info['preflight_error'] = 'missing_segment'
            struct_info['missing_segment'] = selection
            return struct_info
            
        if requested_atoms and psf_atoms_present:
            missing_atoms = requested_atoms - psf_atoms_present
            if missing_atoms:
                missing_str = " ".join(sorted(list(missing_atoms)))
                logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Structural atom selection contains atom identifiers not present in the topology.")
                logging.error(f"        Missing identifiers : {missing_str}")
                logging.error(f"        Requested selection : {struct_info['flags']}")
                struct_info['compatible'] = False
                struct_info['preflight_error'] = 'missing_atoms'
                struct_info['missing_atoms'] = missing_str
                return struct_info

    # PREFLIGHT DCD/PSF CHECK & BOUNDS VALIDATION
    logging.info(f"    {Colors.LIGHT_BLUE}[*]{Colors.RESET} Executing preflight DCD/PSF compatibility check...")
    max_sampled_frame = int(np.max(frames))
    dcd_frames = None
    dcd_atoms = None
    psf_atoms = None

    with tempfile.TemporaryDirectory(dir=abs_results_dir) as tmp_dir:
        out, err, rc_preflight = execute_carma_sandbox(["carma", "-v", "-first", "1", "-last", "1", "-pdb"] + extraction_flags + [abs_original_dcd, abs_original_psf], run_dir=tmp_dir)
        combined_out = out + "\n" + err
        combined_lower = combined_out.lower()
        
        for line in combined_out.splitlines():
            line_lower = line.lower()
            if 'coordinate sets' in line_lower and dcd_frames is None:
                m = re.search(r'coordinate sets\s+is\s+(\d+)', line_lower)
                if m: dcd_frames = int(m.group(1))
                
            m_mismatch = re.search(r'number of atoms in psf and dcd do not match\s*\((\d+)\s+vs\s+(\d+)\)', line_lower)
            if m_mismatch:
                psf_atoms = int(m_mismatch.group(1))
                dcd_atoms = int(m_mismatch.group(2))
                
            m_dcd = re.search(r'number of atoms in sets is\s+(\d+)', line_lower)
            if m_dcd and dcd_atoms is None:
                dcd_atoms = int(m_dcd.group(1))

        struct_info['dcd_atoms'] = dcd_atoms
        struct_info['psf_atoms'] = psf_atoms
        struct_info['dcd_frames'] = dcd_frames
        struct_info['max_frame'] = max_sampled_frame

        if rc_preflight != 0:
            struct_info['compatible'] = False
            struct_info['preflight_error'] = 'generic'
            
            if 'less than two' in combined_lower and 'atoms found' in combined_lower:
                struct_info['preflight_error'] = 'selection'
                if selection:
                    logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Requested segment '{selection}' or selection '{struct_info['flags']}' matched fewer than the required atoms.")
                else:
                    logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Structural atom selection '{struct_info['flags']}' matched fewer than the required atoms.")
            elif m_mismatch or (dcd_atoms and psf_atoms and dcd_atoms != psf_atoms):
                struct_info['preflight_error'] = 'mismatch'
                logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Structural trajectory and topology are incompatible.")
                logging.error(f"        Trajectory atom count : {dcd_atoms}")
                logging.error(f"        Topology atom count   : {psf_atoms}")
                logging.error(f"    {Colors.RED}[-]{Colors.RESET} DCD/PSF atom counts differ.")
            else:
                logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Structural preflight failed; see execution log.")
            return struct_info
            
        if dcd_frames is not None and max_sampled_frame > dcd_frames:
            logging.error(f"    {Colors.RED}[-]{Colors.RESET} Critical error: Maximum sampled frame ID ({max_sampled_frame}) exceeds available DCD coordinate sets ({dcd_frames}).")
            struct_info['bounds_valid'] = False
            return struct_info
            
        logging.info(f"    {Colors.BLUE}[+]{Colors.RESET} Preflight check passed: structural inputs, atom selection, and frame bounds are valid.")
    
    total_clusters = len(unique_clusters)
    struct_info['core_max_rmsd_angstrom'] = {}
    fh = get_clusterMD_file_handler()
    
    # Cluster extraction loop
    for index, cid in enumerate(unique_clusters):
        success = _process_single_cluster(
            cid, index, total_clusters, labels, frames, data_nd, 
            abs_original_dcd, abs_original_psf, abs_results_dir, 
            extraction_flags, config, carma_stats, cluster_rmsds, cluster_reps, cluster_struct_counts, fh,
            representative=representatives.get(cid) if representatives is not None else None,
            core_max_rmsds=struct_info['core_max_rmsd_angstrom'],
        )
        if success:
            struct_info['success_count'] += 1
        else:
            struct_info['failed_count'] += 1

    logging.info(f"\n    {Colors.LIGHT_BLUE}[*]{Colors.RESET} Processing unassigned configurations (label -1)...")
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        with tempfile.TemporaryDirectory(dir=abs_results_dir) as tmp_dir:
            noise_file = os.path.join(tmp_dir, "temp_frames_noise.dat")
            np.savetxt(noise_file, frames[noise_mask], fmt='%d')
            t_sub = time.time()
            _, _, rc_noise = execute_carma_sandbox(["carma", "-v", "-sort", "temp_frames_noise.dat", abs_original_dcd], run_dir=tmp_dir)
            carma_stats['sort'] += (time.time() - t_sub)
            if rc_noise == 0 and os.path.exists(os.path.join(tmp_dir, "carma.reordered.dcd")): 
                os.rename(os.path.join(tmp_dir, "carma.reordered.dcd"), os.path.join(abs_results_dir, "unassigned_noise.dcd"))

    logging.info(f"{Colors.BLUE}[+]{Colors.RESET} Structural extraction stage finished.")
    return struct_info


class CarmaStructuralBackend:
    """Existing CARMA discovery, preflight, extraction, fitting and filtering workflow.

    Command construction stays private to this module. Compound CARMA operations
    remain together to preserve their temporary-file and selected-topology semantics.
    """
    def process(self, request):
        from dataclasses import replace
        request = replace(request, timings={'sort': 0.0, 'rmsd': 0.0, 'super': 0.0},
                          rmsds={}, representative_frames={}, retained_counts={})
        unique_clusters = np.unique(request.clustering.labels[request.clustering.labels != -1])
        info = execute_structural_extraction(
            unique_clusters, request.clustering.labels, request.projection.frame_indices,
            request.projection.coordinates, request.target_analysis_dir, request.results_dir,
            request.config, request.timings, request.rmsds, request.representative_frames,
            request.retained_counts, request.skip_processing, request.representatives,
            request.trajectory_file, request.topology_file,
            getattr(request, "trajectory_discovery", None),
            getattr(request, "topology_discovery", None))
        return StructuralResult(info or {}, request.rmsds, request.representative_frames,
                                request.retained_counts, request.timings)