"""Read and validate precomputed PCA/dPCA projection coordinates."""

import logging
import math
import numpy as np
import os
import subprocess
from collections import Counter
from clustermd.constants import Colors
from clustermd.models import ProjectionDataset, ProjectionValidationError

def ingest_trajectory_data(file_path, step=1):
    """Read a strided projection table with validated frame IDs and finite coordinates.

The first carma DAT column represents a positive original-frame ID;
malformed sampled rows are recorded and skipped."""
    if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step < 1:
        raise ProjectionValidationError("Projection ingestion stride must be a positive integer.")
    if os.path.getsize(file_path) == 0:
        raise ProjectionValidationError("Coordinate file is empty.")

    logging.info(f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Pre-allocating memory and loading data (Stride: {step})...")
    
    try:
        out = subprocess.check_output(['wc', '-l', file_path])
        total_lines = int(out.split()[0])
        with open(file_path, "rb") as tail:
            tail.seek(-1, os.SEEK_END)
            total_lines += int(tail.read(1) != b"\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)
            
    parse_stats = {
        'total_lines': total_lines,
        'selected_by_stride': 0,
        'valid_sampled': 0,
        'malformed_sampled': 0,
        'first_sampled_frame': None,
        'last_sampled_frame': None,
        'sampled_monotonic': True,
        'sampled_duplicates': False
    }
            
    col_counts = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 500:
                break
            parts = line.split()
            if parts:
                col_counts.append(len(parts))
                
    if not col_counts:
        raise ProjectionValidationError("Coordinate file contains no parseable data.")
        
    num_cols = Counter(col_counts).most_common(1)[0][0]
            
    if num_cols < 2:
        raise ProjectionValidationError("Projection requires frame IDs and at least one coordinate column.")
    subset_size = math.ceil(total_lines / step)
    data = np.zeros((subset_size, num_cols))
    
    last_frame_val = None
    seen_frames = set()
    
    with open(file_path, 'r') as f:
        idx = 0
        for i, line in enumerate(f):
            if i % step == 0:
                parse_stats['selected_by_stride'] += 1
                parts = line.split()
                if parts:
                    if len(parts) == num_cols:
                        try:
                            frame_val = float(parts[0])
                            if not frame_val.is_integer() or frame_val <= 0:
                                raise ProjectionValidationError(f"Frame IDs must be positive integers. Found '{parts[0]}' at line {i+1}.")
                            
                            frame_int = int(frame_val)
                            if parse_stats['first_sampled_frame'] is None:
                                parse_stats['first_sampled_frame'] = frame_int
                            parse_stats['last_sampled_frame'] = frame_int
                            
                            if last_frame_val is not None and frame_int <= last_frame_val:
                                parse_stats['sampled_monotonic'] = False
                            if frame_int in seen_frames:
                                parse_stats['sampled_duplicates'] = True
                                
                            seen_frames.add(frame_int)
                            last_frame_val = frame_int
                            
                            row_vals = [float(x) for x in parts]
                            if not all(math.isfinite(x) for x in row_vals):
                                raise ProjectionValidationError(f"Non-finite value (NaN/Inf) detected at line {i+1}. Analysis aborted.")
                                
                            data[idx] = row_vals
                            idx += 1
                            parse_stats['valid_sampled'] += 1
                        except ValueError:
                            parse_stats['malformed_sampled'] += 1
                            logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Format anomaly at line {i+1}: non-numeric data. Row skipped.")
                    else:
                        parse_stats['malformed_sampled'] += 1
                        logging.warning(f"{Colors.YELLOW}[WARN]{Colors.RESET} Format anomaly at line {i+1}: expected {num_cols} cols, found {len(parts)}. Frame skipped.")
                        
    return data[:idx], parse_stats


def read_projection(file_path, analysis_step=1):
    raw_data, parse_stats = ingest_trajectory_data(file_path, step=analysis_step)
    coordinates = raw_data[:, 1:]
    feature_type = ("dPCA" if "dpca" in os.path.basename(file_path).lower()
                    else "PCA" if "pca" in os.path.basename(file_path).lower() else "unknown")
    return ProjectionDataset(
        coordinates=coordinates,
        sample_indices=np.arange(len(raw_data), dtype=np.int64),
        frame_indices=raw_data[:, 0].astype(int),
        feature_names=tuple(f"coordinate_{i+1}" for i in range(coordinates.shape[1])),
        feature_type=feature_type, source=os.path.abspath(file_path),
        metadata={"parse_stats": parse_stats, "analysis_step": analysis_step,
                  "projection_stride": None, "available_dimensions": coordinates.shape[1]})

class PrecomputedProjectionBackend:
    """Loads an already-computed PCA/dPCA coordinate file into ProjectionDataset."""

    def create_projection(self, spec):
        if spec.projection_file is None:
            raise ValueError("A precomputed projection file is required for this input reader.")
        return read_projection(spec.projection_file, spec.clustering.analysis_step)


def obtain_projection(spec):
    """Compatibility entry point for the current precomputed-file projection path."""
    return PrecomputedProjectionBackend().create_projection(spec)