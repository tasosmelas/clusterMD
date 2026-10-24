"""Inspect observed cluster-label changes and diagnostic records."""

import logging
import sys
from clustermd.constants import Colors
from clustermd.characterization.label_changes import hunt_chronological_label_changes
from clustermd.projection.readers import ingest_trajectory_data
from clustermd.characterization.label_changes import map_observed_label_changes
from clustermd.characterization.label_changes import print_top_observed_changes

def run_diagnostic_validation(args, dat_file):
    """Bypasses the full pipeline and inspects chronological observed label changes."""
    raw_data, _ = ingest_trajectory_data(dat_file, step=1)
        
    frames = raw_data[:, 0].astype(int)
    labels = raw_data[:, 1].astype(int)
    data_nd = raw_data[:, 2:]
    
    label_changes = map_observed_label_changes(labels)
        
    logging.info("\n" + "="*85)
    if len(args.transitions) == 2:
        logging.info(f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} DIAGNOSTIC MODE: INSPECTING OBSERVED CHANGES FOR CLUSTER PAIR {tuple(sorted(args.transitions))}")
    else:
        logging.info(f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} DIAGNOSTIC MODE: OBSERVED CHANGE OVERVIEW")
    logging.info("="*85)
    
    print_top_observed_changes(label_changes, args.top)
    
    if len(args.transitions) == 2:
        hunt_chronological_label_changes(tuple(sorted(args.transitions)), labels, frames, data_nd, args.events)
        
    logging.info(f"{Colors.BLUE}[+]{Colors.RESET} Validation complete. Exiting smoothly without triggering extraction pipeline.\n")
    sys.exit(0)

