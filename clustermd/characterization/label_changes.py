"""Summarize observed changes between assigned cluster labels."""

import logging
from clustermd.constants import Colors
from clustermd.utils.formatting import format_record

def map_observed_label_changes(labels):
    """Count undirected changes between consecutive assigned labels.

Frames labeled -1 are omitted from the counted sequence, so changes
may bridge one or more intervening unassigned frames; counts are not rates."""
    valid_labels = labels[labels != -1]
    label_changes = {}         
    for i in range(len(valid_labels) - 1):
        sA, sB = valid_labels[i], valid_labels[i+1]
        if sA != sB:
            edge = tuple(sorted([sA, sB]))
            label_changes[edge] = label_changes.get(edge, 0) + 1
    return label_changes


def print_top_observed_changes(label_changes, top_n):
    """Outputs the most frequent observed changes between assigned cluster labels."""
    sorted_changes = sorted(label_changes.items(), key=lambda item: item[1], reverse=True)
    
    logging.info("\n    OBSERVED LABEL CHANGES")
    logging.info("    " + "-"*49)
    logging.info(f"    {'Cluster pair':<32}{'Observed changes':>17}")
    logging.info("    " + "-"*49)
    
    for edge, count in sorted_changes[:top_n]:
        pair_str = f"{edge[0]} <--> {edge[1]}"
        logging.info(f"    {pair_str:<32}{count:>17d}")
        
    logging.info("    " + "-"*49)
    logging.info(f"    {'Unique cluster pairs':<32}{len(label_changes):>17d}")
    
    return sorted_changes


def hunt_chronological_label_changes(target_edge, labels, frames, data_nd, max_events):
    """Inspects the chronology of observed changes between assigned cluster labels."""
    logging.info(f"\n{Colors.LIGHT_BLUE}[*]{Colors.RESET} Inspecting chronological change events for cluster pair {target_edge}...")
    events_found = 0
    last_valid_state, last_valid_idx = None, None
    intervening_noise_indices = []
    
    for idx, label in enumerate(labels):
        if label != -1:
            if last_valid_state is not None and last_valid_state != label:
                current_edge = tuple(sorted([last_valid_state, label]))
                if current_edge == target_edge:
                    events_found += 1
                    if events_found <= max_events:
                        logging.info(f"\n{Colors.LIGHT_BLUE}--- Observed Change Event {events_found} ---{Colors.RESET}")
                        logging.info(f"PREVIOUS ASSIGNED : {format_record(data_nd[last_valid_idx], frames[last_valid_idx], last_valid_state)}")
                        if intervening_noise_indices:
                            logging.info(f"  ... [ {len(intervening_noise_indices)} intervening unassigned frames (-1) ] ...")
                            for f_idx in intervening_noise_indices[:2]: logging.info(f"  UNASSIGNED: {format_record(data_nd[f_idx], frames[f_idx], -1)}")
                            if len(intervening_noise_indices) > 2: logging.info("  ...      (additional unassigned frames hidden to save space)")
                        logging.info(f"NEXT ASSIGNED     : {format_record(data_nd[idx], frames[idx], label)}")
            last_valid_state, last_valid_idx = label, idx
            intervening_noise_indices = []
        else:
            if last_valid_state is not None: intervening_noise_indices.append(idx)
            
    if events_found == 0:
        logging.error(f"{Colors.YELLOW}[WARN]{Colors.RESET} No observed label changes found between Cluster {target_edge[0]} and Cluster {target_edge[1]}.")
    elif events_found > max_events:
        logging.info(f"\n{Colors.BLUE}[+]{Colors.RESET} ... and {events_found - max_events} additional events hidden.")
    logging.info("="*85 + "\n")

