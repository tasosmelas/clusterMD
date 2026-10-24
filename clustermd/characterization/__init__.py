"""clusterMD characterization components."""

from dataclasses import dataclass
import logging
import os

from clustermd.constants import Colors
from clustermd.characterization.populations import calculate_populations_and_energies
from clustermd.characterization.label_changes import (
    hunt_chronological_label_changes,
    map_observed_label_changes,
    print_top_observed_changes,
)


@dataclass(frozen=True)
class CharacterizationData:
    """Intermediate characterization outputs required by later workflow stages."""
    states: object
    counts: object
    population_map: dict
    free_energies: dict
    k_B_T: float
    label_changes: dict
    top_edges: list


def characterize_clusters(
    labels,
    frames,
    coordinates,
    temperature,
    top_changes,
    diagnostic_pair,
    max_events,
    results_dir,
    skip_label_changes,
):
    """Run backend-independent cluster characterization and report results."""
    logging.info(
        f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} Mapping observed changes between assigned cluster labels..."
    )

    states, counts, population_map, free_energies, k_B_T = (
        calculate_populations_and_energies(labels, temperature)
    )

    if not skip_label_changes:
        label_changes = map_observed_label_changes(labels)
        sorted_changes = print_top_observed_changes(label_changes, top_changes)
        top_edges = [edge for edge, count in sorted_changes[:top_changes]]

        changes_file = os.path.join(results_dir, "observed_label_changes_report.txt")
        with open(changes_file, "w") as cf:
            cf.write("============================================================\n")
            cf.write("OBSERVED LABEL CHANGES\n")
            cf.write("============================================================\n\n")
            cf.write(f"{'Cluster pair':<40}{'Observed changes':>20}\n")
            cf.write("-" * 60 + "\n")
            for edge, count in sorted_changes:
                pair_str = f"{edge[0]} <--> {edge[1]}"
                cf.write(f"{pair_str:<40}{count:>20d}\n")
            cf.write("-" * 60 + "\n")
            cf.write(f"{'Unique cluster pairs':<40}{len(label_changes):>20d}\n")
        logging.info(
            f"{Colors.BLUE}[+]{Colors.RESET} Observed-change report saved to: {changes_file}"
        )

        if diagnostic_pair is not None and len(diagnostic_pair) == 2:
            logging.info("\n" + "=" * 85)
            logging.info(
                f"{Colors.LIGHT_BLUE}[*]{Colors.RESET} "
                f"DIAGNOSTIC MODE: INSPECTING OBSERVED CHANGES FOR CLUSTER PAIR "
                f"{tuple(sorted(diagnostic_pair))}"
            )
            logging.info("=" * 85)
            hunt_chronological_label_changes(
                tuple(sorted(diagnostic_pair)),
                labels,
                frames,
                coordinates,
                max_events,
            )
    else:
        label_changes, top_edges = {}, []
        logging.info("    -> Skipped due to chronological validation failure.")

    return CharacterizationData(
        states=states,
        counts=counts,
        population_map=population_map,
        free_energies=free_energies,
        k_B_T=k_B_T,
        label_changes=label_changes,
        top_edges=top_edges,
    )
