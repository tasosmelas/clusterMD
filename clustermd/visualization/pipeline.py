"""Visualization orchestration preserving original plot guards and status wording."""
import os
import time
import logging
from clustermd.constants import Colors
from clustermd.visualization.landscapes import (
    get_shared_limits, render_conformational_landscape, render_fes_connectivity,
    render_pcoa_connectivity)

def render_outputs(data_nd, n_dimensions, unique_clusters, labels, k_B_T,
                   num_states, states, population_map, label_changes, top_edges,
                   results_dir, skip_label_changes):
    # 4. VISUALIZATION
    plot_status = {
        'landscape': 'Completed', 
        'fes_connectivity': 'Completed', 
        'pcoa_connectivity': 'Completed'
    }
    
    logging.info(f"\n{Colors.LIGHT_BLUE}[*]{Colors.RESET} Generating static conformational projections and free-energy surfaces...")
    t0_plots = time.time()
    
    pc1 = data_nd[:, 0]
    pc2 = data_nd[:, 1] if n_dimensions >= 2 else None

    if n_dimensions >= 2:
        xlim, x_degen = get_shared_limits(pc1)
        ylim, y_degen = get_shared_limits(pc2)
        zero_var_2d = x_degen or y_degen
        
        render_conformational_landscape(pc1, pc2, xlim, ylim, "1", "2", unique_clusters, labels, zero_var_2d, os.path.join(results_dir, "landscape_PC1_vs_PC2"))
        
        if zero_var_2d:
            logging.warning(f"    {Colors.YELLOW}[WARN]{Colors.RESET} Zero variance detected in primary dimensions. Bypassing free-energy surface connectivity overlay.")
            plot_status['fes_connectivity'] = 'Skipped (zero variance in projection)'
        else:
            if num_states > 0:
                if skip_label_changes:
                    logging.info("    -> Rendering population-derived FES without connectivity...")
                else:
                    logging.info("    -> Rendering FES with observed cluster connectivity...")
                render_fes_connectivity(pc1, pc2, unique_clusters, labels, k_B_T, xlim, ylim, label_changes, top_edges, os.path.join(results_dir, "FES_Connectivity_PC1_vs_PC2.png"))
                if skip_label_changes:
                    plot_status['fes_connectivity'] = 'FES completed; connectivity not performed (chronological validation failure)'
            else:
                logging.info(f"    -> Bypassing FES connectivity overlay (0 clusters assigned).")
                plot_status['fes_connectivity'] = 'Skipped (0 clusters assigned)'
    else:
        logging.info(f"    -> Bypassing 2D visualizations (1D input detected).")
        for p in ['landscape', 'fes_connectivity']:
            plot_status[p] = 'Skipped (1D input)'

    if n_dimensions >= 3:
        zlim, z_degen = get_shared_limits(data_nd[:, 2])
        if not (x_degen or z_degen):
            render_conformational_landscape(data_nd[:, 0], data_nd[:, 2], xlim, zlim, "1", "3", unique_clusters, labels, False, os.path.join(results_dir, "landscape_PC1_vs_PC3"))
        if not (y_degen or z_degen):
            render_conformational_landscape(data_nd[:, 1], data_nd[:, 2], ylim, zlim, "2", "3", unique_clusters, labels, False, os.path.join(results_dir, "landscape_PC2_vs_PC3"))

    if not skip_label_changes:
        if num_states > 2:
            logging.info(f"    -> Rendering principal coordinate analysis (PCoA) of cluster medians...")
            render_pcoa_connectivity(num_states, data_nd, labels, states, population_map, label_changes, top_edges, os.path.join(results_dir, "PCoA_Connectivity_Cluster_Medians.png"))
        else:
            plot_status['pcoa_connectivity'] = 'Skipped (requires >2 clusters)'
    else:
        plot_status['pcoa_connectivity'] = 'Not performed (chronological validation failure)'

    return plot_status, time.time() - t0_plots

