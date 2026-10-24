"""Define consistent plotting styles for analysis figures."""

import numpy as np

class PlotStyle:
    """Centralized formatting standard for all scientific figures."""
    FIG_SIZE = (10, 8)
    DPI = 600
    # Absolute margins guarantee the internal axes rect is pixel-identical across all 2D plots
    MARGINS = {'left': 0.12, 'right': 0.80, 'bottom': 0.12, 'top': 0.88}
    CBAR_AXES = [0.82, 0.12, 0.03, 0.76] 
    LEGEND_BBOX = (1.02, 0.5)

    CMAP_SCATTER = 'turbo'
    CMAP_FES = 'turbo_r'
    
    NOISE_COLOR = 'lightgrey'
    NOISE_SIZE = 0.5
    NOISE_ALPHA = 0.5
    CLUSTER_SIZE = 1.0
    CLUSTER_ALPHA = 0.8
    
    TITLE_FMT = {'fontsize': 16, 'fontweight': 'bold', 'pad': 15}
    LABEL_FMT = {'fontsize': 14, 'fontweight': 'bold'}
    TICK_SIZE = 12

    @staticmethod
    def apply_standard_layout(fig, ax, title, xlabel, ylabel, xlim=None, ylim=None):
        fig.subplots_adjust(**PlotStyle.MARGINS)
        ax.set_title(title, **PlotStyle.TITLE_FMT)
        ax.set_xlabel(xlabel, **PlotStyle.LABEL_FMT)
        ax.set_ylabel(ylabel, **PlotStyle.LABEL_FMT)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.tick_params(labelsize=PlotStyle.TICK_SIZE)

    @staticmethod
    def get_cluster_colors(unique_clusters, cmap_name):
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(cmap_name)
        n_clusters = len(unique_clusters)
        return {cid: cmap(i / max(1, n_clusters - 1)) for i, cid in enumerate(unique_clusters)}

    @staticmethod
    def repel_labels(node_positions, xlim, ylim):
        """Deterministic spring-repel algorithm to prevent label overlap."""
        labels_pos = {k: np.array(v, dtype=float) for k, v in node_positions.items()}
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        min_dist_x = x_range * 0.05
        min_dist_y = y_range * 0.05
        
        for _ in range(150):
            moved = False
            forces = {k: np.zeros(2) for k in labels_pos}
            for k1, p1 in labels_pos.items():
                forces[k1] += (node_positions[k1] - p1) * 0.1  # Spring to origin
                for k2, p2 in labels_pos.items():
                    if k1 == k2: continue
                    dx = p1[0] - p2[0]
                    dy = p1[1] - p2[1]
                    dist_x = abs(dx)
                    dist_y = abs(dy)
                    if dist_x < min_dist_x and dist_y < min_dist_y:
                        tie_sign = -1.0 if k1 < k2 else 1.0
                        repel_x = (
                            (min_dist_x - dist_x) * np.sign(dx)
                            if dx != 0
                            else (min_dist_x - dist_x) * tie_sign
                        )
                        repel_y = (
                            (min_dist_y - dist_y) * np.sign(dy)
                            if dy != 0
                            else (min_dist_y - dist_y) * tie_sign
                        )
                        forces[k1] += np.array([repel_x, repel_y]) * 0.3
                        moved = True
            for k in labels_pos:
                labels_pos[k] += forces[k]
            if not moved:
                break
        return labels_pos

