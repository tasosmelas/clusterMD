"""Visualize conformational projections and free-energy landscapes."""

from clustermd.characterization.free_energy import calculate_free_energy

import numpy as np
from clustermd.constants import MAX_FREE_ENERGY_KCAL
from clustermd.visualization.style import PlotStyle

def render_conformational_landscape(x, y, lx, ly, p1, p2, unique_clusters, labels, zero_var, out_prefix):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=PlotStyle.FIG_SIZE, dpi=PlotStyle.DPI) 
    
    if np.any(labels == -1):
        ax.scatter(x[labels == -1], y[labels == -1], color=PlotStyle.NOISE_COLOR, s=PlotStyle.NOISE_SIZE, alpha=PlotStyle.NOISE_ALPHA, label='Unassigned / noise')
        
    colors = PlotStyle.get_cluster_colors(unique_clusters, PlotStyle.CMAP_SCATTER)
    for cid in unique_clusters:
        mask = (labels == cid)
        ax.scatter(x[mask], y[mask], color=colors[cid], s=PlotStyle.CLUSTER_SIZE, alpha=PlotStyle.CLUSTER_ALPHA, label=f'Cluster {cid}')
        
    PlotStyle.apply_standard_layout(fig, ax, "Conformational Projection", f"Principal Component {p1}", f"Principal Component {p2}", lx, ly)
    
    if zero_var:
        ax.text(0.5, 0.5, "Zero-variance projection\n(coincident configurations)", transform=ax.transAxes, ha='center', va='center', fontsize=14, fontweight='bold', color='gray', alpha=0.7)
    elif len(unique_clusters) <= 35: 
        ax.legend(markerscale=5, loc='center left', bbox_to_anchor=PlotStyle.LEGEND_BBOX, framealpha=0.8)
        
    fig.savefig(f"{out_prefix}.png", dpi=PlotStyle.DPI)
    plt.close(fig)


def render_fes_connectivity(pc1, pc2, unique_clusters, labels, k_B_T, xl12, yl12, label_changes, top_edges, out_file):
    import matplotlib.pyplot as plt
    
    X, Y, free_energy = calculate_free_energy(pc1, pc2, k_B_T, xl12, yl12)

    fig, ax = plt.subplots(figsize=PlotStyle.FIG_SIZE, dpi=PlotStyle.DPI)
    contour = ax.contourf(X, Y, free_energy, levels=np.linspace(0, MAX_FREE_ENERGY_KCAL, 16), cmap=PlotStyle.CMAP_FES, extend='max')
    ax.contour(X, Y, free_energy, levels=np.linspace(0, MAX_FREE_ENERGY_KCAL, 16), colors='black', alpha=0.1, linewidths=0.5)
    
    node_positions = {}
    for cid in unique_clusters:
        mask = (labels == cid)
        node_positions[cid] = (np.median(pc1[mask]), np.median(pc2[mask]))

    if top_edges:
        max_w = max([label_changes[e] for e in top_edges]) if top_edges else 1
        # Reverse drawing order ensures highest-count (thickest) edges are rendered on top
        for edge in reversed(top_edges):
            sA, sB = edge
            if sA in node_positions and sB in node_positions:
                w = (label_changes[edge] / max_w) * 4.0 + 0.5
                ax.plot([node_positions[sA][0], node_positions[sB][0]], [node_positions[sA][1], node_positions[sB][1]], color='black', alpha=0.6, linewidth=w, zorder=2)
                
    labels_pos = PlotStyle.repel_labels(node_positions, xl12, yl12)
    for cid in unique_clusters:
        nx, ny = node_positions[cid]
        lx, ly = labels_pos[cid]
        if np.linalg.norm(np.array([nx, ny]) - np.array([lx, ly])) > 1e-3:
            ax.plot([nx, lx], [ny, ly], color='black', linewidth=0.5, zorder=3)
        ax.scatter(nx, ny, color='black', s=10, zorder=3)
        ax.text(lx, ly, str(cid), fontsize=9, fontweight='bold', color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='circle,pad=0.2'), zorder=4)
        
    PlotStyle.apply_standard_layout(fig, ax, "Population-Derived Free-Energy Landscape\n& Observed Cluster Connectivity", "Principal Component 1", "Principal Component 2", xl12, yl12)
    
    cax = fig.add_axes(PlotStyle.CBAR_AXES)
    cbar = fig.colorbar(contour, cax=cax)
    cbar.set_label('Relative free energy (kcal/mol)', **PlotStyle.LABEL_FMT)
    
    if len(top_edges) > 0:
        ax.text(0.02, 0.98, f"{len(top_edges)} highest-count observed cluster-pair changes", transform=ax.transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    fig.savefig(out_file, dpi=PlotStyle.DPI)
    plt.close(fig)


def render_pcoa_connectivity(num_states, data_nd, labels, states, population_map, label_changes, top_edges, out_file):
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import squareform, pdist
    cluster_medians = np.array([np.median(data_nd[labels == state], axis=0) for state in states])
    D = squareform(pdist(cluster_medians, metric='euclidean'))
    H = np.eye(num_states) - np.ones((num_states, num_states)) / num_states
    evals, evecs = np.linalg.eigh(-0.5 * H.dot(D**2).dot(H))
    idx = np.argsort(evals)[::-1]; evals, evecs = evals[idx], evecs[:, idx]
    pcoa_coords = evecs[:, :2] * np.sqrt(np.maximum(evals[:2], 0))
    pos_pcoa = {state: (pcoa_coords[i, 0], pcoa_coords[i, 1]) for i, state in enumerate(states)}
    
    pad_x = max((np.max(pcoa_coords[:, 0]) - np.min(pcoa_coords[:, 0])) * 0.15, 0.1)
    pad_y = max((np.max(pcoa_coords[:, 1]) - np.min(pcoa_coords[:, 1])) * 0.15, 0.1)
    xlim = (np.min(pcoa_coords[:, 0]) - pad_x, np.max(pcoa_coords[:, 0]) + pad_x)
    ylim = (np.min(pcoa_coords[:, 1]) - pad_y, np.max(pcoa_coords[:, 1]) + pad_y * 1.5) # Asymmetric top pad
    
    fig, ax = plt.subplots(figsize=PlotStyle.FIG_SIZE, dpi=PlotStyle.DPI)
    
    if top_edges:
        max_w = max([label_changes[e] for e in top_edges]) if top_edges else 1
        for edge in reversed(top_edges):
            sA, sB = edge
            if sA in pos_pcoa and sB in pos_pcoa:
                w = (label_changes[edge] / max_w) * 4.0 + 0.5
                ax.plot([pos_pcoa[sA][0], pos_pcoa[sB][0]], [pos_pcoa[sA][1], pos_pcoa[sB][1]], color='black', alpha=0.6, linewidth=w, zorder=1)
                
    colors = PlotStyle.get_cluster_colors(states, PlotStyle.CMAP_SCATTER)
    labels_pos = PlotStyle.repel_labels(pos_pcoa, xlim, ylim)
    
    for state in states:
        cx, cy = pos_pcoa[state]
        lx, ly = labels_pos[state]
        node_size = float(np.log10(population_map[state]) ** 3.5) * 8
        ax.scatter(cx, cy, s=node_size, color=colors[state], edgecolors='white', linewidths=1.5, zorder=2)
        if np.linalg.norm(np.array([cx, cy]) - np.array([lx, ly])) > 1e-3:
            ax.plot([cx, lx], [cy, ly], color='black', linewidth=0.5, zorder=3)
        ax.text(lx, ly, str(state), color='black', fontweight='bold', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.1, boxstyle='round'), zorder=4)
        
    PlotStyle.apply_standard_layout(fig, ax, "Principal-Coordinate Representation\nof Cluster Medians & Connectivity", "PCoA Dimension 1", "PCoA Dimension 2", xlim, ylim)
    
    if len(top_edges) > 0:
        ax.text(0.02, 0.98, f"{len(top_edges)} highest-count observed cluster-pair changes", transform=ax.transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
    fig.savefig(out_file, dpi=PlotStyle.DPI)
    plt.close(fig)


def get_shared_limits(pc_data, pad_ratio=0.05, min_pad=0.1):
    """Calculates shared axis bounds robust to zero-variance input."""
    ptp = np.ptp(pc_data)
    if ptp == 0:
        return (pc_data[0] - min_pad, pc_data[0] + min_pad), True
    pad = max(ptp * pad_ratio, min_pad)
    return (np.min(pc_data) - pad, np.max(pc_data) + pad), False

