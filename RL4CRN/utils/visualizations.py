# ===== fast topology graph (sparse, thresholded by Hamming distance) =====
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  # pip install scikit-learn

def hamming_radius_graph(X_bool: np.ndarray, t: int):
    """
    Build a sparse graph of pairs with Hamming distance <= t.
    Returns a scipy.sparse CSR with weights = Hamming counts (integers).
    """
    X_bool = np.asarray(X_bool, dtype=np.uint8)  # {0,1}
    n, d = X_bool.shape
    radius = t / d  # sklearn uses normalized Hamming in [0,1]
    nn = NearestNeighbors(metric="hamming", radius=radius, n_jobs=-1)
    nn.fit(X_bool)
    G_norm = nn.radius_neighbors_graph(mode="distance")   # normalized distances in [0,1]
    G_counts = G_norm.multiply(d).astype(np.float64)      # convert to counts (kept as float for nx)
    G_counts.setdiag(0.0); G_counts.eliminate_zeros()
    return G_counts

def plot_sparse_distance_graph(G_counts_csr, counts, title="Topological Diversity Graph of IOCRNs",
                               figsize=(7,7), with_edge_labels=False, seed=42):
    """
    Plot a sparse distance graph.
    - G_counts_csr: CSR matrix, weight = Hamming count (<= t)
    - counts: per-node sizes (e.g., frequency)
    Returns: matplotlib Figure
    """
    G = nx.from_scipy_sparse_array(G_counts_csr)  # weight = distance (Hamming count)

    # Spring layout using inverse distance as weight (shorter -> stronger attraction)
    invw = {(u, v): 1.0 / (d["weight"] + 1e-9) for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, invw, name="invw")
    pos = nx.spring_layout(G, weight="invw", seed=seed)

    # Edge transparency: smaller distance -> more opaque
    w = np.array([d["weight"] for *_, d in G.edges(data=True)], dtype=float)
    if w.size:
        a = 1.0 - (w - w.min()) / (np.ptp(w) + 1e-9)
        alpha = 0.15 + 0.85 * a
        edge_color = [(0, 0, 0, a_) for a_ in alpha]
    else:
        edge_color = []

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=1.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=np.asarray(counts) * 200, ax=ax)

    # Node labels (counts)
    labels = {i: str(counts[i]) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color="white", ax=ax)

    # Optional edge labels (can be slow on many edges)
    if with_edge_labels and w.size and w.size <= 1500:
        edge_labels = {(u, v): f"{d['weight']:.0f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.5, rotate=False, font_size=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7), ax=ax
        )

    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    return fig

def topology_graph(crn_list, t=10, figsize=(7,7), with_edge_labels=False):
    """
    Build unique boolean signatures from crn_list, make a sparse Hamming-≤t graph, and plot it.
    Assumes each `crn` has a .get_bool_signature() -> 1D boolean/0-1 array.
    Returns: matplotlib Figure
    """
    # Stack signatures; get uniques and their counts (which set node sizes/labels)
    crn_topologies = np.stack([crn.get_bool_signature() for crn in crn_list]).astype(bool)
    unique_topologies, inv, counts = np.unique(crn_topologies, axis=0, return_inverse=True, return_counts=True)

    # Build sparse graph under threshold t and plot
    G_counts_csr = hamming_radius_graph(unique_topologies, t=t)
    return plot_sparse_distance_graph(G_counts_csr, counts, figsize=figsize, with_edge_labels=with_edge_labels)

# --- Example usage ---
# fig = topology_graph(crn_list, t=10, figsize=(8,8))
# fig.savefig("topology_graph.png", dpi=300)