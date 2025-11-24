import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  
from itertools import combinations_with_replacement
from matplotlib.patches import Rectangle

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from matplotlib.patches import Rectangle

def plot_reactant_product_heatmap(
    iocrns,
    species_labels,
    max_order,
    num_reactions_template=0,
    title="Usage of reactant→product complexes across reaction networks",
    cmap_name="YlGnBu",
    max_ticks=20,
    figsize=(8, 7),
):
    """
    Build and plot a reactant→product complex heatmap directly from a list of iocrns.

    Parameters
    ----------
    iocrns : list
        List of reaction network objects. Each must have an attribute
        `reactions`, which is an ordered list. Each reaction must have:
          - reaction.reactant_labels : list[str]
          - reaction.product_labels  : list[str]
        where these lists represent a complex (with repetition, order aligned
        with `species_labels`).

    species_labels : list of str
        Species names (e.g. ['X_1', 'Z_1', 'Z_2']). Complexes are built from
        these species.

    max_order : int
        Maximum complex size. All complexes (multisets) of sizes 0..max_order
        over `species_labels` are generated, including the empty complex [].

    num_reactions_template : int, optional
        Number of initial reactions in each iocrn to treat as template reactions.
        They are NOT counted in the heatmap, but their cells are highlighted
        with a hatched box.

    title : str, optional
        Title of the plot.

    cmap_name : str, optional
        Name of a matplotlib sequential colormap ('YlGnBu', 'Blues', 'Greys', ...).

    max_ticks : int, optional
        Maximum number of tick labels shown on each axis. Ticks are thinned
        uniformly if the number of complexes exceeds this.

    figsize : tuple, optional
        Figure size (width, height) for matplotlib.

    Behavior
    --------
    - Zero cells (no non-template reactions with that pattern) are rendered white.
    - Nonzero cells are colored with `cmap_name` and annotated with their count.
    - Cells corresponding to template reactions are outlined with a hatched box.
    - x-axis: product complexes; y-axis: reactant complexes.
    """

    # --- generate complexes (multisets up to max_order) ---
    def all_complexes(labels, o):
        complexes = [[]]  # empty complex
        for k in range(1, o + 1):
            for combo in combinations_with_replacement(labels, k):
                complexes.append(list(combo))
        return complexes

    complexes = all_complexes(species_labels, max_order)
    K = len(complexes)

    # map complex (as tuple) -> index
    complex_to_idx = {tuple(c): i for i, c in enumerate(complexes)}

    # --- build reactant_product_array from iocrns + track template cells ---
    arr = np.zeros((K, K), dtype=int)
    special_cells = set()  # cells containing at least one template reaction

    for iocrn in iocrns:
        for i, reaction in enumerate(iocrn.reactions):
            r_key = tuple(reaction.reactant_labels)
            p_key = tuple(reaction.product_labels)
            try:
                ri = complex_to_idx[r_key]
                pi = complex_to_idx[p_key]
            except KeyError:
                raise ValueError(
                    f"Reaction complex {r_key} or {p_key} not found in generated complexes. "
                    "Check species_labels / max_order consistency."
                )

            if i < num_reactions_template:
                # Mark template cell, but do not increment count
                special_cells.add((ri, pi))
                continue

            arr[ri, pi] += 1

    # --- helpers for LaTeX labels ---
    def species_to_latex(s):
        """Convert 'X_1' → '\\mathbf{X}_{1}'."""
        parts = s.split('_')
        base = parts[0]
        if len(parts) == 1:
            return r'\mathbf{' + base + '}'
        sub = '_'.join(parts[1:])
        return r'\mathbf{' + base + '}_{' + sub + '}'

    def complex_to_latex(c):
        """Convert a complex (list of species labels) to LaTeX, joined by '+'."""
        if len(c) == 0:
            return r'\varnothing'
        return " + ".join(species_to_latex(s) for s in c)

    axis_labels_latex = [r"$" + complex_to_latex(c) + r"$" for c in complexes]

    # --- plotting ---
    data = np.ma.masked_equal(arr, 0)      # zeros → masked → white
    vmax = data.max() if data.count() > 0 else 1

    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        vmin=1 if data.count() > 0 else 0,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of IOCRNS")

    # annotate counts in nonzero cells
    n_rows, n_cols = arr.shape
    for ri in range(n_rows):
        for pi in range(n_cols):
            val = arr[ri, pi]
            if val > 0:
                ax.text(
                    pi, ri, str(val),
                    ha="center", va="center",
                    fontsize=6,
                    color="black",
                )

    # draw hatched boxes for template cells
    for (ri, pi) in special_cells:
        rect = Rectangle(
            (pi - 0.5, ri - 0.5),  # bottom-left corner of the cell
            1, 1,
            fill=False,
            hatch="///",
            linewidth=1.0,
            edgecolor="black",
        )
        ax.add_patch(rect)

    # tick thinning
    n = len(axis_labels_latex)
    if n <= max_ticks:
        tick_idx = np.arange(n)
    else:
        tick_idx = np.linspace(0, n - 1, max_ticks, dtype=int)

    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels([axis_labels_latex[i] for i in tick_idx], rotation=90)
    ax.set_yticklabels([axis_labels_latex[i] for i in tick_idx])

    ax.set_xlabel("Products")
    ax.set_ylabel("Reactants")
    ax.set_title(title)

    # gridlines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.3)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    plt.show()

def plot_reactant_product_scatter(
    iocrns,
    perf,
    species_labels,
    max_order,
    num_reactions_template=0,
    title="Reaction patterns colored by network performance",
    cmap_name="viridis",
    max_ticks=20,
    figsize=(8, 7),
    jitter_scale=0.3,
):
    # --- checks ---
    perf = np.asarray(perf, dtype=float)
    if len(perf) != len(iocrns):
        raise ValueError("perf must have same length as iocrns.")

    # --- complexes ---
    def all_complexes(labels, o):
        complexes = [[]]
        for k in range(1, o + 1):
            for combo in combinations_with_replacement(labels, k):
                complexes.append(list(combo))
        return complexes

    complexes = all_complexes(species_labels, max_order)
    K = len(complexes)
    complex_to_idx = {tuple(c): i for i, c in enumerate(complexes)}

    # --- LaTeX labels ---
    def species_to_latex(s):
        parts = s.split('_')
        base = parts[0]
        if len(parts) == 1:
            return r'\mathbf{' + base + '}'
        sub = '_'.join(parts[1:])
        return r'\mathbf{' + base + '}_{' + sub + '}'

    def complex_to_latex(c):
        if len(c) == 0:
            return r'\varnothing'
        return " + ".join(species_to_latex(s) for s in c)

    axis_labels_latex = [r"$" + complex_to_latex(c) + r"$" for c in complexes]

    # --- collect scatter points + special cells ---
    xs, ys, cs = [], [], []
    special_cells = set()  # (ri, pi) for template reactions
    rng = np.random.default_rng()

    for net_idx, iocrn in enumerate(iocrns):
        p = perf[net_idx]
        for i, reaction in enumerate(iocrn.reactions):
            r_key = tuple(reaction.reactant_labels)
            p_key = tuple(reaction.product_labels)
            try:
                ri = complex_to_idx[r_key]
                pi = complex_to_idx[p_key]
            except KeyError:
                raise ValueError(
                    f"Reaction complex {r_key} or {p_key} not in generated complexes. "
                    "Check species_labels / max_order."
                )

            if i < num_reactions_template:
                # mark this cell as special, but don't plot a point
                special_cells.add((ri, pi))
                continue

            x = pi + rng.uniform(-jitter_scale, jitter_scale)
            y = ri + rng.uniform(-jitter_scale, jitter_scale)
            xs.append(x)
            ys.append(y)
            cs.append(p)

    xs, ys, cs = map(np.array, (xs, ys, cs))

    # --- plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    # background grid
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(-0.5, K - 0.5)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.3)
    ax.tick_params(which="minor", length=0)

    # scatter points (non-template reactions)
    sc = ax.scatter(
        xs, ys,
        c=cs,
        cmap=cmap_name,
        s=10,
        alpha=0.8,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Performance")

    # draw dashed boxes for template cells
    for (ri, pi) in special_cells:
        rect = Rectangle(
        (pi - 0.5, ri - 0.5),
        1, 1,
        fill=True,
        facecolor="none",  # transparent but still shows hatch
        hatch="///",
        linewidth=1.0,
        edgecolor="black",
    )
        ax.add_patch(rect)

    # tick thinning
    n = len(axis_labels_latex)
    if n <= max_ticks:
        tick_idx = np.arange(n)
    else:
        tick_idx = np.linspace(0, n - 1, max_ticks, dtype=int)

    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels([axis_labels_latex[i] for i in tick_idx], rotation=90)
    ax.set_yticklabels([axis_labels_latex[i] for i in tick_idx])

    ax.set_xlabel("Products")
    ax.set_ylabel("Reactants")
    ax.set_title(title)

    fig.tight_layout()
    plt.show()

def hamming_radius_graph(X_bool: np.ndarray, t: int):
    """ Build a sparse graph of pairs with Hamming distance <= t.
    Returns a scipy.sparse CSR with weights = Hamming counts (integers). """

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
    """ Plot a sparse distance graph.
    - G_counts_csr: CSR matrix, weight = Hamming count (<= t)
    - counts: per-node sizes (e.g., frequency)
    Returns: matplotlib Figure. """
    
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
    """ Build unique boolean signatures from crn_list, make a sparse Hamming-≤t graph, and plot it.
    Assumes each `crn` has a .get_bool_signature() -> 1D boolean/0-1 array.
    Returns: matplotlib Figure """
    
    # Stack signatures; get uniques and their counts (which set node sizes/labels)
    crn_topologies = np.stack([crn.get_bool_signature() for crn in crn_list]).astype(bool)
    unique_topologies, inv, counts = np.unique(crn_topologies, axis=0, return_inverse=True, return_counts=True)

    # Build sparse graph under threshold t and plot
    G_counts_csr = hamming_radius_graph(unique_topologies, t=t)
    return plot_sparse_distance_graph(G_counts_csr, counts, figsize=figsize, with_edge_labels=with_edge_labels)

# --- Example usage ---
# fig = topology_graph(crn_list, t=10, figsize=(8,8))
# fig.savefig("topology_graph.png", dpi=300)