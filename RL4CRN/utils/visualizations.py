"""
Plotting and topology-analysis utilities for IOCRN / RL4CRN experiments.

This module provides visualization helpers for:

- reaction-pattern usage across many networks (reactant→product heatmaps and scatters),
- topology/diversity graphs built from boolean CRN signatures (Hamming-distance graphs),
- ensemble trajectory plots for top-performing networks, and
- compact visualization of logic targets (truth tables).

Several functions support both publication-style matplotlib output and optional
interactive Plotly rendering for exploratory analysis.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors  
from itertools import combinations_with_replacement
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

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
    Build and plot a reactant→product complex heatmap from a list of IOCRNs.

    The heatmap counts how often each *reactant complex* maps to each *product complex*
    across the provided networks, ignoring the first `num_reactions_template` reactions
    in each IOCRN (treated as fixed/template). Template reactant→product cells are
    highlighted with a hatched outline.

    A *complex* is represented by a multiset of species labels (with repetition allowed),
    e.g. ['X_1', 'X_1', 'Z_1'] corresponds to X_1:2 + Z_1:1. All complexes of size
    0..`max_order` over `species_labels` are generated (including the empty complex).

    Args:
        iocrns : list
            List of reaction network objects. Each must have an attribute `reactions`
            (ordered list). Each reaction must have:
            - reaction.reactant_labels : list[str]
            - reaction.product_labels  : list[str]
            The labels should be sorted consistently with `species_labels` and allow
            repetition to encode stoichiometry.

        species_labels : list[str]
            Species names used to enumerate all possible complexes.

        max_order : int
            Maximum complex size. Complexes of sizes 0..max_order are included.

        num_reactions_template : int, optional
            Number of initial reactions in each IOCRN that are treated as template
            reactions (not counted). Their cells are marked with hatching.

        title : str, optional
            Plot title.

        cmap_name : str, optional
            Name of a matplotlib sequential colormap.

        max_ticks : int, optional
            Max tick labels shown on each axis. Labels are uniformly thinned if needed.

        figsize : tuple, optional
            Matplotlib figure size.

    Returns:
        None, The function displays the plot via matplotlib and does not return a value.

    Raises:
        ValueError
            If any reaction's reactant or product complex is not found among the
            generated complexes (usually indicates a mismatch in `species_labels`
            or `max_order`).
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

    ax.set_xlabel("products")
    ax.set_ylabel("reactants")
    ax.set_title(title)

    # gridlines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.3)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations_with_replacement

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
    use_plotly=False,
    plot_dic=None,
    marker_size=8,
):
    """
    Plot a reactant→product scatter map where each non-template reaction is a point.

    Each reaction contributes one point at integer coordinates:
      - x = index(product_complex)
      - y = index(reactant_complex)
    Complexes are enumerated as all multisets of sizes 0..`max_order` over
    `species_labels` (including the empty complex). Points are jittered by
    `jitter_scale` to reduce overplotting. Points are colored by per-network
    performance values (`perf`), so all reactions from the same IOCRN share the
    same color.

    Template reactions (first `num_reactions_template` reactions in each IOCRN)
    are not plotted; their cells are instead marked (matplotlib: hatched boxes,
    Plotly: dashed rectangles).

    Args:
        iocrns : list
            List of IOCRN-like objects with `.reactions`. Each reaction must provide:
            - reaction.reactant_labels : list[str]
            - reaction.product_labels  : list[str]

        perf : array-like
            Performance value per IOCRN (same length as `iocrns`). Lower/higher is not
            interpreted here; it is used directly for coloring.

        species_labels : list[str]
            Species names used to enumerate complexes.

        max_order : int
            Maximum complex size used to build the complex index.

        num_reactions_template : int, optional
            Number of initial reactions treated as templates (excluded from scatter).

        title : str, optional
            Plot title.

        cmap_name : str, optional
            Colormap name for point coloring (matplotlib or Plotly colorscale).

        max_ticks : int, optional
            Max tick labels shown on each axis (labels are uniformly thinned).

        figsize : tuple, optional
            Figure size. If Plotly is used, this is interpreted in "matplotlib inches"
            and converted approximately to pixels.

        jitter_scale : float, optional
            Uniform jitter range added to both x and y coordinates.

        use_plotly : bool, optional
            If True, use Plotly for an interactive scatter plot with rich hover text.
            If False, use matplotlib.

        plot_dic : dict, optional
            Matplotlib rcParams override dictionary (used only when `use_plotly=False`).

        marker_size : float, optional
            Scatter marker size (matplotlib points^2 or Plotly marker size).

    Returns:
        matplotlib.figure.Figure or None
            If `use_plotly=False`, returns the matplotlib Figure. If `use_plotly=True`,
            renders the Plotly figure and returns None.

    Raises:
        ValueError
            If `perf` length does not match `iocrns`, or if complexes are not found
            in the generated complex list (mismatch in `species_labels` / `max_order`).
    """
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

    # --- Label Generation (HTML for Plotly, LaTeX for Matplotlib) ---
    def format_species(s, is_html=False):
        parts = s.split('_')
        base = parts[0]
        sub = '_'.join(parts[1:]) if len(parts) > 1 else ''
        
        if is_html:
            # HTML: <b>X</b><sub>1</sub>
            lbl = f"<b>{base}</b>"
            if sub:
                lbl += f"<sub>{sub}</sub>"
            return lbl
        else:
            # LaTeX: \mathbf{X}_{1}
            lbl = r'\mathbf{' + base + '}'
            if sub:
                lbl += r'_{' + sub + '}'
            return lbl

    def format_complex(c, is_html=False):
        if len(c) == 0:
            return "Ø" if is_html else r'\varnothing'
        separator = "+" if is_html else "\!+\!"
        return separator.join(format_species(s, is_html) for s in c)

    # Generate the correct labels for the chosen backend
    if use_plotly:
        axis_labels = [format_complex(c, is_html=True) for c in complexes]
    else:
        axis_labels = [r"$" + format_complex(c, is_html=False) + r"$" for c in complexes]

    # --- collect scatter points + special cells ---
    xs, ys, cs = [], [], []
    hover_texts = []
    special_cells = set()
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
                special_cells.add((ri, pi))
                continue

            x = pi + rng.uniform(-jitter_scale, jitter_scale)
            y = ri + rng.uniform(-jitter_scale, jitter_scale)
            xs.append(x)
            ys.append(y)
            cs.append(p)
            
            if use_plotly:
                txt = (
                    f"<b>CRN Index:</b> {net_idx}<br>"
                    f"<b>Performance:</b> {p:.4f}<br>"
                    f"<b>Reaction:</b> {reaction}<br>"
                    f"<b>Full CRN:</b><br>{str(iocrn).replace(chr(10), '<br>')}"
                )
                hover_texts.append(txt)

    xs, ys, cs = map(np.array, (xs, ys, cs))

    # --- Tick thinning logic ---
    n = len(axis_labels)
    if n <= max_ticks:
        tick_idx = np.arange(n)
    else:
        tick_idx = np.linspace(0, n - 1, max_ticks, dtype=int)
    
    tick_labels = [axis_labels[i] for i in tick_idx]

    # ---------------------------------------------------------
    # PLOTLY IMPLEMENTATION
    # ---------------------------------------------------------
    if use_plotly:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(
                size=8,
                color=cs,
                colorscale=cmap_name,
                showscale=True,
                colorbar=dict(title="loss"),
                opacity=0.8
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>"
        ))

        # Shapes list for Template boxes AND Grid lines
        shapes = []

        # 1. Template Boxes (Dashed Rectangles)
        for (ri, pi) in special_cells:
            shapes.append(dict(
                type="rect",
                x0=pi - 0.5, x1=pi + 0.5,
                y0=ri - 0.5, y1=ri + 0.5,
                line=dict(color="black", width=1, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="above"
            ))
        
        # 2. Manual Grid Lines (at -0.5, 0.5, 1.5 ...)
        # This frames the integer coordinates (0, 1, 2) in the center of the cells
        grid_color = "rgba(0,0,0,0.1)"
        grid_locs = np.arange(-0.5, K, 1) # -0.5, 0.5, 1.5 ... K-0.5
        
        # Vertical Grid Lines
        for x in grid_locs:
            shapes.append(dict(
                type="line",
                x0=x, x1=x,
                y0=-0.5, y1=K-0.5,
                line=dict(color=grid_color, width=1),
                layer="below"
            ))
            
        # Horizontal Grid Lines
        for y in grid_locs:
            shapes.append(dict(
                type="line",
                x0=-0.5, x1=K-0.5,
                y0=y, y1=y,
                line=dict(color=grid_color, width=1),
                layer="below"
            ))

        fig.update_layout(
            title=title,
            shapes=shapes,
            xaxis=dict(
                title="products",
                range=[-0.5, K - 0.5],
                tickmode='array',
                tickvals=tick_idx,
                ticktext=tick_labels,
                zeroline=False,
                showgrid=False,  # Disable default grid (which cuts through 0, 1, 2)
            ),
            yaxis=dict(
                title="reactants",
                range=[-0.5, K - 0.5],
                tickmode='array',
                tickvals=tick_idx,
                ticktext=tick_labels,
                zeroline=False,
                showgrid=False, # Disable default grid
            ),
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            plot_bgcolor='white',
            hoverlabel=dict(align="left")
        )

        fig.show()
        return

    # ---------------------------------------------------------
    # MATPLOTLIB IMPLEMENTATION (Original)
    # ---------------------------------------------------------
    if plot_dic is not None:
        plt.rcParams.update(plot_dic)
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(-0.5, K - 0.5)
    
    # Major ticks (Labels at integers 0, 1, 2...)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    
    # Minor ticks (Grid lines at half-integers -0.5, 0.5...)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    ax.xaxis.tick_top()                 
    ax.xaxis.set_label_position('top')  

    
    # Grid settings: Turn Major OFF, Turn Minor ON
    ax.grid(which="major", visible=False)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.3)
    
    # Hide minor tick marks themselves
    ax.tick_params(which="minor", length=0)

    sc = ax.scatter(
        xs, ys,
        c=cs,
        cmap=cmap_name,
        s=marker_size,
        alpha=0.8,
        edgecolors="none",
    )

    class OneDecimalScalarFormatter(mticker.ScalarFormatter):
        def _set_format(self):
            # Force tick labels to one decimal in mantissa (e.g., 1.2) while keeping ×10^n on top
            self.format = r'$\mathdefault{%1.1f}$' if self._useMathText else '%1.1f'

    cbar = fig.colorbar(sc, ax=ax)

    fmt = OneDecimalScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))   # always show scientific notation (×10^n)
    cbar.formatter = fmt
    cbar.update_ticks()

    # Exponent placement (vertical colorbar)
    cbar.ax.yaxis.set_offset_position('left')  # optional
    cbar.ax.yaxis.get_offset_text().set_va('bottom')

    cbar.set_label("loss")

    for (ri, pi) in special_cells:
        rect = Rectangle(
            (pi - 0.5, ri - 0.5),
            1, 1,
            fill=True,
            facecolor="none",
            hatch="//////",
            linewidth=0.5,
            edgecolor="black",
            linestyle='--'
        )
        ax.add_patch(rect)

    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("products")
    ax.set_ylabel("reactants")
    ax.set_title(title)

    fig.tight_layout()
    plt.show()
    return fig

def hamming_radius_graph(X_bool: np.ndarray, t: int):
    """
    Build a sparse graph connecting all pairs within a Hamming distance threshold.

    This function constructs a sparse neighbor graph where an undirected edge
    exists between two boolean signatures if their Hamming distance is <= `t`.

    Internally it uses scikit-learn's radius neighbor graph with the normalized
    Hamming distance in [0, 1], then rescales back to Hamming counts.

    Args:
        X_bool (np.ndarray): 
            Boolean or {0,1} array of shape (n_samples, n_bits). Each row is a topology
            signature.

        t (int): 
            Hamming distance threshold in *bit counts* (0..n_bits).

    Returns:
        scipy.sparse.csr_matrix
            CSR sparse matrix of shape (n_samples, n_samples) with nonzero entries
            for edges where distance <= t. Values are Hamming distances in counts
            (stored as float for downstream NetworkX compatibility).
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
    Plot a sparse distance graph (typically from Hamming thresholds).

    Nodes represent unique topologies. Edge weights are interpreted as distances.
    For the spring layout, distances are inverted to act as attraction strengths.

    Args:
        G_counts_csr : scipy.sparse.csr_matrix
            Sparse adjacency with edge weights representing distances (e.g. Hamming
            counts). Zero entries indicate no edge.

        counts : array-like
            Node sizes/labels, typically frequency counts per unique topology. Must
            have length equal to number of nodes.

        title : str, optional
            Plot title.

        figsize : tuple, optional
            Matplotlib figure size.

        with_edge_labels : bool, optional
            If True, draw edge labels showing integer distances. This can be slow for
            large graphs.

        seed : int, optional
            Random seed for layout reproducibility.

    Returns:
        matplotlib.figure.Figure
            The created matplotlib figure.
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
    Build and plot a topology diversity graph from a list of CRNs.

    Each CRN is expected to implement:

      - crn.get_bool_signature() -> 1D boolean/{0,1} array

    The function:

    1. extracts signatures,
    2. collapses duplicates and counts frequencies,
    3. builds a sparse graph connecting signatures within Hamming distance <= t,
    4. plots the resulting graph.

    Args:
        crn_list : list
            List of CRN/IOCRN objects with `.get_bool_signature()`.

        t : int, optional
            Hamming distance threshold (bit counts) for edge inclusion.

        figsize : tuple, optional
            Matplotlib figure size.

        with_edge_labels : bool, optional
            Whether to draw distance labels on edges (may be slow).

    Returns:
        matplotlib.figure.Figure
            The created matplotlib figure.
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



# --- better plotting of distance graphs --- 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
import community.community_louvain as community_louvain # pip install python-louvain (optional)

def build_hamming_graph(X_bool: np.ndarray, perf=None, alpha=0.0, t: int = None, k: int = 5):
    """
    Build a sparse distance graph from boolean signatures using Hamming distance.

    Two modes are supported:

    - k-NN graph (if `k` is provided): connect each node to its k nearest neighbors.
    - radius graph (if `t` is provided): connect all nodes within Hamming distance <= t.

    Optionally, edge distances can be *stretched* by performance gaps:
        dist_new(u,v) = dist_hamming(u,v) * (1 + alpha * |perf[u] - perf[v]|)
    This can make high/low performing regions separate more clearly in layouts.

    Args:
        X_bool : np.ndarray
            Boolean or {0,1} array of shape (n_samples, n_bits).

        perf : array-like or None, optional
            Per-node performance values aligned with rows of `X_bool`.

        alpha : float, optional
            Strength of performance-based stretching. If 0.0, no stretching is applied.

        t : int or None, optional
            Hamming radius threshold in counts. Used if `k` is None.

        k : int or None, optional
            Number of neighbors for k-NN graph. If provided, takes precedence over `t`.

    Returns:
        scipy.sparse.csr_matrix
            Sparse matrix with edge weights representing (possibly stretched) distances.
    """
    X_bool = np.asarray(X_bool, dtype=np.uint8)
    n, d = X_bool.shape
    
    # 1. Build Structural Graph (Hamming Distance)
    if k is not None:
        nn = NearestNeighbors(metric="hamming", n_neighbors=k+1, n_jobs=-1)
        nn.fit(X_bool)
        G_sparse = nn.kneighbors_graph(mode="distance")
    elif t is not None:
        radius = t / d
        nn = NearestNeighbors(metric="hamming", radius=radius, n_jobs=-1)
        nn.fit(X_bool)
        G_sparse = nn.radius_neighbors_graph(mode="distance")
    else:
        raise ValueError("Must provide either 't' (radius) or 'k' (neighbors).")

    # Convert normalized Hamming distance [0,1] back to integer counts [0, d]
    # This matrix represents 'Structural Distance'
    G_dist = G_sparse.multiply(d)
    G_dist.setdiag(0.0)
    G_dist.eliminate_zeros()
    
    # 2. Apply Performance Penalty (Reward-Weighted Topology)
    if perf is not None and alpha > 0:
        # Ensure perf is array
        perf = np.asarray(perf)
        
        # Iterate over non-zero edges (sparse efficient)
        rows, cols = G_dist.nonzero()
        
        # Calculate performance delta for every edge
        # |Perf[u] - Perf[v]|
        delta_p = np.abs(perf[rows] - perf[cols])
        
        # Calculate multiplier: 1 + alpha * delta
        penalty = 1.0 + (alpha * delta_p)
        
        # Apply penalty to structural distances
        # New Dist = Old Dist * Penalty
        # Nodes with different performance become "farther" apart
        new_data = G_dist.data * penalty
        
        # Update graph matrix
        G_dist.data = new_data
        
        print(f"Applied reward weighting (alpha={alpha}). Edges stretched by fitness gaps.")

    return G_dist

def plot_topology_graph(G_counts_csr, counts, 
                        node_values=None, 
                        layout_method="spring", 
                        color_by="community", 
                        figsize=(10, 10), 
                        seed=42,
                        title="CRN Topology",
                        label_percentile=90, 
                        unique=False, 
                        crn_ids=None, 
                        plot_dic=None, 
                        graph_dic=None):
    """
    Plot a topology graph with flexible coloring and layout options.

    The input sparse matrix is interpreted as a distance graph. Layout uses
    inverse-distance attraction for spring-style layouts (or falls back as needed).

    Coloring modes:

    - 'community': color nodes by detected communities (Louvain/greedy modularity).
    - 'value': color nodes by `node_values` (e.g. min loss per topology).
    - 'count': color nodes by log1p(counts).
    - 'dual': face colored by `node_values` and border colored by community id.

    Args:
        G_counts_csr : scipy.sparse.csr_matrix
            Sparse adjacency with edge weights as distances.

        counts : array-like
            Per-node counts used for node sizing (and optionally labels).

        node_values : array-like or None, optional
            Per-node scalar values used for coloring in 'value' or 'dual' mode.

        layout_method : str, optional
            Layout method identifier. Currently uses spring-like layouts by default.
            ('mds' name is accepted but may fall back depending on graph properties.)

        color_by : str, optional
            One of {'community', 'degree', 'count', 'value', 'dual'}.
            Note: 'degree' is reserved but not explicitly implemented here; it will
            behave like the default if not set up.

        figsize : tuple, optional
            Matplotlib figure size.

        seed : int, optional
            Random seed for layout reproducibility.

        title : str, optional
            Plot title.

        label_percentile : float, optional
            Percentile threshold for showing node labels (based on `counts`) when
            `unique=False`.

        unique : bool, optional
            If True, assumes each node corresponds to a unique CRN instance (so counts
            are typically all ones) and labels can use `crn_ids`.

        crn_ids : array-like or None, optional
            Optional IDs to label nodes when `unique=True`.

        plot_dic : dict or None, optional
            Matplotlib rcParams updates applied before plotting.

        graph_dic : dict or None, optional
            Plot styling configuration (node size multipliers, linewidths, etc.).

    Returns:
        matplotlib.figure.Figure
            The created matplotlib figure (not shown automatically here).
    """
    if graph_dic is None:
        graph_dic = {"fontsize": 8, 
                     "edgewidth": 0.8,
                     "nodesize_multiplier": 300,
                     "nodesize_offset": 50,
                     "show_colorbar": True,
                     "innersize_ratio": 0.8,
                     "inner_node_linewidth": 1.5,
                     "outer_node_linewidth": 3.0
                     }

    G = nx.from_scipy_sparse_array(G_counts_csr)
    
    # 1. Layout Calculation
    # Note: Spring layout uses 'weight' as ATTRACTION. 
    # Our G_counts_csr contains DISTANCE.
    # We must invert distance to get attraction weight for the layout engine.
    inv_weights = {(u, v): 1.0 / (d["weight"] + 1e-6) for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, inv_weights, name="attraction")

    print(f"Computing layout ({layout_method})...")
    if layout_method == "mds":
        try:
            pos = nx.kamada_kawai_layout(G, weight='weight') # Kamada-Kawai uses distance (weight) directly
        except:
            pos = nx.spring_layout(G, seed=seed, weight='attraction', k=2.0/np.sqrt(len(G.nodes)))
    else:
        # Standard spring layout (uses attraction)
        pos = nx.spring_layout(G, seed=seed, weight='attraction', k=1.5/np.sqrt(len(G.nodes)+1), iterations=100)

    # 2. Helper for Communities (Louvain maximizes modularity using 'attraction' weights)
    def get_partition(graph):
        try:
            # Louvain needs attraction weights (stronger weight = same community)
            return community_louvain.best_partition(graph, weight='attraction')
        except:
            try:
                communities = nx.community.greedy_modularity_communities(graph, weight='attraction')
                part = {}
                for idx, comm in enumerate(communities):
                    for node in comm: part[node] = idx
                return part
            except:
                return None

    # 3. Node Coloring Setup
    cmap = plt.cm.tab20
    show_colorbar = False
    node_colors = "#4285F4"
    node_edge_colors = "white" # Default border
    node_linewidths = graph_dic["inner_node_linewidth"]
    is_dual_mode = False

    # 4. Apply Coloring Mode
    if color_by == "dual" and node_values is not None:
        # Face = Performance (Loss), Edge = Community
        node_colors = node_values
        
        # Use Greyscale for Performance to avoid clashing with Cluster Hue
        # Greys_r: Low Value (Good) = Dark, High Value (Bad) = Light
        cmap = plt.cm.viridis 
        show_colorbar = graph_dic["show_colorbar"]
        is_dual_mode = True
        
        partition = get_partition(G)
        if partition:
            comm_ids = [partition[n] for n in G.nodes()]
            
            # Use a high-saturation spectral map for clusters to contrast with Grey
            edge_cmap = plt.cm.gist_rainbow
            num_comms = max(comm_ids) + 1 if comm_ids else 1
            
            # Map IDs to colors
            node_edge_colors = [edge_cmap(i / num_comms) for i in comm_ids]
            node_linewidths = graph_dic["outer_node_linewidth"] # Thicker borders for the cluster identity
            print("Dual Coloring: Face=Loss (Greyscale), Edge=Cluster (Rainbow) with Black Separator")
        else:
            print("Warning: Could not detect communities for dual coloring.")
            is_dual_mode = False # Fallback
            
    elif color_by == "value" and node_values is not None:
        node_colors = node_values
        cmap = plt.cm.viridis_r # Reversed for loss
        show_colorbar = True
        
    elif color_by == "community":
        partition = get_partition(G)
        if partition:
            node_colors = [partition[n] for n in G.nodes()]
            cmap = plt.cm.tab20
            
    elif color_by == "count":
        node_colors = np.log1p(counts)
        cmap = plt.cm.viridis
        show_colorbar = True

    # 5. Draw Setup
    if plot_dic is not None:
        plt.rcParams.update(plot_dic)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Edge Visualization
    edges = G.edges(data=True)
    # Visualization opacity based on original distance (weight)
    # Lower distance (weight) = Stronger connection = Darker edge
    weights = np.array([d.get("weight", 1.0) for u, v, d in edges])
    if len(edges) > 0:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            # Mapping: Short dist -> Alpha 0.4, Long dist -> Alpha 0.05
            alphas = 0.4 - 0.35 * (weights - w_min) / (w_max - w_min)
        else:
            alphas = np.full(len(edges), 0.2)
        edge_colors = [(0, 0, 0, a) for a in alphas]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=graph_dic["edgewidth"], ax=ax)
    
    # Node Visualization (Size scaling)
    node_sizes = np.log1p(counts) * graph_dic["nodesize_multiplier"] + graph_dic["nodesize_offset"]
    
    if is_dual_mode:
        # --- Dual Mode: Two-step drawing for black separator ---
        
        # 1. Draw outer thick border (Cluster ID) with transparent face
        nx.draw_networkx_nodes(G, pos, 
                                node_size=node_sizes,
                                node_color="none", # Transparent face
                                edgecolors=node_edge_colors, 
                                linewidths=graph_dic["outer_node_linewidth"], # Thick cluster border
                                ax=ax)
        
        # 2. Draw inner face (Performance) with thin black border
        # Scale down size slightly to fit inside the outer border
        inner_sizes = node_sizes * graph_dic["innersize_ratio"]
        sc = nx.draw_networkx_nodes(G, pos, 
                                    node_size=inner_sizes,
                                    node_color=node_colors, 
                                    cmap=cmap, 
                                    edgecolors="black", # The black separator line
                                    linewidths=graph_dic["inner_node_linewidth"],     # Thin separator
                                    ax=ax)
    else:
        # --- Standard Mode: Single draw ---
        sc = nx.draw_networkx_nodes(G, pos, 
                                    node_size=node_sizes,
                                    node_color=node_colors, 
                                    cmap=cmap, 
                                    edgecolors=node_edge_colors, 
                                    linewidths=graph_dic["inner_node_linewidth"],
                                    ax=ax)
    
    if show_colorbar and cmap is not None:
        # sc is the handle to the inner, performance-colored nodes in dual mode
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        label = "Performance (Min Loss)" if color_by in ["value", "dual"] else "Frequency (Log)"
        cbar.set_label(label)

    # Labels for top nodes
    if unique is False:
        if counts is not None:
            threshold = np.percentile(counts, label_percentile) if len(counts) > 20 else 0
            labels = {n: str(counts[n]) for n in G.nodes() if counts[n] >= threshold}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', ax=ax)
    else:
        if crn_ids is not None:
            labels = {n: str(crn_ids[n]) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=graph_dic["fontsize"], font_color='black', ax=ax)

    ax.set_axis_off()
    # ax.set_title(f"{title}\n({len(G.nodes)} Unique Topologies)", fontsize=14)
    
    return fig

def visualize_crn_diversity(crn_list, perf=None, k=5, t=None, 
                            layout_method="spring", label_percentile=90, 
                            alpha=0.0, unique=False, plot_dic=None, graph_dic=None, figsize=(10,10)): # <--- NEW: Alpha parameter
    """
    High-level wrapper to visualize IOCRN topological diversity.

    Extracts boolean signatures from CRNs, builds a sparse Hamming-distance graph
    (k-NN or radius), optionally stretches edges by performance differences, and
    plots the resulting topology map.

    Expected CRN interface:

      - crn.get_bool_signature() -> 1D boolean/{0,1} array

    Args:
        crn_list : list
            List of CRN/IOCRN objects with `.get_bool_signature()`.

        perf : array-like or None, optional
            Per-CRN performance values. If `unique=False`, performance is aggregated
            per unique topology by taking the minimum loss for that topology.

        k : int, optional
            Number of neighbors for k-NN graph construction (used if `t` is None).

        t : int or None, optional
            Hamming radius threshold in counts. If provided, radius graph is used.

        layout_method : str, optional
            Layout method forwarded to `plot_topology_graph`.

        label_percentile : float, optional
            Controls which nodes get labels (based on `counts`) when `unique=False`.

        alpha : float, optional
            Performance stretch coefficient. If > 0 and `perf` is provided, edges are
            multiplied by (1 + alpha * |perf_u - perf_v|).

        unique : bool, optional
            If False, signatures are deduplicated into unique topologies.
            If True, each CRN is treated as its own node.

        plot_dic, graph_dic : dict or None, optional
            Plot configuration dictionaries forwarded to plotting.

        figsize : tuple, optional
            Matplotlib figure size.

    Returns:
        matplotlib.figure.Figure or None
            The created matplotlib figure, or None if signatures cannot be extracted.
    """

    # 1. Extract Signatures
    try:
        crn_topologies = np.stack([crn.get_bool_signature() for crn in crn_list]).astype(bool)
    except AttributeError:
        print("Error: Objects in crn_list must implement .get_bool_signature()")
        return

    # 3. Build Graph (Passing alpha and aggregated performance)
    max_perf_per_node = None
    color_mode = "dual" # Default
    if unique is False:
        # 2. Deduplicate and Aggregate Performance
        unique_sigs, inverse_indices, counts = np.unique(crn_topologies, axis=0, return_inverse=True, return_counts=True)
        
        print(f"Found {len(unique_sigs)} unique topologies from {len(crn_list)} samples.")

        if perf is not None:
            perf = np.asarray(perf)
            if len(perf) != len(crn_list):
                print("Warning: perf length does not match crn_list length. Ignoring performance coloring.")
            else:
                max_perf_per_node = np.full(len(unique_sigs), np.inf)
                np.minimum.at(max_perf_per_node, inverse_indices, perf)
                color_mode = "dual" 
                print("Performance data aggregated (Min Loss per topology).")
        G_matrix = build_hamming_graph(unique_sigs, perf=max_perf_per_node, alpha=alpha, t=t, k=k)
    else:
        G_matrix = build_hamming_graph(crn_topologies, perf=perf, alpha=alpha, t=t, k=k)
    
    # 4. Plot
    print_labels = graph_dic.get("print_labels", False) if graph_dic else False
    crn_ids = list(range(len(crn_list))) if unique else None
    counts = [1] * len(crn_list) if unique else counts
    fig = plot_topology_graph(G_matrix, counts, 
                              node_values=max_perf_per_node if not unique else perf, 
                              color_by=color_mode, 
                              layout_method=layout_method,
                              title="CRN Topological Diversity",
                              label_percentile=label_percentile, 
                              unique=unique, 
                              crn_ids=crn_ids if print_labels else None, 
                              plot_dic=plot_dic,
                              graph_dic=graph_dic, 
                              figsize=figsize)
    plt.show()
    return fig


### --- Multi IC trainsient response plot --- ###

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import math

def plot_trajectory_ensemble(
    all_iocrns, 
    ic, 
    u_list, 
    time_horizon, 
    n_top=50, 
    figsize=(15, 15),
    cmap_name="viridis_r", # Reversed: Low Loss (Best) = Yellow/Bright, High Loss = Purple/Dark
    alpha_min=0.1,
    alpha_max=0.6,
    highlight_best=True
):
    """
    Plot an ensemble of trajectories for the top-performing IOCRNs.

    Networks are sorted by `crn.last_task_info['reward']` (ascending assumed best),
    and the top `n_top` are simulated (or retrieved from cache if available).
    For each input scenario, trajectories from all selected networks are rendered
    as a LineCollection colored by loss value.

    Caching behavior:

    - If `crn.last_task_info` contains cached transient response fields
      (type == 'transient response', plus outputs/time_horizon), those trajectories
      are reused.
    - Otherwise, trajectories are simulated via `crn.transient_response(...)`.

    Args:
        all_iocrns : list
            List of IOCRN objects.

        ic : object
            Initial condition helper with method `ic.get_ic(crn)` returning x0_list.

        u_list : list
            List of input scenarios passed to the CRN simulator.

        time_horizon : np.ndarray
            Time grid for simulation (and/or matching cached trajectories).

        n_top : int, optional
            Number of best networks to include.

        figsize : tuple, optional
            Matplotlib figure size.

        cmap_name : str, optional
            Colormap used to color trajectories by loss (default reversed so low loss
            is brighter).

        alpha_min, alpha_max : float, optional
            Alpha range used to fade worse trajectories (higher loss -> lower alpha).

        highlight_best : bool, optional
            If True, overlays the best network trajectories (currently red dashed).

    Returns
    -------
    None
        Displays the plot via matplotlib.
    """
    
    # 1. Sort by Loss (Ascending: Best -> Worst)
    sorted_iocrns = sorted(all_iocrns, key=lambda x: x.last_task_info.get('reward', np.inf), reverse=False)
    
    # Slice top N
    n_top = min(n_top, len(sorted_iocrns))
    top_iocrns = sorted_iocrns[:n_top]
    
    if n_top == 0:
        print("No IOCRNs to plot.")
        return

    print(f"Preparing top {n_top} trajectories (checking cache first)...")

    ensemble_data = {} 
    losses = []
    
    # Initialize dimensions placeholders
    num_scenarios = 0
    num_species = 0
    initialized = False

    for i, crn in enumerate(top_iocrns):
        loss = crn.last_task_info.get('reward', np.nan)
        
        # --- CACHE CHECK ---
        cached_info = getattr(crn, 'last_task_info', {})
        
        use_cache = (
            cached_info.get('type') == 'transient response' 
            and 'outputs' in cached_info 
            and 'time_horizon' in cached_info
        )
        
        if use_cache:
            try:
                raw_outputs = cached_info['outputs']
                y_list = [out.T for out in raw_outputs] 
                t_grid = cached_info['time_horizon']
            except Exception:
                use_cache = False

        if not use_cache:
            x0 = ic.get_ic(crn)
            _, _, y_list, _ = crn.transient_response(u_list, x0, time_horizon)
            t_grid = time_horizon

        # --- VALIDATION & SETUP ---
        if not y_list or len(y_list) == 0:
            continue
            
        current_scenarios = len(y_list)
        if y_list[0].shape[0] == 0: continue
        current_species = y_list[0].shape[1]

        if not initialized:
            num_scenarios = current_scenarios
            num_species = current_species
            # Structure: data[scenario_idx][species_idx] = list of segments
            for sc_idx in range(num_scenarios):
                ensemble_data[sc_idx] = {sp_idx: [] for sp_idx in range(num_species)}
            initialized = True
        
        if current_scenarios != num_scenarios or current_species != num_species:
            continue

        losses.append(loss)

        # --- COLLECT SEGMENTS ---
        for sc_idx, traj_array in enumerate(y_list):
            traj_len = traj_array.shape[0]
            if traj_len != len(t_grid):
                t_current = t_grid[:traj_len]
            else:
                t_current = t_grid

            for sp_idx in range(num_species):
                points = np.column_stack([t_current, traj_array[:, sp_idx]])
                ensemble_data[sc_idx][sp_idx].append(points)

    # 3. Setup Grid Plot (Square-ish layout)
    if not initialized:
        print("No valid trajectory data found.")
        return

    # Calculate grid dimensions
    n_cols = int(math.ceil(math.sqrt(num_scenarios)))
    n_rows = int(math.ceil(num_scenarios / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=figsize, 
                             sharex=True, sharey=True, # Global sharing for cleaner grid
                             squeeze=False)
    
    # Flatten axes for easy indexing
    axes_flat = axes.flatten()

    # Normalize Loss for Colormap
    losses = np.array(losses)
    if len(losses) > 0:
        norm = mcolors.Normalize(vmin=losses.min(), vmax=losses.max())
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(cmap_name)

    # Line styles for different species to distinguish them in the same plot
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))] 

    # 4. Plot Ensembles
    print("Rendering ensemble...")
    
    for sc_idx in range(num_scenarios):
        ax = axes_flat[sc_idx]
        ax.set_title(f"Input {sc_idx+1}: {u_list[sc_idx]}")
        
        for sp_idx in range(num_species):
            segments = ensemble_data[sc_idx][sp_idx]
            if not segments:
                continue

            # A. Add LineCollection (Background Ensemble)
            # Use a specific linestyle for this species
            ls = linestyles[sp_idx % len(linestyles)]
            
            lc = LineCollection(segments, cmap=cmap, norm=norm, linestyles=ls)
            lc.set_array(losses)
            
            # Dynamic Alpha
            if len(losses) > 1 and losses.max() > losses.min():
                n_loss = (losses - losses.min()) / (losses.max() - losses.min())
                alphas = alpha_max - (n_loss * (alpha_max - alpha_min))
            else:
                alphas = np.full(len(losses), alpha_max)
                
            lc.set_alpha(alphas)
            lc.set_linewidth(1.0)
            ax.add_collection(lc)
            
            # B. Highlight Best (#1)
            if highlight_best and len(segments) > 0:
                best_segment = segments[0]
                # Label only once per species (in the first scenario) to keep legend clean
                label = f'Best (Sp {sp_idx+1})' if sc_idx == 0 else None
                ax.plot(best_segment[:, 0], best_segment[:, 1], 
                        color='red', linestyle='--', linewidth=2.0, label=label, zorder=10)

        ax.autoscale()
        ax.set_xlim(time_horizon[0], time_horizon[-1])
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for k in range(num_scenarios, n_rows * n_cols):
        axes_flat[k].set_visible(False)

    # Labels (Outer grid only)
    for ax in axes[-1, :]: # Bottom row
        ax.set_xlabel("Time")
    for ax in axes[:, 0]:  # Left column
        ax.set_ylabel("Concentration")

    # 5. Colorbar
    if len(losses) > 0:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label(f"Loss (Top {len(losses)} CRNs)")
    
    # Legend (Top Left Plot)
    if highlight_best:
        axes_flat[0].legend(loc='upper right', fontsize='small', framealpha=0.9)

    plt.suptitle(f"Top {len(losses)} Trajectories (Colored by Loss)", fontsize=16)
    plt.show()


def plot_truth_table(u_list, r_list, title='Truth Table for Target Logic Function', figsize=(6, 4), silent=False):
    """
    Plot a heatmap representation of a logic truth table (target outputs).

    Inputs are displayed along the x-axis as tuples (e.g. (0, 1, 0)).
    Outputs are displayed along the y-axis:

    - If r_list is shape (N,), a single row "Output" is shown.
    - If r_list is shape (N, 1), it is treated as a single output row.
    - If r_list is shape (N, M), it is transposed to (M, N) and each output is a row.

    Each cell is annotated with the numeric value (integer if near-integer, else
    2-decimal float). A colorbar indicates output intensity.

    Args:
        u_list : list
            List of input combinations. Each element can be a list/array/scalar.

        r_list : list or np.ndarray
            Target outputs aligned with u_list.

        title : str, optional
            Plot title.

        figsize : tuple, optional
            Matplotlib figure size.

        silent : bool, optional
            If True, does not call plt.show() (useful in notebooks/pipelines).

    Returns:
        matplotlib.figure.Figure
            The created figure (useful for saving).
    """
    
    # 1. Prepare Data
    # Ensure inputs are formatted nicely for labels
    # Converts numpy arrays e.g. [0. 1.] to string "(0, 1)" or just "0, 1"
    xticklabels = []
    for u in u_list:
        if isinstance(u, (np.ndarray, list)):
            # Flatten and format integers if possible, else floats
            u_clean = np.array(u).flatten()
            # convert to int
            u_clean = [int(x) for x in u_clean]
            label = str(tuple(u_clean))
        else:
            label = str(u)
        xticklabels.append(label)

    # Ensure outputs are in a (rows, cols) format for imshow
    # r_list is typically shape (N_samples, N_outputs) or just (N_samples,)
    outputs = np.array(r_list)
    if outputs.ndim == 1:
        # If 1D, reshape to (1, N) so it plots as a single horizontal strip
        outputs = outputs.reshape(1, -1)
    elif outputs.ndim == 2 and outputs.shape[1] == 1:
        # If (N, 1), transpose to (1, N) for the standard horizontal "truth table row" look
        outputs = outputs.T
    else:
        # If (N, M) where M > 1, we typically want inputs on X-axis (columns), outputs on Y-axis (rows)
        # So we transpose it to be (M, N)
        outputs = outputs.T

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a clean colormap (e.g., 'Blues', 'viridis', or 'coolwarm')
    # 'Blues' works well for 0->1 intensity.
    im = ax.imshow(outputs, cmap='Blues', aspect='auto', vmin=np.min(outputs), vmax=np.max(outputs))

    # 3. Annotate Cells with Values
    # Loop over data dimensions and create text annotations.
    rows, cols = outputs.shape
    
    # Determine threshold for switching text color (for readability)
    threshold = (outputs.max() + outputs.min()) / 2.0

    for i in range(rows):
        for j in range(cols):
            val = outputs[i, j]
            text_color = "white" if val > threshold else "black"
            # Format: if integer-like, no decimals, otherwise 2 decimals
            if abs(val - round(val)) < 1e-5:
                val_str = f"{int(round(val))}"
            else:
                val_str = f"{val:.2f}"
                
            ax.text(j, i, val_str, ha="center", va="center", 
                    color=text_color, fontsize=10, fontweight='bold')

    # 4. Formatting Ticks and Grid
    # X-axis
    ax.set_xticks(np.arange(cols))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.set_xlabel('Input Combinations', fontsize=12, fontweight='bold')
    
    # Y-axis
    ax.set_yticks(np.arange(rows))
    if rows == 1:
        ax.set_yticklabels(['Output'])
    else:
        ax.set_yticklabels([f'Out {i+1}' for i in range(rows)])

    # Add minor grid lines to separate cells nicely
    ax.set_xticks(np.arange(cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Remove major grid
    ax.grid(which="major", visible=False)

    # 5. Finishing Touches
    ax.set_title(title, fontsize=14, pad=15)
    
    # Add colorbar but keep it aligned properly
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Output Value', rotation=270, labelpad=15)

    plt.tight_layout()
    if not silent:
        plt.show()

    # return the figure for further use if needed
    return fig


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

def plot_truth_table_transposed_nature(
    u_list,
    actual_outputs,
    logic_function,
    title="Truth table",
    figsize=(2.2, 1.2),
    silent=True,
):
    """
    Nature Methods–style transposed truth table (Refined):
    - Font size 6 for data.
    - First column (labels) colored to match sections (Green/Red).
    - Shading intensity reduced (pastel/lighter).
    """

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }

    def make_gradient_cmap(base_color):
        base_rgb = mcolors.to_rgb(base_color)
        return mcolors.LinearSegmentedColormap.from_list(
            f"grad_{base_color}", ["#FFFFFF", mcolors.to_hex(base_rgb)], N=256
        )

    def to_cell_color(val, cmap, vmin=0.0, vmax=1.0):
        if np.isnan(val):
            return (0.97, 0.97, 0.97, 1.0) # very light grey for NaNs
        t = (val - vmin) / (vmax - vmin + 1e-12)
        t = float(np.clip(t, 0.0, 1.0))
        
        # REDUCED INTENSITY:
        # Map 0.0 -> 0.05 (almost white)
        # Map 1.0 -> 0.60 (pastel, not full saturation)
        t = 0.05 + 0.55 * t 
        return cmap(t)

    def best_text_color(rgba):
        r, g, b = rgba[:3]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#111111" if lum > 0.6 else "#FFFFFF"

    with mpl.rc_context(rc):
        # ----- Data prep -----
        u_matrix = np.array(u_list, dtype=float)
        if u_matrix.ndim == 1:
            u_matrix = u_matrix.reshape(-1, 1)
        n_samples, n_inputs = u_matrix.shape

        targets = []
        for row in u_matrix:
            try:
                targets.append(float(logic_function(row)))
            except Exception:
                targets.append(np.nan)
        targets = np.array(targets, dtype=float)

        actuals = np.array(actual_outputs, dtype=float).flatten()
        if actuals.size != n_samples:
             actuals = np.resize(actuals, n_samples) 

        # ----- Colormaps (Pastel/Lighter Base) -----
        # Using lighter hex codes for base colors to keep shading gentle
        base_green = "#74c476"  # Pastel Green
        base_red   = "#709cd6"  # Pastel Red
        
        cmap_input = make_gradient_cmap(base_green)
        cmap_output = make_gradient_cmap(base_red)

        # Background colors for the label column (static light shades)
        # We pick a very light version of the base colors for the labels
        label_bg_input = mcolors.to_rgba("#e5f5e0") # Very pale green
        label_bg_output = mcolors.to_rgba("#d2e9fe") # Very pale red

        # ----- Build table data -----
        row_labels = [f"$\\bf{{u_{{{i+1}}}}}$" for i in range(n_inputs)] + ["Target", "Output"]
        cell_text, cell_colors = [], []

        # 1. Inputs
        for i in range(n_inputs):
            vals = u_matrix[:, i]
            cell_text.append([f"{x:g}" for x in vals])
            cell_colors.append([to_cell_color(x, cmap_input) for x in vals])

        # 2. Target
        cell_text.append([("—" if np.isnan(t) else f"{t:.2f}") for t in targets])
        cell_colors.append([to_cell_color(t, cmap_output) for t in targets])

        # 3. Output
        cell_text.append([f"{a:.2f}" for a in actuals])
        cell_colors.append([to_cell_color(a, cmap_output) for a in actuals])

        # ----- Figure -----
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        top_margin = 0.16
        fig.subplots_adjust(top=1 - top_margin)
        bbox = [0.0, 0.0, 1.0, 1.0]

        tbl = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            rowLabels=row_labels,
            cellLoc="center",
            loc="center",
            bbox=bbox,
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)

        # ----- Styling -----
        grid_col = "#D9D9D9"
        outer_col = "#A6A6A6"
        label_edge = "#B7C0D1"
        label_text = "#111111"

        idx_target = n_inputs
        idx_output = n_inputs + 1

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(grid_col)
            cell.set_linewidth(0.4)

            # --- Row Labels (First Column) ---
            if col == -1:
                # Apply specific shading based on row type
                if row < n_inputs:
                    cell.set_facecolor(label_bg_input)
                else:
                    cell.set_facecolor(label_bg_output)
                
                cell.set_edgecolor(label_edge)
                cell.set_linewidth(0.6)
                cell.set_text_props(color=label_text, weight="bold", ha="right")
                cell.set_fontsize(7) # Slightly larger for labels
            
            # --- Data Cells ---
            else:
                tc = best_text_color(cell.get_facecolor())
                weight = "bold" if row == idx_output else "normal"
                cell.set_text_props(color=tc, weight=weight)
                cell.set_fontsize(6)

        # ----- Separation line (black) -----
        total_rows = len(row_labels)
        row_h = bbox[3] / total_rows
        y_sep = bbox[1] + bbox[3] - (n_inputs * row_h)

        ax.add_line(
            plt.Line2D(
                [bbox[0], bbox[0] + bbox[2]],
                [y_sep, y_sep],
                transform=ax.transAxes,
                color="#000000",
                linewidth=0.9,
                zorder=10,
            )
        )

        # ----- Outer border -----
        ax.add_patch(
            patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                transform=ax.transAxes,
                fill=False,
                edgecolor=outer_col,
                linewidth=0.8,
                zorder=10,
            )
        )

        if not silent:
            plt.show()

        return fig, ax
    
def get_outputs(iocrn):
    """
    Helper function to extract the final output values from an IOCRN's last_task_info.
    Assumes outputs are stored in `iocrn.last_task_info['outputs']` as a list of arrays,
    where the last element of the first array corresponds to the final output values.

    Args:
        iocrn : object
            An IOCRN object with a `last_task_info` attribute containing 'outputs'.
    Returns:
        list
            A list of final output values extracted from the IOCRN's last_task_info.
    """
    return [o[0][-1] for o in iocrn.last_task_info['outputs']]