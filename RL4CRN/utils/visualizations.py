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
):
    """
    Plots a scatter map of reactants vs products indices.
    If use_plotly=True, generates an interactive plot where hovering shows the CRN.
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
        separator = " + "
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
                colorbar=dict(title="Performance"),
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
                title="Products",
                range=[-0.5, K - 0.5],
                tickmode='array',
                tickvals=tick_idx,
                ticktext=tick_labels,
                zeroline=False,
                showgrid=False,  # Disable default grid (which cuts through 0, 1, 2)
            ),
            yaxis=dict(
                title="Reactants",
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
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(-0.5, K - 0.5)
    
    # Major ticks (Labels at integers 0, 1, 2...)
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    
    # Minor ticks (Grid lines at half-integers -0.5, 0.5...)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
    
    # Grid settings: Turn Major OFF, Turn Minor ON
    ax.grid(which="major", visible=False)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.3)
    
    # Hide minor tick marks themselves
    ax.tick_params(which="minor", length=0)

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

    for (ri, pi) in special_cells:
        rect = Rectangle(
            (pi - 0.5, ri - 0.5),
            1, 1,
            fill=True,
            facecolor="none",
            hatch="///",
            linewidth=1.0,
            edgecolor="black",
            linestyle='--'
        )
        ax.add_patch(rect)

    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

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



# --- better plotting of distance graphs --- 

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
import community.community_louvain as community_louvain # pip install python-louvain (optional)

def build_hamming_graph(X_bool: np.ndarray, perf=None, alpha=0.0, t: int = None, k: int = 5):
    """ 
    Builds a graph from boolean signatures.
    If perf and alpha > 0 are provided, edge weights (distances) are penalized 
    by the difference in performance.
    
    Distance_new(u,v) = Hamming(u,v) * (1 + alpha * |Perf(u) - Perf(v)|)
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
                        label_percentile=90):
    """
    Enhanced plotting function.
    - color_by: 'community', 'degree', 'count', 'value', or 'dual' (Value + Community Edge).
    """
    
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
    node_linewidths = 0.5
    is_dual_mode = False

    # 4. Apply Coloring Mode
    if color_by == "dual" and node_values is not None:
        # Face = Performance (Loss), Edge = Community
        node_colors = node_values
        
        # Use Greyscale for Performance to avoid clashing with Cluster Hue
        # Greys_r: Low Value (Good) = Dark, High Value (Bad) = Light
        cmap = plt.cm.viridis_r 
        show_colorbar = True
        is_dual_mode = True
        
        partition = get_partition(G)
        if partition:
            comm_ids = [partition[n] for n in G.nodes()]
            
            # Use a high-saturation spectral map for clusters to contrast with Grey
            edge_cmap = plt.cm.gist_rainbow
            num_comms = max(comm_ids) + 1 if comm_ids else 1
            
            # Map IDs to colors
            node_edge_colors = [edge_cmap(i / num_comms) for i in comm_ids]
            node_linewidths = 3.0 # Thicker borders for the cluster identity
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
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.8, ax=ax)
    
    # Node Visualization (Size scaling)
    node_sizes = np.log1p(counts) * 300 + 50
    
    if is_dual_mode:
        # --- Dual Mode: Two-step drawing for black separator ---
        
        # 1. Draw outer thick border (Cluster ID) with transparent face
        nx.draw_networkx_nodes(G, pos, 
                                node_size=node_sizes,
                                node_color="none", # Transparent face
                                edgecolors=node_edge_colors, 
                                linewidths=node_linewidths, # Thick cluster border
                                ax=ax)
        
        # 2. Draw inner face (Performance) with thin black border
        # Scale down size slightly to fit inside the outer border
        inner_sizes = node_sizes * 0.8
        sc = nx.draw_networkx_nodes(G, pos, 
                                    node_size=inner_sizes,
                                    node_color=node_colors, 
                                    cmap=cmap, 
                                    edgecolors="black", # The black separator line
                                    linewidths=0.5,     # Thin separator
                                    ax=ax)
    else:
        # --- Standard Mode: Single draw ---
        sc = nx.draw_networkx_nodes(G, pos, 
                                    node_size=node_sizes,
                                    node_color=node_colors, 
                                    cmap=cmap, 
                                    edgecolors=node_edge_colors, 
                                    linewidths=node_linewidths,
                                    ax=ax)
    
    if show_colorbar and cmap is not None:
        # sc is the handle to the inner, performance-colored nodes in dual mode
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        label = "Performance (Min Loss)" if color_by in ["value", "dual"] else "Frequency (Log)"
        cbar.set_label(label)

    # Labels for top nodes
    if counts is not None:
        threshold = np.percentile(counts, label_percentile) if len(counts) > 20 else 0
        labels = {n: str(counts[n]) for n in G.nodes() if counts[n] >= threshold}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black', font_weight='bold', ax=ax)

    ax.set_axis_off()
    ax.set_title(f"{title}\n({len(G.nodes)} Unique Topologies)", fontsize=14)
    
    return fig

def visualize_crn_diversity(crn_list, perf=None, k=5, t=None, 
                            layout_method="spring", label_percentile=90, 
                            alpha=0.0): # <--- NEW: Alpha parameter
    """ 
    Wrapper to process CRN list and plot. 
    Args:
        crn_list: List of IOCRN objects.
        perf: Optional list/array of performance scores corresponding to crn_list.
        k: Neighbors for k-NN graph.
        t: Radius for radius graph.
        alpha: Reward-weighting coefficient. 
               If > 0, edges between nodes with different performance are stretched.
               High alpha (e.g. 10.0) creates distinct clusters for high/low performance.
    """
    # 1. Extract Signatures
    try:
        crn_topologies = np.stack([crn.get_bool_signature() for crn in crn_list]).astype(bool)
    except AttributeError:
        print("Error: Objects in crn_list must implement .get_bool_signature()")
        return

    # 2. Deduplicate and Aggregate Performance
    unique_sigs, inverse_indices, counts = np.unique(crn_topologies, axis=0, return_inverse=True, return_counts=True)
    
    print(f"Found {len(unique_sigs)} unique topologies from {len(crn_list)} samples.")

    max_perf_per_node = None
    color_mode = "community" # Default

    if perf is not None:
        perf = np.asarray(perf)
        if len(perf) != len(crn_list):
            print("Warning: perf length does not match crn_list length. Ignoring performance coloring.")
        else:
            max_perf_per_node = np.full(len(unique_sigs), np.inf)
            np.minimum.at(max_perf_per_node, inverse_indices, perf)
            color_mode = "dual" 
            print("Performance data aggregated (Min Loss per topology).")

    # 3. Build Graph (Passing alpha and aggregated performance)
    G_matrix = build_hamming_graph(unique_sigs, perf=max_perf_per_node, alpha=alpha, t=t, k=k)
    
    # 4. Plot
    fig = plot_topology_graph(G_matrix, counts, 
                              node_values=max_perf_per_node, 
                              color_by=color_mode, 
                              layout_method=layout_method,
                              title="CRN Topological Diversity",
                              label_percentile=label_percentile)
    plt.show()


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
    Simulates (or retrieves cached) and plots an ensemble of trajectories for the top-performing CRNs.
    Colors trajectories by their loss value and highlights the global best.
    Arranges scenarios in a square-ish grid layout. Distinct species use different line styles.

    Args:
        all_iocrns: List of IOCRN objects.
        ic: Initial Condition object.
        u_list: List of input scenarios.
        time_horizon: Time array for simulation (used if re-simulation is needed).
        n_top: Number of top CRNs to plot.
        cmap_name: Colormap for the ensemble lines.
        highlight_best: If True, plots the #1 CRN in thick black lines.
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