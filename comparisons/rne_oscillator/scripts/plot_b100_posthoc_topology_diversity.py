#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import cloudpickle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "comparisons/rne_oscillator/figures"
RAW_ROOT = ROOT / "comparisons/rne_oscillator/data/raw"

METHOD = "rl4crn_rne_oscillator_trace_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20"
ANALYSIS_CSV = FIG_DIR / "rne_oscillator_analysis_8rxn_stop_risksched085to09_lr1e4_entropy005_s4_c1_rnetol_d256_depth3_p20_n400.csv"
OUT_PREFIX = FIG_DIR / "rne_oscillator_b100_posthoc_topology_diversity"


def seed_label(seed) -> str:
    s = str(seed)
    if s.startswith("seed"):
        return s
    return f"seed{int(float(s)):03d}"


def signature_hash(sig: np.ndarray) -> str:
    packed = np.packbits(np.asarray(sig, dtype=np.uint8))
    return hashlib.sha1(packed.tobytes()).hexdigest()[:12]


def hamming_graph(unique_signatures: np.ndarray, radius: int):
    n, d = unique_signatures.shape
    nn = NearestNeighbors(metric="hamming", radius=radius / d, n_jobs=-1)
    nn.fit(unique_signatures.astype(np.uint8))
    graph = nn.radius_neighbors_graph(mode="distance")
    graph = graph.multiply(d).astype(float)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def nearest_topology_graph(unique_signatures: np.ndarray, n_neighbors: int):
    X = unique_signatures.astype(np.uint8)
    n, d = X.shape
    nn = NearestNeighbors(metric="hamming", n_neighbors=min(n_neighbors + 1, n), n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            G.add_edge(i, int(j), weight=float(dist * d))
    return G


def main() -> None:
    df = pd.read_csv(ANALYSIS_CSV)
    df = df[df["rne_posthoc_success"].fillna(False).astype(bool)].copy()
    df["seed_label"] = df["seed"].map(seed_label)

    crns = []
    signatures = []
    for row in df.itertuples(index=False):
        path = RAW_ROOT / METHOD / row.seed_label / "best_network.pkl"
        with path.open("rb") as f:
            crn = cloudpickle.load(f)
        crns.append(crn)
        signatures.append(np.asarray(crn.get_bool_signature()).astype(bool))

    X = np.stack(signatures).astype(bool)
    unique_sigs, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    n_unique = len(unique_sigs)

    min_loss = np.full(n_unique, np.inf)
    seeds_by_topology: list[list[str]] = [[] for _ in range(n_unique)]
    for idx, topo_idx in enumerate(inverse):
        min_loss[topo_idx] = min(min_loss[topo_idx], float(df.iloc[idx]["best_loss"]))
        seeds_by_topology[topo_idx].append(str(df.iloc[idx]["seed_label"]))

    summary = pd.DataFrame(
        {
            "topology_id": np.arange(n_unique),
            "topology_hash": [signature_hash(sig) for sig in unique_sigs],
            "count": counts,
            "min_best_loss": min_loss,
            "seeds": [";".join(seeds) for seeds in seeds_by_topology],
        }
    ).sort_values(["count", "min_best_loss"], ascending=[False, True])

    out_csv = OUT_PREFIX.with_suffix(".csv")
    summary.to_csv(out_csv, index=False)

    layout_radius = 5
    display_radius = 5
    nearest_neighbors = 3
    G_sparse = hamming_graph(unique_sigs, layout_radius)
    G = nx.from_scipy_sparse_array(G_sparse)
    G_nearest = nearest_topology_graph(unique_sigs, nearest_neighbors)
    G.add_edges_from(G_nearest.edges(data=True))
    inv_dist = {(u, v): 1.0 / (d["weight"] + 1e-9) for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, inv_dist, "attraction")
    pos = nx.spring_layout(G, weight="attraction", seed=7, iterations=250, k=1.1 / np.sqrt(max(n_unique, 1)))
    coords = np.asarray(list(pos.values()), dtype=float)
    coord_min = coords.min(axis=0)
    coord_span = np.ptp(coords, axis=0)
    coord_span[coord_span == 0] = 1.0
    pos = {
        node: tuple(0.06 + 0.88 * (np.asarray(point) - coord_min) / coord_span)
        for node, point in pos.items()
    }

    G_display = G_nearest.copy()
    G_display.add_edges_from(
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if float(d["weight"]) <= display_radius
    )
    G_display.add_nodes_from(G.nodes)

    node_sizes = 55 + 145 * np.log1p(counts)

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    display_weights = np.asarray([d["weight"] for _, _, d in G_display.edges(data=True)], dtype=float)
    if len(display_weights):
        lo, hi = float(display_weights.min()), float(display_weights.max())
        for u, v, d in G_display.edges(data=True):
            scaled = (float(d["weight"]) - lo) / (hi - lo + 1e-9)
            alpha = 0.62 - 0.47 * scaled
            width = 1.15 - 0.55 * scaled
            nx.draw_networkx_edges(
                G_display,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                edge_color=[(0.16, 0.16, 0.16, alpha)],
                width=width,
            )
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=min_loss,
        cmap="viridis_r",
        edgecolors="white",
        linewidths=0.45,
    )

    repeated = {i: str(int(counts[i])) for i in range(n_unique) if counts[i] > 1}
    if repeated:
        nx.draw_networkx_labels(G, pos, labels=repeated, font_size=7, font_color="black", ax=ax)

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Best loss per topology")

    for count in sorted(set(counts)):
        if count > 1:
            ax.scatter([], [], s=55 + 145 * np.log1p(count), color="#777777", edgecolor="white", label=f"{count} CRNs")
    if np.max(counts) == 1:
        ax.scatter([], [], s=55 + 145 * np.log1p(1), color="#777777", edgecolor="white", label="1 CRN")
    size_legend = ax.legend(title="Topology count", loc="lower left", frameon=True, fontsize=7, title_fontsize=7)
    ax.add_artist(size_legend)
    distance_handles = [
        Line2D([0], [0], color=(0.16, 0.16, 0.16, 0.62), lw=1.15, label="near"),
        Line2D([0], [0], color=(0.16, 0.16, 0.16, 0.15), lw=0.60, label="far"),
    ]
    ax.legend(
        handles=distance_handles,
        title="Topology distance",
        loc="lower right",
        frameon=True,
        fontsize=7,
        title_fontsize=7,
    )

    ax.set_title("B=100 posthoc-valid topology diversity")
    ax.text(
        0.01,
        0.99,
        f"{n_unique} topologies",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.set_axis_off()

    fig.savefig(OUT_PREFIX.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_PREFIX.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Posthoc-valid CRNs: {len(crns)}")
    print(f"Unique topologies: {n_unique}")
    print(f"Repeated topology nodes: {int(np.sum(counts > 1))}")
    print(f"Maximum topology count: {int(np.max(counts))}")
    print(f"CSV: {out_csv}")
    print(f"PNG: {OUT_PREFIX.with_suffix('.png')}")
    print(f"PDF: {OUT_PREFIX.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
