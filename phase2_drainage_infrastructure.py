"""
PHASE 2: Drainage Infrastructure Setup
=========================================
Builds the drain network graph using NetworkX.
Connects micro-zones to drain nodes.
Assigns infrastructure attributes per zone.
Output: drain_network.parquet, drain_edges.parquet, phase2_drain_network.png
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 42
OUTPUT_DIR  = "data"
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Drain network hierarchy
# PRIMARY: major trunk drains along main roads (every 10 zones)
# SECONDARY: neighbourhood drains (every 5 zones)
# TERTIARY: block-level drains (every zone connects to nearest secondary)

PRIMARY_SPACING   = 10   # Every 10 grid cells
SECONDARY_SPACING = 5    # Every 5 grid cells


def build_drain_network(city_df: pd.DataFrame) -> nx.Graph:
    """
    Builds a hierarchical drain network graph.
    Nodes = drain junctions
    Edges = pipes between junctions
    """
    grid_n = int(np.sqrt(len(city_df)))
    G = nx.Graph()

    print(f"[Phase 2] Building drain network for {grid_n}x{grid_n} grid...")

    # ── Step 1: Create junction nodes at grid intersections ──────────────
    for i in range(0, grid_n + 1, SECONDARY_SPACING):
        for j in range(0, grid_n + 1, SECONDARY_SPACING):
            is_primary = (i % PRIMARY_SPACING == 0) and (j % PRIMARY_SPACING == 0)
            node_id    = f"J_{i}_{j}"
            x_m = j * 200
            y_m = i * 200
            G.add_node(node_id,
                       x_m        = x_m,
                       y_m        = y_m,
                       node_type  = "primary" if is_primary else "secondary",
                       capacity   = 500 if is_primary else 250)

    # ── Step 2: Connect junctions with pipes (edges) ─────────────────────
    junction_list = list(G.nodes())
    for node in junction_list:
        x = G.nodes[node]["x_m"]
        y = G.nodes[node]["y_m"]
        # Connect to right and up neighbours
        right = f"J_{int(y//200)}_{int(x//200) + SECONDARY_SPACING}"
        up    = f"J_{int(y//200) + SECONDARY_SPACING}_{int(x//200)}"
        for neighbour in [right, up]:
            if neighbour in G.nodes():
                pipe_type = "primary" if (
                    G.nodes[node]["node_type"] == "primary" and
                    G.nodes[neighbour]["node_type"] == "primary"
                ) else "secondary"
                G.add_edge(node, neighbour,
                           pipe_type = pipe_type,
                           length_m  = SECONDARY_SPACING * 200,
                           capacity  = 400 if pipe_type == "primary" else 180)

    print(f"[Phase 2] Drain graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G


def assign_zones_to_drain_nodes(city_df: pd.DataFrame,
                                 G: nx.Graph) -> pd.DataFrame:
    """
    Each micro-zone is assigned its nearest drain junction node.
    Also assigns infrastructure health score (0–100).
    """
    node_positions = {
        n: (G.nodes[n]["x_m"], G.nodes[n]["y_m"]) for n in G.nodes()
    }
    node_list = list(node_positions.keys())
    node_coords = np.array(list(node_positions.values()))  # (M, 2)

    zone_x = city_df["x_m"].values[:, None]   # (N, 1)
    zone_y = city_df["y_m"].values[:, None]

    node_x = node_coords[:, 0][None, :]       # (1, M)
    node_y = node_coords[:, 1][None, :]

    distances = np.sqrt((zone_x - node_x)**2 + (zone_y - node_y)**2)
    nearest_idx = distances.argmin(axis=1)

    city_df = city_df.copy()
    city_df["nearest_drain_node"] = [node_list[i] for i in nearest_idx]
    city_df["dist_to_drain_m"]    = distances.min(axis=1).round(1)

    # ── Infrastructure Health Score ───────────────────────────────────────
    # Based on age: newer = healthier. 100 = perfect, 0 = failed.
    city_df["infra_health_score"] = (
        100 - city_df["drain_age_yrs"] * 1.8 + np.random.uniform(-5, 5, len(city_df))
    ).clip(30, 100).round(1)

    # ── Blockage Probability (initial) ───────────────────────────────────
    # Higher for old drains, industrial and dense residential zones
    blockage_base = city_df["drain_age_yrs"] / 35 * 0.3
    lu_factor = city_df["land_use"].map({
        "residential_dense" : 0.10,
        "residential_light" : 0.05,
        "commercial"        : 0.08,
        "industrial"        : 0.15,
        "green_space"       : 0.02,
        "mixed_use"         : 0.07,
    })
    city_df["blockage_prob_initial"] = (blockage_base + lu_factor).clip(0, 0.5).round(4)

    # ── Ideal Flow Efficiency ─────────────────────────────────────────────
    city_df["ideal_flow_efficiency"] = (city_df["infra_health_score"] / 100 * 0.85).round(3)

    print(f"[Phase 2] Zone-to-drain assignment complete.")
    print(f"  Avg distance to drain node: {city_df['dist_to_drain_m'].mean():.0f}m")
    print(f"  Avg infrastructure health: {city_df['infra_health_score'].mean():.1f}/100")
    return city_df


def save_network(city_df: pd.DataFrame, G: nx.Graph):
    # Save updated city zones
    city_df.to_csv(os.path.join(OUTPUT_DIR, "city_zones.csv"), index=False)

    # Save edges as DataFrame
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "from_node" : u,
            "to_node"   : v,
            "pipe_type" : data["pipe_type"],
            "length_m"  : data["length_m"],
            "capacity"  : data["capacity"],
        })
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(os.path.join(OUTPUT_DIR, "drain_edges.csv"), index=False)
    print(f"[Phase 2] Network saved → data/city_zones.csv + data/drain_edges.csv")


def visualize_network(city_df: pd.DataFrame, G: nx.Graph):
    """Plot drain network overlaid on land-use map."""
    grid_n = int(np.sqrt(len(city_df)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 2 — Drainage Infrastructure", fontsize=14, fontweight="bold")

    # ── Left: Drain network graph ─────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Drain Network (Primary=thick, Secondary=thin)")
    ax.set_facecolor("#e8f4f8")

    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
        x1, y1 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
        lw    = 2.5 if data["pipe_type"] == "primary" else 0.8
        color = "#1a237e" if data["pipe_type"] == "primary" else "#64b5f6"
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=0.8)

    for node in G.nodes():
        x, y = G.nodes[node]["x_m"], G.nodes[node]["y_m"]
        nt   = G.nodes[node]["node_type"]
        ax.scatter(x, y,
                   c    = "#d32f2f" if nt == "primary" else "#ff8a65",
                   s    = 30 if nt == "primary" else 10,
                   zorder=5)

    ax.set_xlabel("X (metres)"); ax.set_ylabel("Y (metres)")
    ax.set_xlim(0, grid_n * 200); ax.set_ylim(0, grid_n * 200)

    # ── Right: Infrastructure Health Score map ────────────────────────────
    health_grid = city_df["infra_health_score"].values.reshape(grid_n, grid_n)
    im = axes[1].imshow(health_grid, cmap="RdYlGn", vmin=30, vmax=100, origin="lower")
    axes[1].set_title("Infrastructure Health Score (100=perfect)")
    axes[1].set_xlabel("Grid Column"); axes[1].set_ylabel("Grid Row")
    plt.colorbar(im, ax=axes[1], label="Health Score")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase2_drain_network.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 2] Visualization saved → {out}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def run(city_df: pd.DataFrame = None):
    if city_df is None:
        city_df = pd.read_csv(os.path.join(OUTPUT_DIR, "city_zones.csv"))

    G        = build_drain_network(city_df)
    city_df  = assign_zones_to_drain_nodes(city_df, G)
    save_network(city_df, G)
    visualize_network(city_df, G)
    return city_df, G


if __name__ == "__main__":
    run()