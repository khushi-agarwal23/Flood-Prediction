"""
PHASE 6: Self-Learning Weight Adjustment Analysis
===================================================
Analyses how drift component weights (w1, w2, w3) evolved
per zone over the 10-year simulation.

Reveals:
  - Which drift component dominates in each land-use type
  - How weights shift over time (early vs mature state)
  - Zone-specific adaptation patterns
  - Convergence of weights across clusters

Output: weight_evolution.parquet, phase6_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

OUTPUT_DIR  = "data"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT EVOLUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def extract_weight_evolution(sim_df: pd.DataFrame,
                              city_df: pd.DataFrame,
                              n_sample_zones: int = 100) -> pd.DataFrame:
    """
    Extracts weight trajectory for a sample of zones over all years.
    w1 = Hydraulic Deviation weight
    w2 = Stress Mismatch weight
    w3 = Flow Efficiency weight
    """
    print(f"\n[Phase 6] Extracting weight evolution for {n_sample_zones} sample zones...")

    # Sample zones across all land-use types
    sample_zones = []
    for lu in city_df["land_use"].unique():
        lu_zones = city_df[city_df["land_use"] == lu]["zone_id"].values
        n_take   = min(n_sample_zones // len(city_df["land_use"].unique()), len(lu_zones))
        chosen   = np.random.choice(lu_zones, n_take, replace=False)
        sample_zones.extend(chosen.tolist())

    # Get weight data at monthly intervals (every 30 days)
    monthly_days = sim_df["day"].unique()[::30]
    weight_data  = sim_df[
        (sim_df["zone_id"].isin(sample_zones)) &
        (sim_df["day"].isin(monthly_days))
    ][["day", "zone_id", "w1", "w2", "w3", "drift_index", "flood_event"]].copy()

    # Merge land use
    weight_data = weight_data.merge(
        city_df[["zone_id", "land_use", "drain_material", "drain_age_yrs"]],
        on="zone_id"
    )
    weight_data["year"] = (weight_data["day"] / 365).round(2)

    out = os.path.join(OUTPUT_DIR, "weight_evolution.csv")
    weight_data.to_csv(out, index=False)
    print(f"[Phase 6] Weight evolution saved → {out}  ({len(weight_data)} records)")
    return weight_data


def analyse_weight_convergence(sim_df: pd.DataFrame,
                                city_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each zone, computes:
    - Initial vs final weight comparison
    - Dominant drift component (which weight became largest)
    - Weight stability (variance over last 2 years)
    """
    print(f"\n[Phase 6] Analysing weight convergence per zone...")

    # Initial state (day 30 to allow warmup)
    initial = sim_df[sim_df["day"] == 30][["zone_id","w1","w2","w3"]].copy()
    initial.columns = ["zone_id","w1_initial","w2_initial","w3_initial"]

    # Final state (last day)
    last_day = sim_df["day"].max()
    final    = sim_df[sim_df["day"] == last_day][["zone_id","w1","w2","w3"]].copy()
    final.columns = ["zone_id","w1_final","w2_final","w3_final"]

    # Weight variance over last 2 years (stability)
    last_2yr = sim_df[sim_df["day"] >= last_day - 730]
    stability = last_2yr.groupby("zone_id").agg(
        w1_var = ("w1", "var"),
        w2_var = ("w2", "var"),
        w3_var = ("w3", "var"),
    ).reset_index()

    # Combine
    conv_df = initial.merge(final, on="zone_id").merge(stability, on="zone_id")
    conv_df = conv_df.merge(
        city_df[["zone_id","land_use","drain_material","grid_row","grid_col"]],
        on="zone_id"
    )

    # Dominant component
    final_cols = ["w1_final","w2_final","w3_final"]
    comp_names = ["Hydraulic(d1)","StressMismatch(d2)","FlowEfficiency(d3)"]
    dominant_idx = conv_df[final_cols].values.argmax(axis=1)
    conv_df["dominant_component"] = [comp_names[i] for i in dominant_idx]

    # Weight shift (how much each weight changed)
    conv_df["w1_shift"] = (conv_df["w1_final"] - conv_df["w1_initial"]).round(4)
    conv_df["w2_shift"] = (conv_df["w2_final"] - conv_df["w2_initial"]).round(4)
    conv_df["w3_shift"] = (conv_df["w3_final"] - conv_df["w3_initial"]).round(4)

    print(f"\n[Phase 6] Dominant component by land-use:")
    dom_table = conv_df.groupby(["land_use","dominant_component"]).size().unstack(fill_value=0)
    print(dom_table.to_string())

    out = os.path.join(OUTPUT_DIR, "weight_convergence.csv")
    conv_df.to_csv(out, index=False)
    return conv_df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def visualize_weight_trajectories(weight_evo: pd.DataFrame):
    """
    Shows how w1, w2, w3 evolve over time per land-use type.
    """
    land_uses = weight_evo["land_use"].unique()
    fig, axes = plt.subplots(len(land_uses), 3,
                              figsize=(16, 3.5 * len(land_uses)), sharex=True)

    fig.suptitle("Phase 6 — Drift Weight Evolution by Land-Use",
                 fontsize=13, fontweight="bold")

    w_colors = {"w1": "#e53935", "w2": "#1e88e5", "w3": "#43a047"}
    w_labels = {
        "w1": "w1 (Hydraulic Deviation)",
        "w2": "w2 (Stress Mismatch)",
        "w3": "w3 (Flow Efficiency)"
    }

    for row_idx, lu in enumerate(sorted(land_uses)):
        lu_data = weight_evo[weight_evo["land_use"] == lu]

        for col_idx, (w, color) in enumerate(w_colors.items()):
            ax = axes[row_idx, col_idx] if len(land_uses) > 1 else axes[col_idx]

            # Plot individual zone trajectories (light)
            for zid in lu_data["zone_id"].unique()[:20]:
                zdata = lu_data[lu_data["zone_id"] == zid].sort_values("year")
                ax.plot(zdata["year"], zdata[w],
                        color=color, alpha=0.15, linewidth=0.8)

            # Plot mean trajectory (bold)
            mean_traj = lu_data.groupby("year")[w].mean()
            ax.plot(mean_traj.index, mean_traj.values,
                    color=color, linewidth=2.0, label=w_labels[w])

            ax.axhline(1/3, color="gray", linestyle="--", alpha=0.5,
                       linewidth=0.8, label="Initial (1/3)")
            ax.set_ylim(0, 0.8)
            ax.set_ylabel(w_labels[w], fontsize=8)
            ax.set_title(f"{lu}" if col_idx == 1 else "", fontsize=9)
            ax.grid(True, alpha=0.2)

    # X-axis labels on bottom row
    for col_idx in range(3):
        ax = axes[-1, col_idx] if len(land_uses) > 1 else axes[col_idx]
        ax.set_xlabel("Year")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase6_weight_trajectories.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 6] Weight trajectory plot saved → {out}")


def visualize_dominant_component_map(conv_df: pd.DataFrame):
    """Spatial map of which drift component dominates per zone."""
    grid_n = int(np.sqrt(conv_df["zone_id"].max() + 1))

    comp_map    = {
        "Hydraulic(d1)":    0,
        "StressMismatch(d2)": 1,
        "FlowEfficiency(d3)": 2
    }
    comp_colors = ["#e53935", "#1e88e5", "#43a047"]
    cmap        = plt.matplotlib.colors.ListedColormap(comp_colors)

    dom_grid = conv_df.sort_values(["grid_row","grid_col"])[
        "dominant_component"].map(comp_map).values.reshape(grid_n, grid_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 6 — Self-Learned Dominant Drift Component",
                 fontsize=13, fontweight="bold")

    im = axes[0].imshow(dom_grid, cmap=cmap, vmin=0, vmax=2, origin="lower")
    axes[0].set_title("Dominant Drift Component (Learned per Zone)")
    axes[0].set_xlabel("Grid Col"); axes[0].set_ylabel("Grid Row")
    cbar = plt.colorbar(im, ax=axes[0], ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["d1 Hydraulic", "d2 Stress", "d3 Flow Eff"])

    # Final weight triangle (ternary-like scatter)
    ax2 = axes[1]
    lu_colors_map = {
        "residential_dense" : "#e53935",
        "residential_light" : "#fb8c00",
        "commercial"        : "#8e24aa",
        "industrial"        : "#546e7a",
        "green_space"       : "#43a047",
        "mixed_use"         : "#1e88e5",
    }
    for lu, grp in conv_df.groupby("land_use"):
        ax2.scatter(grp["w1_final"], grp["w2_final"],
                    c=lu_colors_map.get(lu, "gray"),
                    label=lu, alpha=0.4, s=8)
    ax2.set_xlabel("Final w1 (Hydraulic)")
    ax2.set_ylabel("Final w2 (Stress Mismatch)")
    ax2.set_title("Final Weight Distribution by Land-Use")
    ax2.legend(markerscale=3, fontsize=8, loc="upper right")
    ax2.axhline(1/3, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)
    ax2.axvline(1/3, color="gray", linestyle="--", alpha=0.4, linewidth=0.7)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase6_dominant_component_map.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 6] Dominant component map saved → {out}")


def visualize_weight_shift_heatmap(conv_df: pd.DataFrame):
    """Heatmap of weight shifts (initial vs final) by land-use."""
    shift_summary = conv_df.groupby("land_use").agg(
        w1_shift_mean = ("w1_shift", "mean"),
        w2_shift_mean = ("w2_shift", "mean"),
        w3_shift_mean = ("w3_shift", "mean"),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase 6 — Average Weight Shift Over 10 Years by Land-Use",
                 fontsize=13, fontweight="bold")

    im = ax.imshow(shift_summary.values, cmap="RdBu_r",
                   vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Δw1 (Hydraulic)", "Δw2 (Stress)", "Δw3 (Flow Eff)"])
    ax.set_yticks(range(len(shift_summary.index)))
    ax.set_yticklabels(shift_summary.index)

    # Annotate cells
    for i in range(len(shift_summary.index)):
        for j in range(3):
            val = shift_summary.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(val) < 0.1 else "white")

    plt.colorbar(im, ax=ax, label="Weight Shift (Final − Initial)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase6_weight_shift_heatmap.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 6] Weight shift heatmap saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(sim_10yr: pd.DataFrame = None,
        city_df: pd.DataFrame  = None):

    if city_df is None:
        city_df  = pd.read_csv(os.path.join(OUTPUT_DIR, "city_zones.csv"))
    if sim_10yr is None:
        sim_10yr = pd.read_csv(os.path.join(OUTPUT_DIR, "simulation_10yr.csv"))

    weight_evo  = extract_weight_evolution(sim_10yr, city_df, n_sample_zones=120)
    conv_df     = analyse_weight_convergence(sim_10yr, city_df)

    visualize_weight_trajectories(weight_evo)
    visualize_dominant_component_map(conv_df)
    visualize_weight_shift_heatmap(conv_df)

    print("\n[Phase 6] ✓ Self-learning analysis complete.")
    return weight_evo, conv_df


if __name__ == "__main__":
    run()