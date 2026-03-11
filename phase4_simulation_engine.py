"""
PHASE 4: Simulation Engine — Deep Analysis Module
===================================================
Loads simulation output from Phase 3 and performs:
  - Degradation trajectory analysis per zone
  - Drift memory accumulation curves
  - Threshold evolution over time
  - Zone clustering by infrastructure behaviour
  - Chronic vs temporary flood classification

This module analyses the already-generated simulation data
and provides deeper modelling insights.

Output: zone_profiles.parquet, chronic_zones.parquet, phase4_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

OUTPUT_DIR  = "data"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ZONE PROFILE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_zone_profiles(sim_df: pd.DataFrame,
                         city_df: pd.DataFrame,
                         label: str = "10yr") -> pd.DataFrame:
    """
    Computes a rich per-zone summary across the entire simulation.
    Each zone gets a profile vector capturing its behaviour over time.
    """
    print(f"\n[Phase 4] Building zone profiles from {label} simulation...")

    n_years = sim_df["day"].max() // 365 + 1
    sim_df  = sim_df.copy()
    sim_df["year"] = (sim_df["day"] // 365) + 1

    # ── Per-zone full-period statistics ──────────────────────────────────
    zone_stats = sim_df.groupby("zone_id").agg(
        total_flood_days       = ("flood_event",        "sum"),
        flood_rate             = ("flood_event",        "mean"),
        avg_degradation        = ("degradation_factor", "mean"),
        final_degradation      = ("degradation_factor", "last"),
        max_degradation        = ("degradation_factor", "max"),
        avg_drift_memory       = ("drift_memory",       "mean"),
        final_drift_memory     = ("drift_memory",       "last"),
        max_drift_memory       = ("drift_memory",       "max"),
        avg_load_ratio         = ("load_ratio",         "mean"),
        max_load_ratio         = ("load_ratio",         "max"),
        avg_d1                 = ("d1_hydraulic",       "mean"),
        avg_d2                 = ("d2_stress",          "mean"),
        avg_d3                 = ("d3_efficiency",      "mean"),
        final_w1               = ("w1",                 "last"),
        final_w2               = ("w2",                 "last"),
        final_w3               = ("w3",                 "last"),
        avg_adaptive_thresh    = ("adaptive_thresh",    "mean"),
        final_adaptive_thresh  = ("adaptive_thresh",    "last"),
    ).reset_index()

    # ── Year-over-year flood trend (slope of annual flood count) ─────────
    annual = sim_df.groupby(["zone_id", "year"])["flood_event"].sum().reset_index()
    annual.columns = ["zone_id", "year", "annual_floods"]

    def flood_trend(group):
        if len(group) < 2:
            return 0.0
        return np.polyfit(group["year"], group["annual_floods"], 1)[0]

    trend = annual.groupby("zone_id").apply(flood_trend).reset_index()
    trend.columns = ["zone_id", "flood_trend_slope"]
    zone_stats = zone_stats.merge(trend, on="zone_id")

    # ── Early vs Late flood comparison ───────────────────────────────────
    half = n_years // 2
    early = sim_df[sim_df["year"] <= half].groupby("zone_id")["flood_event"].sum()
    late  = sim_df[sim_df["year"] > half].groupby("zone_id")["flood_event"].sum()
    flood_accel = (late - early).reset_index()
    flood_accel.columns = ["zone_id", "flood_acceleration"]
    zone_stats = zone_stats.merge(flood_accel, on="zone_id")

    # ── Merge city attributes ─────────────────────────────────────────────
    zone_stats = zone_stats.merge(
        city_df[["zone_id", "land_use", "drain_material", "drain_age_yrs",
                 "drain_capacity", "infra_health_score", "elevation_m",
                 "grid_row", "grid_col"]],
        on="zone_id"
    )

    out = os.path.join(OUTPUT_DIR, f"zone_profiles_{label}.csv")
    zone_stats.to_csv(out, index=False)
    print(f"[Phase 4] Zone profiles saved → {out}  ({len(zone_stats)} zones)")
    return zone_stats


# ─────────────────────────────────────────────────────────────────────────────
# CHRONIC vs TEMPORARY FLOOD CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_flood_type(sim_df: pd.DataFrame,
                         zone_profiles: pd.DataFrame,
                         label: str = "10yr") -> pd.DataFrame:
    """
    Classifies each zone's flood pattern:
    - CHRONIC: consistently high drift memory + frequent floods (degradation-driven)
    - ACUTE:   sudden flood spikes during extreme rainfall only
    - SAFE:    rarely floods
    """
    print(f"\n[Phase 4] Classifying flood types ({label})...")

    # Thresholds for classification
    CHRONIC_FLOOD_RATE  = 0.08   # >8% days flooded = chronic risk
    ACUTE_MAX_LOAD      = 1.8    # Max load ratio spike = acute
    SAFE_FLOOD_RATE     = 0.02   # <2% days flooded = safe

    profiles = zone_profiles.copy()

    conditions = []
    for _, row in profiles.iterrows():
        if row["flood_rate"] > CHRONIC_FLOOD_RATE and row["final_drift_memory"] > 0.15:
            conditions.append("CHRONIC")
        elif row["flood_rate"] > SAFE_FLOOD_RATE and row["max_load_ratio"] > ACUTE_MAX_LOAD:
            conditions.append("ACUTE")
        elif row["flood_rate"] <= SAFE_FLOOD_RATE:
            conditions.append("SAFE")
        else:
            conditions.append("MODERATE")

    profiles["flood_classification"] = conditions

    dist = pd.Series(conditions).value_counts()
    print(f"[Phase 4] Flood classification distribution:")
    for cls, cnt in dist.items():
        print(f"  {cls:10s}: {cnt:5d} zones ({cnt/len(profiles)*100:.1f}%)")

    out = os.path.join(OUTPUT_DIR, f"chronic_zones_{label}.csv")
    profiles.to_csv(out, index=False)
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# ZONE CLUSTERING (Behavioural Segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_zones(zone_profiles: pd.DataFrame,
                   label: str = "10yr",
                   n_clusters: int = 5) -> pd.DataFrame:
    """
    K-Means clustering on zone behaviour features.
    Groups zones by their infrastructure + flood behaviour fingerprint.
    """
    print(f"\n[Phase 4] Clustering zones into {n_clusters} behavioural groups...")

    feature_cols = [
        "avg_degradation", "final_drift_memory", "flood_rate",
        "avg_d1", "avg_d2", "avg_d3",
        "final_w1", "final_w2", "final_w3",
        "flood_trend_slope", "flood_acceleration"
    ]

    X = zone_profiles[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    zone_profiles = zone_profiles.copy()
    zone_profiles["cluster"] = km.fit_predict(X_scaled)

    # Describe each cluster
    print(f"[Phase 4] Cluster profiles:")
    cluster_desc = zone_profiles.groupby("cluster").agg(
        n_zones       = ("zone_id",       "count"),
        avg_flood_rate= ("flood_rate",    "mean"),
        avg_deg       = ("avg_degradation","mean"),
        dominant_lu   = ("land_use",      lambda x: x.mode()[0]),
    )
    print(cluster_desc.to_string())

    # PCA for 2D cluster visualization
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    zone_profiles["pca1"] = X_pca[:, 0]
    zone_profiles["pca2"] = X_pca[:, 1]

    out = os.path.join(OUTPUT_DIR, f"zone_profiles_{label}.csv")
    zone_profiles.to_csv(out, index=False)
    return zone_profiles


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def visualize_degradation_trajectories(sim_df: pd.DataFrame,
                                        zone_profiles: pd.DataFrame,
                                        label: str):
    """Degradation curves for sample zones from each cluster."""
    clusters = zone_profiles["cluster"].unique()
    n_sample = 5   # Zones per cluster to plot

    fig, axes = plt.subplots(len(clusters), 1,
                              figsize=(14, 3 * len(clusters)), sharex=True)
    if len(clusters) == 1:
        axes = [axes]

    fig.suptitle(f"Degradation Trajectories by Cluster — {label}",
                 fontsize=13, fontweight="bold")

    for ax, cl in zip(axes, sorted(clusters)):
        sample_zones = zone_profiles[zone_profiles["cluster"] == cl]["zone_id"].values[:n_sample]
        for zid in sample_zones:
            zdata = sim_df[sim_df["zone_id"] == zid][["day", "degradation_factor"]]
            ax.plot(zdata["day"], zdata["degradation_factor"],
                    linewidth=0.8, alpha=0.7)
        ax.set_title(f"Cluster {cl}  (n={len(zone_profiles[zone_profiles['cluster']==cl])} zones)")
        ax.set_ylabel("Degradation Factor")
        ax.set_ylim(0, 0.8)
        ax.grid(True, alpha=0.3)
        # Year markers
        n_yrs = sim_df["day"].max() // 365 + 1
        for yr in range(1, n_yrs + 1):
            ax.axvline(yr * 365, color="red", alpha=0.2, linewidth=0.7)

    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"phase4_{label}_degradation_trajectories.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 4] Degradation trajectories saved → {out}")


def visualize_zone_clusters(zone_profiles: pd.DataFrame, label: str):
    """3-panel cluster visualization."""
    grid_n = int(np.sqrt(zone_profiles["zone_id"].max() + 1))

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3)
    fig.suptitle(f"Phase 4 — Zone Behavioural Clusters ({label})",
                 fontsize=13, fontweight="bold")

    colors  = plt.cm.tab10(np.linspace(0, 1, zone_profiles["cluster"].nunique()))
    cmap    = plt.cm.get_cmap("tab10", zone_profiles["cluster"].nunique())

    # ── Panel 1: Spatial cluster map ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cluster_grid = zone_profiles.sort_values(["grid_row", "grid_col"])[
        "cluster"].values.reshape(grid_n, grid_n)
    im1 = ax1.imshow(cluster_grid, cmap=cmap, origin="lower",
                     vmin=0, vmax=zone_profiles["cluster"].nunique()-1)
    ax1.set_title("Spatial Cluster Distribution")
    ax1.set_xlabel("Grid Col"); ax1.set_ylabel("Grid Row")
    plt.colorbar(im1, ax=ax1, label="Cluster")

    # ── Panel 2: PCA scatter ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for cl in sorted(zone_profiles["cluster"].unique()):
        sub = zone_profiles[zone_profiles["cluster"] == cl]
        ax2.scatter(sub["pca1"], sub["pca2"], label=f"Cl {cl}",
                    alpha=0.4, s=5, color=colors[cl])
    ax2.set_title("PCA — Zone Behaviour Space")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.legend(markerscale=3, fontsize=8)

    # ── Panel 3: Flood rate vs degradation by cluster ─────────────────────
    ax3 = fig.add_subplot(gs[2])
    for cl in sorted(zone_profiles["cluster"].unique()):
        sub = zone_profiles[zone_profiles["cluster"] == cl]
        ax3.scatter(sub["avg_degradation"], sub["flood_rate"] * 100,
                    label=f"Cl {cl}", alpha=0.4, s=8, color=colors[cl])
    ax3.set_title("Flood Rate vs Avg Degradation")
    ax3.set_xlabel("Avg Degradation Factor")
    ax3.set_ylabel("Flood Rate (%)")
    ax3.legend(markerscale=2, fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"phase4_{label}_clusters.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 4] Cluster visualization saved → {out}")


def visualize_flood_classification_map(zone_profiles: pd.DataFrame, label: str):
    """Spatial map of chronic / acute / safe zones."""
    grid_n = int(np.sqrt(zone_profiles["zone_id"].max() + 1))

    cls_map   = {"SAFE": 0, "MODERATE": 1, "ACUTE": 2, "CHRONIC": 3}
    cls_colors = ["#4caf50", "#ffeb3b", "#ff9800", "#f44336"]
    cmap = plt.matplotlib.colors.ListedColormap(cls_colors)

    cls_grid = zone_profiles.sort_values(["grid_row", "grid_col"])[
        "flood_classification"].map(cls_map).values.reshape(grid_n, grid_n)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cls_grid, cmap=cmap, vmin=0, vmax=3, origin="lower")
    ax.set_title(f"Flood Risk Classification — {label}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Grid Column"); ax.set_ylabel("Grid Row")

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["SAFE", "MODERATE", "ACUTE", "CHRONIC"])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"phase4_{label}_flood_classification.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 4] Classification map saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(sim_10yr: pd.DataFrame = None,
        city_df: pd.DataFrame  = None):

    if city_df is None:
        city_df  = pd.read_csv(os.path.join(OUTPUT_DIR, "city_zones.csv"))
    if sim_10yr is None:
        sim_10yr = pd.read_csv(os.path.join(OUTPUT_DIR, "simulation_10yr.csv"))

    # 10-year profiles
    profiles_10 = build_zone_profiles(sim_10yr, city_df, label="10yr")
    profiles_10 = classify_flood_type(sim_10yr, profiles_10, label="10yr")
    profiles_10 = cluster_zones(profiles_10, label="10yr", n_clusters=5)

    visualize_degradation_trajectories(sim_10yr, profiles_10, label="10yr")
    visualize_zone_clusters(profiles_10, label="10yr")
    visualize_flood_classification_map(profiles_10, label="10yr")

    # Also do 5yr profiles
    try:
        sim_5yr  = pd.read_csv(os.path.join(OUTPUT_DIR, "simulation_5yr.csv"))
        profiles_5 = build_zone_profiles(sim_5yr, city_df, label="5yr")
        profiles_5 = classify_flood_type(sim_5yr, profiles_5, label="5yr")
        profiles_5 = cluster_zones(profiles_5, label="5yr", n_clusters=5)
        visualize_flood_classification_map(profiles_5, label="5yr")
    except FileNotFoundError:
        print("[Phase 4] 5yr simulation not found, skipping.")

    print("\n[Phase 4] ✓ Engine analysis complete.")
    return profiles_10


if __name__ == "__main__":
    run()