"""
PHASE 7: Visualization Dashboard
==================================
Comprehensive matplotlib-based dashboard for the flood prediction system.
Generates:
  1. City overview panel (land use, elevation, drain health)
  2. Simulation timeline (5yr vs 10yr comparison)
  3. Interactive-style zone risk snapshot at any given day
  4. Maintenance priority + flood classification maps
  5. Weight learning summary
  6. Year-over-year flood trend comparison
  7. Single-zone deep dive report

Run: python phase7_visualization.py
     python phase7_visualization.py --day 3000 --zone 612
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import argparse
import os

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    """Load all computed datasets."""
    print("[Phase 7] Loading data...")
    data = {}

    files = {
        "city"        : "city_zones.csv",
        "sim_5yr"     : "simulation_5yr.csv",
        "sim_10yr"    : "simulation_10yr.csv",
        "profiles_10" : "zone_profiles_10yr.csv",
        "maint"       : "maintenance_priority.csv",
        "weight_evo"  : "weight_evolution.csv",
        "weight_conv" : "weight_convergence.csv",
        "ml_preds"    : "flood_predictions_ml.csv",
        "annual_10"   : "annual_summary_10yr.csv",
    }

    for key, filename in files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found (run earlier phases first)")
            data[key] = None

    return data


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 1: CITY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_city_overview(data: dict):
    city_df = data["city"]
    if city_df is None:
        return

    grid_n = int(np.sqrt(len(city_df)))
    LU_NAMES = sorted(city_df["land_use"].unique())
    lu_map   = {k: i for i, k in enumerate(LU_NAMES)}

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("City Overview Dashboard", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Land use
    ax1 = fig.add_subplot(gs[0, 0])
    lu_grid = city_df["land_use"].map(lu_map).values.reshape(grid_n, grid_n)
    cmap1   = plt.cm.get_cmap("Set2", len(LU_NAMES))
    im1     = ax1.imshow(lu_grid, cmap=cmap1, vmin=0, vmax=len(LU_NAMES)-1, origin="lower")
    ax1.set_title("Land-Use Types", fontweight="bold")
    patches = [mpatches.Patch(color=cmap1(i), label=lu) for i, lu in enumerate(LU_NAMES)]
    ax1.legend(handles=patches, loc="lower right", fontsize=6, framealpha=0.8)

    # 2. Elevation
    ax2 = fig.add_subplot(gs[0, 1])
    elev_grid = city_df["elevation_m"].values.reshape(grid_n, grid_n)
    im2 = ax2.imshow(elev_grid, cmap="terrain", origin="lower")
    ax2.set_title("Elevation (m)", fontweight="bold")
    plt.colorbar(im2, ax=ax2)

    # 3. Initial drain capacity
    ax3 = fig.add_subplot(gs[0, 2])
    cap_grid = city_df["drain_capacity"].values.reshape(grid_n, grid_n)
    im3 = ax3.imshow(cap_grid, cmap="Blues", origin="lower")
    ax3.set_title("Initial Drain Capacity (mm/hr)", fontweight="bold")
    plt.colorbar(im3, ax=ax3)

    # 4. Infrastructure health
    ax4 = fig.add_subplot(gs[1, 0])
    health_grid = city_df["infra_health_score"].values.reshape(grid_n, grid_n)
    im4 = ax4.imshow(health_grid, cmap="RdYlGn", vmin=30, vmax=100, origin="lower")
    ax4.set_title("Infrastructure Health Score", fontweight="bold")
    plt.colorbar(im4, ax=ax4)

    # 5. Drain age
    ax5 = fig.add_subplot(gs[1, 1])
    age_grid = city_df["drain_age_yrs"].values.reshape(grid_n, grid_n)
    im5 = ax5.imshow(age_grid, cmap="hot_r", origin="lower")
    ax5.set_title("Drain Age (years)", fontweight="bold")
    plt.colorbar(im5, ax=ax5)

    # 6. Runoff coefficient
    ax6 = fig.add_subplot(gs[1, 2])
    rc_grid = city_df["runoff_coeff"].values.reshape(grid_n, grid_n)
    im6 = ax6.imshow(rc_grid, cmap="Oranges", origin="lower")
    ax6.set_title("Runoff Coefficient", fontweight="bold")
    plt.colorbar(im6, ax=ax6)

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_xlabel("Grid Col"); ax.set_ylabel("Grid Row")

    out = os.path.join(OUTPUT_DIR, "dashboard_01_city_overview.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Phase 7] Dashboard 1 saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 2: SIMULATION TIMELINE
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_simulation_timeline(data: dict):
    sim_5yr  = data["sim_5yr"]
    sim_10yr = data["sim_10yr"]
    if sim_5yr is None or sim_10yr is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=False)
    fig.suptitle("Simulation Timeline — 5yr vs 10yr",
                 fontsize=15, fontweight="bold")

    def daily_avg(sim):
        return sim.groupby("day").agg(
            rainfall        = ("rainfall_mm",       "mean"),
            flood_pct       = ("flood_event",       "mean"),
            avg_degradation = ("degradation_factor","mean"),
            avg_drift       = ("drift_memory",      "mean"),
        ).reset_index()

    d5  = daily_avg(sim_5yr)
    d10 = daily_avg(sim_10yr)

    metrics = [
        ("rainfall",        "Avg Daily Rainfall (mm)",           "#1565c0"),
        ("flood_pct",       "% Zones Flooded",                   "#c62828"),
        ("avg_degradation", "Avg Degradation Factor",            "#6a1b9a"),
        ("avg_drift",       "Avg Drift Memory",                  "#e65100"),
    ]

    for ax, (col, title, color) in zip(axes, metrics):
        ax.plot(d10["day"], d10[col], color=color, linewidth=0.7, alpha=0.9, label="10yr")
        ax.plot(d5["day"],  d5[col],  color=color, linewidth=1.0, alpha=0.5,
                linestyle="--", label="5yr")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        # Year markers
        n_yrs = d10["day"].max() // 365 + 1
        for yr in range(1, n_yrs + 1):
            ax.axvline(yr * 365, color="red", alpha=0.15, linewidth=0.8)
            ax.text(yr * 365 - 100, ax.get_ylim()[1] * 0.95,
                    f"Y{yr}", fontsize=6, color="red", alpha=0.7)

    axes[-1].set_xlabel("Day")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dashboard_02_simulation_timeline.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Phase 7] Dashboard 2 saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 3: DAY SNAPSHOT (Risk State on a Specific Day)
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_day_snapshot(data: dict, day: int = 3000):
    sim_10yr = data["sim_10yr"]
    city_df  = data["city"]
    if sim_10yr is None or city_df is None:
        return

    day = min(day, sim_10yr["day"].max())
    snapshot = sim_10yr[sim_10yr["day"] == day].merge(
        city_df[["zone_id","grid_row","grid_col"]], on="zone_id")

    grid_n = int(np.sqrt(len(city_df)))
    year   = day // 365 + 1
    doy    = day % 365

    def to_grid(col):
        return snapshot.sort_values(["grid_row","grid_col"])[col].values.reshape(grid_n, grid_n)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Day {day} Snapshot  |  Year {year}, Day-of-Year {doy}",
                 fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

    panels = [
        ("flood_event",        "Flood Event (1=flooded)",      "Reds",    None, None),
        ("load_ratio",         "Load Ratio",                   "OrRd",    0,    3),
        ("degradation_factor", "Degradation Factor",           "YlOrRd",  0,    0.75),
        ("drift_memory",       "Drift Memory",                 "PuRd",    0,    0.5),
        ("adaptive_thresh",    "Adaptive Threshold",           "YlGn_r",  0.8,  1.4),
        ("rainfall_mm",        "Today's Rainfall (mm)",        "Blues",   0,    300),
    ]

    for idx, (col, title, cmap, vmin, vmax) in enumerate(panels):
        ax  = fig.add_subplot(gs[idx // 3, idx % 3])
        kw  = {"origin": "lower"}
        if vmin is not None:
            kw["vmin"] = vmin; kw["vmax"] = vmax
        im  = ax.imshow(to_grid(col), cmap=cmap, **kw)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Col"); ax.set_ylabel("Row")
        plt.colorbar(im, ax=ax)

    out = os.path.join(OUTPUT_DIR, f"dashboard_03_day_{day}_snapshot.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Phase 7] Dashboard 3 (day {day}) saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 4: RISK & MAINTENANCE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_risk_and_maintenance(data: dict):
    maint    = data["maint"]
    profiles = data["profiles_10"]
    city_df  = data["city"]
    if any(x is None for x in [maint, profiles, city_df]):
        return

    grid_n = int(np.sqrt(len(city_df)))

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Risk & Maintenance Dashboard — 10yr Summary",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    def make_grid(df, col):
        return df.sort_values(["grid_row","grid_col"])[col].values.reshape(grid_n, grid_n)

    # 1. Total flood days
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(make_grid(profiles, "total_flood_days"),
                     cmap="hot_r", origin="lower")
    ax1.set_title("Total Flood Days (10yr)", fontweight="bold")
    plt.colorbar(im1, ax=ax1, label="days")

    # 2. Flood rate
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(make_grid(profiles, "flood_rate") * 100,
                     cmap="Reds", origin="lower", vmin=0, vmax=30)
    ax2.set_title("Flood Rate (%)", fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="%")

    # 3. Flood acceleration (late - early)
    ax3 = fig.add_subplot(gs[0, 2])
    accel_grid = make_grid(profiles, "flood_acceleration")
    im3 = ax3.imshow(accel_grid, cmap="RdBu_r", origin="lower",
                     vmin=-50, vmax=50)
    ax3.set_title("Flood Acceleration\n(Late 5yr − Early 5yr days)", fontweight="bold")
    plt.colorbar(im3, ax=ax3)

    # 4. Maintenance priority score
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(make_grid(maint, "maintenance_priority_score"),
                     cmap="YlOrRd", origin="lower", vmin=0, vmax=1)
    ax4.set_title("Maintenance Priority Score", fontweight="bold")
    plt.colorbar(im4, ax=ax4)

    # 5. Priority tier
    ax5 = fig.add_subplot(gs[1, 1])
    tier_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    cmap5    = plt.matplotlib.colors.ListedColormap(
        ["#4caf50","#ffeb3b","#ff9800","#f44336"])
    tier_grid = maint.sort_values(["grid_row","grid_col"])[
        "priority_tier"].map(tier_map).values.reshape(grid_n, grid_n)
    im5 = ax5.imshow(tier_grid, cmap=cmap5, origin="lower", vmin=0, vmax=3)
    ax5.set_title("Maintenance Priority Tier", fontweight="bold")
    cbar5 = plt.colorbar(im5, ax=ax5, ticks=[0,1,2,3])
    cbar5.ax.set_yticklabels(["LOW","MEDIUM","HIGH","CRITICAL"])

    # 6. Flood classification
    ax6 = fig.add_subplot(gs[1, 2])
    cls_map  = {"SAFE":0,"MODERATE":1,"ACUTE":2,"CHRONIC":3}
    cls_cmap = plt.matplotlib.colors.ListedColormap(
        ["#4caf50","#ffeb3b","#ff9800","#f44336"])
    cls_grid = profiles.sort_values(["grid_row","grid_col"])[
        "flood_classification"].map(cls_map).values.reshape(grid_n, grid_n)
    im6 = ax6.imshow(cls_grid, cmap=cls_cmap, origin="lower", vmin=0, vmax=3)
    ax6.set_title("Flood Risk Classification", fontweight="bold")
    cbar6 = plt.colorbar(im6, ax=ax6, ticks=[0,1,2,3])
    cbar6.ax.set_yticklabels(["SAFE","MODERATE","ACUTE","CHRONIC"])

    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.set_xlabel("Col"); ax.set_ylabel("Row")

    out = os.path.join(OUTPUT_DIR, "dashboard_04_risk_maintenance.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Phase 7] Dashboard 4 saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 5: YEAR-OVER-YEAR TREND
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_yoy_trends(data: dict):
    annual = data["annual_10"]
    city_df = data["city"]
    if annual is None or city_df is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Year-Over-Year Trends (10yr Simulation)",
                 fontsize=14, fontweight="bold")

    # Merge land use
    annual = annual.merge(city_df[["zone_id","land_use"]], on="zone_id")

    # 1. City-wide avg flood days per year
    city_annual = annual.groupby("year")["flood_days"].mean()
    axes[0,0].bar(city_annual.index, city_annual.values, color="#c62828", alpha=0.8)
    axes[0,0].set_title("Avg Flood Days per Zone per Year"); axes[0,0].set_xlabel("Year")
    axes[0,0].set_ylabel("Flood Days"); axes[0,0].grid(True, alpha=0.3, axis="y")

    # 2. Flood days by land-use per year
    lu_annual = annual.groupby(["year","land_use"])["flood_days"].mean().unstack()
    for col in lu_annual.columns:
        axes[0,1].plot(lu_annual.index, lu_annual[col], marker="o", markersize=4, label=col)
    axes[0,1].set_title("Avg Flood Days by Land-Use per Year")
    axes[0,1].set_xlabel("Year"); axes[0,1].set_ylabel("Flood Days")
    axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.3)

    # 3. Avg degradation per year
    deg_annual = annual.groupby("year")["avg_degradation"].mean()
    axes[1,0].plot(deg_annual.index, deg_annual.values, color="#6a1b9a",
                   marker="o", markersize=5)
    axes[1,0].fill_between(deg_annual.index, 0, deg_annual.values,
                            alpha=0.2, color="#6a1b9a")
    axes[1,0].set_title("City-Wide Avg Degradation per Year")
    axes[1,0].set_xlabel("Year"); axes[1,0].set_ylabel("Degradation Factor")
    axes[1,0].grid(True, alpha=0.3)

    # 4. Distribution of flood days (violin-style histogram per year)
    years = sorted(annual["year"].unique())
    positions = years
    data_per_year = [annual[annual["year"]==y]["flood_days"].values for y in years]
    axes[1,1].violinplot(data_per_year, positions=positions, showmedians=True)
    axes[1,1].set_title("Distribution of Zone Flood Days per Year")
    axes[1,1].set_xlabel("Year"); axes[1,1].set_ylabel("Flood Days")
    axes[1,1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "dashboard_05_yoy_trends.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Phase 7] Dashboard 5 saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD 6: SINGLE ZONE DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────

def dashboard_zone_deep_dive(data: dict, zone_id: int = None):
    sim_10yr = data["sim_10yr"]
    city_df  = data["city"]
    maint    = data["maint"]
    if sim_10yr is None or city_df is None:
        return

    # Auto-select a CRITICAL zone if none provided
    if zone_id is None and maint is not None:
        critical = maint[maint["priority_tier"] == "CRITICAL"]
        zone_id  = int(critical["zone_id"].iloc[len(critical)//2]) if len(critical) > 0 else 612
    elif zone_id is None:
        zone_id = 612

    zdata = sim_10yr[sim_10yr["zone_id"] == zone_id].sort_values("day")
    zmeta = city_df[city_df["zone_id"] == zone_id].iloc[0]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Zone {zone_id} Deep Dive  |  "
        f"Land-Use: {zmeta['land_use']}  |  "
        f"Drain: {zmeta['drain_material']} ({zmeta['drain_age_yrs']}yr old)  |  "
        f"Base Capacity: {zmeta['drain_capacity']:.1f} mm/hr",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.3)

    days = zdata["day"].values

    def add_year_lines(ax):
        for yr in range(1, 11):
            ax.axvline(yr * 365, color="red", alpha=0.2, linewidth=0.7)

    # 1. Rainfall + flood events
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(days, zdata["rainfall_mm"], color="#90caf9", alpha=0.6, width=1, label="Rainfall")
    flood_days = zdata[zdata["flood_event"] == 1]["day"]
    ax1.scatter(flood_days, [zdata["rainfall_mm"].max()*1.05]*len(flood_days),
                color="red", s=3, alpha=0.7, label="Flood Event")
    ax1.set_title("Daily Rainfall & Flood Events"); ax1.set_ylabel("mm/day")
    ax1.legend(fontsize=8); add_year_lines(ax1)

    # 2. Degradation & Drift Memory
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(days, zdata["degradation_factor"], color="#6a1b9a", linewidth=0.8,
             label="Degradation")
    ax2.plot(days, zdata["drift_memory"],        color="#e65100", linewidth=0.8,
             linestyle="--", label="Drift Memory")
    ax2.set_title("Degradation Factor & Drift Memory"); ax2.set_ylabel("Value")
    ax2.legend(fontsize=8); ax2.set_ylim(0, 0.8); add_year_lines(ax2)

    # 3. Load Ratio vs Adaptive Threshold
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(days, zdata["load_ratio"],      color="#1565c0", linewidth=0.7, label="Load Ratio")
    ax3.plot(days, zdata["adaptive_thresh"], color="#c62828", linewidth=0.9,
             linestyle="--", label="Adaptive Threshold")
    ax3.fill_between(days,
        zdata["load_ratio"], zdata["adaptive_thresh"],
        where=zdata["load_ratio"] > zdata["adaptive_thresh"],
        alpha=0.3, color="red", label="Flood Zone")
    ax3.set_title("Load Ratio vs Adaptive Threshold"); ax3.set_ylabel("Ratio")
    ax3.legend(fontsize=8); add_year_lines(ax3)

    # 4. Drift components (d1, d2, d3)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(days, zdata["d1_hydraulic"],   color="#e53935", linewidth=0.6, label="d1 Hydraulic")
    ax4.plot(days, zdata["d2_stress"],      color="#1e88e5", linewidth=0.6, label="d2 Stress")
    ax4.plot(days, zdata["d3_efficiency"],  color="#43a047", linewidth=0.6, label="d3 Efficiency")
    ax4.set_title("Drift Components d1, d2, d3"); ax4.set_ylabel("Value")
    ax4.legend(fontsize=8); add_year_lines(ax4)

    # 5. Weight evolution (w1, w2, w3)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(days, zdata["w1"], color="#e53935", linewidth=0.8, label="w1")
    ax5.plot(days, zdata["w2"], color="#1e88e5", linewidth=0.8, label="w2")
    ax5.plot(days, zdata["w3"], color="#43a047", linewidth=0.8, label="w3")
    ax5.axhline(1/3, color="gray", linestyle="--", alpha=0.5, linewidth=0.7, label="1/3 baseline")
    ax5.set_title("Self-Learned Weights (w1, w2, w3)"); ax5.set_ylabel("Weight")
    ax5.legend(fontsize=8); ax5.set_ylim(0, 0.8); add_year_lines(ax5)

    for ax in [ax2, ax3, ax4, ax5]:
        ax.set_xlabel("Day")

    out = os.path.join(OUTPUT_DIR, f"dashboard_06_zone_{zone_id}_deep_dive.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[Phase 7] Dashboard 6 (zone {zone_id}) saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ALERT SYSTEM — Daily Zone Risk Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_flood_alert_report(data: dict, day: int = 3000):
    """
    Prints a flood alert report for a given day,
    listing zones at risk with severity.
    """
    sim_10yr = data["sim_10yr"]
    city_df  = data["city"]
    if sim_10yr is None or city_df is None:
        return

    day = min(day, sim_10yr["day"].max())
    snapshot = sim_10yr[sim_10yr["day"] == day].merge(
        city_df[["zone_id","land_use","drain_capacity","grid_row","grid_col"]],
        on="zone_id"
    )

    flooded  = snapshot[snapshot["flood_event"] == 1].copy()
    flooded["severity"] = pd.cut(
        flooded["load_ratio"],
        bins=[0, 1.3, 1.6, 2.0, 999],
        labels=["MILD","MODERATE","SEVERE","CRITICAL"]
    )

    year = day // 365 + 1
    doy  = day % 365

    print("\n" + "="*60)
    print(f"  FLOOD ALERT REPORT — Day {day} (Year {year}, Day-of-Year {doy})")
    print("="*60)
    print(f"  Total zones:   {len(snapshot)}")
    print(f"  Flooded zones: {len(flooded)} ({len(flooded)/len(snapshot)*100:.1f}%)")
    if len(flooded) > 0:
        print(f"\n  Severity breakdown:")
        print(flooded["severity"].value_counts().to_string())
        print(f"\n  Top 10 highest risk zones:")
        top10 = flooded.nlargest(10, "load_ratio")[
            ["zone_id","land_use","load_ratio","degradation_factor","drift_memory","severity"]]
        print(top10.to_string(index=False))
    print("="*60 + "\n")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, f"alert_report_day_{day}.txt")
    with open(report_path, "w") as f:
        f.write(f"FLOOD ALERT REPORT — Day {day} (Year {year}, Day-of-Year {doy})\n")
        f.write(f"Total zones: {len(snapshot)}\n")
        f.write(f"Flooded zones: {len(flooded)} ({len(flooded)/len(snapshot)*100:.1f}%)\n\n")
        if len(flooded) > 0:
            f.write(flooded.sort_values("load_ratio", ascending=False).to_string())
    print(f"[Phase 7] Alert report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(snapshot_day: int = 3000, zone_id: int = None):
    data = load_all_data()

    print("\n[Phase 7] Generating all dashboards...")
    dashboard_city_overview(data)
    dashboard_simulation_timeline(data)
    dashboard_day_snapshot(data, day=snapshot_day)
    dashboard_risk_and_maintenance(data)
    dashboard_yoy_trends(data)
    dashboard_zone_deep_dive(data, zone_id=zone_id)
    generate_flood_alert_report(data, day=snapshot_day)

    print("\n[Phase 7] ✓ All dashboards generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Dashboard Generator")
    parser.add_argument("--day",  type=int, default=3000,
                        help="Day to snapshot (0-3649 for 10yr sim)")
    parser.add_argument("--zone", type=int, default=None,
                        help="Zone ID for deep-dive analysis")
    args = parser.parse_args()
    run(snapshot_day=args.day, zone_id=args.zone)