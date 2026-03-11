"""
PHASE 1: Virtual City Construction
====================================
Creates a virtual city grid of micro-zones with spatial coordinates,
land-use types, elevation, and zone properties.
Each micro-zone is a 200m x 200m cell in a 10km x 10km city.
Output: city_zones.parquet — master city GeoDataFrame equivalent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CITY_SIZE_KM   = 10          # Total city width/height in km
ZONE_SIZE_M    = 200         # Each micro-zone is 200m x 200m
GRID_N         = CITY_SIZE_KM * 1000 // ZONE_SIZE_M   # 50 x 50 = 2500 zones
RANDOM_SEED    = 42
OUTPUT_DIR     = "data"

np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LAND-USE TYPES & RUNOFF COEFFICIENTS
# Higher coefficient = more rainfall becomes runoff
# ─────────────────────────────────────────────
LAND_USE_TYPES = {
    "residential_dense" : {"runoff_coeff": 0.80, "weight": 0.30},
    "residential_light" : {"runoff_coeff": 0.55, "weight": 0.20},
    "commercial"        : {"runoff_coeff": 0.90, "weight": 0.15},
    "industrial"        : {"runoff_coeff": 0.85, "weight": 0.10},
    "green_space"       : {"runoff_coeff": 0.20, "weight": 0.15},
    "mixed_use"         : {"runoff_coeff": 0.65, "weight": 0.10},
}

LAND_USE_NAMES   = list(LAND_USE_TYPES.keys())
LAND_USE_WEIGHTS = [v["weight"] for v in LAND_USE_TYPES.values()]
LAND_USE_RUNOFF  = {k: v["runoff_coeff"] for k, v in LAND_USE_TYPES.items()}

# ─────────────────────────────────────────────
# DRAINAGE MATERIAL TYPES
# Older/cheaper materials degrade faster
# ─────────────────────────────────────────────
DRAIN_MATERIALS = {
    "concrete"   : {"base_capacity": 120, "degradation_rate": 0.0003},
    "pvc"        : {"base_capacity": 100, "degradation_rate": 0.0002},
    "cast_iron"  : {"base_capacity": 140, "degradation_rate": 0.0004},
    "clay"       : {"base_capacity":  80, "degradation_rate": 0.0006},
    "hdpe"       : {"base_capacity": 110, "degradation_rate": 0.00015},
}

MATERIAL_NAMES   = list(DRAIN_MATERIALS.keys())
MATERIAL_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.15]


def generate_synthetic_elevation(grid_n: int) -> np.ndarray:
    """
    Generates a realistic synthetic elevation map.
    City centre is slightly elevated; edges slope downward.
    Added Perlin-like noise for natural variation.
    Returns a (grid_n x grid_n) 2D array of elevation in metres.
    """
    x = np.linspace(-1, 1, grid_n)
    y = np.linspace(-1, 1, grid_n)
    xx, yy = np.meshgrid(x, y)

    # Base terrain: gentle hill at city centre
    base_elevation = 30 * np.exp(-(xx**2 + yy**2) / 0.8)

    # Add multi-scale noise for natural look
    noise = (
        5  * np.sin(xx * 5) * np.cos(yy * 5) +
        2  * np.sin(xx * 12 + 0.5) * np.cos(yy * 11) +
        1  * np.random.randn(grid_n, grid_n)
    )

    elevation = base_elevation + noise + 10   # Minimum ~10m above sea level
    return elevation


def assign_land_use_spatially(grid_n: int) -> np.ndarray:
    """
    Assigns land-use with spatial logic:
    - City centre → commercial/mixed
    - Mid ring → residential dense
    - Outer ring → residential light / green space
    - Random industrial clusters
    """
    land_use_grid = np.empty((grid_n, grid_n), dtype=object)
    centre = grid_n // 2

    for i in range(grid_n):
        for j in range(grid_n):
            dist = np.sqrt((i - centre)**2 + (j - centre)**2)
            r = np.random.random()

            if dist < 8:                        # CBD core
                lu = "commercial" if r < 0.6 else "mixed_use"
            elif dist < 16:                     # Inner ring
                lu = "residential_dense" if r < 0.55 else (
                     "commercial" if r < 0.70 else "mixed_use")
            elif dist < 25:                     # Middle ring
                lu = "residential_dense" if r < 0.40 else (
                     "residential_light" if r < 0.70 else (
                     "industrial" if r < 0.85 else "green_space"))
            else:                               # Outer ring
                lu = "residential_light" if r < 0.45 else (
                     "green_space" if r < 0.70 else (
                     "industrial" if r < 0.80 else "residential_dense"))

            land_use_grid[i, j] = lu

    return land_use_grid


def build_city_dataframe(grid_n: int) -> pd.DataFrame:
    """
    Builds the master city DataFrame.
    Each row = one micro-zone.
    """
    print(f"[Phase 1] Building virtual city: {grid_n}x{grid_n} = {grid_n**2} micro-zones")

    elevation_grid = generate_synthetic_elevation(grid_n)
    land_use_grid  = assign_land_use_spatially(grid_n)

    records = []
    zone_id = 0

    for i in range(grid_n):
        for j in range(grid_n):
            lu       = land_use_grid[i, j]
            material = np.random.choice(MATERIAL_NAMES, p=MATERIAL_WEIGHTS)
            age_yrs  = np.random.randint(2, 35)   # Drain age in years

            # Base drain capacity adjusted for age (older = lower capacity)
            base_cap   = DRAIN_MATERIALS[material]["base_capacity"]
            age_factor = max(0.60, 1.0 - age_yrs * 0.008)
            drain_cap  = round(base_cap * age_factor, 2)

            records.append({
                # Identity
                "zone_id"          : zone_id,
                "grid_row"         : i,
                "grid_col"         : j,

                # Spatial (metres from city origin)
                "x_m"              : j * ZONE_SIZE_M + ZONE_SIZE_M / 2,
                "y_m"              : i * ZONE_SIZE_M + ZONE_SIZE_M / 2,
                "elevation_m"      : round(elevation_grid[i, j], 2),

                # Land use
                "land_use"         : lu,
                "runoff_coeff"     : LAND_USE_RUNOFF[lu],

                # Drainage infrastructure
                "drain_material"   : material,
                "drain_age_yrs"    : age_yrs,
                "drain_capacity"   : drain_cap,   # mm/hr the drain can handle
                "degradation_rate" : DRAIN_MATERIALS[material]["degradation_rate"],

                # Initial states (all zones start healthy)
                "degradation_factor"  : 0.0,
                "soil_saturation"     : np.random.uniform(10, 30),  # Initial soil moisture
                "drift_memory"        : 0.0,
                "drift_w1"            : 1/3,
                "drift_w2"            : 1/3,
                "drift_w3"            : 1/3,
            })

            zone_id += 1

    df = pd.DataFrame(records)
    print(f"[Phase 1] City DataFrame created: {df.shape[0]} zones, {df.shape[1]} attributes")
    return df


def save_city(df: pd.DataFrame):
    path = os.path.join(OUTPUT_DIR, "city_zones.csv")
    df.to_csv(path, index=False)
    print(f"[Phase 1] City saved → {path}")


def visualize_city(df: pd.DataFrame):
    """Generate 3 city overview plots."""
    grid_n = int(np.sqrt(len(df)))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Virtual City — Micro-Zone Overview", fontsize=14, fontweight="bold")

    # --- Plot 1: Land Use ---
    lu_map = {k: i for i, k in enumerate(LAND_USE_NAMES)}
    lu_grid = df["land_use"].map(lu_map).values.reshape(grid_n, grid_n)
    cmap1 = plt.cm.get_cmap("Set2", len(LAND_USE_NAMES))
    im1 = axes[0].imshow(lu_grid, cmap=cmap1, vmin=0, vmax=len(LAND_USE_NAMES)-1)
    axes[0].set_title("Land Use Types")
    axes[0].set_xlabel("Grid Column"); axes[0].set_ylabel("Grid Row")
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(len(LAND_USE_NAMES)))
    cbar1.ax.set_yticklabels(LAND_USE_NAMES, fontsize=7)

    # --- Plot 2: Elevation ---
    elev_grid = df["elevation_m"].values.reshape(grid_n, grid_n)
    im2 = axes[1].imshow(elev_grid, cmap="terrain")
    axes[1].set_title("Elevation (metres)")
    axes[1].set_xlabel("Grid Column")
    plt.colorbar(im2, ax=axes[1], label="m")

    # --- Plot 3: Drain Capacity ---
    cap_grid = df["drain_capacity"].values.reshape(grid_n, grid_n)
    im3 = axes[2].imshow(cap_grid, cmap="Blues")
    axes[2].set_title("Initial Drain Capacity (mm/hr)")
    axes[2].set_xlabel("Grid Column")
    plt.colorbar(im3, ax=axes[2], label="mm/hr")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase1_city_overview.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 1] Visualization saved → {out}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def run():
    df = build_city_dataframe(GRID_N)
    save_city(df)
    visualize_city(df)

    # Summary stats
    print("\n[Phase 1] Land-use distribution:")
    print(df["land_use"].value_counts().to_string())
    print(f"\n[Phase 1] Elevation range: {df['elevation_m'].min():.1f}m "
          f"– {df['elevation_m'].max():.1f}m")
    print(f"[Phase 1] Drain capacity range: {df['drain_capacity'].min():.1f} "
          f"– {df['drain_capacity'].max():.1f} mm/hr")
    return df


if __name__ == "__main__":
    run()