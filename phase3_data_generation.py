"""
PHASE 3: Generating 5-Year and 10-Year Parameter Datasets
============================================================
FIXED VERSION — Realistic degradation parameters so that:
  - ~55% zones stay SAFE
  - ~25% zones are MODERATE risk
  - ~12% zones are HIGH risk
  - ~8%  zones are CRITICAL

Key fixes from previous version:
  - DEG_CAP     : 0.75 → 0.30  (max 30% capacity loss, not 75%)
  - SPIKE_PROB  : 0.05 → 0.008 (spike every ~125 days, not every 20)
  - BASE_THRESH : 1.20 → 1.55  (harder to trigger a flood)
  - ALPHA       : 0.30 → 0.18  (threshold adjusts more slowly)
  - BETA        : 0.85 → 0.80  (drift memory less sticky)
  - deg_rate    : scaled 0.12x (10x lower baseline)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 42
OUTPUT_DIR  = "data"
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# RAINFALL MODEL
# ─────────────────────────────────────────────────────────────────────────────

def generate_rainfall(n_days: int, n_zones: int) -> np.ndarray:
    """
    Seasonal rainfall per zone per day (mm/day).
    Monsoon peak around day 180, dry near day 0/365.
    Extreme events: ~5 per year, affecting only 30% of zones each time.
    """
    days     = np.arange(n_days)
    seasonal = 80 + 35 * np.sin(2 * np.pi * (days % 365) / 365 - np.pi / 2)
    seasonal = seasonal[:, None]                            # (n_days, 1)

    zone_offset = np.random.uniform(-8, 8, (1, n_zones))   # spatial variation
    noise       = np.random.normal(0, 10, (n_days, n_zones))

    # Extreme events: ~5/year, localised
    n_extreme      = int(n_days / 365 * 5)
    extreme_days   = np.random.choice(n_days, n_extreme, replace=False)
    extreme_matrix = np.zeros((n_days, n_zones))
    for d in extreme_days:
        affected = np.random.random(n_zones) < 0.30
        extreme_matrix[d, affected] = np.random.uniform(60, 150, int(affected.sum()))

    rainfall = np.clip(seasonal + zone_offset + noise + extreme_matrix, 3, 280)
    return rainfall.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(city_df: pd.DataFrame,
                   n_days:  int,
                   label:   str = "sim") -> pd.DataFrame:

    n_zones = len(city_df)
    print(f"\n[Phase 3] Starting {label}: {n_days} days × {n_zones} zones")

    # Static zone parameters
    drain_cap    = city_df["drain_capacity"].values.astype(np.float32)
    runoff_coeff = city_df["runoff_coeff"].values.astype(np.float32)
    ideal_eff    = city_df["ideal_flow_efficiency"].values.astype(np.float32)

    # Scale down raw degradation rates → realistic 10-yr drift
    deg_rate = city_df["degradation_rate"].values.astype(np.float32) * 0.12

    # Initial states
    soil_sat   = city_df["soil_saturation"].values.astype(np.float32).copy()
    deg_factor = np.zeros(n_zones, dtype=np.float32)
    drift_mem  = np.zeros(n_zones, dtype=np.float32)
    w1 = np.full(n_zones, 1/3, dtype=np.float32)
    w2 = np.full(n_zones, 1/3, dtype=np.float32)
    w3 = np.full(n_zones, 1/3, dtype=np.float32)

    # ── TUNED HYPERPARAMETERS (calibrated from actual load ratio distribution) ─
    # Load ratio percentiles: P90=0.75, P92=0.79, P95=0.86, P97=0.93
    # BASE_THRESH=0.80 → ~8% avg zone-days flood; high-runoff zones flood much more
    # → produces realistic spread: ~40% SAFE, ~30% MOD, ~18% HIGH, ~12% CRITICAL
    BETA        = 0.80
    ALPHA       = 0.08
    BASE_THRESH = 0.85   # P95 of load ratio → ~5% avg; CRITICAL zones = ~8-10%
    ETA         = 0.008  # Weight learning rate
    DEG_CAP     = 0.30   # Max 30% capacity loss over drain lifetime
    SPIKE_PROB  = 0.008  # ~0.8% chance per zone/day of blockage spike
    SPIKE_MAX   = 0.018  # Max single blockage magnitude

    print(f"  DEG_CAP={DEG_CAP} | BASE_THRESH={BASE_THRESH} | SPIKE_P={SPIKE_PROB}")

    # Generate full rainfall matrix upfront
    print(f"[Phase 3] Generating rainfall...")
    rainfall_matrix = generate_rainfall(n_days, n_zones)

    all_chunks = []
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")

    for day in range(n_days):
        R = rainfall_matrix[day]

        # 1. Soil saturation memory
        soil_sat = 0.70 * soil_sat + 0.30 * R

        # 2. Effective runoff
        eff_runoff = R * (0.5 + 0.5 * np.clip(soil_sat / 200, 0, 1))

        # 3. Expected discharge
        exp_discharge = eff_runoff * runoff_coeff

        # 4. Degradation
        daily_inc  = deg_rate * np.maximum(0.3, 1 + 0.4 * np.random.randn(n_zones))
        spike_mask = np.random.random(n_zones) < SPIKE_PROB
        spike_amt  = np.random.uniform(0, SPIKE_MAX, n_zones) * spike_mask
        deg_factor = np.clip(deg_factor + daily_inc + spike_amt, 0.0, DEG_CAP)

        # 5. Observed discharge
        obs_discharge = exp_discharge * (1.0 - deg_factor)

        # 6. Drift components
        safe_exp     = np.where(exp_discharge > 0, exp_discharge, 1e-6)
        d1           = np.clip((exp_discharge - obs_discharge) / safe_exp, 0, 1)
        load_ratio   = exp_discharge / drain_cap
        stress_ratio = obs_discharge / drain_cap
        d2           = np.abs(load_ratio - stress_ratio)
        safe_R       = np.where(R > 0, R, 1e-6)
        flow_eff     = obs_discharge / safe_R
        d3           = np.abs(ideal_eff - flow_eff)

        # 7. Composite Drift Index
        drift_index = w1 * d1 + w2 * d2 + w3 * d3

        # 8. Drift persistence memory
        drift_mem = BETA * drift_index + (1 - BETA) * drift_mem

        # 9. Adaptive threshold
        adaptive_thresh = BASE_THRESH - ALPHA * drift_mem

        # 10. Flood decision
        flood_event = (load_ratio > adaptive_thresh).astype(np.int8)

        # 11. Self-adaptive weight learning
        error_signal = flood_event.astype(np.float32) - load_ratio
        w1_new = np.clip(w1 + ETA * error_signal * d1, 0.05, 2.0)
        w2_new = np.clip(w2 + ETA * error_signal * d2, 0.05, 2.0)
        w3_new = np.clip(w3 + ETA * error_signal * d3, 0.05, 2.0)
        ws = w1_new + w2_new + w3_new
        w1 = w1_new / ws
        w2 = w2_new / ws
        w3 = w3_new / ws

        # Store
        day_df = pd.DataFrame({
            "day"               : day,
            "date"              : str(dates[day].date()),
            "zone_id"           : city_df["zone_id"].values,
            "rainfall_mm"       : R.round(2),
            "soil_saturation"   : soil_sat.round(2),
            "eff_runoff"        : eff_runoff.round(2),
            "exp_discharge"     : exp_discharge.round(2),
            "obs_discharge"     : obs_discharge.round(2),
            "degradation_factor": deg_factor.round(4),
            "d1_hydraulic"      : d1.round(4),
            "d2_stress"         : d2.round(4),
            "d3_efficiency"     : d3.round(4),
            "drift_index"       : drift_index.round(4),
            "drift_memory"      : drift_mem.round(4),
            "load_ratio"        : load_ratio.round(4),
            "adaptive_thresh"   : adaptive_thresh.round(4),
            "flood_event"       : flood_event,
            "w1"                : w1.round(4),
            "w2"                : w2.round(4),
            "w3"                : w3.round(4),
        })
        all_chunks.append(day_df)

        if (day + 1) % 365 == 0:
            yr = (day + 1) // 365
            fp = flood_event.mean() * 100
            print(f"  Year {yr:2d} | Avg deg: {deg_factor.mean():.3f} | "
                  f"Flood zones: {fp:.1f}% | Drift: {drift_mem.mean():.3f}")

    print(f"[Phase 3] Assembling {len(all_chunks)} day records...")
    result_df = pd.concat(all_chunks, ignore_index=True)

    out = os.path.join(OUTPUT_DIR, f"simulation_{label}.csv")
    result_df.to_csv(out, index=False)
    print(f"[Phase 3] Saved → {out} ({len(result_df):,} rows)")

    # Save final zone state for Streamlit forecasting
    final_state = city_df.copy()
    final_state["degradation_factor"] = deg_factor
    final_state["soil_saturation"]    = soil_sat
    final_state["drift_memory"]       = drift_mem
    final_state["drift_w1"]           = w1
    final_state["drift_w2"]           = w2
    final_state["drift_w3"]           = w3
    final_state.to_csv(
        os.path.join(OUTPUT_DIR, f"city_state_after_{label}.csv"), index=False)

    return result_df


def compute_annual_summary(sim_df: pd.DataFrame, label: str) -> pd.DataFrame:
    sim_df = sim_df.copy()
    sim_df["year"] = (sim_df["day"] // 365) + 1
    summary = sim_df.groupby(["zone_id","year"]).agg(
        flood_days       = ("flood_event",        "sum"),
        avg_degradation  = ("degradation_factor", "mean"),
        avg_drift_memory = ("drift_memory",        "mean"),
        max_load_ratio   = ("load_ratio",          "max"),
        avg_rainfall     = ("rainfall_mm",         "mean"),
    ).reset_index()
    out = os.path.join(OUTPUT_DIR, f"annual_summary_{label}.csv")
    summary.to_csv(out, index=False)
    print(f"[Phase 3] Annual summary → {out}")
    return summary


def visualize_and_check(sim_df: pd.DataFrame, city_df: pd.DataFrame, label: str):
    """
    Two charts:
    1. Simulation timeline (rainfall, flood%, degradation, drift)
    2. Risk distribution map — confirm NOT all red
    """
    daily_avg = sim_df.groupby("day").agg(
        rainfall        = ("rainfall_mm",        "mean"),
        flood_pct       = ("flood_event",        "mean"),
        avg_degradation = ("degradation_factor", "mean"),
        avg_drift       = ("drift_memory",        "mean"),
    ).reset_index()

    n_years = sim_df["day"].max() // 365 + 1
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Phase 3 — Simulation Overview ({label}) — REALISTIC DATA",
                 fontsize=14, fontweight="bold")

    panels = [
        (axes[0,0], "rainfall",        "Daily Rainfall (mm)",    "#1565c0"),
        (axes[0,1], "flood_pct",       "% Zones Flooded",        "#c62828"),
        (axes[1,0], "avg_degradation", "Avg Degradation Factor", "#6a1b9a"),
        (axes[1,1], "avg_drift",       "Avg Drift Memory",       "#e65100"),
    ]
    days = daily_avg["day"].values
    for ax, col, title, color in panels:
        ax.plot(days, daily_avg[col], color=color, linewidth=0.7, alpha=0.85)
        ax.set_title(title); ax.set_xlabel("Day")
        ax.grid(True, alpha=0.25)
        for yr in range(1, n_years + 1):
            ax.axvline(yr * 365, color="gray", alpha=0.3, linewidth=0.6)

    axes[0,1].axhline(5,  color="green",  linestyle="--", alpha=0.5, linewidth=0.8, label="Safe (<5%)")
    axes[0,1].axhline(15, color="orange", linestyle="--", alpha=0.5, linewidth=0.8, label="High (>15%)")
    axes[0,1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"phase3_{label}_overview.png"), dpi=120)
    plt.close()

    # Risk distribution check
    grid_n = int(np.sqrt(len(city_df)))
    zone_floods = sim_df.groupby("zone_id")["flood_event"].sum().reset_index()
    zone_floods.columns = ["zone_id","total_flood_days"]
    n_days = sim_df["day"].max() + 1
    zone_floods["flood_rate"] = zone_floods["total_flood_days"] / n_days
    zone_floods["risk"] = pd.cut(
        zone_floods["flood_rate"],
        bins=[-0.001, 0.03, 0.08, 0.15, 1.0],
        labels=["SAFE","MODERATE","HIGH","CRITICAL"]
    )

    dist = zone_floods["risk"].value_counts()
    print(f"\n[Phase 3] Risk distribution ({label}):")
    total = len(zone_floods)
    for r in ["SAFE","MODERATE","HIGH","CRITICAL"]:
        cnt = int(dist.get(r, 0))
        bar = "█" * int(cnt / total * 40)
        print(f"  {r:10s}: {cnt:5d} ({cnt/total*100:.1f}%)  {bar}")

    merged = city_df.merge(zone_floods, on="zone_id")
    risk_num = {"SAFE":0,"MODERATE":1,"HIGH":2,"CRITICAL":3}
    risk_grid = merged.sort_values(["grid_row","grid_col"])[
        "risk"].map(risk_num).values.reshape(grid_n, grid_n)

    cmap = matplotlib.colors.ListedColormap(["#4caf50","#ffeb3b","#ff9800","#f44336"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Risk Distribution — {label}", fontsize=13, fontweight="bold")

    im = axes[0].imshow(risk_grid, cmap=cmap, vmin=0, vmax=3, origin="lower")
    axes[0].set_title("Flood Risk Classification Map")
    cbar = plt.colorbar(im, ax=axes[0], ticks=[0,1,2,3])
    cbar.ax.set_yticklabels(["SAFE","MODERATE","HIGH","CRITICAL"])

    labels  = ["SAFE","MODERATE","HIGH","CRITICAL"]
    colors  = ["#4caf50","#ffeb3b","#ff9800","#f44336"]
    counts  = [int(dist.get(r,0)) for r in labels]
    axes[1].bar(labels, counts, color=colors, edgecolor="white", width=0.6)
    axes[1].set_title("Zone Count by Risk Level")
    axes[1].set_ylabel("Number of Zones")
    for i,(cnt,lbl) in enumerate(zip(counts,labels)):
        if cnt > 0:
            axes[1].text(i, cnt+2, f"{cnt/sum(counts)*100:.1f}%",
                         ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"phase3_{label}_risk_distribution.png"), dpi=120)
    plt.close()
    print(f"[Phase 3] Risk distribution chart saved.")


def run(city_df: pd.DataFrame = None):
    if city_df is None:
        city_df = pd.read_csv(os.path.join(OUTPUT_DIR, "city_zones.csv"))

    print(f"\n{'='*55}")
    print(f"  Phase 3: Realistic Simulation (FIXED PARAMETERS)")
    print(f"  Target: ~55% SAFE | ~25% MODERATE | ~12% HIGH | ~8% CRITICAL")
    print(f"{'='*55}")

    sim_5yr  = run_simulation(city_df, n_days=365*5,  label="5yr")
    compute_annual_summary(sim_5yr,  "5yr")
    visualize_and_check(sim_5yr,  city_df, "5yr")

    sim_10yr = run_simulation(city_df, n_days=365*10, label="10yr")
    compute_annual_summary(sim_10yr, "10yr")
    visualize_and_check(sim_10yr, city_df, "10yr")

    print("\n[Phase 3] ✓ Complete.")
    return sim_5yr, sim_10yr


if __name__ == "__main__":
    run()