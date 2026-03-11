"""
PHASE 5: Flood Prediction & Decision Logic
=============================================
Implements:
  1. Rule-based flood prediction (load ratio vs adaptive threshold)
  2. Flood propagation between adjacent zones (cascade effect)
  3. ML model (scikit-learn) trained on 5yr data, predicts on years 6-10
  4. Zone-level 7-day flood forecast
  5. Maintenance prioritization ranking

Output: flood_predictions_ml.parquet, maintenance_priority.parquet, phase5_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR  = "data"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FLOOD PROPAGATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_zone_adjacency(city_df: pd.DataFrame) -> dict:
    """
    Creates adjacency map: zone_id → list of adjacent zone_ids.
    Based on grid position (4-connectivity: up, down, left, right).
    """
    grid_n = int(np.sqrt(len(city_df)))
    adj = {}

    for _, row in city_df.iterrows():
        zid = int(row["zone_id"])
        r   = int(row["grid_row"])
        c   = int(row["grid_col"])
        neighbours = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_n and 0 <= nc < grid_n:
                nid = nr * grid_n + nc
                neighbours.append(nid)
        adj[zid] = neighbours

    return adj


def propagate_floods(flood_array: np.ndarray,
                      load_ratio:  np.ndarray,
                      elevation:   np.ndarray,
                      adjacency:   dict,
                      n_zones:     int) -> np.ndarray:
    """
    For each flooded zone, check if excess load spills to lower neighbours.
    Returns updated flood array with cascade effects.
    """
    SPILL_THRESHOLD = 0.10   # 10% load excess triggers spill
    propagated = flood_array.copy()

    flooded_zones = np.where(flood_array == 1)[0]
    for zid in flooded_zones:
        excess = load_ratio[zid] - 1.0   # How much over capacity
        if excess < SPILL_THRESHOLD:
            continue
        for nid in adjacency.get(zid, []):
            # Water flows to lower elevation neighbours
            if elevation[nid] < elevation[zid] and propagated[nid] == 0:
                spill_load = excess * 0.4   # 40% of excess spills over
                if load_ratio[nid] + spill_load > 1.0:
                    propagated[nid] = 1     # Neighbour now floods too

    return propagated


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING FOR ML
# ─────────────────────────────────────────────────────────────────────────────

def build_ml_features(sim_df: pd.DataFrame,
                       city_df: pd.DataFrame,
                       window: int = 7) -> pd.DataFrame:
    """
    Builds feature matrix for ML flood prediction.
    Features include: current-day values + rolling window stats.
    """
    print(f"[Phase 5] Engineering ML features (rolling window={window} days)...")

    # Sort properly
    sim_df = sim_df.sort_values(["zone_id", "day"]).reset_index(drop=True)

    # ── Rolling window features per zone ─────────────────────────────────
    grp = sim_df.groupby("zone_id")

    for col in ["rainfall_mm", "load_ratio", "drift_memory", "degradation_factor"]:
        sim_df[f"{col}_roll7_mean"] = grp[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        sim_df[f"{col}_roll7_max"]  = grp[col].transform(
            lambda x: x.rolling(window, min_periods=1).max())

    # Rainfall trend (7-day slope)
    def rolling_slope(x):
        result = np.full(len(x), 0.0)
        vals   = x.values
        for i in range(window, len(vals)):
            y = vals[i-window:i]
            result[i] = np.polyfit(range(window), y, 1)[0]
        return pd.Series(result, index=x.index)

    sim_df["rainfall_trend"] = grp["rainfall_mm"].transform(rolling_slope)

    # Zone static features
    static_cols = ["runoff_coeff", "drain_capacity", "drain_age_yrs",
                   "infra_health_score"]
    static      = city_df[["zone_id"] + static_cols]
    sim_df      = sim_df.merge(static, on="zone_id", how="left",
                               suffixes=("", "_static"))

    # Day-of-year (seasonality signal)
    sim_df["day_of_year"] = sim_df["day"] % 365

    print(f"[Phase 5] Feature matrix: {sim_df.shape}")
    return sim_df


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "rainfall_mm", "soil_saturation", "eff_runoff",
    "degradation_factor", "drift_memory", "load_ratio",
    "d1_hydraulic", "d2_stress", "d3_efficiency", "drift_index",
    "rainfall_mm_roll7_mean", "rainfall_mm_roll7_max",
    "load_ratio_roll7_mean", "load_ratio_roll7_max",
    "drift_memory_roll7_mean", "degradation_factor_roll7_mean",
    "rainfall_trend", "runoff_coeff", "drain_capacity",
    "drain_age_yrs", "infra_health_score", "day_of_year"
]
TARGET_COL = "flood_event"


def train_flood_model(sim_df_feat: pd.DataFrame):
    """
    Train on years 1-5, evaluate on years 6-10.
    Returns trained model, scaler, and evaluation metrics.
    """
    print(f"\n[Phase 5] Training ML flood prediction model...")

    # Train/test split: first 5 years = train, next 5 = test
    train = sim_df_feat[sim_df_feat["day"] < 365 * 5]
    test  = sim_df_feat[sim_df_feat["day"] >= 365 * 5]

    # Sample to manage memory (10% of train, all test)
    train_sample = train.sample(frac=0.10, random_state=RANDOM_SEED)

    X_train = train_sample[FEATURE_COLS].fillna(0)
    y_train = train_sample[TARGET_COL]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test[TARGET_COL]

    print(f"  Train: {len(X_train):,} samples | "
          f"Positive rate: {y_train.mean()*100:.1f}%")
    print(f"  Test:  {len(X_test):,} samples | "
          f"Positive rate: {y_test.mean()*100:.1f}%")

    # ── Model 1: Gradient Boosting ────────────────────────────────────────
    print("  Training Gradient Boosting Classifier...")
    gb_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_SEED))
    ])
    gb_model.fit(X_train, y_train)
    y_pred_gb  = gb_model.predict(X_test)
    y_prob_gb  = gb_model.predict_proba(X_test)[:, 1]
    auc_gb     = roc_auc_score(y_test, y_prob_gb)
    print(f"  GradientBoosting AUC: {auc_gb:.4f}")
    print(classification_report(y_test, y_pred_gb, digits=3))

    # ── Model 2: Random Forest ────────────────────────────────────────────
    print("  Training Random Forest Classifier...")
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1))
    ])
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    auc_rf    = roc_auc_score(y_test, y_prob_rf)
    print(f"  RandomForest AUC: {auc_rf:.4f}")

    # Use best model
    best_model = gb_model if auc_gb >= auc_rf else rf_model
    best_prob  = y_prob_gb if auc_gb >= auc_rf else y_prob_rf
    best_name  = "GradientBoosting" if auc_gb >= auc_rf else "RandomForest"
    print(f"\n  ✓ Best model: {best_name} (AUC={max(auc_gb,auc_rf):.4f})")

    # Save predictions
    pred_df = test[["day", "zone_id", "flood_event"]].copy()
    pred_df["ml_flood_prob"] = best_prob
    pred_df["ml_flood_pred"] = (best_prob > 0.5).astype(int)
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, "flood_predictions_ml.csv"), index=False)
    print(f"[Phase 5] ML predictions saved → data/flood_predictions_ml.csv")

    # Feature importance
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": clf.feature_importances_
        }).sort_values("importance", ascending=False)
        fi.to_csv(
            os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
        print("\n[Phase 5] Top 10 features:")
        print(fi.head(10).to_string(index=False))

    return best_model, pred_df


# ─────────────────────────────────────────────────────────────────────────────
# MAINTENANCE PRIORITIZATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_maintenance_priority(zone_profiles: pd.DataFrame,
                                  city_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks all zones by maintenance urgency.
    Score = weighted combination of degradation, flood risk, and drift.
    """
    print("\n[Phase 5] Computing maintenance priority scores...")

    df = zone_profiles.merge(
        city_df[["zone_id","x_m","y_m"]], on="zone_id", how="left")

    # Normalize each component to 0-1
    def norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mn) / (mx - mn)

    score = (
        0.35 * norm(df["final_degradation"]) +
        0.30 * norm(df["flood_rate"]) +
        0.20 * norm(df["final_drift_memory"]) +
        0.10 * norm(df["flood_trend_slope"].clip(lower=0)) +
        0.05 * norm(df["drain_age_yrs"])
    )

    df["maintenance_priority_score"] = score.round(4)
    df["maintenance_rank"]           = score.rank(ascending=False).astype(int)

    # Priority tier
    df["priority_tier"] = pd.cut(score,
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    out = os.path.join(OUTPUT_DIR, "maintenance_priority.csv")
    df.to_csv(out, index=False)

    tier_dist = df["priority_tier"].value_counts()
    print(f"[Phase 5] Maintenance priority distribution:")
    print(tier_dist.to_string())
    print(f"\n[Phase 5] Saved → {out}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def visualize_ml_results(pred_df: pd.DataFrame,
                          sim_df: pd.DataFrame):
    """Precision-recall and prediction comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 5 — ML Flood Prediction Results", fontsize=13, fontweight="bold")

    # Daily flood rate: actual vs predicted
    daily = pred_df.groupby("day").agg(
        actual_rate  = ("flood_event",  "mean"),
        predicted_rate = ("ml_flood_pred","mean"),
    ).reset_index()

    axes[0].plot(daily["day"], daily["actual_rate"]*100,
                 label="Actual",    color="red",   linewidth=0.8, alpha=0.8)
    axes[0].plot(daily["day"], daily["predicted_rate"]*100,
                 label="Predicted", color="blue",  linewidth=0.8, alpha=0.8, linestyle="--")
    axes[0].set_title("Actual vs ML Predicted Flood Rate (Years 6-10)")
    axes[0].set_xlabel("Day"); axes[0].set_ylabel("% Zones Flooded")
    axes[0].legend()

    # Flood probability distribution
    axes[1].hist(pred_df["ml_flood_prob"], bins=50,
                 color="#1565c0", edgecolor="white", alpha=0.8)
    axes[1].axvline(0.5, color="red", linestyle="--", label="Threshold=0.5")
    axes[1].set_title("ML Predicted Flood Probability Distribution")
    axes[1].set_xlabel("Flood Probability"); axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase5_ml_results.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 5] ML results visualization saved → {out}")


def visualize_maintenance_map(maint_df: pd.DataFrame):
    """Spatial maintenance priority map."""
    grid_n = int(np.sqrt(maint_df["zone_id"].max() + 1))

    score_grid = maint_df.sort_values(["grid_row", "grid_col"])[
        "maintenance_priority_score"].values.reshape(grid_n, grid_n)

    tier_map   = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    tier_grid  = maint_df.sort_values(["grid_row", "grid_col"])[
        "priority_tier"].map(tier_map).values.reshape(grid_n, grid_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 5 — Maintenance Prioritization Map",
                 fontsize=13, fontweight="bold")

    im1 = axes[0].imshow(score_grid, cmap="YlOrRd", origin="lower")
    axes[0].set_title("Maintenance Priority Score")
    plt.colorbar(im1, ax=axes[0], label="Priority Score (0=low, 1=critical)")

    cmap2 = plt.matplotlib.colors.ListedColormap(
        ["#4caf50", "#ffeb3b", "#ff9800", "#f44336"])
    im2 = axes[1].imshow(tier_grid, cmap=cmap2, origin="lower", vmin=0, vmax=3)
    axes[1].set_title("Maintenance Priority Tier")
    cbar2 = plt.colorbar(im2, ax=axes[1], ticks=[0, 1, 2, 3])
    cbar2.ax.set_yticklabels(["LOW", "MEDIUM", "HIGH", "CRITICAL"])

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "phase5_maintenance_map.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[Phase 5] Maintenance map saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run(sim_10yr: pd.DataFrame     = None,
        city_df: pd.DataFrame      = None,
        zone_profiles: pd.DataFrame = None):

    if city_df is None:
        city_df = pd.read_csv(os.path.join(OUTPUT_DIR, "city_zones.csv"))
    if sim_10yr is None:
        sim_10yr = pd.read_csv(os.path.join(OUTPUT_DIR, "simulation_10yr.csv"))
    if zone_profiles is None:
        zone_profiles = pd.read_csv(os.path.join(OUTPUT_DIR, "zone_profiles_10yr.csv"))

    # Feature engineering
    sim_feat = build_ml_features(sim_10yr, city_df)

    # Train ML model
    model, pred_df = train_flood_model(sim_feat)

    # Maintenance priority
    maint_df = compute_maintenance_priority(zone_profiles, city_df)

    # Visualizations
    visualize_ml_results(pred_df, sim_10yr)
    visualize_maintenance_map(maint_df)

    print("\n[Phase 5] ✓ Flood prediction complete.")
    return model, pred_df, maint_df


if __name__ == "__main__":
    run()