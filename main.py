"""
MAIN RUNNER — Urban Flood Prediction System
=============================================
Executes all 7 phases in sequence.

Usage:
    python main.py                       # Full pipeline (all phases)
    python main.py --phases 4 5 6 7      # Re-run specific phases only
    python main.py --phases 1 2 4 5 6 7  # Skip Phase 3 (use cached sim)
    python main.py --day 2500            # Dashboard snapshot day
    python main.py --zone 800            # Deep dive zone ID

Estimated runtime on 2500 zones:
    Phase 1-2:  ~10 seconds
    Phase 3:    ~15-25 minutes (10yr simulation)
    Phase 4-6:  ~5-10 minutes
    Phase 7:    ~2 minutes
"""

import argparse
import time
import os
import sys
import traceback
import pandas as pd

# ── Use ABSOLUTE path so all phases write to the same data/ folder
# regardless of which directory you run python from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _patch_output_dir(module):
    """Inject absolute OUTPUT_DIR into a phase module so it saves files correctly."""
    module.OUTPUT_DIR = OUTPUT_DIR
    if hasattr(module, "DATA_DIR"):
        module.DATA_DIR = OUTPUT_DIR


def phase_banner(n: int, name: str):
    print(f"\n{'='*60}")
    print(f"  PHASE {n}: {name}")
    print(f"{'='*60}")


def check_dependencies():
    """Check all required packages are installed before starting."""
    required = {
        "numpy":      "numpy",
        "pandas":     "pandas",
        "matplotlib": "matplotlib",
        "sklearn":    "scikit-learn",
        "networkx":   "networkx",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("\n  ❌ MISSING PACKAGES — install these first:\n")
        for pkg in missing:
            print(f"     pip install {pkg}")
        print()
        sys.exit(1)
    else:
        print("  ✓ All dependencies found")


def run_phase(n, name, fn):
    """
    Runs a phase function and catches ALL errors, printing full traceback.
    Returns (success: bool, result: any).
    """
    phase_banner(n, name)
    t = time.time()
    try:
        result  = fn()
        elapsed = time.time() - t
        print(f"  ⏱ Phase {n} completed in {elapsed:.1f}s")
        return True, result

    except ImportError as e:
        pkg = str(e).split("'")[1] if "'" in str(e) else str(e).split()[-1]
        print(f"\n  ❌ Phase {n} FAILED — missing package: {e}")
        print(f"     Fix:  pip install {pkg}")
        print(f"     Then: python main.py --phases {n}")
        return False, None

    except FileNotFoundError as e:
        print(f"\n  ❌ Phase {n} FAILED — file not found: {e}")
        print(f"     Make sure earlier phases ran successfully first.")
        return False, None

    except Exception as e:
        print(f"\n  ❌ Phase {n} FAILED — {type(e).__name__}: {e}")
        print("  ── Full traceback ──")
        traceback.print_exc()
        print("  ────────────────────")
        return False, None


def run_full_pipeline(phases_to_run, snapshot_day, zone_id):
    total_start = time.time()
    results     = {}
    failed      = []

    # ── PHASE 1 ──────────────────────────────────────────────────────────────
    if 1 in phases_to_run:
        def _p1():
            import phase1_city_construction as p1
            _patch_output_dir(p1)
            return p1.run()
        ok, res = run_phase(1, "Virtual City Construction", _p1)
        if ok:
            results["city_df"] = res
        else:
            failed.append(1)
            p = os.path.join(OUTPUT_DIR, "city_zones.csv")
            if os.path.exists(p):
                results["city_df"] = pd.read_csv(p)
                print(f"  ↩  Loaded existing city_zones.csv as fallback")
    else:
        p = os.path.join(OUTPUT_DIR, "city_zones.csv")
        if os.path.exists(p):
            results["city_df"] = pd.read_csv(p)
            print("[Skipped] Phase 1 — loaded city_zones.csv")
        else:
            print("[Skipped] Phase 1 — ⚠️  city_zones.csv not found, add phase 1 to --phases")
            results["city_df"] = None

    # ── PHASE 2 ──────────────────────────────────────────────────────────────
    if 2 in phases_to_run:
        def _p2():
            import phase2_drainage_infrastructure as p2
            _patch_output_dir(p2)
            return p2.run(results.get("city_df"))
        ok, res = run_phase(2, "Drainage Infrastructure Setup", _p2)
        if ok:
            results["city_df"], results["drain_graph"] = res
        else:
            failed.append(2)
            results["drain_graph"] = None
    else:
        p = os.path.join(OUTPUT_DIR, "city_zones.csv")
        if os.path.exists(p) and results.get("city_df") is None:
            results["city_df"] = pd.read_csv(p)
        results["drain_graph"] = None
        print("[Skipped] Phase 2")

    # ── PHASE 3 ──────────────────────────────────────────────────────────────
    if 3 in phases_to_run:
        def _p3():
            import phase3_data_generation as p3
            _patch_output_dir(p3)
            return p3.run(results.get("city_df"))
        ok, res = run_phase(3, "5-Year & 10-Year Data Generation", _p3)
        if ok:
            results["sim_5yr"], results["sim_10yr"] = res
        else:
            failed.append(3)
            results["sim_5yr"]  = None
            results["sim_10yr"] = None
    else:
        print("[Skipped] Phase 3 — loading cached simulation files")
        for label, key in [("10yr", "sim_10yr"), ("5yr", "sim_5yr")]:
            p = os.path.join(OUTPUT_DIR, f"simulation_{label}.csv")
            if os.path.exists(p):
                results[key] = pd.read_csv(p)
                print(f"  ↩  Loaded simulation_{label}.csv")
            else:
                results[key] = None
                print(f"  ⚠️  simulation_{label}.csv not found")

    # ── PHASE 4 ──────────────────────────────────────────────────────────────
    if 4 in phases_to_run:
        def _p4():
            import phase4_simulation_engine as p4
            _patch_output_dir(p4)
            return p4.run(results.get("sim_10yr"), results.get("city_df"))
        ok, res = run_phase(4, "Simulation Engine Analysis", _p4)
        if ok:
            results["zone_profiles"] = res
        else:
            failed.append(4)
            results["zone_profiles"] = None
    else:
        p = os.path.join(OUTPUT_DIR, "zone_profiles_10yr.csv")
        results["zone_profiles"] = pd.read_csv(p) if os.path.exists(p) else None
        print("[Skipped] Phase 4")

    # ── PHASE 5 ──────────────────────────────────────────────────────────────
    if 5 in phases_to_run:
        def _p5():
            import phase5_flood_prediction as p5
            _patch_output_dir(p5)
            return p5.run(
                results.get("sim_10yr"),
                results.get("city_df"),
                results.get("zone_profiles"),
            )
        ok, res = run_phase(5, "Flood Prediction & ML", _p5)
        if ok:
            results["model"], results["ml_preds"], results["maint"] = res
        else:
            failed.append(5)
    else:
        print("[Skipped] Phase 5")

    # ── PHASE 6 ──────────────────────────────────────────────────────────────
    if 6 in phases_to_run:
        def _p6():
            import phase6_self_learning as p6
            _patch_output_dir(p6)
            return p6.run(results.get("sim_10yr"), results.get("city_df"))
        ok, res = run_phase(6, "Self-Learning Weight Analysis", _p6)
        if ok:
            results["weight_evo"], results["weight_conv"] = res
        else:
            failed.append(6)
    else:
        print("[Skipped] Phase 6")

    # ── PHASE 7 ──────────────────────────────────────────────────────────────
    if 7 in phases_to_run:
        def _p7():
            import phase7_visualization as p7
            _patch_output_dir(p7)
            p7.run(snapshot_day=snapshot_day, zone_id=zone_id)
        ok, _ = run_phase(7, "Visualization Dashboard", _p7)
        if not ok:
            failed.append(7)

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    if not failed:
        print(f"  ✅ PIPELINE COMPLETE")
    else:
        print(f"  ⚠️  PIPELINE FINISHED WITH ERRORS IN PHASES: {failed}")
        print(f"\n  Fix the errors shown above, then re-run ONLY failed phases:")
        print(f"     python main.py --phases {' '.join(str(p) for p in failed)}")
        print(f"\n  (Phase 3 won't re-run since it's not in --phases,")
        print(f"   so the 15-minute simulation won't repeat)")

    print(f"\n  Total time: {total_time/60:.1f} minutes")
    print(f"  Output folder: {OUTPUT_DIR}")

    saved_csv = [f for f in sorted(os.listdir(OUTPUT_DIR)) if f.endswith(".csv")]
    saved_png = [f for f in sorted(os.listdir(OUTPUT_DIR)) if f.endswith(".png")]

    print(f"\n  CSV files saved ({len(saved_csv)}):")
    for f in saved_csv:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"    ✓ {f}  ({size} KB)")

    print(f"\n  PNG dashboards saved ({len(saved_png)}):")
    for f in saved_png:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
        print(f"    ✓ {f}  ({size} KB)")

    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban Flood Prediction System — Main Runner")

    parser.add_argument("--phases", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6, 7],
                        help="Which phases to run (default: all)")
    parser.add_argument("--day",  type=int, default=3000,
                        help="Day index for dashboard snapshot (default: 3000)")
    parser.add_argument("--zone", type=int, default=None,
                        help="Zone ID for deep-dive report (default: auto)")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   ADAPTIVE MICRO-ZONE URBAN FLOOD PREDICTION SYSTEM      ║
║   Based on Multi-Vector Drainage Infrastructure Drift     ║
╚══════════════════════════════════════════════════════════╝

  Phases to run : {args.phases}
  Snapshot day  : {args.day}
  Zone deep dive: {args.zone or 'auto'}
  Output folder : {OUTPUT_DIR}
""")

    print("  Checking dependencies...")
    check_dependencies()

    run_full_pipeline(
        phases_to_run = args.phases,
        snapshot_day  = args.day,
        zone_id       = args.zone,
    )