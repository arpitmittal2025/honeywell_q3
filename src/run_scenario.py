"""
Closed-Loop Execution Harness
==============================

Runs the autonomous choke controller in closed loop against a simulator
(mock or real) for each demonstration scenario.

Scenarios
---------
A -- Startup to Target:  choke starts at 0%, controller ramps to target Q.
B -- Target Tracking:    target changes mid-run (100 -> 150 bbl/hr at t=24).
C -- Infeasible Target:  target exceeds max safe rate -> graceful fallback.

Usage
-----
    python src/run_scenario.py --scenario A
    python src/run_scenario.py --scenario B
    python src/run_scenario.py --scenario C
    python src/run_scenario.py --scenario all
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator_interface import create_simulator
from src.model_identification import WellModel, fit_all
from src.controller import ChokeController, ControllerConfig, ControlDecision
from src.step_test import run_step_tests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ====================================================================== #
#  Scenario definitions                                                   #
# ====================================================================== #

@dataclass
class ScenarioConfig:
    """Defines a demonstration scenario."""
    name: str
    description: str
    n_steps: int                       # total simulation hours
    initial_choke: float               # starting choke %
    target_schedule: Dict[int, float]  # {time_step: target_Q}


SCENARIOS = {
    "A": ScenarioConfig(
        name="A - Startup to Target",
        description="Controller brings the well from startup (choke=0%) to target Q=120 bbl/hr.",
        n_steps=72,
        initial_choke=0.0,
        target_schedule={0: 120.0},
    ),
    "B": ScenarioConfig(
        name="B - Target Tracking",
        description="Target changes from 100 -> 150 bbl/hr at t=24 hr.",
        n_steps=72,
        initial_choke=0.0,
        target_schedule={0: 100.0, 24: 150.0},
    ),
    "C": ScenarioConfig(
        name="C - Infeasible Target",
        description="Target Q=500 bbl/hr exceeds max safe rate -> controller settles at max safe.",
        n_steps=72,
        initial_choke=0.0,
        target_schedule={0: 500.0},
    ),
}

# Controller constraint defaults
DEFAULT_CONTROLLER_CONFIG = ControllerConfig(
    target_Q=0.0,  # will be set per scenario
    whp_limits=(200.0, 3000.0),
    flp_limits=(150.0, 2500.0),
    bhp_limits=(2000.0, 5000.0),
    max_ramp_rate=5.0,
    candidate_resolution=0.5,
)


# ====================================================================== #
#  Closed-loop runner                                                     #
# ====================================================================== #

def run_closed_loop(
    scenario: ScenarioConfig,
    model: WellModel,
    sim_type: str = "mock",
) -> pd.DataFrame:
    """
    Run the closed-loop controller for one scenario.

    Returns
    -------
    pd.DataFrame
        Time-series log with columns:
        t, choke, choke_applied, Q, WHP, FLP, BHP, target_Q,
        predicted_Q, reason, n_feasible
    """
    sim = create_simulator(sim_type)
    sim.reset()

    config = ControllerConfig(
        target_Q=0.0,
        whp_limits=DEFAULT_CONTROLLER_CONFIG.whp_limits,
        flp_limits=DEFAULT_CONTROLLER_CONFIG.flp_limits,
        bhp_limits=DEFAULT_CONTROLLER_CONFIG.bhp_limits,
        max_ramp_rate=DEFAULT_CONTROLLER_CONFIG.max_ramp_rate,
        candidate_resolution=DEFAULT_CONTROLLER_CONFIG.candidate_resolution,
    )

    ctrl = ChokeController(model, config)
    ctrl.reset(scenario.initial_choke)

    # Set initial target
    sorted_targets = sorted(scenario.target_schedule.items())
    target_idx = 0
    config.target_Q = sorted_targets[0][1]

    # Initial state read (at starting choke)
    Q, WHP, FLP, BHP = sim.step(scenario.initial_choke)
    current_state = {"Q": Q, "WHP": WHP, "FLP": FLP, "BHP": BHP}

    records = []

    for t in range(scenario.n_steps):
        # Update target if schedule dictates
        while target_idx < len(sorted_targets) - 1 and t >= sorted_targets[target_idx + 1][0]:
            target_idx += 1
        config.target_Q = sorted_targets[target_idx][1]

        # Controller decides
        decision = ctrl.compute_action(current_state)

        # Apply the selected choke to the simulator
        Q, WHP, FLP, BHP = sim.step(decision.selected_choke)
        current_state = {"Q": Q, "WHP": WHP, "FLP": FLP, "BHP": BHP}

        # Log
        records.append({
            "t": t,
            "choke_cmd": decision.selected_choke,
            "Q": Q,
            "WHP": WHP,
            "FLP": FLP,
            "BHP": BHP,
            "target_Q": config.target_Q,
            "predicted_Q": decision.predicted_Q,
            "reason": decision.reason,
            "n_feasible": decision.n_feasible,
            "n_candidates": decision.n_candidates,
        })

    return pd.DataFrame(records)


# ====================================================================== #
#  Validation checks                                                      #
# ====================================================================== #

def validate_results(df: pd.DataFrame, scenario: ScenarioConfig) -> Dict[str, bool]:
    """Check that the run satisfies all constraints."""
    cfg = DEFAULT_CONTROLLER_CONFIG

    checks = {}

    # 1. Choke ramp rate
    choke_diffs = df["choke_cmd"].diff().abs().dropna()
    max_ramp = choke_diffs.max()
    checks["ramp_rate_ok"] = max_ramp <= cfg.max_ramp_rate + 0.01  # small tolerance
    if not checks["ramp_rate_ok"]:
        print(f"  WARNING: RAMP RATE VIOLATION: max delta_choke = {max_ramp:.2f}% (limit {cfg.max_ramp_rate}%)")

    # 2. WHP limits
    whp_ok = (df["WHP"] >= cfg.whp_limits[0] - 1.0).all() and (df["WHP"] <= cfg.whp_limits[1] + 1.0).all()
    checks["whp_ok"] = whp_ok
    if not whp_ok:
        print(f"  WARNING: WHP VIOLATION: range [{df['WHP'].min():.0f}, {df['WHP'].max():.0f}] psi")

    # 3. FLP limits
    flp_ok = (df["FLP"] >= cfg.flp_limits[0] - 1.0).all() and (df["FLP"] <= cfg.flp_limits[1] + 1.0).all()
    checks["flp_ok"] = flp_ok
    if not flp_ok:
        print(f"  WARNING: FLP VIOLATION: range [{df['FLP'].min():.0f}, {df['FLP'].max():.0f}] psi")

    # 4. BHP limits
    bhp_ok = (df["BHP"] >= cfg.bhp_limits[0] - 1.0).all() and (df["BHP"] <= cfg.bhp_limits[1] + 1.0).all()
    checks["bhp_ok"] = bhp_ok
    if not bhp_ok:
        print(f"  WARNING: BHP VIOLATION: range [{df['BHP'].min():.0f}, {df['BHP'].max():.0f}] psi")

    # 5. Choke bounds
    choke_ok = (df["choke_cmd"] >= 0.0).all() and (df["choke_cmd"] <= 100.0).all()
    checks["choke_bounds_ok"] = choke_ok

    all_ok = all(checks.values())
    checks["all_constraints_satisfied"] = all_ok

    return checks


# ====================================================================== #
#  Plotting                                                               #
# ====================================================================== #

def plot_scenario(df: pd.DataFrame, scenario: ScenarioConfig,
                  checks: Dict[str, bool], save_dir: str) -> str:
    """
    Generate the 5-panel scenario plot.

    Returns path to saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    cfg = DEFAULT_CONTROLLER_CONFIG

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    status = "[PASS] ALL CONSTRAINTS SATISFIED" if checks["all_constraints_satisfied"] else "[WARN] CONSTRAINT VIOLATIONS"
    fig.suptitle(f"Scenario {scenario.name}\n{scenario.description}\n{status}",
                 fontsize=14, fontweight="bold", y=0.98)

    t = df["t"]

    # Panel 1: Oil Flow Rate (Q) vs Target
    ax = axes[0]
    ax.plot(t, df["Q"], color="#2196F3", linewidth=2.0, label="Actual Q")
    ax.step(t, df["target_Q"], color="#F44336", linewidth=1.5, linestyle="--",
            where="post", label="Target Q")
    ax.set_ylabel("Oil Rate (bbl/hr)", fontsize=11)
    ax.set_title("Oil Flow Rate - Target vs. Actual", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: WHP
    ax = axes[1]
    ax.plot(t, df["WHP"], color="#FF9800", linewidth=1.5)
    ax.axhline(cfg.whp_limits[0], color="red", linestyle=":", alpha=0.5, label=f"Min {cfg.whp_limits[0]}")
    ax.axhline(cfg.whp_limits[1], color="red", linestyle=":", alpha=0.5, label=f"Max {cfg.whp_limits[1]}")
    ax.set_ylabel("WHP (psi)", fontsize=11)
    ax.set_title("Wellhead Pressure", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: FLP
    ax = axes[2]
    ax.plot(t, df["FLP"], color="#4CAF50", linewidth=1.5)
    ax.axhline(cfg.flp_limits[0], color="red", linestyle=":", alpha=0.5, label=f"Min {cfg.flp_limits[0]}")
    ax.axhline(cfg.flp_limits[1], color="red", linestyle=":", alpha=0.5, label=f"Max {cfg.flp_limits[1]}")
    ax.set_ylabel("FLP (psi)", fontsize=11)
    ax.set_title("Flowline Pressure", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: BHP
    ax = axes[3]
    ax.plot(t, df["BHP"], color="#E91E63", linewidth=1.5)
    ax.axhline(cfg.bhp_limits[0], color="red", linestyle=":", alpha=0.5, label=f"Min {cfg.bhp_limits[0]}")
    ax.axhline(cfg.bhp_limits[1], color="red", linestyle=":", alpha=0.5, label=f"Max {cfg.bhp_limits[1]}")
    ax.set_ylabel("BHP (psi)", fontsize=11)
    ax.set_title("Bottom-Hole Pressure", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 5: Choke Position
    ax = axes[4]
    ax.step(t, df["choke_cmd"], where="post", color="#9C27B0", linewidth=2.0)
    ax.set_ylabel("Choke (%)", fontsize=11)
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_title("Choke Position (Controller Output)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"scenario_{scenario.name[0].lower()}_plots.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Plot saved: {fpath}")

    return fpath


# ====================================================================== #
#  Console summary                                                        #
# ====================================================================== #

def print_summary(df: pd.DataFrame, scenario: ScenarioConfig,
                  checks: Dict[str, bool]) -> None:
    """Print a concise summary to the console."""
    print(f"\n{'='*60}")
    print(f"  Scenario {scenario.name}")
    print(f"{'='*60}")
    print(f"  Description : {scenario.description}")
    print(f"  Duration    : {scenario.n_steps} hours")
    print(f"  Final Q     : {df['Q'].iloc[-1]:.1f} bbl/hr")
    print(f"  Final Target: {df['target_Q'].iloc[-1]:.1f} bbl/hr")
    print(f"  Final Choke : {df['choke_cmd'].iloc[-1]:.1f}%")
    print(f"  Q Range     : [{df['Q'].min():.1f}, {df['Q'].max():.1f}] bbl/hr")
    print(f"  WHP Range   : [{df['WHP'].min():.0f}, {df['WHP'].max():.0f}] psi")
    print(f"  FLP Range   : [{df['FLP'].min():.0f}, {df['FLP'].max():.0f}] psi")
    print(f"  BHP Range   : [{df['BHP'].min():.0f}, {df['BHP'].max():.0f}] psi")
    print(f"  Max Ramp    : {df['choke_cmd'].diff().abs().max():.2f}%")

    # Settling time estimate (within 5% of target)
    target = df["target_Q"].iloc[-1]
    within_band = (df["Q"] - target).abs() < 0.05 * target
    if within_band.any():
        settle_time = within_band.idxmax()
        print(f"  Settle Time : ~{df['t'].iloc[settle_time]} hr (5% band)")
    else:
        print(f"  Settle Time : Not settled (target may be infeasible)")

    print(f"\n  Constraint checks:")
    for check, ok in checks.items():
        symbol = "[PASS]" if ok else "[FAIL]"
        print(f"    {symbol} {check}")


# ====================================================================== #
#  Model preparation                                                      #
# ====================================================================== #

def prepare_model() -> WellModel:
    """
    Load a fitted model from disk, or fit one from step-test data.
    """
    model_path = os.path.join(PROJECT_ROOT, "data", "model_params.json")

    if os.path.exists(model_path):
        print("[harness] Loading fitted model from disk...")
        return WellModel.load(model_path)

    # No model on disk — run step tests and fit
    print("[harness] No fitted model found. Running step tests + fitting...")
    csv_path = os.path.join(PROJECT_ROOT, "data", "step_test_results.csv")

    if os.path.exists(csv_path):
        df_step = pd.read_csv(csv_path)
    else:
        df_step = run_step_tests("mock")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_step.to_csv(csv_path, index=False)

    model = fit_all(df_step)
    model.save(model_path)
    return model


# ====================================================================== #
#  Main orchestrator                                                      #
# ====================================================================== #

def run_scenario(scenario_key: str, model: WellModel) -> None:
    """Run a single scenario end-to-end."""
    scenario = SCENARIOS[scenario_key]

    print(f"\n{'#'*60}")
    print(f"  Running Scenario {scenario.name}")
    print(f"{'#'*60}")

    # Run closed loop
    df = run_closed_loop(scenario, model, sim_type="mock")

    # Validate
    checks = validate_results(df, scenario)

    # Save CSV log
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_out = os.path.join(results_dir, f"scenario_{scenario_key.lower()}_log.csv")
    df.to_csv(csv_out, index=False)
    print(f"  -> Log saved: {csv_out}")

    # Plot
    plot_scenario(df, scenario, checks, results_dir)

    # Summary
    print_summary(df, scenario, checks)


def main():
    parser = argparse.ArgumentParser(
        description="Run closed-loop choke control scenarios."
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=["A", "B", "C", "all"],
        default="all",
        help="Which scenario to run (default: all).",
    )
    parser.add_argument(
        "--sim-type",
        choices=["mock", "real"],
        default="mock",
        help="Simulator to use (default: mock).",
    )
    args = parser.parse_args()

    # Prepare model
    model = prepare_model()

    # Run requested scenarios
    if args.scenario == "all":
        for key in ["A", "B", "C"]:
            run_scenario(key, model)
    else:
        run_scenario(args.scenario, model)

    print("\n" + "=" * 60)
    print("  All scenarios complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
