"""
Open-Loop Step Testing
======================

Runs step tests across the choke operating range to characterize the
well's process response.  Each step is held for enough intervals to let
the first-order dynamics settle, then the next step is applied.

Outputs
-------
- ``data/step_test_results.csv``  — raw time-series data
- ``results/step_tests/*.png``    — per-variable step-response plots
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running as ``python src/step_test.py`` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator_interface import create_simulator


# ====================================================================== #
#  Configuration                                                          #
# ====================================================================== #

CHOKE_STEPS = np.arange(0, 105, 10)  # 0%, 10%, 20%, …, 100%
HOLD_INTERVALS = 15                   # hold each step for 15 hours (5× τ)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ====================================================================== #
#  Step-test runner                                                       #
# ====================================================================== #

def run_step_tests(sim_type: str = "mock") -> pd.DataFrame:
    """
    Execute a staircase of choke step changes and record responses.

    Returns
    -------
    pd.DataFrame
        Columns: t, choke, Q, WHP, FLP, BHP
    """
    sim = create_simulator(sim_type)
    sim.reset()

    records = []
    t = 0

    for choke_pct in CHOKE_STEPS:
        for _ in range(HOLD_INTERVALS):
            Q, WHP, FLP, BHP = sim.step(float(choke_pct))
            records.append({
                "t": t,
                "choke": choke_pct,
                "Q": Q,
                "WHP": WHP,
                "FLP": FLP,
                "BHP": BHP,
            })
            t += 1

    # Step back down to verify symmetry
    for choke_pct in reversed(CHOKE_STEPS):
        for _ in range(HOLD_INTERVALS):
            Q, WHP, FLP, BHP = sim.step(float(choke_pct))
            records.append({
                "t": t,
                "choke": choke_pct,
                "Q": Q,
                "WHP": WHP,
                "FLP": FLP,
                "BHP": BHP,
            })
            t += 1

    return pd.DataFrame(records)


# ====================================================================== #
#  Plotting                                                               #
# ====================================================================== #

def plot_step_tests(df: pd.DataFrame, save_dir: str | None = None) -> None:
    """Generate step-response plots for all process variables."""
    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "results", "step_tests")
    os.makedirs(save_dir, exist_ok=True)

    variables = {
        "Q":   ("Oil Flow Rate (bbl/hr)",   "#2196F3"),
        "WHP": ("Wellhead Pressure (psi)",   "#FF9800"),
        "FLP": ("Flowline Pressure (psi)",   "#4CAF50"),
        "BHP": ("Bottom-Hole Pressure (psi)", "#E91E63"),
    }

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Open-Loop Step Test Responses", fontsize=16, fontweight="bold")

    # Choke position (top panel)
    ax = axes[0]
    ax.step(df["t"], df["choke"], where="post", color="#9C27B0", linewidth=1.5)
    ax.set_ylabel("Choke (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.set_title("Choke Position (Input)", fontsize=12)

    for idx, (var, (label, color)) in enumerate(variables.items(), start=1):
        ax = axes[idx]
        ax.plot(df["t"], df[var], color=color, linewidth=1.0, alpha=0.8)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_title(label, fontsize=12)

    axes[-1].set_xlabel("Time (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "step_test_all.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[step_test] Saved combined plot -> {save_dir}/step_test_all.png")

    # Individual per-variable plots
    for var, (label, color) in variables.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Step Response - {label}", fontsize=14, fontweight="bold")

        ax1.step(df["t"], df["choke"], where="post", color="#9C27B0", linewidth=1.5)
        ax1.set_ylabel("Choke (%)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["t"], df[var], color=color, linewidth=1.2)
        ax2.set_ylabel(label)
        ax2.set_xlabel("Time (hours)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"step_test_{var}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[step_test] Saved individual plots -> {save_dir}/")


# ====================================================================== #
#  Steady-state gain estimation                                           #
# ====================================================================== #

def estimate_steady_state_gains(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the steady-state gain for each variable by taking the last
    measurement at each choke setting (during the up-staircase only).
    """
    # Use only the first half (ascending staircase)
    n_up = len(CHOKE_STEPS) * HOLD_INTERVALS
    df_up = df.iloc[:n_up].copy()

    # Get last reading at each choke setting
    ss = df_up.groupby("choke").agg({
        "Q": "last", "WHP": "last", "FLP": "last", "BHP": "last"
    }).reset_index()

    print("\n[step_test] Steady-state estimates:")
    print(ss.to_string(index=False))

    # Compute approximate gains (dY / dU)
    print("\n[step_test] Approximate gains (dY / d_choke%):")
    for var in ["Q", "WHP", "FLP", "BHP"]:
        diffs = ss[var].diff() / ss["choke"].diff()
        avg_gain = diffs.dropna().mean()
        print(f"  d{var}/du ~ {avg_gain:.3f}  (per 1% choke)")

    return ss


# ====================================================================== #
#  CLI entry point                                                        #
# ====================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("  Open-Loop Step Testing (Mock Simulator)")
    print("=" * 60)

    df = run_step_tests("mock")

    # Save raw data
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "step_test_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[step_test] Saved data -> {csv_path}  ({len(df)} rows)")

    # Plots
    plot_step_tests(df)

    # Gain estimates
    estimate_steady_state_gains(df)

    print("\n[step_test] Done.")
