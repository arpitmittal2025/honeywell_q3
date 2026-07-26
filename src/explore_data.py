"""
Reference Dataset Exploration
=============================

Loads the reference CSV (if available) **or** generates synthetic data from
the mock simulator, then produces exploration plots for initial understanding
of choke ↔ process-variable relationships.

Outputs
-------
- ``results/exploration/choke_vs_Q.png``
- ``results/exploration/choke_vs_pressures.png``
- ``results/exploration/steady_state_curves.png``
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator_interface import create_simulator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "exploration")


def generate_synthetic_steady_state(n_points: int = 101) -> pd.DataFrame:
    """
    Generate synthetic steady-state data by running the mock simulator
    to equilibrium at each choke setting.
    """
    sim = create_simulator("mock")
    choke_values = np.linspace(0, 100, n_points)
    records = []

    for choke in choke_values:
        sim.reset()
        # Run for 30 steps to reach steady state (10× τ)
        for _ in range(30):
            Q, WHP, FLP, BHP = sim.step(float(choke))
        records.append({
            "choke": choke,
            "Q": Q, "WHP": WHP, "FLP": FLP, "BHP": BHP,
        })

    return pd.DataFrame(records)


def load_or_generate_data() -> Tuple[pd.DataFrame, str]:
    """Load reference CSV if available; otherwise generate synthetic data."""
    csv_path = os.path.join(DATA_DIR, "Autonomous_Choke_Control_Simulated_Dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df, "reference"
    else:
        print("[explore] Reference CSV not found — generating synthetic data from mock simulator.")
        df = generate_synthetic_steady_state()
        return df, "synthetic"


from typing import Tuple


def plot_choke_vs_flow(df: pd.DataFrame, source: str) -> None:
    """Plot choke position vs. oil flow rate."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["choke"], df["Q"], color="#2196F3", alpha=0.7, s=30, edgecolors="white", linewidth=0.5)
    ax.plot(df["choke"], df["Q"], color="#2196F3", alpha=0.4, linewidth=1)
    ax.set_xlabel("Choke Opening (%)", fontsize=13)
    ax.set_ylabel("Oil Flow Rate Q (bbl/hr)", fontsize=13)
    ax.set_title(f"Choke Position vs. Oil Flow Rate ({source} data)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 102)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "choke_vs_Q.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explore] Saved -> {RESULTS_DIR}/choke_vs_Q.png")


def plot_choke_vs_pressures(df: pd.DataFrame, source: str) -> None:
    """Plot choke position vs. WHP, FLP, BHP on one figure."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Choke Position vs. Pressures ({source} data)", fontsize=15, fontweight="bold")

    pressure_vars = [
        ("WHP", "Wellhead Pressure (psi)", "#FF9800"),
        ("FLP", "Flowline Pressure (psi)", "#4CAF50"),
        ("BHP", "Bottom-Hole Pressure (psi)", "#E91E63"),
    ]

    for ax, (var, label, color) in zip(axes, pressure_vars):
        ax.scatter(df["choke"], df[var], color=color, alpha=0.7, s=30,
                   edgecolors="white", linewidth=0.5)
        ax.plot(df["choke"], df[var], color=color, alpha=0.4, linewidth=1)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Choke Opening (%)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "choke_vs_pressures.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explore] Saved -> {RESULTS_DIR}/choke_vs_pressures.png")


def plot_all_on_one(df: pd.DataFrame, source: str) -> None:
    """Combined 4-panel steady-state characteristic curves."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Steady-State Characteristic Curves ({source} data)",
                 fontsize=15, fontweight="bold")

    plot_defs = [
        ("Q", "Oil Flow Rate (bbl/hr)", "#2196F3"),
        ("WHP", "Wellhead Pressure (psi)", "#FF9800"),
        ("FLP", "Flowline Pressure (psi)", "#4CAF50"),
        ("BHP", "Bottom-Hole Pressure (psi)", "#E91E63"),
    ]

    for ax, (var, label, color) in zip(axes.flat, plot_defs):
        ax.scatter(df["choke"], df[var], color=color, alpha=0.7, s=25,
                   edgecolors="white", linewidth=0.5)
        ax.plot(df["choke"], df[var], color=color, alpha=0.4, linewidth=1)
        ax.set_xlabel("Choke (%)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "steady_state_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explore] Saved -> {RESULTS_DIR}/steady_state_curves.png")


def print_gain_summary(df: pd.DataFrame) -> None:
    """Print approximate gains from the steady-state data."""
    print("\n[explore] Gain Direction Summary:")
    print("-" * 50)
    for var in ["Q", "WHP", "FLP", "BHP"]:
        v_min = df[var].iloc[0]
        v_max = df[var].iloc[-1]
        choke_range = df["choke"].iloc[-1] - df["choke"].iloc[0]
        gain = (v_max - v_min) / max(choke_range, 1.0)
        direction = "UP" if gain > 0 else "DOWN"
        print(f"  {var:4s}: {direction} as choke opens  "
              f"(gain = {gain:+.3f} per 1% choke)")
    print("-" * 50)


# ====================================================================== #
#  CLI entry point                                                        #
# ====================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("  Reference Data Exploration")
    print("=" * 60)

    df, source = load_or_generate_data()
    print(f"[explore] Loaded {len(df)} data points ({source})")

    plot_choke_vs_flow(df, source)
    plot_choke_vs_pressures(df, source)
    plot_all_on_one(df, source)
    print_gain_summary(df)

    print("\n[explore] Done.")
