"""
Generate Final Publication-Quality Plots
=========================================

Reads scenario CSV logs from ``results/`` and produces polished,
presentation-ready figures for each scenario.

Outputs to ``results/final/``.

Usage
-----
    python src/generate_final_plots.py
    python src/generate_final_plots.py --sim-source real
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Constraint limits (must match controller config)
WHP_LIMITS = (200.0, 3000.0)
FLP_LIMITS = (150.0, 2500.0)
BHP_LIMITS = (2000.0, 5000.0)
RAMP_RATE  = 5.0


# ====================================================================== #
#  Plot styling                                                           #
# ====================================================================== #

COLORS = {
    "Q":      "#1976D2",
    "target": "#D32F2F",
    "WHP":    "#F57C00",
    "FLP":    "#388E3C",
    "BHP":    "#C2185B",
    "choke":  "#7B1FA2",
    "limit":  "#EF5350",
    "band":   "#FFCDD2",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})


# ====================================================================== #
#  Per-scenario plot                                                      #
# ====================================================================== #

def _compute_settling_time(df: pd.DataFrame, band_pct: float = 0.05) -> str:
    """Estimate when Q first enters and stays within ±band of final target."""
    target = df["target_Q"].iloc[-1]
    q = df["Q"].values
    t = df["t"].values
    tol = band_pct * target

    # Walk backwards to find last exit from band
    settled_idx = None
    for i in range(len(q) - 1, -1, -1):
        if abs(q[i] - target) > tol:
            settled_idx = i + 1
            break

    if settled_idx is None:
        return "< 1 hr"
    elif settled_idx >= len(q):
        return "Not settled"
    else:
        return f"~{t[settled_idx]} hr"


def _add_constraint_band(ax, lo, hi, y_data):
    """Add light red shading outside the constraint limits."""
    y_min, y_max = y_data.min(), y_data.max()
    margin = (y_max - y_min) * 0.15
    ax_lo = min(lo - margin * 0.5, y_min - margin)
    ax_hi = max(hi + margin * 0.5, y_max + margin)

    # Shade below lower limit
    if ax_lo < lo:
        ax.axhspan(ax_lo, lo, color=COLORS["band"], alpha=0.3, zorder=0)
    # Shade above upper limit
    if ax_hi > hi:
        ax.axhspan(hi, ax_hi, color=COLORS["band"], alpha=0.3, zorder=0)

    ax.axhline(lo, color=COLORS["limit"], linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Limit ({lo:.0f})")
    ax.axhline(hi, color=COLORS["limit"], linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Limit ({hi:.0f})")
    ax.set_ylim(ax_lo, ax_hi)


def plot_scenario_final(df: pd.DataFrame, scenario_label: str,
                        description: str, save_dir: str) -> str:
    """
    Generate a polished 5-panel plot for one scenario.

    Returns the path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)

    # Compute metrics for title
    final_Q = df["Q"].iloc[-1]
    final_target = df["target_Q"].iloc[-1]
    settle = _compute_settling_time(df)
    max_ramp = df["choke_cmd"].diff().abs().max()

    fig.suptitle(
        f"Scenario {scenario_label}\n{description}\n"
        f"Final Q = {final_Q:.1f} bbl/hr  |  Settling ≈ {settle}  |  "
        f"Max Δchoke = {max_ramp:.1f}%",
        fontsize=15, fontweight="bold", y=0.98,
    )

    t = df["t"]

    # ── Panel 1: Oil Flow Rate vs Target ──────────────────────────────
    ax = axes[0]
    ax.plot(t, df["Q"], color=COLORS["Q"], linewidth=2.0, label="Actual Q",
            zorder=3)
    ax.step(t, df["target_Q"], color=COLORS["target"], linewidth=1.8,
            linestyle="--", where="post", label="Target Q", zorder=2)

    # Shade ±5% band around target
    targets_unique = df["target_Q"].unique()
    for tgt in targets_unique:
        mask = df["target_Q"] == tgt
        t_start = df.loc[mask, "t"].iloc[0]
        t_end = df.loc[mask, "t"].iloc[-1]
        ax.axhspan(tgt * 0.95, tgt * 1.05, xmin=0, xmax=1,
                    color=COLORS["Q"], alpha=0.07, zorder=0)

    ax.set_ylabel("Oil Rate (bbl/hr)")
    ax.set_title("Oil Flow Rate — Target vs. Actual")
    ax.legend(loc="lower right")

    # ── Panel 2: WHP ──────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, df["WHP"], color=COLORS["WHP"], linewidth=1.8)
    _add_constraint_band(ax, WHP_LIMITS[0], WHP_LIMITS[1], df["WHP"])
    ax.set_ylabel("WHP (psi)")
    ax.set_title("Wellhead Pressure (WHP)")
    ax.legend(loc="best", fontsize=9)

    # ── Panel 3: FLP ──────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, df["FLP"], color=COLORS["FLP"], linewidth=1.8)
    _add_constraint_band(ax, FLP_LIMITS[0], FLP_LIMITS[1], df["FLP"])
    ax.set_ylabel("FLP (psi)")
    ax.set_title("Flowline Pressure (FLP)")
    ax.legend(loc="best", fontsize=9)

    # ── Panel 4: BHP ──────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, df["BHP"], color=COLORS["BHP"], linewidth=1.8)
    _add_constraint_band(ax, BHP_LIMITS[0], BHP_LIMITS[1], df["BHP"])
    ax.set_ylabel("BHP (psi)")
    ax.set_title("Bottom-Hole Pressure (BHP)")
    ax.legend(loc="best", fontsize=9)

    # ── Panel 5: Choke Position ───────────────────────────────────────
    ax = axes[4]
    ax.step(t, df["choke_cmd"], where="post", color=COLORS["choke"],
            linewidth=2.0)
    ax.set_ylabel("Choke (%)")
    ax.set_xlabel("Time (hours)")
    ax.set_title("Choke Position (Controller Output)")
    ax.set_ylim(-5, 105)

    # Annotate ramp rate limit
    ax.text(0.98, 0.95, f"Max ramp rate: ±{RAMP_RATE:.0f}%/hr",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                      edgecolor=COLORS["choke"], alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f"scenario_{scenario_label[0].lower()}_final.png"
    fpath = os.path.join(save_dir, fname)
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [final] Saved: {fpath}")
    return fpath


# ====================================================================== #
#  Summary dashboard                                                      #
# ====================================================================== #

def plot_summary_dashboard(logs: dict, save_dir: str) -> str:
    """
    Generate a single-page summary dashboard comparing all scenarios.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Autonomous Choke Controller — Scenario Comparison Dashboard",
                 fontsize=16, fontweight="bold")

    scenario_info = {
        "A": ("Startup to Target", "#1976D2"),
        "B": ("Target Tracking",   "#388E3C"),
        "C": ("Infeasible Target", "#C2185B"),
    }

    # Left column: Q vs Target for each scenario
    for idx, (key, (label, color)) in enumerate(scenario_info.items()):
        ax = axes[idx, 0]
        df = logs[key]
        ax.plot(df["t"], df["Q"], color=color, linewidth=1.8, label="Actual Q")
        ax.step(df["t"], df["target_Q"], color="#999", linewidth=1.2,
                linestyle="--", where="post", label="Target")
        ax.set_ylabel("Q (bbl/hr)")
        ax.set_title(f"Scenario {key}: {label}")
        ax.legend(fontsize=8, loc="lower right")
        if idx == 2:
            ax.set_xlabel("Time (hours)")

    # Right column: Choke position for each scenario
    for idx, (key, (label, color)) in enumerate(scenario_info.items()):
        ax = axes[idx, 1]
        df = logs[key]
        ax.step(df["t"], df["choke_cmd"], where="post", color=color, linewidth=1.8)
        ax.set_ylabel("Choke (%)")
        ax.set_ylim(-5, 105)
        ax.set_title(f"Scenario {key}: Choke Position")
        if idx == 2:
            ax.set_xlabel("Time (hours)")

    plt.tight_layout()
    fpath = os.path.join(save_dir, "summary_dashboard.png")
    plt.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [final] Saved: {fpath}")
    return fpath


# ====================================================================== #
#  Performance metrics table                                              #
# ====================================================================== #

def print_performance_table(logs: dict) -> str:
    """Print and return a markdown-formatted performance summary table."""
    rows = []
    for key in ["A", "B", "C"]:
        df = logs[key]
        final_Q = df["Q"].iloc[-1]
        final_target = df["target_Q"].iloc[-1]
        settle = _compute_settling_time(df)
        max_ramp = df["choke_cmd"].diff().abs().max()
        final_choke = df["choke_cmd"].iloc[-1]

        # Constraint compliance
        whp_ok = (df["WHP"] >= WHP_LIMITS[0] - 1).all() and (df["WHP"] <= WHP_LIMITS[1] + 1).all()
        flp_ok = (df["FLP"] >= FLP_LIMITS[0] - 1).all() and (df["FLP"] <= FLP_LIMITS[1] + 1).all()
        bhp_ok = (df["BHP"] >= BHP_LIMITS[0] - 1).all() and (df["BHP"] <= BHP_LIMITS[1] + 1).all()
        ramp_ok = max_ramp <= RAMP_RATE + 0.01

        all_ok = whp_ok and flp_ok and bhp_ok and ramp_ok

        rows.append({
            "Scenario": key,
            "Final Q": f"{final_Q:.1f}",
            "Target Q": f"{final_target:.1f}",
            "Settling": settle,
            "Final Choke": f"{final_choke:.1f}%",
            "Max Ramp": f"{max_ramp:.1f}%",
            "Constraints": "PASS" if all_ok else "FAIL",
        })

    table = pd.DataFrame(rows)
    table_str = table.to_markdown(index=False)
    print("\n" + table_str)
    return table_str


# ====================================================================== #
#  CLI entry point                                                        #
# ====================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate final publication-quality plots."
    )
    parser.add_argument(
        "--sim-source",
        choices=["mock", "real"],
        default="mock",
        help="Which scenario logs to use (default: mock).",
    )
    args = parser.parse_args()

    suffix = "" if args.sim_source == "mock" else "_real"
    label = args.sim_source.capitalize()
    save_dir = os.path.join(PROJECT_ROOT, "results", "final")

    print("=" * 60)
    print(f"  Generating Final Plots ({label} Data)")
    print("=" * 60)

    scenarios = {
        "A": "Controller brings the well from startup (choke=0%) to target Q=120 bbl/hr.",
        "B": "Target changes from 100 → 150 bbl/hr at t=24 hr.",
        "C": "Target Q=500 bbl/hr exceeds max safe rate → controller settles at max safe.",
    }

    logs = {}
    for key, desc in scenarios.items():
        csv_path = os.path.join(PROJECT_ROOT, "results",
                                f"scenario_{key.lower()}{suffix}_log.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        logs[key] = df
        plot_scenario_final(df, f"{key} — {desc.split('.')[0]}", desc, save_dir)

    if len(logs) == 3:
        plot_summary_dashboard(logs, save_dir)
        print_performance_table(logs)

    print(f"\n[final] All plots saved to {save_dir}")
    print("[final] Done.")
