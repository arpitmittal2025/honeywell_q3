"""
Dynamic Model Identification
=============================

Model approach: **Steady-state curve + First-order dynamics**

Rather than a pure linear FOPDT (which fails on the nonlinear sqrt
relationship), we use:

1. A nonlinear steady-state mapping f(u) fitted to step-test endpoints.
2. First-order dynamic filter: y(k+1) = alpha * y(k) + (1-alpha) * f_ss(u)

This provides good prediction for the controller because:
- The steady-state curve captures the nonlinear choke-to-output relationship.
- The first-order filter captures the dynamic lag.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List
from scipy.optimize import curve_fit, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ====================================================================== #
#  Steady-state curve models                                              #
# ====================================================================== #

def _ss_power(u, a, b, c):
    """Power-law steady-state:  y = a * (u/100)^b + c"""
    return a * np.power(np.clip(u / 100.0, 1e-6, 1.0), b) + c


def _ss_linear(u, a, b):
    """Linear steady-state:  y = a * u + b"""
    return a * u + b


# ====================================================================== #
#  Model parameters                                                       #
# ====================================================================== #

@dataclass
class VariableModel:
    """Model for one output variable: steady-state curve + time constant."""
    variable: str
    ss_type: str          # "power" or "linear"
    ss_params: list       # parameters for the steady-state function
    tau: float            # time constant (hours)
    Ts: float = 1.0       # sampling interval

    @property
    def alpha(self) -> float:
        """First-order decay factor."""
        return np.exp(-self.Ts / max(self.tau, 0.01))

    def steady_state(self, u: float) -> float:
        """Compute steady-state value for a choke opening."""
        if self.ss_type == "power":
            return _ss_power(u, *self.ss_params)
        else:
            return _ss_linear(u, *self.ss_params)

    def predict_one_step(self, y_current: float, u: float) -> float:
        """One-step-ahead prediction with first-order dynamics."""
        y_ss = self.steady_state(u)
        return self.alpha * y_current + (1.0 - self.alpha) * y_ss


@dataclass
class WellModel:
    """Complete dynamic model for all output variables."""
    models: Dict[str, VariableModel] = field(default_factory=dict)

    def predict_next(
        self,
        current_state: Dict[str, float],
        choke_history: List[float],
        candidate_choke: float,
    ) -> Dict[str, float]:
        """
        Predict the next-step values given current state and a candidate
        choke move.

        Parameters
        ----------
        current_state : dict
            Keys: Q, WHP, FLP, BHP -- current measured values.
        choke_history : list of float
            Recent choke positions (newest last).
        candidate_choke : float
            The proposed choke position (%) for this step.

        Returns
        -------
        dict
            Predicted {Q, WHP, FLP, BHP} at next time step.
        """
        predictions = {}
        for var, m in self.models.items():
            y_current = current_state[var]
            y_next = m.predict_one_step(y_current, candidate_choke)
            predictions[var] = y_next
        return predictions

    # Keep backward-compatible attribute name
    @property
    def params(self):
        return self.models

    def save(self, filepath: str) -> None:
        """Save model parameters to JSON."""
        data = {}
        for var, m in self.models.items():
            data[var] = {
                "ss_type": m.ss_type,
                "ss_params": m.ss_params,
                "tau": m.tau,
                "Ts": m.Ts,
            }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[model] Saved model -> {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "WellModel":
        """Load model parameters from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        model = cls()
        for var, d in data.items():
            model.models[var] = VariableModel(
                variable=var,
                ss_type=d["ss_type"],
                ss_params=d["ss_params"],
                tau=d["tau"],
                Ts=d["Ts"],
            )
        return model


# ====================================================================== #
#  Fitting engine                                                         #
# ====================================================================== #

def _extract_steady_state(df: pd.DataFrame, hold_intervals: int = 15) -> pd.DataFrame:
    """
    Extract steady-state values by taking the last reading at each
    choke setting from the ascending staircase.
    """
    # Detect choke changes and group
    choke_vals = df["choke"].unique()
    # Take ascending portion only (first occurrence of each level)
    ss_records = []
    for choke in sorted(choke_vals):
        subset = df[df["choke"] == choke]
        # Take the last few readings and average them (reduces noise)
        tail = subset.tail(min(5, len(subset)))
        ss_records.append({
            "choke": choke,
            "Q": tail["Q"].mean(),
            "WHP": tail["WHP"].mean(),
            "FLP": tail["FLP"].mean(),
            "BHP": tail["BHP"].mean(),
        })
    return pd.DataFrame(ss_records)


def _fit_steady_state(choke: np.ndarray, y: np.ndarray, var_name: str):
    """Fit the steady-state curve (power-law or linear)."""
    # Try power-law fit
    try:
        if var_name == "Q":
            # Q: a * (u/100)^b + c, expect a>0, 0<b<1
            popt, _ = curve_fit(_ss_power, choke, y,
                                p0=[300.0, 0.5, 0.0],
                                bounds=([0, 0.01, -100], [1000, 2.0, 100]),
                                maxfev=5000)
            y_pred = _ss_power(choke, *popt)
        elif var_name in ("WHP", "BHP"):
            # Decreasing: a<0
            popt, _ = curve_fit(_ss_power, choke, y,
                                p0=[-1500.0, 0.5, y[0]],
                                bounds=([-5000, 0.01, 0], [0, 2.0, 10000]),
                                maxfev=5000)
            y_pred = _ss_power(choke, *popt)
        else:  # FLP: increasing
            popt, _ = curve_fit(_ss_power, choke, y,
                                p0=[600.0, 0.5, y[0]],
                                bounds=([0, 0.01, 0], [5000, 2.0, 5000]),
                                maxfev=5000)
            y_pred = _ss_power(choke, *popt)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        return "power", list(popt), r2
    except Exception:
        # Fallback to linear
        popt, _ = curve_fit(_ss_linear, choke, y)
        y_pred = _ss_linear(choke, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        return "linear", list(popt), r2


def _fit_time_constant(df: pd.DataFrame, var: str, ss_func, ss_params,
                       ss_type: str, Ts: float = 1.0) -> float:
    """
    Fit the time constant tau by simulating the dynamic response and
    minimizing the prediction error.
    """
    u = df["choke"].values
    y = df[var].values
    y0 = y[0]

    def ss_eval(u_val):
        if ss_type == "power":
            return _ss_power(u_val, *ss_params)
        else:
            return _ss_linear(u_val, *ss_params)

    def simulate(tau):
        alpha = np.exp(-Ts / max(tau, 0.01))
        y_sim = np.zeros(len(u))
        y_sim[0] = y0
        for k in range(1, len(u)):
            y_ss = ss_eval(u[k])
            y_sim[k] = alpha * y_sim[k - 1] + (1.0 - alpha) * y_ss
        return y_sim

    def objective(tau):
        y_sim = simulate(tau)
        return np.sum((y - y_sim) ** 2)

    result = minimize_scalar(objective, bounds=(0.1, 20.0), method="bounded")
    return result.x


def fit_all(df: pd.DataFrame, Ts: float = 1.0) -> WellModel:
    """
    Fit models for Q, WHP, FLP, BHP from step-test data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: choke, Q, WHP, FLP, BHP

    Returns
    -------
    WellModel
        Fitted model with parameters for all four variables.
    """
    model = WellModel()

    # Extract steady-state data
    ss_df = _extract_steady_state(df)
    choke_ss = ss_df["choke"].values

    for var in ["Q", "WHP", "FLP", "BHP"]:
        y_ss = ss_df[var].values

        print(f"[model] Fitting {var}...")

        # 1. Fit steady-state curve
        ss_type, ss_params, r2_ss = _fit_steady_state(choke_ss, y_ss, var)
        print(f"  -> SS curve ({ss_type}): params={[f'{p:.3f}' for p in ss_params]}, R^2={r2_ss:.4f}")

        # 2. Fit time constant
        tau = _fit_time_constant(df, var, None, ss_params, ss_type, Ts)
        print(f"  -> Time constant tau = {tau:.2f} hr")

        # 3. Compute overall dynamic fit R^2
        vm = VariableModel(var, ss_type, ss_params, tau, Ts)
        u_all = df["choke"].values
        y_all = df[var].values
        y_pred = np.zeros(len(u_all))
        y_pred[0] = y_all[0]
        for k in range(1, len(u_all)):
            y_pred[k] = vm.predict_one_step(y_pred[k - 1], u_all[k])
        ss_res = np.sum((y_all - y_pred) ** 2)
        ss_tot = np.sum((y_all - y_all.mean()) ** 2)
        r2_dyn = 1.0 - ss_res / max(ss_tot, 1e-12)
        print(f"  -> Dynamic fit R^2 = {r2_dyn:.4f}")

        model.models[var] = vm

    return model


# ====================================================================== #
#  Validation plotting                                                    #
# ====================================================================== #

def plot_model_validation(df: pd.DataFrame, model: WellModel,
                          save_dir: str | None = None) -> None:
    """Plot measured vs. model-predicted responses."""
    import matplotlib.pyplot as plt

    if save_dir is None:
        save_dir = os.path.join(PROJECT_ROOT, "results", "step_tests")
    os.makedirs(save_dir, exist_ok=True)

    u = df["choke"].values
    t = df["t"].values if "t" in df.columns else np.arange(len(df))

    colors = {"Q": "#2196F3", "WHP": "#FF9800", "FLP": "#4CAF50", "BHP": "#E91E63"}
    labels = {
        "Q": "Oil Flow Rate (bbl/hr)",
        "WHP": "Wellhead Pressure (psi)",
        "FLP": "Flowline Pressure (psi)",
        "BHP": "Bottom-Hole Pressure (psi)",
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Model Validation - Measured vs. Predicted", fontsize=16, fontweight="bold")

    for idx, var in enumerate(["Q", "WHP", "FLP", "BHP"]):
        ax = axes[idx]
        y_meas = df[var].values
        vm = model.models[var]

        # Simulate dynamic response
        y_pred = np.zeros(len(u))
        y_pred[0] = y_meas[0]
        for k in range(1, len(u)):
            y_pred[k] = vm.predict_one_step(y_pred[k - 1], u[k])

        ax.plot(t, y_meas, color=colors[var], alpha=0.6, linewidth=1.0, label="Measured")
        ax.plot(t, y_pred, color=colors[var], linewidth=2.0, linestyle="--", label="Model")
        ax.set_ylabel(labels[var], fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (hours)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_validation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[model] Saved validation plot -> {save_dir}/model_validation.png")


# ====================================================================== #
#  CLI entry point                                                        #
# ====================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("  Dynamic Model Identification")
    print("=" * 60)

    csv_path = os.path.join(PROJECT_ROOT, "data", "step_test_results.csv")
    if not os.path.exists(csv_path):
        print(f"[model] Step test data not found at {csv_path}")
        print("[model] Running step tests first...")
        from src.step_test import run_step_tests
        df = run_step_tests("mock")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    model = fit_all(df)

    # Save model
    model_path = os.path.join(PROJECT_ROOT, "data", "model_params.json")
    model.save(model_path)

    # Validation plot
    plot_model_validation(df, model)

    print("\n[model] Done.")
