"""
Autonomous Choke Controller
============================

Brute-force MPC-style single-step-horizon controller.

Algorithm
---------
1. Generate candidate choke moves within ±5% of current position (0.5% resolution).
2. Predict next-step Q, WHP, FLP, BHP for each candidate using the FOPDT model.
3. Reject candidates that violate pressure constraints.
4. Among feasible candidates:
   a. Pick the one closest to the target Q.
   b. If target is unreachable, pick the one maximizing Q (max safe rate).
5. If no candidates are feasible, hold current position (safety).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

from src.model_identification import WellModel


# ====================================================================== #
#  Configuration                                                          #
# ====================================================================== #

@dataclass
class ControllerConfig:
    """Controller tuning parameters and constraints."""

    # Production target
    target_Q: float = 120.0  # bbl/hr

    # Pressure constraints (min, max)
    whp_limits: Tuple[float, float] = (200.0, 3000.0)
    flp_limits: Tuple[float, float] = (150.0, 2500.0)
    bhp_limits: Tuple[float, float] = (2000.0, 5000.0)

    # Choke constraints
    choke_min: float = 0.0
    choke_max: float = 100.0
    max_ramp_rate: float = 5.0        # % per control interval

    # Candidate generation
    candidate_resolution: float = 0.5  # % increments


# ====================================================================== #
#  Decision record (for logging / analysis)                               #
# ====================================================================== #

@dataclass
class ControlDecision:
    """Records the controller's reasoning for one time step."""
    t: int
    current_choke: float
    selected_choke: float
    target_Q: float
    predicted_Q: float
    predicted_WHP: float
    predicted_FLP: float
    predicted_BHP: float
    n_candidates: int
    n_feasible: int
    reason: str  # "target_achieved", "max_safe_rate", "hold_position"


# ====================================================================== #
#  Controller                                                             #
# ====================================================================== #

class ChokeController:
    """
    Autonomous choke controller for a single naturally flowing well.

    Parameters
    ----------
    model : WellModel
        Fitted dynamic model used for one-step-ahead prediction.
    config : ControllerConfig
        Controller settings (target, constraints, resolution).
    """

    def __init__(self, model: WellModel, config: ControllerConfig):
        self.model = model
        self.config = config
        self._choke_history: List[float] = [0.0]  # start at closed
        self._t = 0

    @property
    def current_choke(self) -> float:
        return self._choke_history[-1]

    def set_target(self, target_Q: float) -> None:
        """Update the production target mid-run."""
        self.config.target_Q = target_Q

    def _generate_candidates(self, current_choke: float) -> np.ndarray:
        """
        Generate candidate choke positions within ±ramp_rate of current.
        """
        cfg = self.config
        lo = max(cfg.choke_min, current_choke - cfg.max_ramp_rate)
        hi = min(cfg.choke_max, current_choke + cfg.max_ramp_rate)

        # Generate with specified resolution
        n_steps = int(round((hi - lo) / cfg.candidate_resolution)) + 1
        candidates = np.linspace(lo, hi, n_steps)

        return candidates

    def _check_constraints(self, predicted: Dict[str, float]) -> bool:
        """Return True if predicted values satisfy all pressure constraints."""
        cfg = self.config

        if not (cfg.whp_limits[0] <= predicted["WHP"] <= cfg.whp_limits[1]):
            return False
        if not (cfg.flp_limits[0] <= predicted["FLP"] <= cfg.flp_limits[1]):
            return False
        if not (cfg.bhp_limits[0] <= predicted["BHP"] <= cfg.bhp_limits[1]):
            return False

        return True

    def compute_action(
        self,
        current_state: Dict[str, float],
    ) -> ControlDecision:
        """
        Determine the next choke position.

        Parameters
        ----------
        current_state : dict
            Current measured {Q, WHP, FLP, BHP}.

        Returns
        -------
        ControlDecision
            The selected action and reasoning.
        """
        cfg = self.config
        current_choke = self.current_choke
        candidates = self._generate_candidates(current_choke)
        n_candidates = len(candidates)

        # Evaluate each candidate
        feasible = []  # (choke, predicted_state)
        for u_candidate in candidates:
            predicted = self.model.predict_next(
                current_state,
                self._choke_history,
                u_candidate,
            )
            if self._check_constraints(predicted):
                feasible.append((u_candidate, predicted))

        n_feasible = len(feasible)

        if n_feasible == 0:
            # SAFETY: no feasible move -> hold position
            predicted = self.model.predict_next(
                current_state, self._choke_history, current_choke,
            )
            decision = ControlDecision(
                t=self._t,
                current_choke=current_choke,
                selected_choke=current_choke,
                target_Q=cfg.target_Q,
                predicted_Q=predicted["Q"],
                predicted_WHP=predicted["WHP"],
                predicted_FLP=predicted["FLP"],
                predicted_BHP=predicted["BHP"],
                n_candidates=n_candidates,
                n_feasible=0,
                reason="hold_position",
            )
        else:
            # Find candidate closest to target Q
            best_choke = None
            best_pred = None
            best_error = np.inf

            for u_c, pred in feasible:
                error = abs(pred["Q"] - cfg.target_Q)
                if error < best_error:
                    best_error = error
                    best_choke = u_c
                    best_pred = pred

            # Check if we're actually achieving target or just maximizing Q
            # If target_Q > max predicted Q among feasible, we're at max safe rate
            max_feasible_Q = max(pred["Q"] for _, pred in feasible)

            if cfg.target_Q > max_feasible_Q * 1.05:
                # Target is infeasible — select max Q candidate
                for u_c, pred in feasible:
                    if pred["Q"] == max_feasible_Q:
                        best_choke = u_c
                        best_pred = pred
                        break
                reason = "max_safe_rate"
            else:
                reason = "target_achieved"

            decision = ControlDecision(
                t=self._t,
                current_choke=current_choke,
                selected_choke=best_choke,
                target_Q=cfg.target_Q,
                predicted_Q=best_pred["Q"],
                predicted_WHP=best_pred["WHP"],
                predicted_FLP=best_pred["FLP"],
                predicted_BHP=best_pred["BHP"],
                n_candidates=n_candidates,
                n_feasible=n_feasible,
                reason=reason,
            )

        # Update history
        self._choke_history.append(decision.selected_choke)
        self._t += 1

        return decision

    def reset(self, initial_choke: float = 0.0) -> None:
        """Reset controller state."""
        self._choke_history = [initial_choke]
        self._t = 0


# ====================================================================== #
#  Quick self-test                                                        #
# ====================================================================== #

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.model_identification import WellModel, VariableModel

    # Create a simple hand-tuned model for testing
    model = WellModel()
    model.models["Q"]   = VariableModel("Q",   "power", [300.0, 0.5, 0.0],   tau=3.0)
    model.models["WHP"] = VariableModel("WHP", "power", [-1500.0, 0.5, 2800.0], tau=3.0)
    model.models["FLP"] = VariableModel("FLP", "power", [600.0, 0.5, 200.0],  tau=3.0)
    model.models["BHP"] = VariableModel("BHP", "power", [-900.0, 0.5, 4500.0], tau=3.0)

    config = ControllerConfig(target_Q=120.0)
    ctrl = ChokeController(model, config)

    state = {"Q": 0.0, "WHP": 2800.0, "FLP": 200.0, "BHP": 4500.0}

    print("Step | Choke -> Choke | Pred Q  | Reason")
    print("-" * 55)
    for t in range(20):
        decision = ctrl.compute_action(state)
        print(f" {t:3d}  | {decision.current_choke:5.1f} -> {decision.selected_choke:5.1f} "
              f"| {decision.predicted_Q:7.2f} | {decision.reason}")
        # Simulate crude state update for self-test
        state["Q"] = decision.predicted_Q
        state["WHP"] = decision.predicted_WHP
        state["FLP"] = decision.predicted_FLP
        state["BHP"] = decision.predicted_BHP

