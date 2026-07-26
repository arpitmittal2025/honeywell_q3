"""
Mock Simulator for a Naturally Flowing Oil Well
================================================

Provides a placeholder simulator with the *exact same interface* the real
simulator will have:

    Q, WHP, FLP, BHP = simulator.step(choke_position)

Physics (simplified, directionally correct):
  - Q increases with choke opening (square-root relationship, à la Gilbert).
  - WHP decreases as flow increases (less restriction → lower wellhead pressure).
  - FLP increases with flow (more flow through flowline → higher downstream P).
  - BHP decreases as flow increases (more drawdown).
  - First-order lag dynamics (time constant τ) applied to all outputs.
  - Small Gaussian measurement noise for realism.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MockSimulatorConfig:
    """Tunable parameters for the mock well model."""

    # Steady-state flow relationship
    Q_max: float = 300.0          # bbl/hr — max flow at 100% choke
    flow_exponent: float = 0.5    # sqrt relationship (Gilbert-like)

    # Wellhead Pressure: WHP = WHP_shutin - K_whp * Q
    WHP_shutin: float = 2800.0    # psi — shut-in WHP
    K_whp: float = 5.0            # psi / (bbl/hr)

    # Flowline Pressure: FLP = FLP_base + K_flp * Q
    FLP_base: float = 200.0       # psi — base FLP at zero flow
    K_flp: float = 2.0            # psi / (bbl/hr)

    # Bottom Hole Pressure: BHP = BHP_static - K_bhp * Q
    BHP_static: float = 4500.0    # psi — static reservoir pressure
    K_bhp: float = 3.0            # psi / (bbl/hr)

    # Dynamic response
    tau: float = 3.0              # hours — first-order time constant
    Ts: float = 1.0               # hours — sampling / control interval

    # Measurement noise (standard deviation as fraction of signal)
    noise_fraction: float = 0.005  # 0.5%

    # Random seed for reproducibility (None → random)
    seed: int | None = 42


class MockSimulator:
    """
    A first-order dynamic mock of a naturally flowing oil well.

    Usage
    -----
    >>> sim = MockSimulator()
    >>> Q, WHP, FLP, BHP = sim.step(choke_position=30.0)
    """

    def __init__(self, config: MockSimulatorConfig | None = None):
        self.cfg = config or MockSimulatorConfig()
        self._rng = np.random.default_rng(self.cfg.seed)

        # Decay factor for first-order lag: alpha = exp(-Ts / tau)
        self._alpha = np.exp(-self.cfg.Ts / self.cfg.tau)

        # Internal dynamic state (start at shut-in conditions)
        self._Q = 0.0
        self._WHP = self.cfg.WHP_shutin
        self._FLP = self.cfg.FLP_base
        self._BHP = self.cfg.BHP_static

        # Track current choke for external inspection
        self._choke = 0.0

        # Time step counter
        self._t = 0

    # ------------------------------------------------------------------ #
    #  Steady-state relationships                                         #
    # ------------------------------------------------------------------ #

    def _steady_state(self, choke_pct: float) -> Tuple[float, float, float, float]:
        """Compute steady-state values for a given choke opening (0-100%)."""
        u = np.clip(choke_pct, 0.0, 100.0) / 100.0  # normalise to 0-1

        Q_ss = self.cfg.Q_max * (u ** self.cfg.flow_exponent)
        WHP_ss = self.cfg.WHP_shutin - self.cfg.K_whp * Q_ss
        FLP_ss = self.cfg.FLP_base + self.cfg.K_flp * Q_ss
        BHP_ss = self.cfg.BHP_static - self.cfg.K_bhp * Q_ss

        return Q_ss, WHP_ss, FLP_ss, BHP_ss

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def step(self, choke_position: float) -> Tuple[float, float, float, float]:
        """
        Advance the simulator by one control interval.

        Parameters
        ----------
        choke_position : float
            Choke opening in percent (0–100).

        Returns
        -------
        Q : float   — Oil flow rate (bbl/hr)
        WHP : float — Wellhead pressure (psi)
        FLP : float — Flowline pressure (psi)
        BHP : float — Bottom-hole pressure (psi)
        """
        choke_pct = float(np.clip(choke_position, 0.0, 100.0))
        self._choke = choke_pct

        # Target steady-state for new choke position
        Q_ss, WHP_ss, FLP_ss, BHP_ss = self._steady_state(choke_pct)

        # First-order lag update: y(k+1) = alpha * y(k) + (1-alpha) * y_ss
        a = self._alpha
        self._Q   = a * self._Q   + (1.0 - a) * Q_ss
        self._WHP = a * self._WHP + (1.0 - a) * WHP_ss
        self._FLP = a * self._FLP + (1.0 - a) * FLP_ss
        self._BHP = a * self._BHP + (1.0 - a) * BHP_ss

        # Add measurement noise
        noise = self.cfg.noise_fraction
        Q_out   = max(0.0, self._Q   + self._rng.normal(0, noise * max(self._Q, 1.0)))
        WHP_out = self._WHP + self._rng.normal(0, noise * self._WHP)
        FLP_out = self._FLP + self._rng.normal(0, noise * max(self._FLP, 1.0))
        BHP_out = self._BHP + self._rng.normal(0, noise * self._BHP)

        self._t += 1
        return Q_out, WHP_out, FLP_out, BHP_out

    def reset(self) -> None:
        """Reset the simulator to initial (shut-in) conditions."""
        self._Q = 0.0
        self._WHP = self.cfg.WHP_shutin
        self._FLP = self.cfg.FLP_base
        self._BHP = self.cfg.BHP_static
        self._choke = 0.0
        self._t = 0
        self._rng = np.random.default_rng(self.cfg.seed)

    def get_state(self) -> dict:
        """Return the current internal state for inspection."""
        return {
            "t": self._t,
            "choke": self._choke,
            "Q": self._Q,
            "WHP": self._WHP,
            "FLP": self._FLP,
            "BHP": self._BHP,
        }


# ====================================================================== #
#  Quick self-test                                                        #
# ====================================================================== #
if __name__ == "__main__":
    sim = MockSimulator()
    print("Step | Choke% |    Q    |   WHP   |   FLP   |   BHP")
    print("-" * 60)
    for t in range(20):
        choke = min(t * 5.0, 50.0)  # ramp from 0% to 50%
        Q, WHP, FLP, BHP = sim.step(choke)
        print(f" {t:3d}  | {choke:5.1f}  | {Q:7.2f} | {WHP:7.1f} | {FLP:7.1f} | {BHP:7.1f}")
