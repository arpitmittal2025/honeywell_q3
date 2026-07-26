"""
Simulator Interface — Abstraction Layer
========================================

All code in this project interacts with the well simulator *only* through
this module.  When the real simulator is delivered it will be wrapped in a
new adapter class; no other file needs to change.

Classes
-------
SimulatorInterface  — Abstract base defining the contract.
MockSimulatorAdapter — Wraps the Phase-1 mock simulator.

Factory
-------
create_simulator(sim_type="mock") — Returns the requested adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from src.mock_simulator import MockSimulator, MockSimulatorConfig


# ====================================================================== #
#  Abstract interface                                                     #
# ====================================================================== #

class SimulatorInterface(ABC):
    """Contract that every simulator adapter must satisfy."""

    @abstractmethod
    def step(self, choke_position: float) -> Tuple[float, float, float, float]:
        """
        Advance the simulation by one control interval.

        Parameters
        ----------
        choke_position : float
            Choke opening in percent (0–100).

        Returns
        -------
        Q   : float — Oil flow rate (bbl/hr)
        WHP : float — Wellhead pressure (psi)
        FLP : float — Flowline pressure (psi)
        BHP : float — Bottom-hole pressure (psi)
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulator to initial conditions."""

    @abstractmethod
    def get_state(self) -> dict:
        """Return the current simulator state for inspection / logging."""


# ====================================================================== #
#  Mock adapter                                                           #
# ====================================================================== #

class MockSimulatorAdapter(SimulatorInterface):
    """Wraps :class:`MockSimulator` behind the standard interface."""

    def __init__(self, config: MockSimulatorConfig | None = None):
        self._sim = MockSimulator(config)

    def step(self, choke_position: float) -> Tuple[float, float, float, float]:
        return self._sim.step(choke_position)

    def reset(self) -> None:
        self._sim.reset()

    def get_state(self) -> dict:
        return self._sim.get_state()


# ====================================================================== #
#  Factory                                                                #
# ====================================================================== #

def create_simulator(sim_type: str = "mock", **kwargs) -> SimulatorInterface:
    """
    Create and return a simulator adapter.

    Parameters
    ----------
    sim_type : str
        ``"mock"`` — Phase-1 mock simulator (default).
        ``"real"`` — Placeholder for the real simulator (Phase 2).

    Returns
    -------
    SimulatorInterface
    """
    if sim_type == "mock":
        config = MockSimulatorConfig(**kwargs) if kwargs else None
        return MockSimulatorAdapter(config)
    elif sim_type == "real":
        raise NotImplementedError(
            "Real simulator adapter not yet implemented. "
            "Waiting for real simulator delivery (Phase 2)."
        )
    else:
        raise ValueError(f"Unknown simulator type: {sim_type!r}")


# ====================================================================== #
#  Quick smoke test                                                       #
# ====================================================================== #
if __name__ == "__main__":
    sim = create_simulator("mock")
    sim.reset()
    for t in range(5):
        Q, WHP, FLP, BHP = sim.step(20.0)
        print(f"t={t}  Q={Q:.1f}  WHP={WHP:.0f}  FLP={FLP:.0f}  BHP={BHP:.0f}")
    print("State:", sim.get_state())
