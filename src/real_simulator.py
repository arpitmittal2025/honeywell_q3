"""
Real Simulator Adapter
=======================

Wraps the *real* well simulator (once delivered) behind the same
:class:`SimulatorInterface` contract used by the rest of the project.

How to integrate the real simulator
------------------------------------
1. Place the real simulator file(s) in this directory (``src/``).
2. Update the import below to point at the real module.
3. Adjust ``__init__`` / ``step`` / ``reset`` if the real API differs
   from the expected ``Q, WHP, FLP, BHP = sim.step(choke_position)``
   signature.

Everything else in the project (controller, step tests, scenarios)
will work unchanged — they only call through
``simulator_interface.create_simulator("real")``.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

# Allow running as ``python src/real_simulator.py`` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulator_interface import SimulatorInterface


# ====================================================================== #
#  TODO — Replace this block with the real simulator import               #
# ====================================================================== #
#
#  Example (once the real simulator is available):
#
#      from src.honeywell_simulator import HoneywellWellSimulator
#
#  Then update RealSimulatorAdapter.__init__ to instantiate it.
#
# ====================================================================== #


class RealSimulatorAdapter(SimulatorInterface):
    """
    Adapter wrapping the real well simulator.

    Currently a skeleton — the body of each method shows exactly what
    needs to be wired once the real simulator module is provided.
    """

    def __init__(self, **kwargs):
        """
        Initialize the real simulator.

        Parameters
        ----------
        **kwargs
            Forwarded to the real simulator's constructor.  Adjust as
            needed once the real API is known.
        """
        # ------------------------------------------------------------- #
        #  TODO: Replace the block below with real simulator init.       #
        #                                                                #
        #  Example:                                                      #
        #      self._sim = HoneywellWellSimulator(**kwargs)              #
        # ------------------------------------------------------------- #
        self._sim = None  # placeholder
        self._choke = 0.0
        self._t = 0

        # Validate that the real simulator was loaded
        if self._sim is None:
            raise NotImplementedError(
                "Real simulator not yet integrated.\n"
                "To integrate:\n"
                "  1. Place the real simulator module in src/\n"
                "  2. Update the import at the top of real_simulator.py\n"
                "  3. Set self._sim in __init__ to an instance of the "
                "real simulator\n"
                "  4. Adjust step() / reset() / get_state() if the API "
                "differs."
            )

    # ------------------------------------------------------------------ #
    #  SimulatorInterface methods                                         #
    # ------------------------------------------------------------------ #

    def step(self, choke_position: float) -> Tuple[float, float, float, float]:
        """
        Advance the real simulator by one control interval.

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
        self._choke = float(choke_position)
        self._t += 1

        # ------------------------------------------------------------- #
        #  TODO: Call the real simulator here.                           #
        #                                                                #
        #  If the real API matches the expected signature:               #
        #      Q, WHP, FLP, BHP = self._sim.step(choke_position)        #
        #      return Q, WHP, FLP, BHP                                  #
        #                                                                #
        #  If the real API returns a dict:                               #
        #      result = self._sim.step(choke_position)                   #
        #      return (result["Q"], result["WHP"],                       #
        #              result["FLP"], result["BHP"])                     #
        # ------------------------------------------------------------- #
        Q, WHP, FLP, BHP = self._sim.step(choke_position)
        return Q, WHP, FLP, BHP

    def reset(self) -> None:
        """Reset the real simulator to initial conditions."""
        self._choke = 0.0
        self._t = 0

        # ------------------------------------------------------------- #
        #  TODO: Call the real simulator's reset.                        #
        #      self._sim.reset()                                        #
        # ------------------------------------------------------------- #
        self._sim.reset()

    def get_state(self) -> dict:
        """Return the current simulator state for inspection / logging."""
        # ------------------------------------------------------------- #
        #  TODO: Adapt if the real sim exposes state differently.        #
        # ------------------------------------------------------------- #
        return {
            "t": self._t,
            "choke": self._choke,
            "adapter": "real",
        }


# ====================================================================== #
#  Quick smoke test                                                       #
# ====================================================================== #
if __name__ == "__main__":
    try:
        from src.real_simulator import RealSimulatorAdapter
        sim = RealSimulatorAdapter()
        sim.reset()
        for t in range(5):
            Q, WHP, FLP, BHP = sim.step(20.0)
            print(f"t={t}  Q={Q:.1f}  WHP={WHP:.0f}  FLP={FLP:.0f}  BHP={BHP:.0f}")
    except NotImplementedError as e:
        print(f"[real_simulator] Expected error (real sim not yet provided):\n{e}")
        print("\n[real_simulator] Skeleton is correctly wired. Ready for drop-in.")
