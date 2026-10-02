"""stockbridge - model and solvers for Stockbridge dampers.

The package is organised in four sub-packages:

- [](`slenderpy.stockbridge.core`) - the domain model: [](`~slenderpy.stockbridge.__init__.Mass`), [](`~slenderpy.stockbridge.__init__.Clamp`),
  [](`~slenderpy.stockbridge.__init__.Stockbridge`), the [](`~slenderpy.stockbridge.__init__.Side`) enum and the
  parameter dataclasses.
- [](`slenderpy.stockbridge.solvers`) - free-function solvers
  ([](`~slenderpy.stockbridge.__init__.solve_imposed_force`), [](`~slenderpy.stockbridge.__init__.solve_imposed_acceleration`),
  [](`~slenderpy.stockbridge.__init__.solve_linearized_imposed_force`)).
- [](`slenderpy.stockbridge.plotting`) - matplotlib plotting helpers.
- [](`slenderpy.stockbridge.coupling`) - coupling with the beam model.

The most common symbols are re-exported here for convenience.
"""

from .core.clamp import Clamp
from .core.mass import Mass
from .core.parameters import (
    ClampParameters,
    MassParameters,
    MessengerCableParameters,
)
from .core.side import Side
from .core.stockbridge import Result, Stockbridge
from .coupling.beam_coupling import solve_dynamic_with_sb
from .plotting import (
    plot_clamp,
    plot_clamp_all_versions,
    plot_mass,
    plot_spectrum,
)
from .solvers.imposed_acceleration import solve_imposed_acceleration
from .solvers.imposed_force import solve_imposed_force
from .solvers.linearized import solve_linearized_imposed_force

__all__ = [
    "Clamp",
    "Mass",
    "ClampParameters",
    "MassParameters",
    "MessengerCableParameters",
    "Side",
    "Stockbridge",
    "Result",
    "plot_clamp",
    "plot_clamp_all_versions",
    "plot_mass",
    "plot_spectrum",
    "solve_imposed_acceleration",
    "solve_imposed_force",
    "solve_linearized_imposed_force",
    "solve_dynamic_with_sb",
]
