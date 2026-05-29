"""stockbridge - model and solvers for Stockbridge dampers.

The package is organised in four sub-packages:

- :mod:`stockbridge.core` - the domain model: :class:`Mass`, :class:`Clamp`,
  :class:`Stockbridge`, the :class:`Side` enum and the
  parameter dataclasses.
- :mod:`stockbridge.solvers` - free-function solvers
  (:func:`solve_imposed_force`, :func:`solve_imposed_acceleration`,
  :func:`solve_linearized_imposed_force`).
- :mod:`stockbridge.plotting` - matplotlib plotting helpers.
- :mod:`stockbridge.coupling` - coupling with the beam model.

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
from .core.stockbridge import Stockbridge, Result
from .plotting import (
    plot_clamp,
    plot_clamp_all_versions,
    plot_mass,
    plot_spectrum,
)
from .solvers.imposed_acceleration import solve_imposed_acceleration
from .solvers.imposed_force import solve_imposed_force
from .solvers.linearized import solve_linearized_imposed_force
from .coupling.beam_coupling import solve_dynamic_with_sb