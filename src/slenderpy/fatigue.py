"""Cycle counting of simulation results for fatigue post-processing.

Counts the vibration cycles of a position-dependent output of a cable or
beam run at a fatigue position given as a distance from a support,
e.g. the Poffenberger-Swart point 89 mm from a clamp. Stress models, S-N
curves and damage are out of scope: the counted cycles feed a separate
package.

The counting is ported from :mod:`slenderpy.fatigue`, without its Goodman
correction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import slenderpy.simulation as simulation
from slenderpy.components import Span

# Output columns of count_cycles.
_COLUMNS = ["range", "mean", "count"]

# Weight of a residual half cycle (legacy uc_mult default).
_HALF_CYCLE = 0.5


def _compress(signal):
    """Indices of the turning points of a signal, both ends included.

    Repeated consecutive samples are collapsed first, keeping the first of
    each run: the legacy version kept them, and the zero range they form made
    the rainflow stack drop the true peak, losing cycles.
    """
    # prepending nan keeps the first sample, nan differing from everything
    distinct = np.flatnonzero(np.diff(signal, prepend=np.nan) != 0.0)
    values = signal[distinct]
    before = np.sign(values[1:-1] - values[:-2])
    after = np.sign(values[2:] - values[1:-1])
    turning = np.where(before != after)[0]
    return distinct[np.concatenate(([0], 1 + turning, [len(values) - 1]))]


def _rainflow(turning_points):
    """Rainflow count of a sequence of turning points.

    Four-point stack algorithm of the legacy counter (ASTM E1049-85): a range
    that holds the first point of the stack is a half cycle, any other closed
    range a full cycle, and the ranges left on the stack at the end are half
    cycles. On the output of :func:`_compress`, which collapses repeated
    samples, a zero range cannot occur; zero ranges are still skipped, as a
    guard for direct calls.

    Parameters
    ----------
    turning_points : numpy.ndarray
        Turning points of the signal, see :func:`_compress`.

    Returns
    -------
    list of tuple
        ``(range, mean, count)`` per extracted cycle, in extraction order;
        ``count`` is 1 for a full cycle and 0.5 for a half cycle.
    """
    rows = []
    stack = []
    for point in turning_points:
        stack.append(float(point))
        while len(stack) >= 3 and abs(stack[-2] - stack[-3]) <= abs(
            stack[-1] - stack[-2]
        ):
            lrange = abs(stack[-2] - stack[-3])
            mean = 0.5 * (stack[-2] + stack[-3])
            if len(stack) == 3:
                count = _HALF_CYCLE
                del stack[0]
            else:
                count = 1.0
                del stack[-3:-1]
            if lrange > 0.0:
                rows.append((lrange, mean, count))

    for first, second in zip(stack, stack[1:]):
        lrange = abs(first - second)
        if lrange > 0.0:
            rows.append((lrange, 0.5 * (first + second), _HALF_CYCLE))

    return rows


def count_cycles(
    res: simulation.Results,
    variable: str,
    span: Span,
    distance: float,
    support: str = "left",
) -> pd.DataFrame:
    """Rainflow cycles of a result at a distance from a support.

    The variable is read at the span fraction ``distance / span.length`` from
    support 1 (``"left"``) or ``1 - distance / span.length`` from support 2
    (``"right"``), interpolated linearly between the two stored positions
    around it, so the signal is only as accurate as ``parameters.los`` is
    dense there: store the fatigue position itself in ``los`` when possible.
    It is then compressed to its turning points and counted with the
    rainflow method (ASTM E1049-85), without mean correction.

    Both supports sit at a constant height, so counting ``z`` near a clamp
    gives the Poffenberger-Swart ``Yb`` ranges; the means carry the static
    offset.

    Parameters
    ----------
    res : simulation.Results
        Results of a cable or beam run.
    variable : str
        Position-dependent variable of ``res``, e.g. ``"z"``, ``"y"``,
        ``"curvature"``, ``"moment"``.
    span : Span
        Span of the run; only its length is used.
    distance : float
        Horizontal distance (m) from the support to the fatigue position.
    support : str, optional
        ``"left"`` (support 1) or ``"right"`` (support 2). Default ``"left"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``range``, ``mean`` and ``count``, one row per extracted
        cycle (``count`` 1) or half cycle (``count`` 0.5), in extraction
        order and not aggregated: ``frame.groupby("range")["count"].sum()``
        gives the totals. Empty, with the same columns, when the signal holds
        no cycle.

    Raises
    ------
    ValueError
        If ``variable`` is not a position-dependent variable of ``res``, if
        ``support`` is not ``"left"`` or ``"right"``, if ``distance`` is not
        within ``[0, span.length]``, if the fatigue position lies outside the
        stored positions, or if the signal holds nan.
    """
    if variable not in res.lov():
        raise ValueError(f"variable {variable!r} is not in the results {res.lov()}")
    if res.lov_dims[variable] != 2:
        raise ValueError(f"variable {variable!r} does not depend on the position")
    if support not in ("left", "right"):
        raise ValueError(f"support must be 'left' or 'right', got {support!r}")
    if not (math.isfinite(distance) and 0.0 <= distance <= span.length):
        raise ValueError(
            f"distance must be within [0, {span.length}] m, got {distance}"
        )

    fraction = distance / span.length
    if support == "right":
        fraction = 1.0 - fraction

    stored = np.asarray(res.los(), dtype=float)
    # a stored position up to round-off is that position, as in Results.drop
    match = np.flatnonzero(
        np.isclose(stored, fraction, rtol=0.0, atol=simulation._POSITION_ATOL)
    )
    if match.size > 0:
        fraction = stored[match[0]]
    if stored.size == 0 or not stored[0] <= fraction <= stored[-1]:
        raise ValueError(
            f"fatigue position {fraction} (span fraction) is outside the stored "
            f"positions {stored.tolist()}"
        )

    values = res[variable].values
    if stored.size == 1:
        signal = values[:, 0]
    else:
        # linear interpolation between the two stored positions around it,
        # the same as np.interp at every time
        upper = min(max(int(np.searchsorted(stored, fraction)), 1), stored.size - 1)
        lower = upper - 1
        weight = (fraction - stored[lower]) / (stored[upper] - stored[lower])
        signal = (1.0 - weight) * values[:, lower] + weight * values[:, upper]

    if np.isnan(signal).any():
        raise ValueError(
            f"the {variable!r} signal holds nan (a run stopped early?): crop it "
            "first with Results.drop(tmax=...)"
        )

    rows = _rainflow(signal[_compress(signal)])
    return pd.DataFrame(rows, columns=_COLUMNS)
