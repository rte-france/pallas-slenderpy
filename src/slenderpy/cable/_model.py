"""Pieces of the [Lee1992] cable model shared by the static and dynamic solvers.

The catenary equilibrium of :mod:`slenderpy.cable.static.catenary`
sampled on a grid uniform in arc length, its local triad, the conversions
between the local state and global positions, the finite-difference operators,
the quasi-static stretching condition and the projection of a global force on
the triad.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import slenderpy.fd_utils as fdu
from slenderpy._constant import _GRAVITY
from slenderpy.cable.static import catenary
from slenderpy.components import Conductor, Span


@dataclass(frozen=True)
class _Geometry:
    """Catenary equilibrium sampled on a grid uniform in arc length.

    Attributes
    ----------
    s : arc fraction of each node, uniform in [0, 1].
    ds : spacing of ``s``.
    x, z : global horizontal and vertical coordinates of the nodes (m).
    slope : dz/dx of the catenary at each node.
    et, en, eb : tangent, normal and binormal unit vectors, shape (3, ns).
    length : catenary length (m).
    """

    s: np.ndarray
    ds: np.ndarray
    x: np.ndarray
    z: np.ndarray
    slope: np.ndarray
    et: np.ndarray
    en: np.ndarray
    eb: np.ndarray
    length: float

    def equilibrium(self) -> np.ndarray:
        """Global position of the nodes at rest, shape (3, ns)."""
        return np.stack([self.x, np.zeros_like(self.x), self.z])


def _geometry(conductor: Conductor, span: Span, ns: int) -> _Geometry:
    """Build the equilibrium geometry and its local triad."""
    a = catenary._mechparam(span.tension, conductor.mass, g=_GRAVITY)
    length = catenary.length(
        span.length, span.tension, span.sld, conductor.mass, g=_GRAVITY
    )
    # xm is half the catenary offset, ie the abscissa shift of the low point
    xm = 0.5 * (a * catenary._qfactor(length, span.sld) - span.length)

    s = np.linspace(0.0, 1.0, ns)
    x = a * np.arcsinh(length * s / a + np.sinh(xm / a)) - xm
    # the two ends are exact by construction; keep them free of round-off
    x[0], x[-1] = 0.0, span.length
    # catenary.shape is declared floatArrayLike; it is an array for an array x
    z = np.asarray(
        catenary.shape(
            x, span.length, span.tension, span.sld, conductor.mass, g=_GRAVITY
        ),
        dtype=float,
    )

    slope = np.sinh((x + xm) / a)
    norm = np.sqrt(1.0 + slope**2)
    zero = np.zeros_like(x)
    et = np.stack([np.ones_like(x), zero, slope]) / norm
    en = np.stack([-slope, zero, np.ones_like(x)]) / norm
    eb = np.stack([zero, -np.ones_like(x), zero])

    return _Geometry(
        s=s, ds=np.diff(s), x=x, z=z, slope=slope, et=et, en=en, eb=eb, length=length
    )


def _to_local(
    position: np.ndarray | None, velocity: np.ndarray | None, geom: _Geometry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project a global position and velocity onto the local triad.

    The tangential component of the displacement is dropped: the model derives
    the tangential displacement from the stretching condition, so it cannot be
    imposed. Returns dimensional ``(un, ub, vn, vb)``.
    """
    ns = len(geom.s)
    if position is None:
        un = np.zeros(ns)
        ub = np.zeros(ns)
    else:
        offset = position - geom.equilibrium()
        un = (offset * geom.en).sum(axis=0)
        ub = (offset * geom.eb).sum(axis=0)

    if velocity is None:
        vn = np.zeros(ns)
        vb = np.zeros(ns)
    else:
        vn = (velocity * geom.en).sum(axis=0)
        vb = (velocity * geom.eb).sum(axis=0)

    return un, ub, vn, vb


def _to_global(
    ut: np.ndarray, un: np.ndarray, ub: np.ndarray, geom: _Geometry
) -> np.ndarray:
    """Assemble the global position from a dimensional local state, shape (3, ns)."""
    return geom.equilibrium() + ut * geom.et + un * geom.en + ub * geom.eb


def _operators(ns):
    """First and second derivatives in arc fraction on the uniform grid.

    ``second`` acts on the interior nodes, the ends being pinned at zero.
    ``first`` acts on all nodes; its end rows are first-order one-sided, as
    in the legacy solver: second-order rows change dtension by a few percent
    and are left for a separate check.
    """
    h = 1.0 / (ns - 1)
    second = fdu.second_derivative(ns, h).tocsr()[1:-1, 1:-1]
    first = fdu.first_derivative(ns, h).tolil()
    first[0, :2] = [-1.0 / h, 1.0 / h]
    first[-1, -2:] = [-1.0 / h, 1.0 / h]
    return first.tocsr(), second


def _stretching(un, ub, first, ds, vt2):
    """Tangential offset and axial strain from the quasi-static condition."""
    h = -un / vt2 + 0.5 * ((first * un) ** 2 + (first * ub) ** 2)
    segment = 0.5 * (h[:-1] + h[1:]) * ds
    ut = np.sum(segment) * np.linspace(0.0, 1.0, len(un)) - np.cumsum(
        np.concatenate(([0.0], segment))
    )
    strain = (first * ut) + 0.5 * (
        (first * ut) ** 2 + (first * un) ** 2 + (first * ub) ** 2
    )
    return ut, np.log(np.sqrt(1.0 + 2.0 * strain))


def _project(fy, fz, geom: _Geometry):
    """Global force components along y and z projected on (en, eb)."""
    fn = fy * geom.en[1] + fz * geom.en[2]
    fb = fy * geom.eb[1] + fz * geom.eb[2]
    return fn, fb
