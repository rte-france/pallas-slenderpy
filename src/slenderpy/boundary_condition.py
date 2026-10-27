"""General boundary conditions for finite-difference beam/cable schemes.

A [](`~slenderpy.boundary_condition.BoundaryCondition`) describes, at each end of the domain, one or two
linear relations of the form::

    a * y(x) + b * (dy/dx)(x) + c * (d2y/dx2)(x) = d(t)

and builds the matrix and right-hand side contributions that enforce them in a
finite-difference scheme. The [](`~slenderpy.boundary_condition.hinged`) and [](`~slenderpy.boundary_condition.clamped`) helpers
build the two most common cases (pinned and clamped ends).

The two relations of an order-4 condition are taken at the support with the
centered stencils, through a ghost node one step beyond each support: the
scheme runs on the extended grid of ``n + 2`` nodes, the supports being nodes
1 and ``n``. This is second order, and keeps the discrete energy exact at a
clamped or hinged end, where a one-sided slope ``(y1 - y0) / ds`` is first
order. [](`~slenderpy.boundary_condition.BoundaryCondition.extend`) fills the
ghost nodes of a field given on the span nodes.
"""

from typing import Optional, Tuple

import numpy as np
import scipy as sp


class BoundaryCondition:
    """Object to deal with boundary conditions."""

    def __init__(
        self,
        order: int,
        left: Optional[
            Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        ] = None,
        right: Optional[
            Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]
        ] = None,
        dynamic_values: Optional[Tuple[callable, callable, callable, callable]] = None,
    ) -> None:
        """Init with args.

        Input left and right are two tuples.
        They contain one or two tuples each, depending the number of boundary conditions.
        Each sub tuple contain four floats (a,b,c,d) describing the following boundary conditon:
            a * y(x) + b * (dy/dx)(x) + c * (d2y/dx2)(x) = d(t)

        x is either the left bound or either the right bound depending if the sub tuple belong to left or right.
        The d values are used by the static solve; a time-dependent right-hand
        side goes through ``dynamic_values`` instead.

        If None values are used for left or right, Dirichlet boundary conditions are
        used.

        Parameters
        ----------
        order : int
            Number of boundary conditions (2 or 4).
        left : Optional[ Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]] ], optional
            Coefficients for the left boundary condition(s), by default None
        right : Optional[ Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]] ], optional
            Coefficients for the right boundary condition(s), by default None
        dynamic_values : Optional[Tuple[callable, callable, callable, callable]], optional
            Time-dependent right-hand side for the dynamic solve, by default None
            (the relations keep the value of the initial state). The dynamic
            solve enforces the boundary conditions on the velocity, so each
            callable ``f(x, t)`` returns the time derivative ``d'(t)`` of a
            relation, not ``d(t)``; ``x`` is the bound the relation applies to.
            The order is (left[0], left[1], right[1], right[0]), i.e. the rows
            from the first to the last node; with order 2 only the first and
            last items are used.

        Raises
        ------
        ValueError
            If order different than 2 or 4.
        """

        if order != 2 and order != 4:
            raise ValueError("Order must be 2 or 4")

        self.left = left
        self.right = right

        if order == 4:
            self._check_order4()

        if order == 2:
            self._check_order2()

        self.order = order
        self.dynamic_values = dynamic_values

    def _check_order2(self):
        """Check the validity of boundary conditions for order 2.

        Raises
        ------
        TypeError
            If left and right are not under the proper format.
        ValueError
            If there is not exactly one boundary condition for left and for right.
        ValueError
            If the sub tuples for left and right do not have exactly 4 elements.
        ValueError
            If there is no left boundary condition (i.e. 4 zeros in the sub tuple).
        ValueError
            If there is no right boundary condition (i.e. 4 zeros in the sub tuple).
        """
        if self.left is None:
            self.left = ((1.0, 0.0, 0.0, 0.0),)

        if self.right is None:
            self.right = ((1.0, 0.0, 0.0, 0.0),)

        if (not isinstance(self.left, (tuple, list))) or (
            not isinstance(self.right, (tuple, list))
        ):
            raise TypeError("Inputs left and right must be list or tuples")

        if len(self.left) != 1 or len(self.right) != 1:
            raise ValueError("Need one boundary condition for each extremity")

        if len(self.left[0]) != 4 or len(self.right[0]) != 4:
            raise ValueError("Inputs must have 4 elements for each boundary condition")

        rankL = np.linalg.matrix_rank(self.left)
        rankR = np.linalg.matrix_rank(self.right)

        if rankL < 1:
            raise ValueError("There is no left boundary condition")

        if rankR < 1:
            raise ValueError("There is no right boundary condition")

    def _check_order4(self):
        """Check the validity of boundary conditions for order 4.

        Raises
        ------
        TypeError
            If left and right are not under the proper format.
        ValueError
             If there is not exactly two boundary condition for left and for right.
        ValueError
            If the sub tuples for left and right do not have exactly 4 elements.
        ValueError
            If the left boundary condition are twice the same.
        ValueError
            If the right boundary condition are twice the same.
        """
        if self.left is None:
            self.left = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))

        if self.right is None:
            self.right = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))

        if (not isinstance(self.left, (tuple, list))) or (
            not isinstance(self.right, (tuple, list))
        ):
            raise TypeError("Inputs left and right must be list or tuples")

        if len(self.left) != 2 or len(self.right) != 2:
            raise ValueError("Need two boundary conditions for each extremity")

        if (
            len(self.left[0]) != 4
            or len(self.left[1]) != 4
            or len(self.right[0]) != 4
            or len(self.right[1]) != 4
        ):
            raise ValueError("Inputs must have 4 elements for each boundary condition")

        rankL = np.linalg.matrix_rank(self.left)
        rankR = np.linalg.matrix_rank(self.right)

        if rankL < 2:
            raise ValueError(
                "The left boundary conditions are not linearly independent"
            )

        if rankR < 2:
            raise ValueError(
                "The right boundary conditions are not linearly independent"
            )

    def compute(
        self, n: int, ds: float
    ) -> Tuple[sp.sparse.lil_matrix, np.ndarray[float]]:
        """Compute the matrices of the scheme and the right-hand side linked to the boundary conditions.

        Each relation fills one row: rows 0 and -1 hold the first relation of
        each end, rows 1 and -2 the second one (order 4).

        Order 4: ``n`` counts the nodes of the extended grid, a ghost node one
        step beyond each support, so the supports are nodes 1 and ``n - 2``.
        The relations are taken at the supports with the centered stencils,
        ``(u[2] - u[0]) / (2*ds)`` for the slope and ``(u[0] - 2*u[1] + u[2])
        / ds**2`` for the curvature at the left support, the same three nodes
        around the right one.

        Order 2: ``n`` counts the span nodes, the supports being nodes 0 and
        ``n - 1``, and the derivatives are one-sided, ``(y1 - y0) / ds`` for
        the slope and ``(y0 - 2*y1 + y2) / ds**2`` for the curvature, mirrored
        on the right end.

        Parameters
        ----------
        n : int
            Matrix size, the extended grid for order 4.
        ds : float
            Space discretization step.

        Returns
        -------
        Tuple[sp.sparse.lil_matrix, np.ndarray[float]]
            Matrix and right-hand side for boundary conditions.
        """
        bc_matrix = sp.sparse.lil_matrix((n, n))
        rhs = np.zeros(n)

        if self.order == 4:
            for row, (a, b, c, d) in self._relations():
                columns = slice(0, 3) if row >= 0 else slice(n - 3, n)
                bc_matrix[row, columns] = self._stencil(a, b, c, ds)
                rhs[row] = d
            return bc_matrix, rhs

        a1, b1, c1, d1 = self.left[0]
        a4, b4, c4, d4 = self.right[0]

        bc_matrix[0, 0] = a1 - b1 / ds + c1 / ds**2
        bc_matrix[0, 1] = b1 / ds - 2 * c1 / ds**2
        bc_matrix[0, 2] = c1 / ds**2

        bc_matrix[-1, -1] = a4 + b4 / ds + c4 / ds**2
        bc_matrix[-1, -2] = -b4 / ds - 2 * c4 / ds**2
        bc_matrix[-1, -3] = c4 / ds**2

        rhs[0] = d1
        rhs[-1] = d4

        return bc_matrix, rhs

    def _relations(self):
        """Row and coefficients of each relation of an order-4 condition."""
        return [
            (0, self.left[0]),
            (1, self.left[1]),
            (-2, self.right[1]),
            (-1, self.right[0]),
        ]

    @staticmethod
    def _stencil(a: float, b: float, c: float, ds: float) -> np.ndarray:
        """Weights of a relation on the three nodes around a support.

        The nodes are in ascending ``x``: the ghost, the support and the first
        inner node at the left end, the reverse at the right end.
        """
        return np.array(
            [c / ds**2 - b / (2 * ds), a - 2 * c / ds**2, c / ds**2 + b / (2 * ds)]
        )

    def extend(
        self, y: np.ndarray, ds: float, rhs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extend a nodal field with the ghost nodes of an order-4 condition.

        At each end the ghost is solved from the relation that weighs it most,
        the slope of a clamped end or the curvature of a hinged one; the other
        relation does not involve the ghost there and is left to ``y``.

        Parameters
        ----------
        y : np.ndarray
            Field at the ``n`` span nodes.
        ds : float
            Space discretization step.
        rhs : np.ndarray, optional
            Right-hand side of the relations on the extended grid, as
            [](`~slenderpy.boundary_condition.BoundaryCondition.compute`) or
            [](`~slenderpy.boundary_condition.BoundaryCondition.update_rhs`)
            return it; by default the values ``d`` of the relations.

        Returns
        -------
        np.ndarray
            Field at the ``n + 2`` nodes of the extended grid.

        Raises
        ------
        ValueError
            If the condition is not of order 4, or neither relation of an end
            involves its ghost node.
        """
        if self.order != 4:
            raise ValueError("ghost nodes are only defined for order-4 conditions")

        res = np.zeros(np.size(y) + 2)
        res[1:-1] = y
        relations = self._relations()
        if rhs is None:
            values = [d for _, (_, _, _, d) in relations]
        else:
            values = [rhs[row] for row, _ in relations]
        weights = [self._stencil(a, b, c, ds) for _, (a, b, c, _) in relations]

        # left end: ghost first in its stencil, right end: ghost last
        for ghost, others, pair in ((0, [1, 2], [0, 1]), (-1, [-3, -2], [2, 3])):
            column = 0 if ghost == 0 else 2
            k = max(pair, key=lambda i: abs(weights[i][column]))
            w = weights[k]
            if w[column] == 0.0:
                raise ValueError("no relation of an end involves its ghost node")
            known = np.delete(w, column)
            res[ghost] = (values[k] - known @ res[others]) / w[column]

        return res

    def update_rhs(self, n: int, x: np.ndarray[float], t: float) -> np.ndarray[float]:
        """Evaluate ``dynamic_values`` at time ``t``.

        Parameters
        ----------
        n : int
            Number of nodes.
        x : np.ndarray[float]
            Node positions; only the bounds ``x[0]`` and ``x[-1]`` are used.
        t : float
            Time.

        Returns
        -------
        np.ndarray[float]
            Right-hand side of size ``n``, non-zero on the boundary rows only:
            rows 0 and 1 hold left[0] and left[1], rows -2 and -1 hold right[1]
            and right[0].
        """
        rhs = np.zeros(n)

        rhs[0] = self.dynamic_values[0](x[0], t)
        rhs[-1] = self.dynamic_values[-1](x[-1], t)

        if self.order == 4:
            rhs[1] = self.dynamic_values[1](x[0], t)
            rhs[-2] = self.dynamic_values[-2](x[-1], t)

        return rhs


def hinged(y_left: float = 0, y_right: float = 0) -> BoundaryCondition:
    """Get boundary condition with free derivative and constrained value and curvature.

    Parameters
    ----------
    y_left : float, optional
        Left value for y, by default 0
    y_right : float, optional
        Right value for y, by default 0

    Returns
    -------
    BoundaryCondition
        A BoundaryCondition object.
    """
    return BoundaryCondition(
        4,
        left=((1.0, 0.0, 0.0, y_left), (0.0, 0.0, 1.0, 0.0)),
        right=((1.0, 0.0, 0.0, y_right), (0.0, 0.0, 1.0, 0.0)),
    )


def clamped(y_left: float = 0, y_right: float = 0) -> BoundaryCondition:
    """Get boundary condition with constrained value and zero derivative.

    Parameters
    ----------
    y_left : float, optional
        Left value for y, by default 0
    y_right : float, optional
        Right value for y, by default 0

    Returns
    -------
    BoundaryCondition
        A BoundaryCondition object.
    """
    return BoundaryCondition(
        4,
        left=((1.0, 0.0, 0.0, y_left), (0.0, 1.0, 0.0, 0.0)),
        right=((1.0, 0.0, 0.0, y_right), (0.0, 1.0, 0.0, 0.0)),
    )
