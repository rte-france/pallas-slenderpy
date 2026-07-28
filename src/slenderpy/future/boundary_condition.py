"""General boundary conditions for finite-difference beam/cable schemes.

A :class:`BoundaryCondition` describes, at each end of the domain, one or two
linear relations of the form::

    a * y(x) + b * (dy/dx)(x) + c * (d2y/dx2)(x) = d(t)

and builds the matrix and right-hand side contributions that enforce them in a
finite-difference scheme. The :func:`hinged` and :func:`clamped` helpers
build the two most common cases (pinned and clamped ends).
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
        d can be a function that depend on time.

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
            Function of the right-hand side of the boundary conditions (if not constant), by default None

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

        Parameters
        ----------
        n : int
            Matrix size.
        ds : float
            Space discretization step.

        Returns
        -------
        Tuple[sp.sparse.lil_matrix, np.ndarray[float]]
            Matrix and right-hand side for boundary conditions.
        """
        a1, b1, c1, d1 = self.left[0]
        a4, b4, c4, d4 = self.right[0]

        bc_matrix = sp.sparse.lil_matrix((n, n))
        rhs = np.zeros(n)

        bc_matrix[0, 0] = a1 - b1 / ds + c1 / ds**2
        bc_matrix[0, 1] = b1 / ds - 2 * c1 / ds**2
        bc_matrix[0, 2] = c1 / ds**2

        bc_matrix[-1, -1] = a4 + b4 / ds + c4 / ds**2
        bc_matrix[-1, -2] = -b4 / ds - 2 * c4 / ds**2
        bc_matrix[-1, -3] = c4 / ds**2

        rhs[0] = d1
        rhs[-1] = d4

        if self.order == 4:
            a2, b2, c2, d2 = self.left[1]
            a3, b3, c3, d3 = self.right[1]

            bc_matrix[1, 0] = a2 - b2 / ds + c2 / ds**2
            bc_matrix[1, 1] = b2 / ds - 2 * c2 / ds**2
            bc_matrix[1, 2] = c2 / ds**2

            bc_matrix[-2, -1] = a3 + b3 / ds + c3 / ds**2
            bc_matrix[-2, -2] = -b3 / ds - 2 * c3 / ds**2
            bc_matrix[-2, -3] = c3 / ds**2

            rhs[1] = d2
            rhs[-2] = d3

        return bc_matrix, rhs

    def update_rhs(self, n: int, x: np.ndarray[float], t: float) -> np.ndarray[float]:
        rhs = np.zeros(n)

        rhs[0] = self.dynamic_values[0](x[0], t)
        rhs[-1] = self.dynamic_values[-1](x[-1], t)

        if self.order == 4:
            rhs[1] = self.dynamic_values[1](x[1], t)
            rhs[-2] = self.dynamic_values[-2](x[-2], t)

        return rhs


def hinged(
    y_left: float = 0, y_right: float = 0) -> BoundaryCondition:
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
        left=((1.0, 0.0, 0.0, y_left), (0.0, 0.0, 1.0, 0.)),
        right=((1.0, 0.0, 0.0, y_right), (0.0, 0.0, 1.0, 0.)),
    )


def clamped(
    y_left: float = 0, y_right: float = 0) -> BoundaryCondition:
    """_summary_

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
        left=((1.0, 0.0, 0.0, y_left), (0.0, 1.0, 0.0, 0.)),
        right=((1.0, 0.0, 0.0, y_right), (0.0, 1.0, 0.0, 0.)),
    )