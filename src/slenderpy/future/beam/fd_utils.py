from typing import Tuple, Optional

import numpy as np
import scipy as sp


def first_derivative(n: int, ds: float) -> sp.sparse.spmatrix:
    """Centered scheme, the first and last line have to be completed with BC (order 2)."""

    dinf = -1.0 * np.ones((n - 1,)) / (2 * ds)
    dsup = +1.0 * np.ones((n - 1,)) / (2 * ds)

    dinf[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, dsup], [-1, 1])

    return res


def second_derivative(n: int, ds: float) -> sp.sparse.spmatrix:
    """Centered scheme, the first and last line have to be completed with BC (order 2)."""

    dinf = +1.0 * np.ones((n - 1,)) / ds**2
    diag = -2.0 * np.ones((n,)) / ds**2
    dsup = +1.0 * np.ones((n - 1,)) / ds**2

    dinf[-1] = 0
    diag[0] = 0
    diag[-1] = 0
    dsup[0] = 0

    res = sp.sparse.diags([dinf, diag, dsup], [-1, 0, 1])

    return res


def fourth_derivative(n: int, ds: float) -> sp.sparse.spmatrix:
    """Centered scheme, the two first and two last line have to be completed with BC (order 4)."""

    dinf2 = +1.0 * np.ones((n - 2)) / ds**4
    dinf1 = -4.0 * np.ones((n - 1,)) / ds**4
    diag = +6.0 * np.ones((n,)) / ds**4
    dsup1 = -4.0 * np.ones((n - 1,)) / ds**4
    dsup2 = +1.0 * np.ones((n - 2,)) / ds**4

    dinf2[[-1, -2]] = [0, 0]
    dinf1[[-1, -2, 0]] = [0, 0, 0]
    diag[[0, 1, -2, -1]] = [0, 0, 0, 0]
    dsup1[[0, 1, -1]] = [0, 0, 0]
    dsup2[[0, 1]] = 0

    res = sp.sparse.diags([dinf2, dinf1, diag, dsup1, dsup2], [-2, -1, 0, 1, 2])

    return res


def clean_matrix(order: int, A: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
    """Earase the proper coefficient in the scheme matrix to take into account the boundary conditions."""
    if order == 4:
        A = sp.sparse.csr_matrix.copy(A)

        if A.data.shape[0] == 1:
            A.data[0, 0] = 0
            A.data[0, 1] = 0
            A.data[0, -1] = 0
            A.data[0, -2] = 0

        else:
            A.data[0, 0] = 0
            A.data[0, -3] = 0

            A.data[1, 1] = 0
            A.data[1, -2] = 0

            A.data[2, -1] = 0
            A.data[2, 2] = 0

    return A


def clean_rhs(order: int, rhs: Optional[np.ndarray[float]] = None) -> np.ndarray[float]:
    """Earase the proper coefficient in the right hand-side to take into account the boundary conditions."""
    rhs = np.copy(rhs)

    rhs[0] = 0
    rhs[-1] = 0

    if order == 4:
        rhs[1] = 0
        rhs[-2] = 0

    return rhs


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
        """Check the validity of boundary conditions for order 2."""
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
        """Check the validity of boundary conditions for order 4."""
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
                "The left boundary conditions are not linearly independant"
            )

        if rankR < 2:
            raise ValueError(
                "The right boundary conditions are not linearly independant"
            )

    def compute(
        self, ds: float, n: int
    ) -> Tuple[sp.sparse.spmatrix, np.ndarray[float]]:
        """Compute the matrices of the scheme and the right-hand side linked to the boundary conditions."""
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

    def update_rhs(self, n, x: np.ndarray[float], t: float):
        rhs = np.zeros(n)

        rhs[0] = self.dynamic_values[0](x[0], t)
        rhs[-1] = self.dynamic_values[-1](x[-1], t)

        if self.order == 4:
            rhs[1] = self.dynamic_values[1](x[1], t)
            rhs[-2] = self.dynamic_values[-2](x[-2], t)

        return rhs


def rot_free(y_left=0, y_right=0, d2y_left=0, d2y_right=0):
    """Get boundary condition with free derivative and constrained value and curvature."""
    return BoundaryCondition(
        4,
        left=((1.0, 0.0, 0.0, y_left), (0.0, 0.0, 1.0, d2y_left)),
        right=((1.0, 0.0, 0.0, y_right), (0.0, 0.0, 1.0, d2y_right)),
    )


def rot_none(y_left=0, y_right=0, d2y_left=0, d2y_right=0):
    """Get boundary condition with free curvature and constrained value and derivative."""
    return BoundaryCondition(
        4,
        left=((1.0, 0.0, 0.0, y_left), (0.0, 0.0, 1.0, d2y_left)),
        right=((1.0, 0.0, 0.0, y_right), (0.0, 0.0, 1.0, d2y_right)),
    )
