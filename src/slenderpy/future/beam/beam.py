from typing import Optional

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy import simtools


class Beam:
    """A Beam object."""

    def __init__(
        self,
        length: float,
        boundary_condition: FD.BoundaryCondition,
        tension: float,
        ei_max: float,
        ei_min: Optional[float] = None,
        critical_curvature: Optional[float] = None,
        mass: Optional[float] = None,
    ) -> None:

        self.length = length
        self.bc = boundary_condition
        self.tension = tension
        self.ei_max = ei_max
        self.ei_min = ei_min
        self.critical_curvature = critical_curvature
        self.mass = mass

    def compute_static_bending_moment(
        self,
        curvature: np.ndarray[float],
    ) -> np.ndarray[float]:
        """Compute the bending moment in static if not constant, otherwise return ei_max."""
        if self.critical_curvature is None:
            return self.ei_max * curvature

        else:
            chi_bar = (1 - self.ei_min / self.ei_max) * self.critical_curvature
            return (self.ei_max * chi_bar + self.ei_min * curvature) * (
                1 - np.exp(-curvature / chi_bar)
            )

    def compute_bending_moment(
        self,
        curvature: np.ndarray[float],
        eta: np.ndarray[float] = None,
    ) -> np.ndarray[float]:
        """Compute the bending moment if not constant, otherwise return ei_max."""
        if self.critical_curvature is None:
            return self.ei_max * curvature

        else:
            return (
                self.ei_min * curvature
                + (self.ei_max - self.ei_min) * self.critical_curvature * eta
            )


def compute_curvature(n: int, ds: float, y: np.ndarray[float]) -> np.ndarray[float]:
    """Compute the exact curvature for a given array."""
    y_second = FD.second_derivative(n, ds) @ y
    y_first = FD.first_derivative(n, ds) @ y
    return y_second * (np.ones(n) + y_first**2) ** (-3 / 2.0)


def _solve_static_approx_curvature(
    n: int,
    beam: Beam,
    rhs: np.ndarray[float],
) -> np.ndarray[float]:
    """Solve equation of the form : (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the approximated curvature i.e. (d^2/dx^2)*y.
    """
    ds = beam.length / (n - 1)

    ei = beam.ei_max
    H = beam.tension

    order = beam.bc.order
    D2_border = FD.second_derivative(n, ds)
    D2 = FD.clean_matrix(order, D2_border)
    BC, rhs_bc = beam.bc.compute(ds, n)
    D4 = FD.fourth_derivative(n, ds)
    K = ei * D4 - H * D2
    rhs = FD.clean_rhs(order, rhs)

    A = K + BC
    rhs_tot = rhs + rhs_bc

    sol = sp.sparse.linalg.spsolve(A, rhs_tot)

    if beam.critical_curvature is not None:

        def equation(y):
            curvature = D2_border @ y
            bending_moment = beam.compute_static_bending_moment(curvature)
            return D2 @ bending_moment - beam.tension * D2 @ y + BC @ y - rhs_tot

        result = sp.optimize.root(equation, sol)

        if not result.success:
            print(result.message)

        sol = result.x

    return sol


def _solve_static_exact_curvature(
    n: int, beam: Beam, rhs: np.ndarray[float]
) -> np.ndarray[float]:
    """Solve equation of the form : (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the exact curvature.
    """

    ds = beam.length / (n - 1)
    bc = beam.bc
    Y0 = _solve_static_approx_curvature(n, beam, rhs)

    D2 = FD.second_derivative(n, ds)
    D2 = FD.clean_matrix(bc.order, D2)

    rhs = FD.clean_rhs(bc.order, rhs)
    BC, rhs_bc = bc.compute(ds, n)

    def equation(y):
        curvature = compute_curvature(n, ds, y)
        bending_moment = beam.compute_static_bending_moment(curvature)

        return D2 @ bending_moment - beam.tension * D2 @ y + BC @ y - rhs - rhs_bc

    sol = sp.optimize.root(equation, Y0)

    if not sol.success:
        print(sol.message)

    return sol.x


def _solve_dynamic_approx_curvature(
    beam: Beam,
    parameters: simtools.Parameters,
    initial_position: np.ndarray[float],
    initial_velocity: np.ndarray[float],
    force: callable,
) -> np.ndarray[float]:
    """Solve equation of the form : m*(d^2/dt^2)*y (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the approximated curvature i.e. (d^2/dx^2)*y.
    """
    lspan = beam.length
    nb_space = parameters.ns
    ds = lspan / (nb_space - 1)
    dt = parameters.tf / parameters.nt
    dt2 = dt * 0.5
    x = np.linspace(0.0, lspan, nb_space)
    y_old = initial_position
    v_old = initial_velocity

    order = beam.bc.order
    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(order, D2)
    D4 = FD.fourth_derivative(nb_space, ds)
    BC, rhs_bc = beam.bc.compute(ds, nb_space)
    Id = sp.sparse.identity(nb_space)
    Id = FD.clean_matrix(order, Id)

    ei = beam.ei_max
    H = beam.tension
    mass = beam.mass
    K = ei * D4 - H * D2
    M = mass * Id
    A = M + dt2**2 * K + BC
    B = M - dt2**2 * K

    current_time = parameters.t0 + dt

    lov = ["y"]
    res = simtools.Results(
        lot=parameters.time_vector_output().tolist(), lov=lov, los=parameters.los
    )
    res.save(0, lov, [y_old])

    for k in range(parameters.nt):

        if beam.bc.dynamic_values is not None:
            rhs_bc = beam.bc.update_rhs(nb_space, x, current_time)

        force_previsous = FD.clean_rhs(order, force(x, current_time - dt))
        force_current = FD.clean_rhs(order, force(x, current_time))

        rhs = (
            B @ v_old
            + dt2 * (force_previsous + force_current)
            - dt * K @ y_old
            + rhs_bc
        )

        v_new = sp.sparse.linalg.spsolve(A, rhs)
        y_new = y_old + dt2 * (v_old + v_new)
        y_new[0:2] = v_new[0:2]
        y_new[-2:] = v_new[-2:]

        current_time += dt
        v_old = v_new
        y_old = y_new

        if (k + 1) % parameters.rr == 0:
            res.save((k // parameters.rr) + 1, lov, [y_new])

    return res


def _solve_dynamic_exact_curvature_EI_const(
    beam: Beam,
    parameters: simtools.Parameters,
    initial_position: np.ndarray[float],
    initial_velocity: np.ndarray[float],
    force: callable,
) -> np.ndarray[float]:
    """Solve equation of the form : m*(d^2/dt^2)*y (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the approximated curvature i.e. (d^2/dx^2)*y.
    """
    lspan = beam.length
    nb_space = parameters.ns
    ds = lspan / (nb_space - 1)
    dt = parameters.tf / parameters.nt
    dt2 = dt * 0.5
    x = np.linspace(0.0, lspan, nb_space)
    y_old = initial_position
    v_old = initial_velocity

    order = beam.bc.order
    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(order, D2)
    BC, rhs_bc = beam.bc.compute(ds, nb_space)
    Id = sp.sparse.identity(nb_space)
    Id = FD.clean_matrix(order, Id)

    H = beam.tension
    mass = beam.mass
    K = -H * D2
    M = mass * Id
    A = M + dt2**2 * K + BC
    B = M - dt2**2 * K

    current_time = parameters.t0 + dt

    lov = ["y"]
    res = simtools.Results(
        lot=parameters.time_vector_output().tolist(), lov=lov, los=parameters.los
    )
    res.save(0, lov, [y_old])

    for k in range(parameters.nt):

        if beam.bc.dynamic_values is not None:
            rhs_bc = beam.bc.update_rhs(nb_space, x, current_time)

        force_previsous = FD.clean_rhs(order, force(x, current_time - dt))
        force_current = FD.clean_rhs(order, force(x, current_time))

        curvature_old = compute_curvature(nb_space, ds, y_old)
        bending_moment_old = beam.compute_bending_moment(curvature_old)
        y_picard = y_old

        for _ in range(10):
            curvature_picard = compute_curvature(nb_space, ds, y_picard)
            bending_moment_picard = beam.compute_bending_moment(curvature_picard)
            rhs = (
                B @ v_old
                + dt2 * (force_previsous + force_current)
                - dt2 * D2 @ (bending_moment_old + bending_moment_picard)
                - dt * K @ y_old
                + rhs_bc
            )
            v_new = sp.sparse.linalg.spsolve(A, rhs)
            y_new = y_old + dt2 * (v_old + v_new)

            y_new[0:2] = v_new[0:2]
            y_new[-2:] = v_new[-2:]
            y_picard = y_new

        current_time += dt
        v_old = v_new
        y_old = y_new

        if (k + 1) % parameters.rr == 0:
            res.save((k // parameters.rr) + 1, lov, [y_new])

    return res


def _solve_dynamic_exact_curvature(
    beam: Beam,
    parameters: simtools.Parameters,
    initial_position: np.ndarray[float],
    initial_velocity: np.ndarray[float],
    initial_bending_moment: np.ndarray[float],
    force: callable,
) -> np.ndarray[float]:
    """Solve equation of the form : m*(d^2/dt^2)*y (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the approximated curvature i.e. (d^2/dx^2)*y.
    """
    lspan = beam.length
    nb_space = parameters.ns
    ds = lspan / (nb_space - 1)
    dt = parameters.tf / parameters.nt
    dt2 = dt * 0.5
    x = np.linspace(0.0, lspan, nb_space)
    ei_min = beam.ei_min
    ei_max = beam.ei_max
    chi0 = beam.critical_curvature
    y_old = initial_position
    v_old = initial_velocity
    bending_moment_old = initial_bending_moment
    curvature_old = compute_curvature(nb_space, ds, y_old)
    eta_old = (initial_bending_moment - ei_min * curvature_old) / (
        (ei_max - ei_min) * chi0
    )

    order = beam.bc.order
    D2 = FD.second_derivative(nb_space, ds)
    D2 = FD.clean_matrix(order, D2)
    BC, rhs_bc = beam.bc.compute(ds, nb_space)
    Id = sp.sparse.identity(nb_space)
    Id = FD.clean_matrix(order, Id)

    H = beam.tension
    mass = beam.mass
    K = -H * D2
    M = mass * Id
    A = M + dt2**2 * K + BC
    B = M - dt2**2 * K

    current_time = dt

    lov = ["y", "c", "M"]
    res = simtools.Results(
        lot=parameters.time_vector_output().tolist(), lov=lov, los=parameters.los
    )
    res.save(0, lov, [y_old, curvature_old, bending_moment_old])

    for k in range(parameters.nt):

        if beam.bc.dynamic_values is not None:
            rhs_bc = beam.bc.update_rhs(nb_space, x, current_time)

        force_previsous = FD.clean_rhs(order, force(x, current_time - dt))
        force_current = FD.clean_rhs(order, force(x, current_time))
        y_picard = y_old
        eta_picard = eta_old

        for _ in range(10):
            curvature_picard = compute_curvature(nb_space, ds, y_picard)
            bending_moment_picard = beam.compute_bending_moment(
                curvature_picard, eta_picard
            )
            rhs = (
                B @ v_old
                + dt2 * (force_previsous + force_current)
                - dt2 * D2 @ (bending_moment_old + bending_moment_picard)
                - dt * K @ y_old
                + rhs_bc
            )
            v_new = sp.sparse.linalg.spsolve(A, rhs)
            y_new = y_old + dt2 * (v_old + v_new)

            y_new[0:2] = v_new[0:2]
            y_new[-2:] = v_new[-2:]
            y_picard = y_new
            diff = curvature_picard - curvature_old
            eta_new = (
                eta_old
                + (diff - 0.5 * (diff * np.abs(eta_picard) + np.abs(diff) * eta_picard))
                / chi0
            )
            eta_picard = eta_new

        current_time += dt

        v_old = v_new
        y_old = y_new
        curvature_new = compute_curvature(nb_space, ds, y_old)
        curvature_old = curvature_new
        bending_moment_old = beam.compute_bending_moment(curvature_old, eta_new)

        if (k + 1) % parameters.rr == 0:
            res.save(
                (k // parameters.rr) + 1,
                lov,
                [y_new, curvature_old, bending_moment_old],
            )

    return res
