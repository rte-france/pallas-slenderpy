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
        mass: float,
        ei_max: float, 
    ) -> None:

        self.length = length
        self.bc = boundary_condition
        self.tension = tension
        self.mass = mass
        self.ei_max = ei_max
        

    def _bending_moment(
        self,
        curvature: np.ndarray[float],
    ) -> np.ndarray[float]:
        return self.ei_max*curvature 
    
    def solve_static(
        self,
        n: int,
        rhs: np.ndarray[float],
        approx_curvature: bool, 
    ) -> np.ndarray[float]:

        ds = self.length / (n - 1)

        order = self.bc.order
        D2_border = FD.second_derivative(n, ds)
        D2 = FD.clean_matrix(order, D2_border)
        BC, rhs_bc = self.bc.compute(ds, n)
        D4 = FD.fourth_derivative(n, ds)
        K = self.ei_max* D4 - self.tension * D2
        A = K + BC
        rhs = FD.clean_rhs(order, rhs)
        rhs_tot = rhs + rhs_bc

        sol = sp.sparse.linalg.spsolve(A, rhs_tot)

        if approx_curvature:
            def curvature(y):
                return D2_border@y
            
        else:
            def curvature(y):
                return D2_border@y / np.sqrt((np.ones(n) + ((FD.first_derivative(n,ds))@y)**2) ** (3))

        def equation(y):
            bending_moment = self._bending_moment(curvature(y))
            return D2 @ bending_moment - self.tension * D2 @ y + BC @ y - rhs_tot
        
        result = sp.optimize.root(equation, sol)

        if not result.success:
            print(result.message)

        return result.x
        
        

class BeamEIVariable(Beam):
    def __init__(
        self,
        length: float,
        boundary_condition: FD.BoundaryCondition,
        tension: float,
        mass: float,
        ei_max: float,
        ei_min: float,
        critical_curvature: float,
    ) -> None:

        super().__init__(length,boundary_condition,tension,mass,ei_max)
        self.ei_min = ei_min
        self.critical_curvature = critical_curvature
        self.chi_bar = (1 - self.ei_min / self.ei_max) * self.critical_curvature
    def _bending_moment(
        self,
        curvature: np.ndarray[float],
    ) -> np.ndarray[float]:
            return (self.ei_max * self.chi_bar + self.ei_min * curvature) * (
                1 - np.exp(-curvature / self.chi_bar)
            )
    
    def _bending_moment_dynamic(
        self,
        curvature: np.ndarray[float],
        eta,
    ):
        return (
            self.ei_min * curvature
            + (self.ei_max - self.ei_min) * self.critical_curvature * eta
        )
        


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
    #def one step 
    for k in range(parameters.nt):

        if beam.bc.dynamic_values is not None:
            rhs_bc = beam.bc.update_rhs(nb_space, x, current_time)

        force_previous = FD.clean_rhs(order, force(x, current_time - dt, y_old, v_old))
        force_current = FD.clean_rhs(order, force(x, current_time, y_old, v_old))
        #utiliser un objet type force 

        rhs = (
            B @ v_old
            + dt2 * (force_previous + force_current)
            - dt * K @ y_old
            + rhs_bc
        )

        v_new = sp.sparse.linalg.spsolve(A, rhs)
        #solve banded (pour aller + vite)
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
    """Solve equation of the form : m*(d^2/dt^2)*y + (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the exact curvature.
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

        force_previsous = FD.clean_rhs(order, force(x, current_time - dt, y_old, v_old))
        force_current = FD.clean_rhs(order, force(x, current_time, y_old, v_old))

        curvature_old = compute_curvature(nb_space, ds, y_old)
        bending_moment_old = beam.compute_bending_moment(curvature_old)
        y_picard = y_old

        it = 0
        error = 100
        #fonction picard loop
        #acceder aux paramètre de picard 
        while it < 10 and error > 1e-3:
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
            error = np.linalg.norm(y_picard - y_new)
            y_picard = y_new
            it += 1

        current_time += dt
        v_old = v_new
        y_old = y_new

        if (k + 1) % parameters.rr == 0:
            res.update((k // parameters.rr) + 1, x/lspan, lov, [y_new])

    return res


def _solve_dynamic_exact_curvature(
    beam: Beam,
    parameters: simtools.Parameters,
    initial_position: np.ndarray[float],
    initial_velocity: np.ndarray[float],
    initial_bending_moment: np.ndarray[float],
    force: callable,
) -> np.ndarray[float]:
    """Solve equation of the form : m*(d^2/dt^2)*y + (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
    where M depends on the exact curvature.
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

    current_time = parameters.t0 + dt

    lov = ["y", "v", "c", "M", "eta"]
    res = simtools.Results(
        lot=parameters.time_vector_output().tolist(), lov=lov, los=parameters.los
    )
    res.update(0, x/lspan, lov, [y_old, v_old, curvature_old, bending_moment_old, eta_old])

    for k in range(parameters.nt):

        if beam.bc.dynamic_values is not None:
            rhs_bc = beam.bc.update_rhs(nb_space, x, current_time)

        force_previsous = FD.clean_rhs(order, force(x, current_time - dt, y_old, v_old))
        force_current = FD.clean_rhs(order, force(x, current_time, y_old, v_old))

        y_picard = y_old
        eta_picard = eta_old

        it = 0
        error = 100
        while it < 15 and error > 1e-4 :
            curvature_picard = compute_curvature(nb_space, ds, y_picard)
            bending_moment_picard = beam.compute_bending_moment(curvature_picard, eta_picard)
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
            error = np.linalg.norm(y_picard - y_new)
            y_picard = y_new

            curvature_picard = compute_curvature(nb_space, ds, y_picard)
            diff = curvature_picard - curvature_old
            eta_new = ( eta_old + (diff - 0.5 * diff * np.abs(eta_picard))/ chi0 ) / ( 1 + 0.5*np.abs(diff)/chi0)
            eta_picard = eta_new 
            it += 1

        current_time += dt

        curvature_new = curvature_picard
        bending_moment_new = beam.compute_bending_moment(curvature_new, eta_new)

        if (k + 1) % parameters.rr == 0:
            res.save(
                (k // parameters.rr) + 1,
                lov,
                [y_new, v_new, curvature_new, bending_moment_new, eta_new],
            )

        v_old = v_new
        y_old = y_new
        eta_old = eta_new
        curvature_old = curvature_new
        bending_moment_old = bending_moment_new

    return res