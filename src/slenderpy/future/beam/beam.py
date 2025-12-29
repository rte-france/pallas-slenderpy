from typing import Optional
from abc import ABC

import numpy as np
import scipy as sp

import slenderpy.future.beam.fd_utils as FD
from slenderpy import simtools
from slenderpy import _progress_bar as spb


class Beam(ABC):
    """A Beam object."""

    def __init__(
        self,
        length: float,
        boundary_condition: FD.BoundaryCondition,
        tension: float,
        mass: float,
    ) -> None:
        self.length = length
        self.bc = boundary_condition
        self.tension = tension
        self.mass = mass
    
    def natural_frequencies(self, n) -> np.ndarray:
        return 0.5 * np.linspace(1, n, n) / self.length * np.sqrt(self.tension / self.mass)

    def natural_frequency(self):
        return self.natural_frequencies(n=1)[0]
    
    def natural_frequencies_rot_free(self, n, ei) -> np.ndarray:
        ep = ei / (self.tension * self.length**2)
        nn = np.linspace(1, n, n)
        Wn = nn * np.sqrt(1.0 + ep * (np.pi * nn) ** 2)
        return Wn * self.natural_frequency()
    
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
        BC, rhs_bc = self.bc.compute(n, ds)
        D4 = FD.fourth_derivative(n, ds)
        K = self.get_ei() * D4 - self.tension * D2
        A = K + BC
        rhs = FD.clean_rhs(order, rhs)
        rhs_tot = rhs + rhs_bc

        sol = sp.sparse.linalg.spsolve(A, rhs_tot)

        if approx_curvature:
            def curvature(y):
                return D2_border @ y

        else:
            D1 = FD.first_derivative(n, ds)
            def curvature(y):
                return D2_border @ y / np.sqrt((1 + (D1 @ y) ** 2) ** (3))

        def equation(y):
            bending_moment = self._bending_moment(curvature(y))
            return D2 @ bending_moment - self.tension * D2 @ y + BC @ y - rhs_tot

        result = sp.optimize.root(equation, sol)

        if not result.success:
            print(result.message)

        return result.x
    
    def solve_dynamic(
        self,
        parameters: simtools.Parameters,
        initial_position: np.ndarray[float],
        initial_velocity: np.ndarray[float],
        force: callable,
        approx_curvature: bool,
        initial_bending_moment: Optional[np.ndarray[float]] = None,
        zeta: Optional[float] = 0,
        w0: Optional[float] = None, 
        it_picard: Optional[int] = 15,
        tol_picard: Optional[float] = 1e-4,
    ) -> simtools.Results:
        """Solve equation of the form : m*(d^2/dt^2)*y + (d^2/dx^2)*M - tension*(d^2/dx^2)*y = rhs,
        where M depends on the exact curvature.
        """
        lspan = self.length
        ns = parameters.ns
        ds = lspan / (ns - 1)
        dt = parameters.tf / parameters.nt
        x = np.linspace(0.0, lspan, ns)
        current_time = parameters.t0 + dt

        order = self.bc.order
        D1 = FD.first_derivative(ns, ds)
        D2_border = FD.second_derivative(ns, ds)
        D2 = FD.clean_matrix(order, D2_border)
        D4 = FD.fourth_derivative(ns, ds)
        rhs_bc = np.zeros(ns)

        if approx_curvature:
            K = self.get_ei() * D4 - self.tension * D2
            def curvature(y):
                return D2 @ y

        else:
            K = -self.tension * D2
            def curvature(y):
                return D2_border @ y / np.sqrt((1 + (D1 @ y) ** 2) ** 3)
            
        y_old = initial_position
        v_old = initial_velocity
        curvature_old = curvature(y_old)
        if initial_bending_moment is None:
                initial_bending_moment = self._bending_moment(curvature_old)
        bending_moment_old = initial_bending_moment
        eta_old = self._init_eta(bending_moment_old,curvature_old)
        
        if w0 is None:
            w0 = self.natural_frequency()
        damp = w0 * zeta

        dict = self.build_dict(parameters, damp, K, D2)
        powers_name = ["p_kin", "p_bend", "p_tens", "p_ext", "p_dissip"]
        energies_name = ["e_kin", "e_bend", "e_tens", "e_ext", "e_dissip"]
        picard = ["it_picard"]
        lov = ["y", "v", "c", "M"]
        all_lov = lov + powers_name + energies_name + picard
        res = simtools.Results(
            lot=parameters.time_vector_output().tolist(), lov=all_lov, lov_dims = [2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1], los=parameters.los
        )
        res.update(0, x / lspan, lov, [y_old, v_old, curvature_old, bending_moment_old])
        pb = spb.generate(parameters.pp, parameters.nt, desc=__name__)

        for k in range(parameters.nt):
            if self.bc.dynamic_values is not None:
                rhs_bc = self.bc.update_rhs(ns, x, current_time)

            force_previous = FD.clean_rhs(
                order, force(x, current_time - dt, y_old, v_old)
            )
            force_current = FD.clean_rhs(order, force(x, current_time, y_old, v_old))
                
            v_new, y_new, eta_new, curvature_new, bending_moment_new, it = self.picard_process(
                dict, v_old, y_old, eta_old, rhs_bc, curvature, approx_curvature, force_current + force_previous, it_picard, tol_picard
                )

            if (k + 1) % parameters.rr == 0:
                values = [y_new, v_new, curvature_new, bending_moment_new] + list(self.compute_power(D2, curvature_new, v_old, v_new, y_new, eta_new, force_current, dt, x).values()) + [it]
                res.update((k // parameters.rr) + 1, x / lspan, lov + powers_name + picard, values)
                pb.update(parameters.rr)

            current_time += dt
            v_old = v_new
            y_old = y_new
            eta_old = eta_new
            curvature_old = curvature_new
            bending_moment_old = bending_moment_new

        self.update_energies(res, powers_name, energies_name, parameters.tf / parameters.nr , parameters.nr)
        pb.close()
        res.set_state({"y": y_new, "v": v_new, "c":curvature_new, "M": bending_moment_new})
        return res
    
    def compute_power(
        self, 
        D2: sp.sparse.spmatrix, 
        curvature_new: np.ndarray[float], 
        v_old: np.ndarray[float], 
        v_new: np.ndarray[float], 
        y_new: np.ndarray[float], 
        eta_new: np.ndarray[float],
        force: callable, 
        dt: float, 
        x: np.ndarray[float]) -> dict:

        power = {}
        power["p_kin"] = (self.mass*sp.integrate.simpson(v_new*(v_new - v_old)/dt,x))
        power["p_bend"] = (self.get_ei()*sp.integrate.simpson((D2@curvature_new)*v_new,x))
        power["p_tens"] = (-self.tension*sp.integrate.simpson((D2@y_new)*v_new,x))
        power["p_ext"] = (sp.integrate.trapezoid(-force*v_new, x))
        power["p_dissip"] = np.nan

        return power
    
    @staticmethod
    def update_energies(
        res: simtools.Results, 
        powers_name: list, 
        energies_name: list, 
        dr: float, 
        nr) -> None:

        for k in range(1,nr):
            energies = []
            for powers in powers_name:
                energies.append(sp.integrate.trapezoid(res[powers][1:k+1], dx = dr))
            
            res.update(k, None, energies_name, energies)
        
        for energy in energies_name:
            res.data[energy] -= res.data[energy].min()
    
    def build_dict(self, parameters, damp, K, D2):
        dict = {}
        ns = parameters.ns
        ds = self.length / (ns - 1)
        dt = parameters.tf / parameters.nt
        dt2 = dt * 0.5

        order = self.bc.order
        BC, _ = self.bc.compute(ns, ds)
        Id = sp.sparse.identity(ns)
        Id = FD.clean_matrix(order, Id)
            
        M = self.mass * (1 + dt2*damp) * Id
        dict["A"] = M + dt2**2 * K + BC
        dict["B"] = M - dt2**2 * K

        dict["K"] = K
        dict["D2"] = D2 
        dict["damp"] = damp
        dict["dt"] = dt
        dict["dt2"] = dt2

        return dict


class BeamConst(Beam):

    def  __init__(
        self,
        length: float,
        boundary_condition: FD.BoundaryCondition,
        tension: float,
        mass: float,
        ei: float
    ) -> None:
        self.length = length
        self.bc = boundary_condition
        self.tension = tension
        self.mass = mass
        self.ei = ei 

    def get_ei(self):
        return self.ei
    
    def _init_eta(self, initial_bending_moment, curvature_old):
        return np.nan * np.zeros_like(initial_bending_moment)

    def _bending_moment(
        self,
        curvature: np.ndarray[float],
    ) -> np.ndarray[float]:
        return self.ei * curvature
    
    def picard_process(self, dict, v_old, y_old, eta_old, rhs_bc, curvature, approx_curvature, forces, it_picard, tol_picard):
        A = dict["A"]
        B = dict["B"]
        K = dict["K"]
        D2 = dict["D2"]
        damp = dict["damp"]
        dt = dict["dt"]
        dt2 = dict["dt2"]

        if not approx_curvature:
            curvature_old = curvature(y_old)
            bending_moment_old = self._bending_moment(curvature_old)

        y_picard = y_old 
        v_picard = v_old
        it = 0
        error = 100
        while it < it_picard and error > tol_picard:
            
            if approx_curvature:
                rhs = (
                    B @ v_old 
                    + dt2 * forces 
                    - dt * K @ y_old 
                    - dt *self.mass*damp * v_old 
                    + rhs_bc)

            else:
                curvature_picard = curvature(y_picard)
                bending_moment_picard = self._bending_moment(curvature_picard)
                rhs = (
                    B @ v_old
                    + dt2 * forces
                    - dt2 * D2 @ (bending_moment_old + bending_moment_picard)
                    - dt * K @ y_old
                    - dt*self.mass*damp * v_old
                    + rhs_bc
                )

            v_new = sp.sparse.linalg.spsolve(A,rhs)
            y_new = y_old + dt2 * (v_old + v_new)

            error = np.linalg.norm((v_picard - v_new)/np.linalg.norm(v_new) + (y_picard - y_new)/np.linalg.norm(y_new))
            v_picard = v_new
            y_picard = y_new
            it += 1
        
        curvature_new = curvature(y_new)
        bending_moment_new = self._bending_moment(curvature_new)
        return v_new, y_new, eta_old, curvature_new, bending_moment_new, it


class BeamBW(Beam):
    def __init__(
        self,
        length: float,
        boundary_condition: FD.BoundaryCondition,
        tension: float,
        mass: float,
        ei_min: float,
        ei_max: float,
        critical_curvature: float,
    ) -> None:

        super().__init__(length, boundary_condition, tension, mass)
        self.ei_min = ei_min
        self.ei_max = ei_max
        self.critical_curvature = critical_curvature
        self.chi_bar = (1 - self.ei_min / self.ei_max) * self.critical_curvature

    def get_ei(self):
        return self.ei_min
    
    def _init_eta(self, initial_bending_moment,curvature_old):
        return (initial_bending_moment - self.ei_min * curvature_old) / (
            (self.ei_max - self.ei_min) * self.critical_curvature
        )

    def _bending_moment(self, curvature: np.ndarray[float]) -> np.ndarray[float]:
        c = np.abs(curvature)
        return (self.ei_max * self.chi_bar + self.ei_min * c) * (
            1 - np.exp(-c / self.chi_bar)
        )*np.sign(curvature)

    def _bending_moment_dynamic(self, curvature: np.ndarray[float], eta: np.ndarray[float]) -> np.ndarray[float]:
        return (
            self.ei_min * curvature
            + (self.ei_max - self.ei_min) * self.critical_curvature * eta
        )
    
    def picard_process(self, dict, v_old, y_old, eta_old, rhs_bc, curvature, approx_curvature, forces, it_picard, tol_picard):
        A = dict["A"]
        B = dict["B"]
        K = dict["K"]
        D2 = dict["D2"]
        damp = dict["damp"]
        dt = dict["dt"]
        dt2 = dict["dt2"]

        if not approx_curvature:
            curvature_old = curvature(y_old)
            bending_moment_old = self._bending_moment_dynamic(curvature_old, eta_old)

        y_picard = y_old 
        v_picard = v_old
        eta_picard = eta_old
        it = 0
        error = 100
        while it < it_picard and error > tol_picard:
            
            if approx_curvature:
                rhs = (
                    B @ v_old 
                    + dt2 * forces 
                    - dt2 * (self.ei_max - self.ei_min)*self.critical_curvature*D2@(eta_old + eta_picard)
                    - dt * K @ y_old 
                    - dt *self.mass*damp * v_old 
                    + rhs_bc)

            else:
                bending_moment_picard = self._bending_moment_dynamic(curvature(y_picard), eta_picard)
                rhs = (
                    B @ v_old
                    + dt2 * forces
                    - dt2 * D2 @ (bending_moment_old + bending_moment_picard)
                    - dt * K @ y_old
                    - dt*self.mass*damp * v_old
                    + rhs_bc
                )

            v_new = sp.sparse.linalg.spsolve(A,rhs)
            y_new = y_old + dt2 * (v_old + v_new)

            error = np.linalg.norm((v_picard - v_new)/np.linalg.norm(v_new) + (y_picard - y_new)/np.linalg.norm(y_new))
            v_picard = v_new
            y_picard = y_new

            if approx_curvature:
                    eta_new = (self.critical_curvature*eta_old + dt*D2@v_new - dt2*D2@v_new*np.abs(eta_picard)) / (self.critical_curvature + dt2*np.abs(D2@v_new))

            else:
                diff = curvature(y_picard) - curvature_old
                eta_new = (
                    eta_old
                    + (diff - 0.5 * diff * np.abs(eta_picard)) / self.critical_curvature
                ) / (1 + 0.5 * np.abs(diff) / self.critical_curvature)

            eta_picard = eta_new
            it += 1
        
        curvature_new = curvature(y_new)
        bending_moment_new = self._bending_moment_dynamic(curvature_new, eta_new)
        return v_new, y_new, eta_new, curvature_new, bending_moment_new, it
    
    
    def compute_power(
        self, 
        D2: sp.sparse.spmatrix, 
        curvature_new: np.ndarray[float], 
        v_old: np.ndarray[float], 
        v_new: np.ndarray[float], 
        y_new: np.ndarray[float], 
        eta_new: np.ndarray[float],
        force: callable, 
        dt: float, 
        x: np.ndarray[float]) -> dict:

        power = super().compute_power(D2, curvature_new, v_old, v_new, y_new, eta_new,force, dt, x)
        power["p_dissip"] = (self.ei_max - self.ei_min)*self.critical_curvature*sp.integrate.simpson((D2@eta_new)*v_new,x)
        return power