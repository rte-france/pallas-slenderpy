import numpy as np
import scipy as sp

from .parameters import MassParameters, MessengerCableParameters
from .side import Side


class Mass:
    """Damper mass attached to the clamp via a messenger cable."""

    var_name = ["mass_displacement",
        "mass_rotation",
        "mass_velocity",
        "mass_angular_velocity",
        "force_extremity",
        "moment_extremity",
        "curvature", 
        "hysteresis_variable"]
    
    var_dim = [1,1,1,1,1,1,
               2,2]

    def __init__(
        self,
        mass_parameters: MassParameters,
        cable_parameters: MessengerCableParameters,
        side: Side,
    ) -> None:
        """Init with args. 

        Parameters
        ----------
        mass_parameters : MassParameters
            Inertial and geometric parameters of the mass.
        cable_parameters : MessengerCableParameters
            Discretisation and constitutive parameters of the
            messenger cable.
        side : Side
            :class:`Side` enum indicating left or right.
        """
        self.length_to_clamp = mass_parameters.length_to_clamp
        self.length_to_centroid = mass_parameters.length_to_centroid
        self.mass = mass_parameters.mass
        self.moment_of_inertia = mass_parameters.moment_of_inertia

        nb_space_points = cable_parameters.nb_space_points
        self.nb_space_points = nb_space_points
        self.x = np.linspace(0, self.length_to_clamp, nb_space_points)
        self.nb_unknowns = 6 + 2 * nb_space_points # position, rotation, velocity, angular velocity, force_extremity, moment_extremity (4), curvature(x), hysteresis_variable(x) (2*nb_space_points)

        self.ei_max, self.ei_min, self.chi0 = self._build_messenger_cable_arrays(
            cable_parameters
        )

        self.side = side
        self.epsilon = side.epsilon

        self.mass_matrix = self._build_mass_matrix()
        self.mass_matrix_inv = np.linalg.inv(self.mass_matrix)

    def _build_messenger_cable_arrays(
        self, cable: MessengerCableParameters
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the piecewise-constant arrays along the messenger cable.
        The cable is split into three regions of size
        ``ratio_boundary1 * l``, ``(1 - ratio_boundary1 - ratio_boundary2) * l``
        and ``ratio_boundary2 * l``. The first and third regions take the
        ``*_boundary`` values and the middle one takes the ``*_cable`` values.

        Parameters
        ----------
        cable : MessengerCableParameters
            Messenger cable parameters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            [ei_max, ei_min, chi0] arrays of shape (nb_space_points,) along the cable.
        """
        n = cable.nb_space_points
        x = self.x
        l = self.length_to_clamp

        index1 = np.argmin(np.abs(x - cable.ratio_boundary1 * l))
        if index1 > 0:
            index1 += 1
        index2 = np.argmin(np.abs(x - (1 - cable.ratio_boundary2) * l))
        if index2 == n - 1:
            index2 += 1

        def piecewise(boundary_value: float, cable_value: float) -> np.ndarray:
            return np.array(
                [boundary_value] * index1
                + [cable_value] * (index2 - index1)
                + [boundary_value] * (n - index2)
            )

        ei_max = piecewise(cable.ei_max_boundary, cable.ei_max_cable)
        ei_min = piecewise(cable.ei_min_boundary, cable.ei_min_cable)
        chi0 = piecewise(cable.chi0_boundary, cable.chi0_cable)
        return ei_max, ei_min, chi0

    def _build_mass_matrix(self) -> np.ndarray:
        """Build the 2x2 inertia matrix of the mass."""
        coef = self.mass * self.length_to_centroid
        return np.array(
            [
                [self.mass, -coef],
                [-coef, self.moment_of_inertia + self.length_to_centroid * coef],
            ]
        )

    def compute_exterior_forces(
        self, half_length: float, vert_acc: float, rot_acc: float
    ) -> tuple[float, float]:
        """Force and moment at clamp (force of the cable on the clamp). 

        Parameters
        ----------
        half_length : float
            Half-length of the clamp (m).
        vert_acc : float
            Vertical acceleration of the clamp (m/s^2).
        rot_acc : float
            Rotational acceleration of the clamp (rad/s^2).

        Returns
        -------
        tuple[float, float]
            ``(f_ext, m_ext)``: vertical force and moment.
        """
        f_ext = -self.mass * vert_acc - self.mass * self.epsilon * half_length * rot_acc
        m_ext = (
            self.mass * self.length_to_centroid * vert_acc
            + self.mass * self.length_to_centroid * self.epsilon * half_length * rot_acc
        )
        return f_ext, m_ext

    def build_matrix_acceleration_imposed(
        self, old_curvature_derivative: np.ndarray, dt: float
    ) -> sp.sparse.csr_matrix:
        """Build the system matrix for the imposed-acceleration solver.

        Parameters
        ----------
        old_curvature_derivative : np.ndarray
            Curvature increment from the previous
                time step, shape ``(nb_space_points,)``.
        dt : float
            Time step (s).

        Returns
        -------
        sp.sparse.csr_matrix
            CSR sparse matrix of shape ``(nb_unknowns, nb_unknowns)``.
        """

        n = self.nb_space_points
        A = sp.sparse.lil_matrix((self.nb_unknowns, self.nb_unknowns))
        ht = 0.5 * dt

        # Rows 0..3: Crank-Nicolson kinematics + Newton on the mass.
        A[0, 0] = 1
        A[0, 2] = -ht
        A[1, 1] = 1
        A[1, 3] = -ht
        A[2, 2] = 1
        A[2, 4:6] = self.mass_matrix_inv[0, :] * ht
        A[3, 3] = 1
        A[3, 4:6] = self.mass_matrix_inv[1, :] * ht

        # Rows 4, 5: link contact force/moment to the integral of the
        # cable curvature field via the trapezoidal rule.
        A[4, 0] = 1
        A[5, 1] = 1
        A[4, 6] = -0.5 * (self.x[1] - self.x[0]) * (self.length_to_clamp - self.x[0])
        A[5, 6] = -0.5 * (self.x[1] - self.x[0])
        for k in range(1, n - 1):
            A[4, 6 + k] = -0.5 * (self.x[k + 1] - self.x[k - 1]) * (
                self.length_to_clamp - self.x[k]
            )
            A[5, 6 + k] = -0.5 * (self.x[k + 1] - self.x[k - 1])
        A[4, 6 + n - 1] = -0.5 * (self.x[-1] - self.x[-2]) * (
            self.length_to_clamp - self.x[-1]
        )
        A[5, 6 + n - 1] = -0.5 * (self.x[-1] - self.x[-2])

        # Rows 6..6+2n: Bouc-Wen-like hysteretic moment-curvature law.
        # First n rows are the moment equation, last n rows are the
        # auxiliary eta variable; the |dchi/dt| term is frozen at the
        # previous step (old_curvature_derivative).
        A[6 : 6 + n, 5] = 1
        A[6 : 6 + n, 4] = self.length_to_clamp - self.x
        for k in range(n):
            A[6 + k, 6 + k] = -self.ei_min[k]
            A[6 + k, 6 + n + k] = -(self.ei_max[k] - self.ei_min[k]) * self.chi0[k]
            A[6 + n + k, 6 + n + k] = self.chi0[k] + np.abs(old_curvature_derivative[k])
            A[6 + n + k, 6 + k] = -1

        return A.tocsr()

    def build_rhs_acceleration_imposed(
        self,
        old_unknowns: np.ndarray,
        half_length: float,
        vert_acc_mean: float,
        rot_acc_mean: float,
        dt: float,
    ) -> np.ndarray:
        """Build the right-hand side for the imposed-acceleration solver.

        Parameters
        ----------
        old_unknowns : np.ndarray
            Solution at the previous time step, shape ``(nb_unknowns,)``.
        half_length : float
            Half-length of the clamp (m).
        vert_acc_mean : float
            Mean vertical clamp acceleration (m/s^2). The meab between the previous and current time step, used for the Crank-Nicolson scheme.
        rot_acc_mean : float
            Rotational clamp acceleration (rad/s^2). The mean between the previous and current time step, used for the Crank-Nicolson scheme.
        dt : float
            Time step (s).

        Returns
        -------
        np.ndarray
            1D array of shape ``(nb_unknowns,)``.
        """
        n = self.nb_space_points
        rhs = np.zeros(self.nb_unknowns)
        fext_mean, mext_mean = self.compute_exterior_forces(
            half_length, vert_acc_mean, rot_acc_mean
        )

        ht = 0.5 * dt
        # Explicit half of the kinematic relation.
        rhs[0] = old_unknowns[0] + ht * old_unknowns[2]
        rhs[1] = old_unknowns[1] + ht * old_unknowns[3]
        # Explicit half of Newton on the mass: average of inertial forces
        # at step n and n+1 minus the reaction stored in old_unknowns[4:6].
        rhs[2] = old_unknowns[2] + dt * self.mass_matrix_inv[0, :] @ (
            [
                fext_mean - 0.5 * old_unknowns[4],
                mext_mean - 0.5 * old_unknowns[5],
            ]
        )
        rhs[3] = old_unknowns[3] + dt * self.mass_matrix_inv[1, :] @ (
            [
                fext_mean - 0.5 * old_unknowns[4],
                mext_mean - 0.5 * old_unknowns[5],
            ]
        )
        # Hysteresis update for eta_{n+1}: explicit half from chi_n / eta_n.
        rhs[6 + n : 6 + 2 * n] = (
            -old_unknowns[6 : 6 + n]
            + self.chi0 * old_unknowns[6 + n : 6 + 2 * n]
        )

        return rhs        
