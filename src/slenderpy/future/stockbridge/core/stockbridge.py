import numpy as np
from slenderpy import simtools

from .parameters import ClampParameters, MessengerCableParameters, MassParameters
from .clamp import Clamp
from .mass import Mass
from .side import Side


class Stockbridge:
    """Two-mass stockbridge damper."""

    def __init__(
        self,
        clamp_parameters: ClampParameters,
        mass_right_parameters: MassParameters,
        messenger_cable_right_parameters: MessengerCableParameters,
        mass_left_parameters: MassParameters,
        messenger_cable_left_parameters: MessengerCableParameters,
        K: np.ndarray | None = None,
        C: np.ndarray | None = None,
    ) -> None:
        """Init with args.

        Parameters
        ----------
        clamp_parameters : ClampParameters
            Inertial and geometric parameters of the clamp.
        mass_right_parameters : MassParameters
            Inertial and geometric parameters of the right mass.
        messenger_cable_right_parameters : MessengerCableParameters
            Inertial and geometric parameters of the right messenger cable.
        mass_left_parameters : MassParameters
            Inertial and geometric parameters of the left mass.
        messenger_cable_left_parameters : MessengerCableParameters
            Inertial and geometric parameters of the left messenger cable.
        K : np.ndarray | None, optional
            2x2 linearised stiffness matrix (used by the linearised solver), by default None
        C : np.ndarray | None, optional
            2x2 linearised damping matrix (used by the linearised solver), by default None
        """
        self.clamp = Clamp(clamp_parameters)
        self.mass_right = Mass(mass_right_parameters, messenger_cable_right_parameters, Side.RIGHT)
        self.mass_left = Mass(mass_left_parameters, messenger_cable_left_parameters, Side.LEFT)
        self.K = K
        self.C = C
        self.nr = self.mass_right.nb_space_points
        self.nl = self.mass_left.nb_space_points

        self.mass_matrix_inv = np.linalg.inv(self._build_mass_matrix())
        self.ab = self._build_ab()

    def _build_mass_matrix(self) -> np.ndarray:
        """Block-diagonal global mass matrix of the two masses."""
        M11 = self.mass_right.mass_matrix
        M21 = self.mass_left.mass_matrix
        Zeros = np.zeros((2, 2))
        return np.block([[M11, Zeros], [Zeros, M21]])

    def _build_ab(self) -> tuple[np.ndarray, np.ndarray]:
        """Coupling vectors ``a`` and ``b`` from the equations of motion.

        ``a`` couples vertical clamp acceleration to the masses, while ``b``
        couples rotational clamp acceleration to the masses. Each vector has
        4 components, ordered as
        ``[right_translation, right_rotation, left_translation, left_rotation]``.
        """
        bc = self.clamp.half_length
        m1 = self.mass_right.mass
        m2 = self.mass_left.mass
        eg1 = self.mass_right.length_to_centroid
        eg2 = self.mass_left.length_to_centroid
        eps1 = self.mass_right.epsilon
        eps2 = self.mass_left.epsilon
        a = np.array([m1, -m1 * eg1, m2, -m2 * eg2])
        b = np.array(
            [
                m1 * eps1 * bc,
                -m1 * eg1 * eps1 * bc,
                m2 * eps2 * bc,
                -m2 * eg2 * eps2 * bc,
            ]
        )
        return a, b
    
class Result:
    """Results of a stockbridge simulation, stored in :class:`simtools.Results` objects for the right mass, left mass and clamp."""

    def __init__(self, stockbridge: Stockbridge, time_vector: np.ndarray) -> None:
        """Init with args. 

        Parameters
        ----------
        stockbridge : Stockbridge
            The stockbridge object containing the simulation data.
        time_vector : np.ndarray
            The time vector for the simulation.
        """
        self.right = simtools.Results(
                lot=time_vector,
                lov=stockbridge.mass_right.var_name, 
                lov_dims=stockbridge.mass_right.var_dim,
                los=np.linspace(0, 1, stockbridge.nr)
            )
        
        self.left = simtools.Results(
                lot=time_vector,
                lov=stockbridge.mass_left.var_name, 
                lov_dims=stockbridge.mass_left.var_dim,
                los=np.linspace(0, 1, stockbridge.nl),
            )
        
        self.general = simtools.Results(
                lot=time_vector,
                lov=stockbridge.clamp.var_name, 
                lov_dims=stockbridge.clamp.var_dim,
                # los=[0,1]
            ) 
        
        self.sb = stockbridge

    def update(self, k: int, 
               value_right: np.ndarray, value_left: np.ndarray, 
               acc_clamp: float, acc_ang_clamp: float, 
               force_clamp: float = None, moment_clamp: float = None) -> None:
        """Update the results objects with the values at time iteration ``k``.

        Parameters
        ----------
        k : int
            time iteration index
        value_right : np.ndarray
            The values for the right mass at the current time iteration.
        value_left : np.ndarray
            The values for the left mass at the current time iteration.
        acc_clamp : float
            The acceleration of the clamp at the current time iteration.
        acc_ang_clamp : float
            The angular acceleration of the clamp at the current time iteration.
        force_clamp : float, optional
            The force acting on the clamp at the current time iteration, by default None
        moment_clamp : float, optional
            The moment acting on the clamp at the current time iteration, by default None
        """
        self.right.update(k, self.sb.mass_right.x / self.sb.mass_right.length_to_clamp, self.sb.mass_right.var_name, 
                            [*value_right[0:6], value_right[6:6+self.sb.nr], value_right[6+self.sb.nr:]])
        self.left.update(k, self.sb.mass_left.x / self.sb.mass_left.length_to_clamp, self.sb.mass_left.var_name, 
                            [*value_left[0:6], value_left[6:6+self.sb.nr], value_left[6+self.sb.nr:]])

        if force_clamp is None and moment_clamp is None:
            force_clamp, moment_clamp = self.sb.clamp.compute_forces_at_clamp(
                value_right[4],
                value_left[4],
                value_right[5],
                value_left[5],
                self.sb.mass_right.length_to_clamp,
                self.sb.mass_left.length_to_clamp,
                acc_clamp,
                acc_ang_clamp,
            )

        self.general.update(k, None, self.sb.clamp.var_name, [force_clamp, moment_clamp, acc_clamp, acc_ang_clamp])
