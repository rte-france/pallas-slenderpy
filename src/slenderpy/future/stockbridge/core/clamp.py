from .parameters import ClampParameters


class Clamp:
    """Rigid clamp connecting the messenger cables to the main cable."""
    
    var_name = ["force_clamp", "moment_clamp", "acceleration_clamp", "acceleration_angular_clamp"]

    var_dim = [1,1,1,1]

    def __init__(self, parameters: ClampParameters) -> None:
        """Init with args. 

        Parameters
        ----------
        parameters : ClampParameters
            Inertial and geometric parameters of the clamp.
        """
        self.half_length = parameters.half_length
        self.mass = parameters.mass
        self.moment_of_inertia = parameters.moment_of_inertia

    def compute_forces_at_clamp(
        self,
        F1: float,
        F2: float,
        M1: float,
        M2: float,
        l1: float,
        l2: float,
        vert_acceleration: float,
        rot_acceleration: float,
    ) -> tuple[float, float]:
        """Vertical force and moment that the main cable applies to the clamp.

        Derived from Newton's second law on the clamp, given the contact
        forces and moments transmitted by both messenger cables.

        Args:
            F1: Vertical force from the right messenger cable (N).
            F2: Vertical force from the left messenger cable (N).
            M1: Moment from the right messenger cable (N.m).
            M2: Moment from the left messenger cable (N.m).
            l1: Length to clamp on the right side (m).
            l2: Length to clamp on the left side (m).
            vert_acceleration: Vertical acceleration of the clamp (m/s^2).
            rot_acceleration: Rotational acceleration of the clamp (rad/s^2).

        Returns:
            Tuple ``(Fc, Mc)`` of the vertical force and moment at the clamp.
        """
        Fc = self.mass * vert_acceleration - (F1 + F2)
        Mc = (
            self.moment_of_inertia * rot_acceleration
            + M2
            - M1
            + F2 * (l2 + self.half_length)
            - F1 * (l1 + self.half_length)
        )
        return Fc, Mc
    
