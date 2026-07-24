"""Frozen dataclasses bundling the physical parameters of the model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MassParameters:
    """Inertial and geometric parameters of a damper mass.

    Attributes:
        length_to_clamp: Distance between the clamp and the mass attachment
            on the messenger cable (m).
        length_to_centroid: Distance from the mass attachment to the mass
            centroid along the messenger cable axis (m).
        mass: Mass (kg).
        moment_of_inertia: Moment of inertia about the centroid (kg.m^2).
    """

    length_to_clamp: float
    length_to_centroid: float
    mass: float
    moment_of_inertia: float


@dataclass(frozen=True)
class MessengerCableParameters:
    """Discretisation and constitutive parameters of the messenger cable.

    The cable is split in three regions (boundary, cable, boundary). The
    bending stiffness and the critical curvature are piecewise constant
    along these regions.

    Attributes:
        nb_space_points: Number of nodes along the messenger cable.
        ratio_boundary1: Length of the first boundary region, normalised by
            ``length_to_clamp``.
        ratio_boundary2: Length of the second boundary region, normalised by
            ``length_to_clamp``.
        ei_max_boundary: Maximum bending stiffness in the boundary regions
            (N.m^2).
        ei_max_cable: Maximum bending stiffness in the cable region (N.m^2).
        ei_min_boundary: Minimum bending stiffness in the boundary regions
            (N.m^2).
        ei_min_cable: Minimum bending stiffness in the cable region (N.m^2).
        chi0_boundary: Critical curvature in the boundary regions (1/m).
        chi0_cable: Critical curvature in the cable region (1/m).
    """

    nb_space_points: int
    ratio_boundary1: float
    ratio_boundary2: float
    ei_max_boundary: float
    ei_max_cable: float
    ei_min_boundary: float
    ei_min_cable: float
    chi0_boundary: float
    chi0_cable: float


@dataclass(frozen=True)
class ClampParameters:
    """Inertial and geometric parameters of the clamp.

    Attributes:
        mass: Mass of the clamp (kg).
        moment_of_inertia: Moment of inertia about the clamp centre (kg.m^2).
        half_length: Half-length of the clamp along the main cable (m).
    """

    mass: float
    moment_of_inertia: float
    half_length: float
