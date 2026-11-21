"""Spacer descriptors for bundled conductors.

A bundle of N sub-conductors is held in a fixed geometric arrangement by
spacers installed at discrete locations along the span. Each spacer is
described by its curvilinear location s in ]0, 1[ and a set of edges, where
each edge couples one pair of sub-conductors via a linear spring-dashpot in
the plane perpendicular to the span (the normal-binormal plane of the local
triad). Rotational and torsional stiffnesses are omitted; see
doc/bundle.rst for the rationale and references.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Tuple


@dataclass(frozen=True)
class SpacerEdge:
    """One sub-conductor pair linked by a spacer.

                        iPos-1    iPos  iPos+1
    -----x---------x---------x----x---------x----- cable i
                                ||||||
                                ||||||
                                ||||||
    -----x---------x---------x----x---------x----- cable j
                                spacer

                            >------<
                                width

    By default, the spacer is 1D with the parameters related to 2D set to 0.

    Parameters
    ----------
    i, j : int
        Sub-conductor indices (0-based) of the two endpoints. Must differ.
    width : float
        Spacer width (m) MUST BE THE VALUE OF THE X STEP.
    k_n : float
        Translational stiffness (N/m), in the normal direction.
    c_n : float
        Translational damping (N.s/m), in the normal direction.
    k_b : float
        Translational stiffness (N/m), in the normal direction.
    c_b : float
        Translational damping (N.s/m), in the normal direction.
    j_t : float
        Angular stiffness of the restoring moment about vertical direction, in N/rad.
    j_b : float
        Angular stiffness of the restoring moment about binormal direction, in N/rad.
    delta_n, delta_b : float
        Rest offset (m) between sub-conductor i and sub-conductor j in the
        normal and binormal directions. For most bundles the static
        equilibrium catenaries already encode the geometric offset, so the
        common case is delta_n = delta_b = 0 (the spacer exerts no force at
        rest because the state is the offset from static equilibrium).
    is_2d: bool
        Inicates if the SpacerEdge is 2D or not.
    """

    i: int
    j: int
    k_n: float
    c_n: float
    k_b: float
    c_b: float
    is_2d: bool = False
    delta_n: float = 0.0
    delta_b: float = 0.0
    j_b: float = 0.0
    j_t: float = 0.0
    width: float = 0.0
    l0: float = 1.0
    

    def __post_init__(self) -> None:
        if not self.is_2d:
            object.__setattr__(self, 'delta_n', 0.0)
            object.__setattr__(self, 'delta_b', 0.0)
            object.__setattr__(self, 'j_b', 0.0)
            object.__setattr__(self, 'j_t', 0.0)
            object.__setattr__(self, 'width', 0.0)
            object.__setattr__(self, 'l0', 1.0)

        if self.i == self.j:
            raise ValueError("SpacerEdge endpoints i and j must differ")
        if self.i < 0 or self.j < 0:
            raise ValueError("SpacerEdge indices must be non-negative")
        if self.k_n < 0.0:
            raise ValueError("SpacerEdge k_n must be non-negative")
        if self.c_n < 0.0:
            raise ValueError("SpacerEdge c_n must be non-negative")
        if self.k_b < 0.0:
            raise ValueError("SpacerEdge k_b must be non-negative")
        if self.c_b < 0.0:
            raise ValueError("SpacerEdge c_b must be non-negative")
        if self.j_t < 0.0:
            raise ValueError("SpacerEdge j_t must be non-negative")
        if self.j_b < 0.0:
            raise ValueError("SpacerEdge j_b must be non-negative")
        if self.width < 0.0:
            raise ValueError("SpacerEdge width must be non-negative")




@dataclass(frozen=True)
class Spacer:
    """A spacer at a single curvilinear location coupling sub-conductors.

    Parameters
    ----------
    s : float
        Curvilinear abscissa in ]0, 1[ along the normalized span.
    edges : tuple of SpacerEdge
        Pairs of sub-conductors linked by this spacer. A star spacer with a
        central hub is encoded as a set of edges from the hub-conductor to
        each leaf; a ring spacer is a cycle. All SpacerEdges must have the same dimension (1D or 2D).
    width : float
        Width of the spacers.
    is_2d : bool
        Inicates if the Spacer is 2D or not.
    """

    s: float
    edges: Tuple[SpacerEdge, ...]
    is_2d: bool = False
    width: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.s < 1.0:
            raise ValueError("Spacer s must be in ]0, 1[")
        if len(self.edges) == 0:
            raise ValueError("Spacer must have at least one edge")

        edges_2d_flags = [edge.is_2d for edge in self.edges]

        if len(set(edges_2d_flags)) > 1:
            raise ValueError("All SpacerEdges within a Spacer must have the same dimension (= same is_2d value)")

        width_flags = [edge.width for edge in self.edges]
        if len(set(edges_2d_flags)) > 1:
            raise ValueError("All SpacerEdges within a Spacer must have the same width")


        common_is_2d = edges_2d_flags[0]
        object.__setattr__(self, 'is_2d', common_is_2d)
        common_width = width_flags[0]
        object.__setattr__(self, 'width', common_width)
   


def twin_spacer(s: float, k_n: float, c_n: float, k_b: float, c_b: float, is_2d: bool,
                delta_n: float, delta_b: float, j_b = 0.0, j_t = 0.0, width = 0.0, l0 = 0.0 ) -> Spacer:
    """Build the common two-sub-conductor spacer (edge 0-1, zero rest offset).

    Parameters
    ----------
    s : float
        Curvilinear location in ]0, 1[.
    width : float
        Spacer width (m) MUST BE THE VALUE OF THE X STEP.
    k_n : float
        Translational stiffness (N/m), in the normal direction.
    c_n : float
        Translational damping (N.s/m), in the normal direction.
    k_b : float
        Translational stiffness (N/m), in the normal direction.
    c_b : float
        Translational damping (N.s/m), in the normal direction.
    j_t : float
        Angular stiffness of the restoring moment about vertical direction, in N/rad.
    j_b : float
        Angular stiffness of the restoring moment about binormal direction, in N/rad.
    delta_n, delta_b : float
        Rest offset (m) between sub-conductor i and sub-conductor j in the
        normal and binormal directions. For most bundles the static
        equilibrium catenaries already encode the geometric offset, so the
        common case is delta_n = delta_b = 0 (the spacer exerts no force at
        rest because the state is the offset from static equilibrium).
    is_2d: bool
        Inicates if the SpacerEdge is 2D or not.

    Returns
    -------
    Spacer
    
    """

    return Spacer(s=s, edges=(SpacerEdge(i=0, j=1, k_n=k_n, c_n=c_n, k_b=k_b, c_b=c_b, 
                  is_2d = is_2d, delta_n = delta_n, delta_b = delta_b, j_b = j_b, j_t = j_t, width = width, l0 = l0), ), 
                  is_2d = is_2d, width = width)