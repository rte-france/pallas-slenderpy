"""Static beam shape for both bending models and both curvature options.

Builds a single Conductor and Span, computes the static deflection under the
conductor's own weight for every (bending model, curvature) combination, and
overlays the four shapes on one plot.
"""

import matplotlib.pyplot as plt
import numpy as np

from slenderpy.future._constant import _GRAVITY
from slenderpy.future.beam.static.shape import BendingModel, solve
from slenderpy.future.boundary_condition import hinged, clamped
from slenderpy.future.components import Conductor, Span


def compare_models(span):

    # setup conductor parameters
    conductor = Conductor(
        mass=1.57,
        ei_min=28.28,
        ei_max=2155.07,
        beta_flexion=6.438E-07,
    )

    # cases variations
    cases = [
        (BendingModel.CONSTANT, True),
        (BendingModel.CONSTANT, False),
        (BendingModel.VARYING, True),
        (BendingModel.VARYING, False),
    ]

    # Distributed self-weight (N/m), constant along the span
    n = 1001
    x = np.linspace(0.0, span.length, n)
    rhs = -_GRAVITY * conductor.mass * np.ones(n)

    plt.figure()
    for model, approx_curvature in cases:
        y = solve(
            conductor,
            span,
            rhs=rhs,
            n=n,
            model=model,
            approx_curvature=approx_curvature,
        )
        curvature = "approx" if approx_curvature else "exact"
        plt.plot(x, y, label=f"{model.value}, {curvature} curvature")

    plt.xlabel("Position along span [m]")
    plt.ylabel("Deflection [m]")
    plt.title("Static beam shape under self-weight")
    plt.grid(True)
    plt.legend()



if __name__ == "__main__":

    # two configurations, realistic span and bretelle
    span = Span(length=440.0, tension=2.8E+04, boundary_conditions=hinged())
    bretelle =  Span(length=5.0, tension=10.0, boundary_conditions=clamped())

    # plot shape under gravity
    compare_models(span)
    compare_models(bretelle)

    plt.show()
