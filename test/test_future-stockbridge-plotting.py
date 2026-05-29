"""Smoke tests for the plotting helpers (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from slenderpy.future.stockbridge import (
    Side,
    plot_clamp,
    plot_clamp_all_versions,
    plot_mass,
    plot_spectrum,
    solve_imposed_acceleration,
)


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    yield
    plt.close("all")


def _filled_out(sb):
    nb = 5
    tf = 0.005
    t = np.linspace(0, 0.005, nb)
    dt = 1e-3
    acc = -0.1 * (2 * np.pi * 10) * np.sin(2 * np.pi * 10 * t)
    ang = np.zeros(nb)
    ic1 = np.zeros(sb.mass_right.nb_unknowns)
    ic2 = np.zeros(sb.mass_left.nb_unknowns)
    return solve_imposed_acceleration(sb, tf, ic1, ic2, acc, ang, dt)


def test_plot_clamp_runs(sb):
    out = _filled_out(sb)
    plot_clamp(out)
    assert plt.gcf() is not None


def test_plot_mass_accepts_enum_and_string(sb):
    out = _filled_out(sb)
    plot_mass(out, Side.RIGHT)
    plot_mass(out, "left")
    assert len(plt.get_fignums()) == 2


def test_plot_clamp_all_versions_runs(sb):
    out1 = _filled_out(sb)
    out2 = _filled_out(sb)
    plot_clamp_all_versions(out1, out2, "acceleration_clamp", "force_clamp")
    assert plt.gcf() is not None


def test_plot_spectrum_runs():
    n = 64
    dt = 1e-3
    t = np.arange(n) * dt
    value = np.sin(2 * np.pi * 50 * t)
    plot_spectrum(t, value, dt)
    assert plt.gcf() is not None
