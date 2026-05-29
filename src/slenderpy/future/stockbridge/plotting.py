"""Matplotlib plotting helpers for stockbridge simulation outputs."""

import matplotlib.pyplot as plt
import numpy as np

from .core.stockbridge import Result
from .core.side import Side


def _side_value(side: Side | str) -> str:
    """Return the string value of a :class:`Side` (or pass-through a string)."""
    if isinstance(side, Side):
        return side.value
    return side


def plot_clamp(res: Result) -> None:
    """Plot the clamp acceleration and force over time.

    Parameters
    ----------
    res : Result
        Result object containing the clamp data to plot.
    """
    plt.figure()
    plt.suptitle("Clamp")

    plt.subplot(211)
    plt.plot(res.general["time"], res.general["acceleration_clamp"])
    plt.title("Acceleration")

    plt.subplot(212)
    plt.plot(res.general["time"], res.general["force_clamp"])
    plt.title("Force")
    plt.xlabel("Time (s)")


def plot_mass(res: Result, side: Side | str) -> None:
    """Plot the eight scalar mass quantities for a given side.

    Parameters
    ----------
    res : Result
        Result object containing the mass data to plot.
    side : Side | str
        Which mass to plot (``Side.LEFT``, ``Side.RIGHT``, ``"left"`` or ``"right"``). 
    """
    s =  _side_value(side)
    s_obj = getattr(res, s)

    plt.figure()
    plt.suptitle("Mass")

    plt.subplot(421)
    plt.plot(s_obj["time"], s_obj["mass_displacement"], label=s)
    plt.title("Displacement")

    plt.subplot(422)
    plt.plot(s_obj["time"], s_obj["mass_rotation"], label=s)
    plt.title("Rotation")

    plt.subplot(423)
    plt.plot(s_obj["time"], s_obj["mass_velocity"], label=s)
    plt.title("Velocity")

    plt.subplot(424)
    plt.plot(s_obj["time"], s_obj["mass_angular_velocity"], label=s)
    plt.title("Angular velocity")

    plt.subplot(425)
    plt.plot(s_obj["time"], s_obj["force_extremity"], label=s)
    plt.title("Force")

    plt.subplot(426)
    plt.plot(s_obj["time"], s_obj["moment_extremity"], label=s)
    plt.title("Moment")

    plt.subplot(427)
    plt.plot(s_obj["time"], s_obj["curvature"][:, -1], label=s)
    plt.title("Curvature")
    plt.xlabel("Time (s)")

    plt.subplot(428)
    plt.plot(s_obj["time"], s_obj["hysteresis_variable"][:, -1], label=s)
    plt.title("Hysteresis variable")
    plt.xlabel("Time (s)")
    plt.legend()


def plot_clamp_all_versions(
    res1: Result, res2: Result, input_key: str, output_key: str
) -> None:
    """Compare an imposed-vs-recovered quantity between two simulations.

    Parameters
    ----------
    res1 : Result
        First simulation, where ``input_key`` is imposed and ``output_key`` is computed.
    res2 : Result
        Second simulation, where ``output_key`` is imposed and ``input_key`` is computed (used for comparison).
    input_key : str
        Name of the imposed quantity in ``res1`` (e.g. ``"acceleration_clamp"``).
    output_key : str
        Name of the computed quantity in ``res1`` (e.g. ``"force_clamp"``).
    """
    plt.figure()
    plt.suptitle("Clamp")

    plt.subplot(311)
    plt.plot(res1.general["time"], res1.general[input_key])
    plt.title(input_key + " imposed")

    plt.subplot(312)
    plt.plot(res1.general["time"], res1.general[output_key])
    plt.title(output_key + " computed, then imposed")

    plt.subplot(313)
    plt.plot(res1.general["time"], res2.general[input_key])
    plt.title(input_key + " computed")
    plt.xlabel("Time (s)")


def plot_spectrum(time: np.ndarray, value: np.ndarray, dt: float) -> None:
    """Plot a time signal and the modulus of its Fourier spectrum. 

    Parameters
    ----------
    time : np.ndarray
        Time vector corresponding to the signal.    
    value : np.ndarray
        Time signal array.
    dt : float
        Time step. 
    """
    n = len(time)
    N = n // 2
    f = np.fft.fftfreq(n, d=dt)[:N]
    spectrum = np.abs(np.fft.fft(value / n))[:N]

    plt.figure()

    plt.subplot(211)
    plt.plot(time, value)
    plt.xlabel("time")

    plt.subplot(212)
    plt.plot(f, spectrum)
    plt.xlabel("frequencies")
    plt.xscale("log")
    plt.yscale("log")
