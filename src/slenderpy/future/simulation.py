"""Simulation configuration (:class:`Parameters`), time-series results
(:class:`Results`) and the helpers to plot (:func:`multiplot`) and transform
(:func:`spectrum`) them.

Ported from :mod:`slenderpy.simtools` with bug fixes and modernization. The
public API is kept compatible so existing solvers can migrate with an import
swap. ``Results`` is backed by an :class:`xarray.Dataset` with a ``time`` x
``curv`` (curvilinear abscissa) layout.
"""

from __future__ import annotations

import json
import pickle as pk
import time
import warnings
from collections.abc import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Coordinate names in the underlying dataset.
_TIME = "time"
_CURV = "curv"

# Default figure font sizes.
_TITLE_SIZE = 12
_LABEL_SIZE = 10


def _check_los(los):
    """Check input los (list of positions of interest)."""
    if not isinstance(los, list):
        raise TypeError("input los must be a list")

    tmp = []
    for s in los:
        if not isinstance(s, float):
            raise TypeError("los elements must be floats")
        if s in tmp:
            raise ValueError("los elements must be unique")
        if s <= 0.0 or s >= 1.0:
            raise ValueError("los elements must be in ]0,1[ range")
        if len(tmp) > 0 and s <= tmp[-1]:
            raise ValueError("los elements must be in ascending order")
        tmp.append(s)


class Parameters:
    """Simulation parameters (time stepping and output configuration)."""

    def __init__(
        self,
        ns: int = 11,
        t0: float = 0.0,
        tf: float = 1.0,
        dt: float = 0.01,
        dr: float = 0.01,
        los: list | int = 11,
        pp: bool | dict = False,
    ) -> None:
        """Init with args.

        Parameters
        ----------
        ns : int, optional
            Number of space discretization points. The default is 11.
        t0 : float, optional
            Start time (s). The default is 0.
        tf : float, optional
            End time (s). Must be greater than t0. The default is 1.
        dt : float, optional
            Time step (s), must be positive. The default is 0.01.
        dr : float, optional
            Output step (s), must be at least dt. The default is 0.01.
        los : list or int, optional
            Positions of interest, where snapshots are recorded. A list of
            floats in ]0, 1[, ascending and unique. A positive int >= 2 gives
            that many evenly spaced interior points; an int < 2 gives [0.5].
            The default is 11.
        pp : bool or dict, optional
            If a bool, whether to print progress. If a dict, tqdm progress-bar
            args. The default is False.
        """
        if isinstance(los, int):
            if los >= 2:
                los = np.linspace(0.0, 1.0, los + 2)[1:-1].tolist()
            else:
                los = [0.5]
        Parameters._check_input(ns, t0, tf, dt, dr, los, pp)

        t0, tf, dt, dr = float(t0), float(tf), float(dt), float(dr)
        nt = int(round((tf - t0) / dt))
        nr = int(round((tf - t0) / dr))

        # Reconcile the number of outputs with an integer output rate.
        if nr > nt or nr < 1:
            nr = nt
            rr = 1
        else:
            rr = nt // nr
            nr = nt // rr

        self.ns = ns  # number of elements in discretization
        self.t0 = t0  # start time (s)
        self.tf = tf  # final time (s)
        self.nt = nt  # number of time steps
        self.nr = nr  # number of (time) outputs
        self.rr = rr  # output rate (one output every rr steps)
        self.pp = pp  # print progress (or progress-bar args)
        self.los = los  # curvilinear abscissas of interest

    @staticmethod
    def _check_input(ns, t0, tf, dt, dr, los, pp):
        """Check input values and raise an exception if necessary."""
        if not isinstance(ns, int):
            raise TypeError("input ns must be an int")
        if ns < 6:
            raise ValueError("input ns must be larger than 5")

        tmp = {"t0": t0, "tf": tf, "dt": dt, "dr": dr}
        for key, value in tmp.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"input {key} must be a real number")

        if tf <= t0:
            raise ValueError("input tf must be larger than t0")
        if dt <= 0.0 or dt >= 0.5 * (tf - t0):
            raise ValueError(
                "input dt must be a positive float much smaller than tf-t0"
            )
        if dr < 0.0 or dr < dt:
            raise ValueError("input dr must be larger than (or equal to) dt")

        _check_los(los)

        if not isinstance(pp, (bool, dict)):
            raise TypeError("input pp must be a bool or a dict")

    def time_vector(self) -> np.ndarray:
        """Get the simulation compute times."""
        return np.linspace(self.t0, self.tf, 1 + self.nt)

    def time_vector_output(self) -> np.ndarray:
        """Get the simulation output times."""
        return np.linspace(self.t0, self.tf, 1 + self.nr)


class Results:
    """Object to handle simulation results."""

    def __init__(
        self,
        lot: list[float] | None = None,
        lov: list[str] | None = None,
        lov_dims: Sequence[int] | None = None,
        los: list[float] | None = None,
        filename: str | None = None,
    ) -> None:
        """Init with args.

        If ``filename`` is provided, data is read from that file. Otherwise all
        other args must be given and the results are zero-initialized.

        Parameters
        ----------
        lot : list of float, optional
            Times to store. The default is None.
        lov : list of str, optional
            Variables to store. The default is None.
        lov_dims : sequence of int, optional
            Dimensionality of each variable: 1 (scalar) or 2 (vector over
            positions). Defaults to 2 for every variable.
        los : list of float, optional
            Positions of interest. The default is None.
        filename : str, optional
            File to read. The default is None.
        """
        self.compute_time = None
        self.data = None
        self.state = None
        if filename is not None:
            self.load(filename)
        else:
            self._from_args(lot, lov, lov_dims, los)

    def _from_args(self, lot, lov, lov_dims=None, los=None):
        """Build a zero dataset from input lists."""
        if lov_dims is None:
            lov_dims = 2 * np.ones(len(lov))
        crd = {_TIME: lot, _CURV: los}
        dct = {}
        self.lov_dims = {}
        for index, v in enumerate(lov):
            if lov_dims[index] == 2:
                dct[v] = (
                    [_TIME, _CURV],
                    np.nan * np.zeros((len(lot), len(los))),
                )
            elif lov_dims[index] == 1:
                dct[v] = ([_TIME], np.nan * np.zeros(len(lot)))
            else:
                raise ValueError(f"lov_dims[{v}] must be 1 (scalar) or 2 (vectorial).")
            self.lov_dims[v] = lov_dims[index]
        self.data = xr.Dataset(dct, coords=crd)

    def los(self):
        """Get a list of positions of interest."""
        val = self.data[_CURV].values
        if val.ndim == 0:
            return []
        return list(val)

    def lot(self):
        """Get a list of output times."""
        return list(self.data[_TIME].values)

    def lov(self):
        """Get a list of stored variables."""
        return list(self.data.data_vars.keys())

    def start_timer(self):
        """Start a time measurement."""
        self.compute_time = time.time()

    def stop_timer(self):
        """Stop a time measurement."""
        self.compute_time = time.time() - self.compute_time

    def update(self, k, s, lov, lod):
        """Record a snapshot. Internal or expert use only."""
        los = self.los()
        for i, v in enumerate(lov):
            if self.lov_dims[v] == 2:
                self.data[v][k, :] = np.interp(los, s, lod[i])
            else:
                self.data[v][k] = lod[i]

    def __getitem__(self, key):
        return self.data[key]

    def set_state(self, state):
        """Record state. Internal or expert use only."""
        self.state = state

    def dump(self, filename):
        """Export content as a pickle."""
        with open(filename, "wb") as f:
            pk.dump(self, f)

    def load(self, filename):
        """Load from pickle, restoring every attribute."""
        with open(filename, "rb") as f:
            tmp = pk.load(f)
        self.__dict__.update(tmp.__dict__)

    def drop(
        self,
        lov: list[str] | None = None,
        los: list[float] | None = None,
        tmin: float = 0.0,
        tmax: float = np.inf,
    ) -> None:
        """Drop variables, positions of interest or crop time to save space.

        All happens in place.

        Parameters
        ----------
        lov : list of str, optional
            Variables to drop. If None nothing is dropped. The default is None.
        los : list of float, optional
            Positions to drop. If None nothing is dropped. The default is None.
        tmin : float, optional
            New first time; earlier recorded times are removed. The default is 0.
        tmax : float, optional
            New last time; later recorded times are removed. The default is inf.

        Raises
        ------
        ValueError
            If a position to drop is not found.
        """
        if lov is not None:
            self.data = self.data.drop_vars(lov)

        aot = np.array(self.lot())
        ttk = np.where((aot >= tmin) & (aot <= tmax))[0].tolist()

        vtk = self.los()
        itk = list(range(len(vtk)))
        if los is not None and len(los) > 0:
            for s in los:
                if s not in vtk:
                    raise ValueError(f"var {s} not found")

                itk.remove(vtk.index(s))
                vtk.remove(s)

        crd = {_TIME: aot[ttk], _CURV: vtk}
        dct = {}
        for v in self.lov():
            tmp = self.data[v].values
            if self.lov_dims[v] == 2:
                tmp = tmp[ttk, :][:, itk]
                dct[v] = ([_TIME, _CURV], tmp)
            else:
                tmp = tmp[ttk]
                dct[v] = ([_TIME], tmp)

        self.data = xr.Dataset(dct, coords=crd)

    def to_netcdf(self, **kwargs):
        """Convert data to netcdf format."""
        return self.data.to_netcdf(**kwargs)

    def to_json(self):
        """Convert data to json format."""
        out = {
            _TIME: self.data[_TIME].values.tolist(),
            _CURV: self.data[_CURV].values.tolist(),
        }
        for v in self.lov():
            if self.lov_dims[v] == 2:
                tmp = []
                for i, s in enumerate(self.los()):
                    tmp.append(self.data[v][:, i].values.tolist())
                out[v] = tmp
            else:
                out[v] = self.data[v].values.tolist()

        return json.dumps(out)


def _as_results_list(res: Results | list[Results]) -> list[Results]:
    """Normalize the multiplot input to a list of consistent Results."""
    if isinstance(res, Results):
        return [res]
    if not isinstance(res, list):
        raise TypeError("input res must be a Results or a list of Results")
    if len(res) < 1:
        raise ValueError("input res must not be empty")
    for r in res:
        if not isinstance(r, Results):
            raise TypeError("input res must be a list of Results")

    ref = res[0]
    for r in res[1:]:
        if r.lov() != ref.lov():
            raise ValueError("all Results must store the same variables (lov)")
        if not np.array_equal(r.los(), ref.los()):
            raise ValueError("all Results must store the same positions (los)")
        if r.lov_dims != ref.lov_dims:
            raise ValueError("all Results must store the same variable dimensions")
    return res


def multiplot(
    res: Results | list[Results],
    lb: list[str] | None = None,
    Lref: float = 1.0,
    stl: str = "-",
    log: bool = False,
    t0: float = 0.0,
    tf: float = np.inf,
    fst: int = _TITLE_SIZE,
    fsl: int = _LABEL_SIZE,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot on a single figure one or more Results instances.

    The figure is a grid with one row per stored variable and one column per
    position of interest. A scalar variable (``lov_dims == 1``) does not depend
    on position, so it gets a single axes spanning its whole row.

    Parameters
    ----------
    res : Results or list of Results
        Simulation results to plot. When several are given they must store the
        same variables, positions and dimensions.
    lb : list of str, optional
        One label per Results instance. Defaults to their index.
    Lref : float, optional
        Reference length (m) used to turn the normalized positions into a
        physical abscissa in the column titles. The default is 1.
    stl : str, optional
        Line plot style. The default is '-'.
    log : bool, optional
        Use log-log axes and label the x axis as a frequency, ie plot the
        output of :func:`spectrum`. The default is False.
    t0 : float, optional
        Lower bound of the plotted time (or frequency) window. The default is 0.
    tf : float, optional
        Upper bound of the plotted time (or frequency) window. The default is inf.
    fst : int, optional
        Title font size. The default is 12.
    fsl : int, optional
        Label font size. The default is 10.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ax : numpy.ndarray
        Object array of Axes with shape (number of variables, number of
        positions). Every column of a scalar variable row references the same
        spanning axes.

    Raises
    ------
    TypeError
        If res is not a Results or a list of Results.
    ValueError
        If res is empty, holds no variable, mixes inconsistent Results, or if
        lb does not have one label per Results.
    """
    res = _as_results_list(res)
    ref = res[0]
    lov = ref.lov()
    los = ref.los()
    if len(lov) < 1:
        raise ValueError("input res must store at least one variable")

    nr = len(lov)
    nc = max(len(los), 1)  # a Results with only scalar variables has no position

    if lb is None:
        lb = [str(i) for i in range(len(res))]
    elif len(lb) != len(res):
        raise ValueError("input lb must hold one label per Results")

    # One color per Results, sampled inside viridis to skip its extremes.
    colors: list = ["royalblue"]
    if len(res) > 1:
        cmap = plt.get_cmap("viridis")
        colors = list(cmap(np.linspace(0.0, 1.0, len(res) + 2))[1:-1])

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=nr, ncols=nc)
    ax = np.empty((nr, nc), dtype=object)
    for i, v in enumerate(lov):
        if ref.lov_dims[v] == 2:
            for j in range(nc):
                ax[i, j] = fig.add_subplot(gs[i, j])
        else:
            spanning = fig.add_subplot(gs[i, :])
            for j in range(nc):
                ax[i, j] = spanning

    for k, r in enumerate(res):
        # Select instead of Results.drop so the caller's data stays untouched.
        dat = r.data.sel({_TIME: slice(t0, tf)})
        for i, v in enumerate(lov):
            if ref.lov_dims[v] == 2:
                for j, s in enumerate(los):
                    ax[i, j].plot(
                        dat[_TIME], dat[v].loc[:, s], stl, c=colors[k], label=lb[k]
                    )
            else:
                ax[i, 0].plot(dat[_TIME], dat[v], stl, c=colors[k], label=lb[k])

    for i in range(nr):
        ax[i, 0].set_ylabel(lov[i], fontsize=fsl)
        for j in range(nc):
            ax[i, j].grid(True)
            if log:
                ax[i, j].set_xscale("log")
                ax[i, j].set_yscale("log")

    xlabel = "Freq (Hz)" if log else "Time (s)"
    for j in range(nc):
        ax[-1, j].set_xlabel(xlabel, fontsize=fsl)

    # Column titles belong to the topmost row that actually varies with position.
    titled = next((i for i, v in enumerate(lov) if ref.lov_dims[v] == 2), None)
    if titled is not None:
        for j, s in enumerate(los):
            ax[titled, j].set_title(
                f"@ x={s * Lref:.1E} m ({s * 100.0:.1f} %)", fontsize=fst
            )

    ax[-1, -1].legend()
    fig.tight_layout()

    return fig, ax


def spectrum(res: Results) -> Results:
    """FFT modulus of every variable of a Results.

    The returned Results stores frequencies (Hz) in place of times, keeping the
    same variables, dimensions and positions. Values are the one-sided modulus
    ``abs(fft(x) / n)`` over the first ``n // 2`` bins, the DC bin included;
    this 1/n normalization is the same convention as :mod:`slenderpy.simtools`,
    so a unit-amplitude sine peaks at 0.5.

    Parameters
    ----------
    res : Results
        Results from a simulation, expected to be sampled at a constant rate.

    Returns
    -------
    spc : Results
        Spectrum of each variable of the input.

    Raises
    ------
    ValueError
        If the input holds fewer than two time samples.

    Warns
    -----
    UserWarning
        If the time sampling is not uniform, or if a variable holds NaN (which
        propagates to its whole spectrum).
    """
    lot = np.asarray(res.lot(), dtype=float)
    n = len(lot)
    if n < 2:
        raise ValueError("input res must hold at least two time samples")

    steps = np.diff(lot)
    if not np.allclose(steps, steps[0]):
        warnings.warn(
            "time sampling is not uniform, using the mean time step", stacklevel=2
        )
    dt = float(np.nanmean(steps))

    nf = n // 2
    freq = np.fft.fftfreq(n, d=dt)[:nf]

    lov = res.lov()
    spc = Results(
        lot=freq.tolist(),
        lov=lov,
        lov_dims=[res.lov_dims[v] for v in lov],
        los=res.los(),
    )
    spc.start_timer()
    for v in lov:
        values = res.data[v].values
        if np.isnan(values).any():
            warnings.warn(
                f"variable {v} holds NaN, its spectrum will be NaN", stacklevel=2
            )
        spc.data[v][:] = np.abs(np.fft.fft(values, axis=0) / n)[:nf]
    spc.stop_timer()

    return spc
