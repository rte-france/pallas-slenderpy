"""Simulation configuration (:class:`Parameters`) and time-series results
(:class:`Results`).

Ported from :mod:`slenderpy.simtools` with bug fixes and modernization. The
public API is kept compatible so existing solvers can migrate with an import
swap. ``Results`` is backed by an :class:`xarray.Dataset` with a ``time`` x
``curv`` (curvilinear abscissa) layout.
"""

from __future__ import annotations

import json
import pickle as pk
import time
from collections.abc import Sequence

import numpy as np
import xarray as xr

# Coordinate names in the underlying dataset.
_TIME = "time"
_CURV = "curv"


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
