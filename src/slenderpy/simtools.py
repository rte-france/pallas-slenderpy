"""Simulation tools."""

import json
import pickle as pk
import time
from typing import Union, Optional, List, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm

# figure default font sizes
__tts__ = 12
__lbs__ = 10
__tks__ = 10

# str names in data array
__stime__ = 'time'
__scabs__ = 'curv'


def _check_lov(lov):
    """Check input lov (list of variables)."""
    if not isinstance(lov, list):
        raise TypeError('input lov must be a list')

    tmp = []
    for v in lov:
        if not isinstance(v, str):
            raise TypeError('lov elements must be strings')
        if v in tmp:
            raise ValueError('lov elements must be unique')
        tmp.append(v)


def _check_los(los):
    """Check input loc (list of s, which are positions of interest)."""
    if not isinstance(los, list):
        raise TypeError('input lov must be a list')

    tmp = []
    for s in los:
        if not isinstance(s, float):
            raise TypeError('los elements must be floats')
        if s in tmp:
            raise ValueError('los elements must be unique')
        if s <= 0. or s >= 1.:
            raise ValueError('los elements must be in ]0,1[ range')
        if len(tmp) > 0 and s <= tmp[-1]:
            raise ValueError('los elements must be in ascending order')
        tmp.append(s)


class Parameters:
    """Simulation parameters."""

    def __init__(self,
                 ns: int = 11,
                 t0: float = 0.,
                 tf: float = 1.,
                 dt: float = 0.01,
                 dr: float = 0.01,
                 los: Union[list, int] = 11,
                 pp: Union[bool, dict] = False) -> None:
        """Init with args.

        Parameters
        ----------
        ns : int, optional
            Number of space discretization points. The default is 11.
        t0 : float, optional
            Start time (s). The default is 0..
        tf : float, optional
            End time (s). Must be greater than t0. The default is 1.
        dt : float, optional
            Time step (s). The default is 0.01.
        dr : float, optional
            Output step. It should be a multiple of dt. The default is 0.01.
        los : list or int, optional
            Positions of interest, ie where on the structure the snapshots will
            be recorded. It must be a list of floats in ]0, 1[ range, in
            ascending order and with no duplicates. If a positive int is given,
            an even-spaced discretization of ]0, 1[ will be provided. The
            default is 11.
        pp : bool or dict, optional
            If a bool, whether or not print progress on stderr. If a dict, it must
            be a list of args to provide to a tqdm progress bar. The default is False.

        Returns
        -------
        None.

        """
        if isinstance(los, int):
            if los >= 2:
                los = np.linspace(0., 1., los + 2)[1:-1].tolist()
            else:
                los = [0.5]
        Parameters._check_input(ns, t0, tf, dt, dr, los, pp)

        nt = int(round((tf - t0) / dt))
        nr = int(round((tf - t0) / dr))

        self.ns = ns  # number of elements in discretization
        self.t0 = t0  # start time (s)
        self.tf = tf  # final time (s)
        self.nt = nt  # number of time steps
        self.nr = nr  # number of (time) outputs
        self.pp = pp  # print progress (or pb args)

        self.los = los  # curvilinear abc. of interest

        if self.nr is None or self.nr > self.nt or self.nr < 1:
            self.nr = self.nt
            self.rr = 1  # output rate (computed from nr)
        else:
            self.nr = int(self.nr)
            self.rr = self.nt // self.nr
            self.nr = self.nt // self.rr

    @staticmethod
    def _check_input(ns, t0, tf, dt, dr, los, pp):
        """Check input values and raise exception if necessary."""
        if not isinstance(ns, int):
            raise TypeError('input ns must be a int')
        if ns < 6:
            raise ValueError('input ns must be larger than 5')

        tmp = {'t0': t0, 'tf': tf, 'dt': dt, 'dr': dr}
        for key, value in tmp.items():
            if not isinstance(value, float):
                raise TypeError(f'input {key} must be a float')

        if tf <= t0:
            raise ValueError('input tf must be larger than t0')
        if dt < 0. or dt >= 0.5 * (tf - t0):
            raise ValueError('input dt must be a positive float and much smaller than tf-t0')
        if dr < 0. or dr < dt:
            raise ValueError('input dr must be larger than (or equal to) dt')

        _check_los(los)

        if not isinstance(pp, bool) and not isinstance(pp, dict):
            raise TypeError('input pp must be a bool')

    def time_vector(self):
        """Get the simulation compute times in a numpy.ndarray."""
        return np.linspace(self.t0, self.tf, 1 + self.nt)

    def time_vector_output(self):
        """Get the simulation output times in a numpy.ndarray."""
        return np.linspace(self.t0, self.tf, 1 + self.nr)


# class to handle simulation results
class Results:
    """Object to handle simulation results."""

    def __init__(self,
                 lot: Optional[List[float]] = None,
                 lov: Optional[List[str]] = None,
                 los: Optional[List[float]] = None,
                 filename: Optional[str] = None) -> None:
        """Init with args.

        If arg filename is provided, data will be read from this file. If
        filename is None, all other args must not be None and the Results will
        be zero-initilized.

        Parameters
        ----------
        lot : list of float, optional
            Times to store. The default is None.
        lov : list of str, optional
            Variables to store. The default is None.
        los : list of float, optional
            Positions of interest. The default is None.
        filename : str, optional
            File to read. The default is None.

        Returns
        -------
        None.

        """
        self.compute_time = None
        self.data = None
        self.state = None
        if filename is not None:
            self.load(filename)
        else:
            self._from_args(lot, lov, los)

    def _from_args(self, lot, lov, los):
        """Build zero dataset from input lists."""
        crd = {__stime__: lot, __scabs__: los}
        dct = {}
        for v in lov:
            dct[v] = ([__stime__, __scabs__],
                      np.nan * np.zeros((len(lot), len(los))))
        self.data = xr.Dataset(dct, coords=crd)

    def los(self):
        """Get a list of positions of interest."""
        return list(self.data[__scabs__].values)

    def lot(self):
        """Get a list of output times."""
        return list(self.data[__stime__].values)

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
            self.data[v][k, :] = np.interp(los, s, lod[i])

    def set_state(self, state):
        """Record State. Internal or expert use only."""
        self.state = state

    def dump(self, filename):
        """Export contnent as a pickle."""
        with open(filename, 'wb') as f:
            pk.dump(self, f)

    def load(self, filename):
        """Load from pickle."""
        with open(filename, 'rb') as f:
            tmp = pk.load(f)
            self.compute_time = tmp.compute_time
            self.data = tmp.data

    # use this to drop variables, positions and/or times
    def drop(self,
             lov: Optional[List[str]] = None,
             los: Optional[List[float]] = None,
             tmin: float = 0.,
             tmax: float = np.inf) -> None:
        """Drop variables, positions of interest or crop time to save space.

        All happens "in place".

        Parameters
        ----------
        lov Variables to drop. If None nothing is dropped. The default is None.
        los : list of floats, optional
            Positions to drop. If None nothing is dropped. The default is None.
        tmin : float, optional
            New first time. ALl recorded times before tmin are removed. The
            default is 0.
        tmax : float, optional
            New last time. All recorded times after tmax are removed. The
            default is np.inf.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.
        """
        if lov is not None:
            self.data = self.data.drop(labels=lov)

        aot = np.array(self.lot())
        ttk = np.where((aot >= tmin) & (aot <= tmax))[0].tolist()

        vtk = self.los()
        itk = list(range(len(vtk)))
        if los is not None and len(los) > 0:
            for s in los:
                if s not in vtk:
                    raise ValueError(f'var {s} not found')

                itk.remove(vtk.index(s))
                vtk.remove(s)

        crd = {__stime__: aot[ttk], __scabs__: vtk}
        dct = {}
        for v in self.lov():
            tmp = self.data[v].values[ttk, :][:, itk]
            dct[v] = ([__stime__, __scabs__], tmp)

        self.data = xr.Dataset(dct, coords=crd)

    def to_netcdf(self, **kwargs):
        """Convert data to netcdf format."""
        return self.data.to_netcdf(**kwargs)

    def to_json(self):
        """Convert data to json format."""
        out = {__stime__: self.data[__stime__].values.tolist(),
               __scabs__: self.data[__scabs__].values.tolist()}
        for v in self.lov():
            tmp = []
            for i, s in enumerate(self.los()):
                tmp.append(self.data[v][:, i].values.tolist())
            out[v] = tmp
        return json.dumps(out)


def multiplot(res: Results,
              lb: Optional[List[str]] = None,
              Lref: float = 0.,
              stl: str = '-',
              log: bool = False,
              t0: float = 0.,
              tf: float = np.inf,
              fst: int = __tts__,
              fsl: int = __lbs__) \
        -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot on a figure one or more structvib.simtools.Results instances.

    Parameters
    ----------
    res : structvib.simtools.Results or list of structvib.simtools.Results
        DESCRIPTION.
    lb : list of str, optional
        Labels for each Result instance in res arg. The default is None.
    Lref : float, optional
        Length of reference. The default is 0..
    stl : str, optional
        Line plot style. The default is '-'.
    log : bool, optional
        Use log axis. The default is False.
    t0 : float, optional
        Minimum time for cropping. The default is 0..
    tf : float, optional
        Maximum time fro cropping. The default is np.inf.
    fst : int, optional
        Title font size. The default is __tts__.
    fsl : int, optional
        Label font size. The default is __lbs__.

    Raises
    ------
    TypeError
        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        Figure which has been generated.
    ax : TYPE
        Axes which have been generated.
    """
    if not isinstance(res, Results):
        if not isinstance(res, list):
            raise TypeError('input res must be a Results or a list')
        if len(res) < 1:
            raise ValueError('input res must not be empty')
        lov = res[0].lov()
        los = res[0].los()
        for r in res:
            if not isinstance(r, Results):
                raise TypeError('input res must be a list of Results')
            if len(r.lov()) != len(lov) or np.any(r.lov() != lov):
                raise ValueError('Results object do not have the same lov')
            if len(r.los()) != len(los) or np.any(r.los() != los):
                raise ValueError('Results object do not have the same los')
    else:
        lov = res.lov()
        los = res.los()
        res = [res]

    nr = len(lov)
    nc = len(los)
    # put default label if empty
    if lb is None:
        lb = [str(i) for i in range(len(res))]
    # colormap
    if len(res) == 1:
        cmap = ['royalblue']
    else:
        cmap = cm.viridis(np.linspace(0., 1., len(res) + 2))[1:-1]

    # big loop
    fig, ax = plt.subplots(nrows=nr, ncols=nc)
    if nc == 1 and nr == 1:
        ax = np.array([[ax]])
    elif nc == 1:
        ax = np.array([[ax[i]] for i in range(len(ax))])
    elif nr == 1:
        ax = np.array([ax])
    for k in range(len(res)):
        if not log:
            res[k].drop(tmin=t0, tmax=tf)
        dfk = res[k].data
        for i in range(nr):
            for j in range(nc):
                if not log:
                    ax[i, j].plot(dfk[__stime__], dfk[lov[i]].loc[:, los[j]],
                                  stl, c=cmap[k], label=lb[k])
                else:
                    ax[i, j].loglog(dfk[__stime__], dfk[lov[i]].loc[:, los[j]],
                                    stl, c=cmap[k], label=lb[k])
    for i in range(nr):
        for j in range(nc):
            ax[i, j].grid(True)
    for i in range(nr):
        ax[i, 0].set_ylabel(lov[i], fontsize=fsl)
    for j in range(nc):
        ax[0, j].set_title(f'@ x={los[j] * Lref:.1E} m ({los[j] * 100.:.1f} %)',
                           fontsize=fst)
        if not log:
            ax[-1, j].set_xlabel('Time (s)', fontsize=fsl)
        else:
            ax[-1, j].set_xlabel('Freq (Hz)', fontsize=fsl)
    ax[-1, -1].legend()

    return fig, ax


def spectrum(res: Results) -> Results:
    """FFT module from a structvib.simtools.Results.

    Parameters
    ----------
    res : structvib.simtools.Results
        Results from a simulation.

    Returns
    -------
    spc : structvib.simtools.Results
        Another Results object with frequencies instead of times and the abs
        value of the fft spectrum of each component.
    """
    T = np.nanmean(np.diff(res.data[__stime__]))
    n = len(res.data[__stime__])
    N = n // 2
    f = np.fft.fftfreq(n, d=T)
    f = f[:N]
    spc = Results(lot=f, lov=res.lov(), los=res.los())
    spc.start_timer()
    for v in res.lov():
        for s in res.los():
            spc.data[v].loc[:, s] = np.abs(np.fft.fft(res.data[v].loc[:, s]) / n)[:N]
    spc.stop_timer()
    return spc
