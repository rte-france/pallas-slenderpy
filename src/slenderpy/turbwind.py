"""Turbulent wind models and associated forces."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
from scipy.interpolate import RectBivariateSpline
from structvib import _progress_bar as spb
from structvib import wind


def von_karman_u(f, mean, std, lx):
    """Von Karman spectrum for u component."""
    n = (lx * f / mean)**2
    c = -5. / 6.
    return 4. * lx * std**2 / mean * np.power(1. + 70.7 * n, c)


def von_karman_v(f, mean, std, lx):
    """Von Karman spectrum for v component."""
    n = (lx * f / mean)**2
    c = -11. / 6.
    return 4. * lx * std**2 / mean * np.power(1. + 282.8 * n, c) * (1. + 753.6 * n)


def von_karman_w(f, mean, std, lx):
    """Von Karman spectrum for w component."""
    n = (lx * f / mean)**2
    c = -11. / 6.
    return 4. * lx * std**2 / mean * np.power(1. + 282.8 * n, c) * (1. + 753.6 * n)


class RandomWind1D:
    """Object to generate a wind signal with Von Karman spectrum."""

    def __init__(self,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 lxu: Optional[float] = None,
                 t0: float = 0.,
                 tf: float = 60.,
                 dt: float = 1.,
                 seed: Optional[int] = None) -> None:
        """Init with args.

        Generated wind is a 1D numpy.ndarray accessible from u member.

        Parameters
        ----------
        mean : float
            Mean wind speed (m/s).
        std : float
            Standard deviation of wind speed (m/s).
        lxu : float
            Turbulent length scale (m).
        t0 : float, optional
            Start time (s). The default is 0.
        tf : float, optional
            End time (s). The default is 60.
        dt : float, optional
            Time step (s). The default is 1.
        seed: int, optional.
            Seed for the random number generator. The default is None (random
            seed).

        Returns
        -------
        None.
        """
        # Random number generator.
        self.rng = np.random.default_rng(seed)

        self.fun = von_karman_u
        self.mean = mean
        self.std = std
        self.lxu = lxu

        self.dt = dt
        self.t = np.linspace(t0, tf, 1 + int(np.floor(tf / dt)))
        self.u = self._generate()

    def _generate(self, fmax=None):

        t = self.t

        if fmax is None:
            T = np.nanmean(np.diff(t))
            n = len(t)
            N = n // 2
            f = np.fft.fftfreq(n, d=T)[:N]
            f = f[:N]
            M = N
        else:
            f = np.logspace(np.log(2. * fmax / len(t)), np.log(fmax), len(t) // 2)
            M = len(f)

        p = self.rng.uniform(0, 2. * np.pi, M - 1)
        s = np.sqrt(np.diff(f) * (self.fun(f[1:], self.mean, self.std, self.lxu) +
                                  self.fun(f[:-1], self.mean, self.std, self.lxu)))
        u = self.mean * np.ones_like(t)
        for i in range(1, M - 1):
            u += s[i] * np.cos(2. * np.pi * t * f[i] + p[i])

        u = self.std * (u - np.mean(u)) / np.std(u) + self.mean

        return u


class TurbWind3D:
    """Object to generate a realistic 3D, turbulent wind Field."""

    def __init__(self,
                 mean: float = 3.0,
                 L: float = 100.,
                 N: int = 51,
                 t0: float = 0.,
                 tf: float = 60.,
                 dt: float = 1.,
                 stdU: float = 0.3,
                 stdV: float = 0.3,
                 stdW: float = 0.1,
                 lux: float = 200.,
                 lvx: float = 70.,
                 lwx: float = 30.,
                 cuy: float = 7.,
                 cuz: float = 10.,
                 cvy: float = 70.,
                 cvz: float = 10.,
                 cwy: float = 7.,
                 cwz: float = 10.,
                 ks: int = 1,
                 kt: int = 1,
                 pp: bool = False,
                 seed: Optional[int] = None) -> None:
        """Init with args.

        Wind fields generation is performed at init.

        Parameters
        ----------
        mean : float, optional
            Mean wind speed (m/s) in principal direction. The default is 3.
        L : float, optional
            Lenght (m) of simulation line. The default is 100.
        N : int, optional
            Numer of space points. The default is 51.
        t0 : float, optional
            Start time (s). The default is 0.
        tf : float, optional
            End time (s). The default is 60.
        dt : float, optional
            Time step (s). The default is 1.
        stdU : float, optional
            Standard deviation of wind fluctuations (m/s) for u component. The
            default is 0.3.
        stdV : float, optional
            Standard deviation of wind fluctuations (m/s) for v component. The
            default is 0.3.
        stdW : float, optional
            Standard deviation of wind fluctuations (m/s) for w component. The
            default is 0.1.
        lux : float, optional
            Turbulence length scale (m) for u component. The default is 200.
        lvx : float, optional
            Turbulence length scale (m) for v component. The default is 70.
        lwx : float, optional
            Turbulence length scale (m) for w component. The default is 30.
        cuy : float, optional
            Co-coherence decay coefficient. The default is 7.
        cuz : float, optional
            Co-coherence decay coefficient. The default is 10.
        cvy : float, optional
            Co-coherence decay coefficient. The default is 70.
        cvz : float, optional
            Co-coherence decay coefficient. The default is 10.
        cwy : float, optional
            Co-coherence decay coefficient. The default is 7.
        cwz : float, optional
            Co-coherence decay coefficient. The default is 10.
        ks : int, optional
            Degree of polynomial interpolation in space. The default is 1.
        kt : int, optional
            Degree of polynomial interpolation in time. The default is 1.
        pp : bool, optional
            Print (or not) progress on stderr. The default is False.
        seed: int, optional.
            Seed for the random number generator. The default is None (random
            seed).
        Returns
        -------
        None.
        """
        # Random number generator.
        self.rng = np.random.default_rng(seed)

        # TIME
        self.dt = dt  # time step (s)
        self.tf = tf  # time duration (s)
        self.t = None  # time vector (s)
        self.fs = None  # sampling frequency (Hz)
        self.f = None  # frequency vector (Hz)

        # GRID
        self.L = L  # line length (m)
        self.N = N  # number of points on line
        self.y = None  # line points (m)

        # WIND DATA
        self.stdU = stdU  # std of wind fluctuations for u component (m/s)
        self.stdV = stdV  # std of wind fluctuations for v component (m/s)
        self.stdW = stdW  # std of wind fluctuations for w component (m/s)
        self.lux = lux  # turbulence length scale for u component along wind direction (m)
        self.lvx = lvx  # turbulence length scale for v component along wind direction (m)
        self.lwx = lwx  # turbulence length scale for w component along wind direction (m)
        self.cuy = cuy  # co-coherence decay coefficient for u-component
        self.cuz = cuz  # co-coherence decay coefficient for u-component
        self.cvy = cvy  # co-coherence decay coefficient for v-component
        self.cvz = cvz  # co-coherence decay coefficient for v-component
        self.cwy = cwy  # co-coherence decay coefficient for w-component
        self.cwz = cwz  # co-coherence decay coefficient for w-component

        # RESULTS
        self.uu = None
        self.vv = None
        self.ww = None
        self.iu = None

        # Compute useful values from members and generate wind field
        self.mean = mean * np.ones((self.N,))  # speed profile at eq. (m/s)
        self.t = np.linspace(t0, self.tf, 1 + int(np.floor((tf - t0) / self.dt)))
        self.fs = 1. / self.dt
        self.f = np.arange(1. / (tf - t0), 0.5 * self.fs, 1. / (tf - t0))
        self.y = np.linspace(0., self.L, self.N)

        self._generate(pp)

        # set interpolation
        self.iu = RectBivariateSpline(self.t, self.y, self.uu.T, kx=kt, ky=ks)
        self.iv = RectBivariateSpline(self.t, self.y, self.vv.T, kx=kt, ky=ks)
        self.iw = RectBivariateSpline(self.t, self.y, self.ww.T, kx=kt, ky=ks)

    def get(self, t, y, u=True, v=True, w=False):
        """Get wind components using interpolation."""
        if u:
            uu = self.iu(t, y, grid=True)
        else:
            uu = None
        if v:
            vv = self.iv(t, y, grid=True)
        else:
            vv = None
        if w:
            ww = self.iw(t, y, grid=True)
        else:
            ww = None
        return uu, vv, ww

    def _generate(self, pp=False):
        """Generate of turbulent wind field."""
        nt = len(self.t)
        ix = range(0, self.N)
        kk = np.tile(ix, (len(ix), 1)).T
        mm = np.tile(ix, (len(ix), 1))
        dy = np.abs(self.y[kk] - self.y[mm])
        dz = np.zeros_like(dy)
        um = 0.5 * (self.mean[kk] + self.mean[mm])

        R = np.zeros((3 * self.N, nt))
        pb = spb.generate(pp, len(self.f), desc=__name__)
        for jj in range(len(self.f)):
            cu = np.exp(-self.f[jj] * np.sqrt((self.cuy * dy)**2 + (self.cuz * dz)**2) / um)
            cv = np.exp(-self.f[jj] * np.sqrt((self.cvy * dy)**2 + (self.cvz * dz)**2) / um)
            cw = np.exp(-self.f[jj] * np.sqrt((self.cwy * dy)**2 + (self.cwz * dz)**2) / um)

            su = von_karman_u(self.f[jj], self.mean, self.stdU, self.lux)
            sv = von_karman_v(self.f[jj], self.mean, self.stdV, self.lvx)
            sw = von_karman_w(self.f[jj], self.mean, self.stdW, self.lwx)

            su = cu * np.sqrt(np.matmul(su, su.T))
            sv = cv * np.sqrt(np.matmul(sv, sv.T))
            sw = cw * np.sqrt(np.matmul(sw, sw.T))

            zr = np.zeros_like(su)
            S = np.block([[su, zr, zr], [zr, sv, zr], [zr, zr, sw]])

            phi = 2. * np.pi * self.rng.random(3 * self.N)
            phi = np.tile(phi, (nt, 1)).T

            A = np.cos(2. * np.pi * self.f[jj] * np.tile(self.t, (3 * self.N, 1)) + phi)
            G = np.linalg.cholesky(S)
            R = R + np.sqrt(2. * np.median(np.diff(self.f))) * np.matmul(np.abs(G), A)

            pb.update()

        pb.close()

        self.uu = R[:self.N, :]
        self.vv = R[self.N: 2 * self.N, :]
        self.ww = R[2 * self.N:, :]

        self.uu = self.uu + np.tile(self.mean, (nt, 1)).T
        self.uu = self.stdU * (self.uu - np.mean(self.uu)) / np.std(self.uu) + self.mean[0]
        self.vv = self.stdV * (self.vv - np.mean(self.vv)) / np.std(self.vv) + 0.
        self.ww = self.stdW * (self.ww - np.mean(self.ww)) / np.std(self.ww) + 0.


class Force1D:
    """Force computation from a 1D wind signal."""

    def __init__(self,
                 mean: Optional[float] = None,
                 std: Optional[float] = None,
                 lxu: Optional[float] = None,
                 t0: float = 0.,
                 tf: float = 60.,
                 dt: float = 1.,
                 cd: Optional[float] = None,
                 rho: float = 1.2,
                 d: Optional[float] = None):
        """Init with args.

        Parameters
        ----------
        mean : float
            Mean wind speed (m/s).
        std : float
            Standard deviation of wind speed (m/s).
        lxu : float
            Turbulent length scale (m).
        t0 : float, optional
            Start time (s). The default is 0.
        tf : float, optional
            End time (s). The default is 60.
        dt : float, optional
            Time step (s). The default is 1.
        cd : float
            Drag coefficient.
        rho : float, optionnal
            Air volumic mass (kg/m**3). The default is 1.2.
        d : float
            Cable diameter (m).

        Returns
        -------
        None.
        """
        self.wnd = RandomWind1D(mean=mean, std=std, lxu=lxu, t0=t0, tf=tf, dt=dt)
        self.cd = cd
        self.alpha = 0.5 * rho * d

    def _fast_interp(self, t):
        """A fast interpolation method."""

        def lincom(x, q, i):
            return (1 - q + i) * x[i] + (q - i) * x[i + 1]

        q = (t - self.wnd.t[0]) / self.wnd.dt
        i = int(np.floor(q))

        if i < 0:
            ws = self.wnd.u[0]
        elif i >= len(self.wnd.t) - 1:
            ws = self.wnd.u[-1]
        else:
            ws = lincom(self.wnd.u, q, i)

        return ws

    def __call__(self, s, t, un, ub, vn, vb):
        """Compute force."""
        ws = self._fast_interp(t)

        al = self.alpha * np.ones_like(s)
        sq = np.sqrt((0. - vn)**2 + (ws - vb)**2)
        fn = al * sq * self.cd * (0. - vn)
        fb = al * sq * self.cd * (ws - vb)
        return fn, fb


class Force2D:
    """Force computation from a 2D turbulent wind field."""

    def __init__(self,
                 Lp: float = 100.,
                 N: int = 51,
                 t0: float = 0.,
                 tf: float = 60.,
                 dt: float = 1.,
                 d: float = 0.1,
                 cd: Optional[float] = None,
                 rho: Optional[float] = None,
                 meanU: float = 3.0,
                 stdU: float = 0.3,
                 stdV: float = 0.3,
                 stdW: float = 0.1,
                 lxu: float = 200.,
                 lxv: float = 70.,
                 lxw: float = 30.,
                 cuy: float = 7.,
                 cuz: float = 10.,
                 cvy: float = 70.,
                 cvz: float = 10.,
                 cwy: float = 7.,
                 cwz: float = 10.,
                 T: float = 293.15,
                 p: float = 1.013E+05,
                 phi: float = 0.,
                 pp: bool = False,
                 seed: Optional[int] = None) -> None:
        """Init with args.

        Parameters
        ----------
        Lp : float, optional
            Lenght (m) of simulation line. The default is 100.
        N : int, optional
            Numer of space points. The default is 51.
        t0 : float, optional
            Start time (s). The default is 0.
        tf : float, optional
            End time (s). The default is 60.
        dt : float, optional
            Time step (s). The default is 1.
        d : float
            Cable diameter (m).
        cd : float
            Drag coefficient.
        rho : float, optionnal
            Air volumic mass (kg/m**3). The default is 1.2.
        meanU : float, optional
            Mean wind speed (m/s) in principal direction. The default is 3.
        stdU : float, optional
            Standard deviation of wind fluctuations (m/s) for u component. The
            default is 0.3.
        stdV : float, optional
            Standard deviation of wind fluctuations (m/s) for v component. The
            default is 0.3.
        stdW : float, optional
            Standard deviation of wind fluctuations (m/s) for w component. The
            default is 0.1.
        lxu : float, optional
            Turbulence length scale (m) for u component. The default is 200.
        lxv : float, optional
            Turbulence length scale (m) for v component. The default is 70.
        lxw : float, optional
            Turbulence length scale (m) for w component. The default is 30.
        cuy : float, optional
            Co-coherence decay coefficient. The default is 7.
        cuz : float, optional
            Co-coherence decay coefficient. The default is 10.
        cvy : float, optional
            Co-coherence decay coefficient. The default is 70.
        cvz : float, optional
            Co-coherence decay coefficient. The default is 10.
        cwy : float, optional
            Co-coherence decay coefficient. The default is 7.
        cwz : float, optional
            Co-coherence decay coefficient. The default is 10.
        T : float, optional
            Temperature (K). The default is 293.15.
        p : float, optional
            Pressure (Pa). The default is 1.013E+05.
        phi : float, optional
            Relative humidity (in [0, 1] range). The default is 0.
        pp : bool, optional
            Print (or not) progress on stderr. The default is False.
        seed: int, optional.
            Seed for the random number generator. The default is None (random
            seed).

        Returns
        -------
        None.
        """
        self.tbw = TurbWind3D(mean=meanU, L=Lp, N=N, t0=t0, tf=tf, dt=dt,
                              stdU=stdU, stdV=stdV, stdW=stdW, lux=lxu,
                              lvx=lxv, lwx=lxw, cuy=cuy, cuz=cuz, cvy=cvy,
                              cvz=cvz, cwy=cwy, cwz=cwz, ks=1, kt=1, pp=pp,
                              seed=seed)

        self.cd = cd
        if rho is None:
            rho = wind.air_volumic_mass(T, p, phi)
        self.alpha = 0.5 * rho * d
        self.beta = d / wind.air_kinematic_viscosity(T, p, phi)

    def _fast_interp(self, s, t):
        """A fast interpolation method."""

        def lincom2(x, q, i):
            return (1 - q + i) * x[:, i] + (q - i) * x[:, i + 1]

        def lincom1(x, q, i, j, k):
            r = (1 - q + i) * x[i] + (q - i) * x[i + 1]
            r[j] = x[0]
            r[k] = x[-1]
            return r

        # time
        q = (t - self.tbw.t[0]) / self.tbw.dt
        i = int(np.floor(q))
        if i < 0:
            uu = self.tbw.uu[:, 0]
            ww = self.tbw.ww[:, 0]
        elif i >= len(self.tbw.t) - 1:
            uu = self.tbw.uu[:, -1]
            ww = self.tbw.ww[:, -1]
        else:
            uu = lincom2(self.tbw.uu, q, i)
            ww = lincom2(self.tbw.ww, q, i)

        # space
        q = s * (self.tbw.N - 1)
        i = np.floor(q).astype(int)
        j = np.where(i < 0)[0]
        k = np.where(i >= self.tbw.N - 1)[0]
        i[j] = 0
        i[k] = 0
        uu = lincom1(uu, q, i, j, k)
        ww = lincom1(ww, q, i, j, k)

        return uu, ww

    def __call__(self, s, t, un, ub, vn, vb):
        """Compute force."""
        # wb, wn, _ = self.tbw.get(t, s * self.tbw.L, u=True, v=True, w=False)
        # wb = wb[0, :]
        # wn = wn[0, :]

        wb, wn = self._fast_interp(s, t)
        rn = wn - vn
        rb = wb - vb
        fs = np.sqrt(rn**2 + rb**2)

        if self.cd is None:
            cd = wind.cylinder_drag(fs * self.beta)
        else:
            cd = self.cd

        gm = self.alpha * cd * fs
        fn = gm * rn
        fb = gm * rb
        return fn, fb


if __name__ == '__main__':

    # Some test of wind features

    import matplotlib.pyplot as mpl

    mpl.close('all')


    def _fft(t, s):
        T = np.nanmean(np.diff(t))
        n = len(t)
        N = n // 2
        f = np.fft.fftfreq(n, d=T)
        f = f[:N]
        y = np.abs(np.fft.fft(s) / n)[:N]
        return f, y


    t0 = 7.
    tf = 127.
    dt = 0.1

    um = 3.0
    us = 0.3
    ul = 200.
    vm = 0.0
    vs = 0.3
    vl = 70.
    wm = 0.
    ws = 0.1
    wl = 30.

    N = 3
    t = np.arange(t0, tf, dt)
    f = np.logspace(-6, +4, 1001)

    f1d = Force1D(mean=um, std=us, lxu=ul, t0=t0, tf=tf, dt=dt, d=0.1, cd=1.1)

    f2d = Force2D(meanU=um, Lp=100., N=N, t0=t0, tf=tf, dt=dt, stdU=us,
                  stdV=vs, stdW=ws, lxu=ul, lxv=vl, lxw=wl, d=0.1, cd=1.1)

    r1d = f1d.wnd
    r3d = f2d.tbw

    fig, ax = mpl.subplots(nrows=3, ncols=2)

    ax[0, 1].loglog(f, von_karman_u(f, um, us, ul), '--k')
    ax[1, 1].loglog(f, von_karman_v(f, um, vs, vl), '--k')
    ax[2, 1].loglog(f, von_karman_w(f, um, ws, wl), '--k')

    k = 1
    ax[0, 0].plot(r3d.t, r3d.uu[k, :])
    ax[1, 0].plot(r3d.t, r3d.vv[k, :])
    ax[2, 0].plot(r3d.t, r3d.ww[k, :])

    fu, yu = _fft(r3d.t, r3d.uu[k, :])
    fv, yv = _fft(r3d.t, r3d.vv[k, :])
    fw, yw = _fft(r3d.t, r3d.ww[k, :])

    ax[0, 1].loglog(fu, yu**2 * (tf - t0))
    ax[1, 1].loglog(fv, yv**2 * (tf - t0))
    ax[2, 1].loglog(fw, yw**2 * (tf - t0))

    print(f'[3d, u] mean = {np.mean(r3d.uu[k, :]):+.3f}, std = {np.std(r3d.uu[k, :]):+.3f}')
    print(f'[3d, v] mean = {np.mean(r3d.vv[k, :]):+.3f}, std = {np.std(r3d.vv[k, :]):+.3f}')
    print(f'[3d, w] mean = {np.mean(r3d.ww[k, :]):+.3f}, std = {np.std(r3d.ww[k, :]):+.3f}')

    ax[0, 0].set_title('Wind speed (u dir, m/s)')
    ax[1, 0].set_title('Wind speed (v dir, m/s)')
    ax[2, 0].set_title('Wind speed (w dir, m/s)')

    ax[0, 1].set_title('Wind spectrum (u dir)')
    ax[1, 1].set_title('Wind spectrum (v dir)')
    ax[2, 1].set_title('Wind spectrum (w dir)')

    ax[0, 0].plot(r1d.t, r1d.u)
    fm, ym = _fft(r1d.t, r1d.u)
    ax[0, 1].loglog(fm, ym**2 * (tf - t0))

    print(f'[1d, u] mean = {np.mean(r1d.u):+.3f}, std = {np.std(r1d.u):+.3f}')

    ax[-1, 0].set_xlabel('Time (s)')
    ax[-1, 1].set_xlabel('Freq (Hz)')
    for i in range(3):
        for j in range(2):
            ax[i, j].grid(True)

    t = np.linspace(t0 - 7., tf + 13, 2 * len(t))
    s = np.array([50.])

    f1n = np.zeros_like(t)
    f1b = np.zeros_like(t)
    f2n = np.zeros_like(t)
    f2b = np.zeros_like(t)

    for k, tk in enumerate(t):
        tmp = f1d(s, tk, None, None, 0., 0.)
        f1n[k] = tmp[0][0]
        f1b[k] = tmp[1][0]
        tmp = f2d(s, tk, None, None, 0., 0.)
        f2n[k] = tmp[0][0]
        f2b[k] = tmp[1][0]

    fig, ax = mpl.subplots(nrows=2, ncols=1)
    ax[0].plot(t, f2n)
    ax[0].plot(t, f1n)

    ax[1].plot(t, f2b)
    ax[1].plot(t, f1b)
