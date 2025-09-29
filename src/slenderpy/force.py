"""Force objects."""

from typing import Optional, Union, Tuple, Any

import numpy as np
from slenderpy.wind import air_volumic_mass


class WOP:
    """Wake oscillator parameterss."""

    def __init__(
        self,
        u: Optional[float] = None,
        st: Optional[float] = None,
        cl0: Optional[float] = None,
        eps: Optional[float] = None,
        al: Optional[float] = None,
        bt: Optional[float] = None,
        gm: Optional[float] = None,
    ) -> None:
        """Init with args.

        Parameters
        ----------
        u : float
            Fluid speed (m/s).
        st : float
            Strouhal number.
        cl0 : float
            Lift coefficient.
        eps : float
            Wake oscillator parameter.
        al : float
            Wake oscillator parameter.
        bt : float
            Wake oscillator parameter.
        gm : float
            Wake oscillator parameter.

        Returns
        -------
        None.
        """
        self.u = u
        self.st = st
        self.cl0 = cl0
        self.eps = eps
        self.al = al
        self.bt = bt
        self.gm = gm
        self.rho = air_volumic_mass()

    def __call__(self, s, t):
        raise NotImplementedError()


class Excitation:
    """Sinusoidal vertical excitation."""

    def __init__(
        self,
        f: float = 1.0,
        a: float = 1.0,
        s: float = 0.5,
        m: Union[float, np.ndarray] = 1.0,
        L: float = 1.0,
        t0: float = 0.0,
        tf: float = np.inf,
        gravity: bool = False,
        g: float = 9.81,
    ) -> None:
        """Init with args.

        Parameters
        ----------
        f : float, optional
            Frequency (Hz) of oscillations. The default is 1.
        a : float, optional
            Amplitude (N) of oscillation if positive it will go up first.
            The default is 1.
        s : float, optional
            Point of application of the force (m). It must be in ]0, L[ range.
            The default is 0.5.
        m : float or numpy.ndarray, optional
            Mass per unit length (kg/m). The default is 1.
        L : float, optional
            Length of the structure. The default is 1.
        t0 : float, optional
            Start time of excitation (s). The default is 0.
        tf : float, optional
            End time of excitations (s). The default is np.inf.
        gravity : bool, optional
            Add (or not) gravity forces. The default is False.
        g : float, optional
            Gravitationnal acceleration. The default is 9.81.

        Raises
        ------
        TypeError
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.
        """
        vrl = [f, a, s, L, t0, tf, g]
        vrn = ["f", "a", "s", "L", "t0", "tf", "g"]
        for i in range(len(vrl)):
            if not isinstance(vrl[i], float):
                raise TypeError(f"input {vrn[i]} must be a float")
        if not isinstance(gravity, bool):
            raise TypeError("input gravity must be a bool")
        if t0 >= tf:
            raise ValueError("input t0 must be less than tf")
        if L <= 0.0:
            raise ValueError("input L must be strictly positive")
        if s <= 0.0 or s >= L:
            raise ValueError("input s must be in ]0, L[ range")

        self.f = f
        self.a = a
        self.s = s
        self.m = m
        self.L = L
        self.t0 = t0
        self.tf = tf
        self.gravity = gravity
        self.g = g

    def _gravity(self, s):
        if self.gravity:
            return -self.g * self.m * np.ones_like(s)
        return np.zeros_like(s)

    def __call__(
        self,
        s: np.ndarray,
        t: float,
        un: Optional[Any] = None,
        ub: Optional[Any] = None,
        vn: Optional[Any] = None,
        vb: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Force computation.

        Parameters
        ----------
        s : numpy.ndarray
            Space discretization. Returned arrays must have the same shape.
        t : float
            Time (s).
        un : TYPE, optional
            Structure offset in normal direction. Not used here, the parameter
            is kept for consistency with other force formulations. The default
            is None.
        ub : TYPE, optional
            Structure offset in binormal direction. Not used here, the parameter
            is kept for consistency with other force formulations. The default
            is None.
        vn : TYPE, optional
            Structure velocity in normal direction. Not used here, the parameter
            is kept for consistency with other force formulations. The default
            is None.
        vb : TYPE, optional
            Structure velocity in binormal direction. Not used here, the parameter
            is kept for consistency with other force formulations. The default
            is None.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        tuple
            Two numpy.ndarray with lift and drag forces. In this case the drag
            is always zero. Returned values are in newton per meter and should
            be integrated on the structure to have a "real" force.
        """
        if not isinstance(s, np.ndarray):
            raise TypeError("input s must be a numpy.ndarray")
        if not isinstance(t, float):
            raise TypeError("input t must be a float")

        p = self._gravity(s)

        if self.t0 <= t <= self.tf:
            n = len(s)
            i = max(1, min(n - 2, int(np.round(self.s / self.L * (n - 1)))))
            d = s[i + 1] - s[i - 1]
            a = 2.0 * self.a / d
            p[i] += a * np.sin(2.0 * np.pi * self.f * (t - self.t0))

        return p, np.zeros_like(s)

    def _twodim(self, s, t):
        fn = np.zeros((len(t), len(s)))
        fb = np.zeros((len(t), len(s)))
        for i in range(len(t)):
            fn[i, :], fb[i, :] = self(s, t[i])
        return fn, fb
