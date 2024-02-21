# Slenderpy

_**slenderpy**_ is a python package to simulate the vibrations of elongated structures like cables or beams.

## Installation

### Using pip

To install the package using pip, execute the following command:

```shell script
python -m pip install slenderpy@git+https://github.com/rte-france/pallas-slenderpy
```

### Using conda

(not available yet)

## Building the documentation

First, make sure you have sphinx and the Readthedocs theme installed.

If you use pip, open a terminal and enter the following commands:

```shell script
pip install sphinx
pip install sphinx_rtd_theme
```

If you use conda, open an Anaconda Powershell Prompt and enter the following commands:

```shell script
conda install sphinx
conda install sphinx_rtd_theme
```

Then, in the same terminal or anaconda prompt, go to directory `slenderpy/doc` and build the doc:

```shell script
cd doc
make html
```

The documentation can then be accessed from `doc/_build/html/index.html`.

## Simple usage

This example defines a cable excited for a short time with a point sine force. The cable is simulated
with and without self-damping.

```python
import numpy as np
from slenderpy import cable
from slenderpy import simtools
from slenderpy import force
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Cable definition (EA is the axial stiffness).
cb = cable.SCable(mass=1.57, diameter=0.031, EA=3.76E+07,
                  length=200., tension=3.7E+04, h=0.)

# Number of space samples.
ns = 101

# Vector of space samples. It defines the positions of the states exported by the simulation.
# The positions are normalized (i.e., in interval ]0, 1[).
s = np.linspace(0.0, 1.0, ns)
los = s[1:-1].tolist()

# Set the simulation parameters.
# dt is the simulation time step and dr the time step to save the simulation results.
pm = simtools.Parameters(ns=ns, t0=0., tf=8., dt=1.0E-03, dr=3.0E-02, los=los, pp=True)

# Define a sine excitation (freq=2Hz) at position=5% of cable length, from time 0.2s to time 1.0s.
fr = force.Excitation(f=2., a=0.1, s=(0.05 * cb.Lp), m=1., L=cb.Lp,
                      t0=0.2, tf=1.0, gravity=False)

# Run simulation without self-damping.
res = cable.solve(cb, pm, force=fr)

# Extract values of normal displacement and add zero values at cable's ends.
# We get an array of dimension (nTimes x nPositions).
un = res.data['un'].values
un = np.insert(un, (0, un.shape[1]), values=0, axis=1)

# Get vectors of time and of horizontal position.
time = res.data['time'].values
x = cb.Lp * s

# Run simulation with a non-zero damping factor (zt).
res_damp = cable.solve(cb, pm, force=fr, zt=0.2)

# Extract results and add zero values at cable's ends.
un_damp = res_damp.data['un'].values
un_damp = np.insert(un_damp, (0, un_damp.shape[1]), values=0, axis=1)
```

Add the following code to show the evolution of the vertical displacement in an animation loop:

```python
fig, ax = plt.subplots(figsize=[6, 4], tight_layout=True)
ax.set_xlabel('horizontal position [m]')
ax.set_ylabel('vertical displacement [m]')
line1, = ax.plot(x, np.zeros_like(x), label="zero damping")
line2, = ax.plot(x, np.zeros_like(x), label="non-zero damping")
ax.legend()
xlim = ax.get_xlim()
d = np.max(np.abs(un))
ax.set_ylim([-1.5 * d, 1.5 * d])
ylim = ax.get_ylim()
text = ax.text(0.95 * xlim[0] + 0.05 * xlim[1],
               0.9 * ylim[0] + 0.1 * ylim[1], 't=0s')


def animate(i):
    """The function to call at each frame."""
    line1.set_ydata(un[i, :])
    line2.set_ydata(un_damp[i, :])
    text.set_text("t={:2.3f}s".format(time[i]))
    return line1, line2, text


ani = animation.FuncAnimation(
    fig, animate, interval=20, blit=True,
    frames=time.size, repeat=True)

plt.show()
```
