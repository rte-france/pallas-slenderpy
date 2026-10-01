Structures
==========

Beam
----

The beam follows a classic Euler–Bernoulli beam model and can only
move in the vertical direction. Different solvers are proposed
according to different models used for the bending moment (constant
bending stiffness, bending stiffness varying with curvature, bending
stiffness varying with curvature and hysteresis loop).

Cable
-----

The cable is a homogeneous, one-dimensional elastic continuum. The
flexural, torsional, and shear rigidities of the cable are
negligible. The cable is suspended, i.e. its two ends are fixed. Our
cable solver is based on solving Lee's equation of motion (see
`[Lee1992] <https://link.springer.com/article/10.1007/BF00045648>`_
for more details), which means its direct results are offsets
regarding the cable equilibrium position (catenary equation) in a
local triad. Tools to project the results back into a more
conventionnal triad are provided.



Forces
======

The force which is applied on a structure is by design a user-defined
object, our solver requires a function (or an instance of an object
with a `__call__` method) with the following arguments: space
discretization, time, and the structure state in the form of the
positions and velocities of its points. Several force models are
already implemented.

Zero Force
---------------

This is the default force in all solvers, absolutely nothing happens
if your structure is initialized at an equilibrium position.

Bishop & Hassan
---------------

It is a formulation derived from wind tunnel experiments on a smooth,
fixed cylinder `[X] <http://0.0.0.0>`_. We derived two forces from it,
one in which the structure only sees the flow speed, and the other
which takes the structure velocity into account.

Sinusoidal Excitation
---------------------

This force is used to reproduce laboratory experiments: it is a basic
sinusoidal excitation where you can change the amplitude, the
frequency and the application point.

Turbulent Wind
--------------

Since a constant wind speed is not realistic to describe, we provide
tools to generate a turbulent wind and apply a drag force.

Wake Oscillator
---------------

A typical wake oscillator model is a heuristic model that uses a
single degree of freedom to represent the wake behind a rigid
cylinder. The wake has its own equation of evolution, which is coupled
to the cable's equation of motion. Parameters of the wake oscillator are
fitted from experiments or CFD simulation.



Simulation
==========

The ``future`` solvers share :mod:`slenderpy.future.simulation`.

Parameters
----------

:class:`~slenderpy.future.simulation.Parameters` holds the time stepping and
the output configuration. The run has ``nt = round((tf - t0) / dt)`` steps of
``dt = (tf - t0) / nt`` and a snapshot is stored every ``rr`` steps, so the
effective output step is ``rr * dt``. A warning is raised when either
effective step differs from the requested one, and the last output time may be
slightly earlier than ``tf``.

``los`` lists the positions where snapshots are stored, as span fractions
(horizontal distance from support 1 over the span length) in [0, 1]; 0 and 1
are the supports. An int ``n`` gives ``n`` evenly spaced positions, supports
included.

Results
-------

:class:`~slenderpy.future.simulation.Results` stores each variable on a
``time`` x ``span_frac`` grid (scalars on ``time`` only), in a global frame:
origin at support 1, ``x`` along the span, ``z`` upwards, ``y`` completing a
right-handed triad, SI units.

=============  =========================================  =====  ====
name           meaning                                    cable  beam
=============  =========================================  =====  ====
``x``, ``y``   along-span and out-of-plane position (m)   yes    no
``z``          vertical position (m)                      yes    yes
``vz``         vertical velocity (m/s)                    no     yes
``curvature``  bending curvature (1/m)                    no     yes
``moment``     bending moment (N.m)                       no     yes
``eta``        Bouc-Wen internal variable                 no     yes
``n_iter``     Newton iterations of the step (scalar)     no     yes
``dtension``   dynamic increment of axial force (N)       yes    no
=============  =========================================  =====  ====

The final state of a run, at full space resolution, is kept in
``Results.state`` so that a run can be restarted from it.

Forces
------

A force is any callable ``force(x, t, y, z, vy, vz) -> (fy, fz)`` returning
the force per unit length (N/m) along the global ``y`` and ``z`` axes, from
the horizontal position ``x`` of the nodes, the time and the global position
and velocity of the structure. The beam uses ``fz`` only; the cable projects
both components on its local normal and binormal.

:mod:`slenderpy.future.force` provides:

- :class:`~slenderpy.future.force.core.Gravity`, the weight, for the beam
  only: the cable equilibrium already holds it and the cable solver refuses
  it;
- :class:`~slenderpy.future.force.core.PointExcitation`, a sinusoidal
  vertical force applied at one node;
- :class:`~slenderpy.future.force.wind.WindDrag`, the drag of a
  :class:`~slenderpy.future.force.wind.ConstantWind`,
  :class:`~slenderpy.future.force.wind.UniformTurbulentWind` or
  :class:`~slenderpy.future.force.wind.TurbulentWindField`, with a constant
  drag coefficient or :func:`~slenderpy.future.force.air.cylinder_drag` of
  the local Reynolds number.

Provided forces add up: ``PointExcitation(...) + Gravity(mass) + WindDrag(...)``.

Fatigue
-------

:func:`~slenderpy.future.fatigue.count_cycles` counts the rainflow cycles
(ASTM E1049-85, no mean correction) of a position-dependent result, e.g.
``z`` or ``moment``, at a distance from a support, such as the
Poffenberger-Swart point 89 mm from a clamp. The result is read by linear
interpolation between the stored positions, so store the fatigue position
in ``los``. It returns a table of ranges, means and counts (1 per cycle, 0.5
per half cycle); stress models, S-N curves and damage are left to a
dedicated package.
