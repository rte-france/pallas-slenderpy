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



..
   Simulation
   ==========

   Parameters
   ----------

   Results
   -------
