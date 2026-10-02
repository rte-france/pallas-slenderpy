This document describes the beam solvers available in SlenderPy. More precisly the different numerical schemes implemented to solve the Euler-Bernoulli beam equation.

We have the following notations:

* :math:`EI`: the bending stiffness of the beam, if it is constant along the beam. :math:`E` is the Young modulus and :math:`I` the second moment of area.
* :math:`EI_{max}`: the maximum bending stiffness of the beam (when the bending stiffness is not constant along the beam).
* :math:`EI_{min}`: the minimum bending stiffness of the beam (when the bending stiffness is not constant along the beam).
* :math:`M`: the bending moment. 
* :math:`H`: the tension. 
* :math:`F`: the external forces per unit length applied on the beam.  
* :math:`m`: the mass per unit length of the beam. 
* :math:`\chi(y)`: the curvature of the beam which is a function of the displacement :math:`y`. 
* :math:`\chi_0`: the critical curvature below which the bending moment is close to :math:`EI_{max}` and above which it is close to :math:`EI_{min}`. 
* :math:`\omega_0`: the natural pulsation of the beam (:math:`= 2 \pi f_0` with :math:`f_0` the natural frequency of the beam).
* :math:`\zeta`: the damping ratio of the beam (if equal to 1: critical damping, if less than 1: underdamped, if greater than 1: overdamped).
* :math:`\eta` : the hysteresis variable. 

The unknown of the problem is the vertical displacement of the beam :math:`y(x,t)` where :math:`x` is the position along the beam and :math:`t` the time.

The exact formula for the curvature :math:`\chi` is:

.. math::
    \chi_{exact}(y) = \frac{\partial^2 y}{\partial x ^2 } \frac{1}{\left(1 + \frac{\partial y}{\partial x}^2 \right)^{3/2}}

By making a taylor expansion at order 1 of this formula, under the assumption of small displacements, we get the following approximation for the curvature:

.. math::
    \chi_{approx}(y) = \frac{\partial^2 y}{\partial x ^2 }


The beam solvers are functions of a :class:`~slenderpy.future.components.Conductor`
and a :class:`~slenderpy.future.components.Span`, whose ``boundary_conditions``
must be set. Two choices are made independently, by arguments:

* the bending law (:mod:`slenderpy.future.beam.bending`): ``model="constant"``
  for a constant bending stiffness :math:`EI` (``conductor.ei_max`` by default,
  or the ``ei`` argument), or ``model="varying"`` for a stiffness that falls
  from :math:`EI_{max}` to :math:`EI_{min}` with hysteresis, after the Bouc-Wen
  model, with :math:`\chi_0` = ``conductor.beta_flexion * span.tension``;
* the curvature (:mod:`slenderpy.future.beam.curvature`):
  ``approx_curvature=True`` for :math:`\chi_{approx}`, ``False`` for
  :math:`\chi_{exact}`.

The displacement :math:`y` is the vertical position of the beam, stored as
``z`` in the results (see the output contract of
:mod:`slenderpy.future.simulation`).

Static
======

:func:`slenderpy.future.beam.static.shape.solve` returns the displacement at the
nodes under a nodal load :math:`F(x)`.

Constant model
--------------

The equation solved in the static case is:

.. math::
    &\frac{\partial^2 M(y)}{\partial x ^2 } - H \frac{\partial^2 y}{\partial x ^2 } = F(x) \\
    &M(y) = EI \chi(y)


Varying model
-------------

The equation solved in the static case is:

.. math::
    &\frac{\partial^2 M(y)}{\partial x ^2 } - H \frac{\partial^2 y}{\partial x ^2 } = F(x) \\
    &M(y) = (EI_{max} \bar{\chi} + EI_{min} |\chi(y)|)(1 - \exp(- \frac{|\chi(y)|}{\bar{\chi}})) \mathrm{sign}(\chi(y)) \\
    &\bar{\chi} = (1 - \frac{EI_{min}}{EI_{max}}) \chi_0


Resolution
----------

The space derivatives are discretized with centered finite differences. The
nonlinear system is solved with a damped Newton iteration, starting from the
constant-stiffness linear solution. The Jacobian is assembled analytically from
the tangent of the bending law and the Jacobian of the curvature; each step is
relaxed until it decreases the residual norm. A solve that does not converge
returns ``nan``.

Dynamic
=======

:func:`slenderpy.future.beam.dynamic.solve_dynamic` returns the time history of
the beam under a force ``force(x, t, y, z, vy, vz) -> (fy, fz)`` (see
:mod:`slenderpy.future.force.core`; the beam is planar and uses ``fz``). By
default it starts at rest from the static shape under the force at the initial
time.

Constant model
--------------

The equation solved in the dynamic case is:

.. math::
    &m\frac{\partial^2 y}{\partial t ^2 } + 2m\omega_0 \zeta \frac{\partial y}{\partial t  }  + \frac{\partial^2 M}{\partial x ^2 }   - H \frac{\partial^2 y}{\partial x ^2 } = F(x,t) \\
    &M(y) = EI \chi(y)


Varying model
-------------

The equations solved in the dynamic case are:

.. math::
    &m\frac{\partial^2 y}{\partial t ^2 } + 2m\omega_0 \zeta \frac{\partial y}{\partial t  }  + \frac{\partial^2 M}{\partial x ^2 }   - H \frac{\partial^2 y}{\partial x ^2 } = F(x,t) \\
    &M(y) = EI_{min}\chi(y) + (EI_{max} - EI_{min})\chi_0 \eta \\
    & \chi_0 \frac{\partial \eta}{\partial t} = \frac{\partial \chi}{\partial t} - \frac{1}{2} (\frac{\partial \chi}{\partial t} |\eta| + |\frac{\partial \chi}{\partial t}| \eta)


Time discretization
-------------------

The velocity :math:`v = \frac{\partial y}{\partial t }` is introduced as an
additional unknown and both models are integrated with the same Crank-Nicolson
scheme on the first-order system, solved for the velocity:

.. math::
    &A v^{n+1} = B v^n - \Delta t K y^n - \frac{\Delta t}{2} (G^n + G^{n+1}) + \frac{\Delta t}{2} (F^n + F^{n+1}) \\
    &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n)

with :math:`K = EI_{lin} D_4 - H D_2` the linear part of the equation,
:math:`A = m I_d + \frac{\Delta t}{2} c I_d + \frac{\Delta t^2}{4} K` plus the
boundary rows, :math:`B` the same with the two last signs flipped,
:math:`c = 2 m \omega_0 \zeta` the damping coefficient, and
:math:`G = D_2 M - EI_{lin} D_4 y` the nonlinear remainder. :math:`EI_{lin}` is
:math:`EI` for the constant model and :math:`EI_{min}` for the varying one; the
split between :math:`K` and :math:`G` is purely algebraic.

For the constant model with the approximate curvature :math:`G` vanishes: the
problem is linear, :math:`A` is factorised once and each step is one solve.

In the three other cases the step is solved with a Newton iteration on its
residual, with the tangent

.. math::
    A + \frac{\Delta t^2}{4} \left( D_2 \, \mathrm{diag}\left(\frac{dM}{d\chi}\right) \frac{d\chi}{dy} - EI_{lin} D_4 \right)

and a backtracking that relaxes a step increasing the residual. The hysteresis
variable is advanced fully implicitly, with
:math:`\Delta \chi = \Delta t \frac{d\chi}{dy} v^{n+1}`:

.. math::
    \chi_0 (\eta^{n+1} - \eta^n) = \Delta \chi - \frac{1}{2} (\Delta \chi |\eta^{n+1}| + |\Delta \chi| \eta^{n+1})

Since the sign of :math:`\eta^{n+1}` is the sign of
:math:`\chi_0 \eta^n + \Delta \chi`, this equation has a closed-form solution,
bounded by 1 whatever the step. A fixed-point (Picard) iteration on a frozen
:math:`A` does not converge here: with :math:`EI_{lin} = EI_{min}` and a true
tangent of :math:`EI_{max}` its gain is :math:`EI_{max}/EI_{min}`. A step that
does not converge stops the run; the snapshots already computed are kept and
the remaining ones are left at ``nan``.


Boundary Conditions
===================

For all the previsous resolutions, there are always four boundary conditions. Two for each side of the beam.
Considereing the left side of the beam at :math:`x=0` and the right side at :math:`x=L`, the boundary conditions supported are under the form:

.. math::
    a_1 y(0,t) + b_1 \frac{\partial y}{\partial x}(0,t) + c_1 \frac{\partial^2 y}{\partial x^2}(0,t) = d_1(t) \\
    a_2 y(0,t) + b_2 \frac{\partial y}{\partial x}(0,t) + c_2 \frac{\partial^2 y}{\partial x^2}(0,t) = d_2(t) \\
    a_3 y(L,t) + b_3 \frac{\partial y}{\partial x}(L,t) + c_3 \frac{\partial^2 y}{\partial x^2}(L,t) = d_3(t) \\
    a_4 y(L,t) + b_4 \frac{\partial y}{\partial x}(L,t) + c_4 \frac{\partial^2 y}{\partial x^2}(L,t) = d_4(t) 

Where :math:`a_i, b_i, c_i \in \mathbb{R} \forall i \in \left\{1,2,3,4\right\}` and :math:`d_i(t) \forall i \in \left\{1,2,3,4\right\}` can be function of time for the dynamic case and simply constant in the static case. 
We denote by :math:`y_0` the displacement at :math:`x=0` and by :math:`y_N` the displacement at :math:`x=L` where :math:`N` is the number of nodes along the beam.
Similarly :math:`y_i` is the displacement at the node :math:`i`.
To take into account properly these boundary conditions in the finite difference schemes, we discretize them:

.. math::
    a_1 y_0 + b_1 \frac{y_1 - y_0}{\Delta x} + c_1 \frac{y_0 - 2y_1 + y_2}{\Delta x^2} = d_1 \\
    a_2 y_0 + b_2 \frac{y_1 - y_0}{\Delta x} + c_2 \frac{y_0 - 2y_1 + y_2}{\Delta x^2} = d_2 \\
    a_3 y_N + b_3 \frac{y_N - y_{N-1}}{\Delta x} + c_3 \frac{y_N - 2y_{N-1} + y_{N-2}}{\Delta x^2} = d_3 \\
    a_4 y_N + b_4 \frac{y_N - y_{N-1}}{\Delta x} + c_4 \frac{y_N - 2y_{N-1} + y_{N-2}}{\Delta x^2} = d_4 
    :label: eq:bc

Thus the corresponding matrix and vector of :eq:`eq:bc` are:

.. math::
    A &= \begin{pmatrix}
        a_1 - \frac{b_1}{\Delta x} + \frac{c_1}{\Delta x^2} & \frac{b_1}{\Delta x} - \frac{2c_1}{\Delta x^2}  & \frac{c_1}{\Delta x^2} & 0 & \cdots & \cdots & 0 \\
        a_2 - \frac{b_2}{\Delta x} + \frac{c_2}{\Delta x^2} & \frac{b_2}{\Delta x} - \frac{2c_2}{\Delta x^2}  & \frac{c_2}{\Delta x^2} & 0 & \cdots & \cdots  & 0 \\
        0 &\ddots & \ddots &  \ddots &  \ddots &  \ddots & 0 \\
        0 & \cdots & \cdots & 0 & \frac{c_3}{\Delta x^2} & - \frac{b_3}{\Delta x} - \frac{2c_3}{\Delta x^2} & a_3 + \frac{b_3}{\Delta x} + \frac{c_3}{\Delta x^2} \\
        0 & \cdots & \cdots & 0 & \frac{c_4}{\Delta x^2} & - \frac{b_4}{\Delta x} - \frac{2c_4}{\Delta x^2} & a_4 + \frac{b_4}{\Delta x} + \frac{c_4}{\Delta x^2} 
        \end{pmatrix} \\
    b &= \begin{pmatrix}
    d_1\\
    d_2\\
    0 \\
    \vdots \\
    0 \\
    d_3 \\
    d_4
    \end{pmatrix}
    :label: eq:matrix_bc

The matrix and vector :eq:`eq:matrix_bc` are used for the static resolution. For the dynamic resolution the first linear system to solve in on the velocity, 
we thus derivate with respect to time :eq:`eq:bc` obtaining the same matrix :math:`A` than :eq:`eq:matrix_bc`, since :math:`\frac{\partial y}{\partial t} = v`,  and the vector :math:`b` contains the time derivative of 
:math:`d_i(t) \forall i \in \left\{1,2,3,4\right\}`. 

Thus when using :func:`~slenderpy.future.beam.dynamic.solve_dynamic` the user should set :code:`dynamic_values` with :math:`\frac{\partial d_i}{\partial t} \forall i \in \left\{1,2,3,4\right\}` 
in the :class:`~slenderpy.future.boundary_condition.BoundaryCondition` constructor.