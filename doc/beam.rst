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
* :math:`\chi_0`: the critical curvature below which the bending moment is close the :math:`EI_{max}` and above which it is close to :math:`EI_{min}`. 
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


A :class:`~slenderpy.future.beam.beam.Beam` object can either be a :class:`~slenderpy.future.beam.beam.BeamConst` or a :class:`~slenderpy.future.beam.beam.BeamBW`. 
The first one solves the beam equation with a constant bending stiffness whereas the second one solves it with a variable bending stiffness with hysteresis effects based on Bouc-Wen model.

Static
======

:class:`~slenderpy.future.beam.beam.BeamConst` object
-----------------------------------------------------

The equation solved in the static case is:

.. math::
    &\frac{\partial^2 M(y)}{\partial x ^2 } - H \frac{\partial^2 y}{\partial x ^2 } = F(x) \\ 
    &M(y) = EI \chi(y)


:class:`~slenderpy.future.beam.beam.BeamBW` object
-----------------------------------------------------

The equation solved in the static case is:

.. math::
    &\frac{\partial^2 M(y)}{\partial x ^2 } - H \frac{\partial^2 y}{\partial x ^2 } = F(x) \\
    &M(y) = (EI_{max} \bar{\chi} + EI_{min} |\chi(y)|)(1 - \exp(- \frac{|\chi(y)|}{\bar{\chi}})) \mathrm{sign}(\chi(y)) \\
    &\bar{\chi} = (1 - \frac{EI_{min}}{EI_{max}}) \chi_0


For the resolution, the space derivatives are discretized using finite differences centered schemes. 
Because of the non-linearity of the problem, the function :code:`sp.optimize.root` is used to solve the equation.

Dynamic
=======

:class:`~slenderpy.future.beam.beam.BeamConst` object
-----------------------------------------------------

The equation solved in the dynamic case is:

.. math::
    &m\frac{\partial^2 y}{\partial t ^2 } + 2m\omega_0 \zeta \frac{\partial y}{\partial t  }  + \frac{\partial^2 M}{\partial x ^2 }   - H \frac{\partial^2 y}{\partial x ^2 } = F(x,t) \\
    &M(y) = EI \chi(y)
    :label: eq:beam_dynamic_const

For the temporal discretization, a Crank-Nicolson scheme is used. We also introduce the velocity :math:`v = \frac{\partial y}{\partial t }` as an additional variable.
We rewrite :eq:`eq:beam_dynamic_const` as:

.. math::
    m\frac{\partial v}{\partial t } = F(x,t) - 2m\omega_0 \zeta v - D_2 M  + H D_2 y 

with :math:`D_2` the second order space derivative operator.

The two unknows at each time step :math:`n` are the velocity :math:`v^n` and the displacement :math:`y^n`,
which are two vectors of size the number of nodes along the beam. 
The external force :math:`F^n` is also a vector of the same size and the operator :math:`D_2` is a matrix.
The bending moment :math:`M^n` is also a vector of the same size and it is computed from the displacement :math:`y^n` using the curvature formula.
The Crank-Nicolson scheme applied to this equation reads:

.. math::
    &m \frac{v^{n+1} - v^n}{\Delta t} = \frac{1}{2}(F^{n+1} + F^n - 2m\omega_0 \zeta (v^{n+1} + v^n) - D_2 M^{n+1}  - D_2 M^n + H D_2 y^{n+1} + H D_2 y^n) \\
    & \frac{y^{n+1} - y^n}{\Delta t} = \frac{1}{2} (v^{n+1} + v^n)

We substitute :math:`y^{n+1}` in the first equation and we introduce :math:`K = - H D_2` to get:

.. math::
    &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
    & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
    & \hspace{15em} - \frac{\Delta t}{2} D_2 (M^{n+1} + M^n) \\
    & \hspace{15em} -\Delta t K y^n \\
    &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n)
    :label: eq:beam_dynamic_const_cn

with :math:`I_d` the identity matrix.

The problem in :eq:`eq:beam_dynamic_const_cn` is the term :math:`D_2 M^{n+1}` which is non-linear because of the curvature formula.
In the case where the curvature is approximated, we have :math:`M^{n+1} = EI D_2 y^{n+1}` and the problem becomes linear. 
We can thus rewrite :eq:`eq:beam_dynamic_const_cn` as:

.. math::
    &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
    & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
    & \hspace{15em} -\Delta t K y^n \\
    &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n)
    :label: eq:beam_dynamic_const_cn_approx

with :math:`K = EI D_4 - H D_2` where :math:`D_4` is the fourth order space derivative operator.

In the case of the exact curvature formula, we could replace :math:`M^{n+1}` by its value at the previous time step :math:`M^n` to get an explicit scheme.
But unfortunately this scheme was not always stable for the beam equation.
Thus we use Picard iterations to solve the non-linear problem at each time step. Here is the process:

* Initialize :math:`y^p = y^n`
* Iterate until convergence:
  
  * Compute :math:`M^p = EI \chi_{exact}(y^p)`
  * Compute :math:`v^{n+1}` and :math:`y^{n+1}`
  
  .. math::
      &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
      & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
      & \hspace{15em} - \frac{\Delta t}{2} D_2 (M^p + M^n) \\
      & \hspace{15em} -\Delta t K y^n \\
      &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n)
  
  * Update :math:`y^p = y^{n+1}`


:class:`~slenderpy.future.beam.beam.BeamBW` object
----------------------------------------------------- 

The equations solved in the dynamic case are:

.. math::
    &m\frac{\partial^2 y}{\partial t ^2 } + 2m\omega_0 \zeta \frac{\partial y}{\partial t  }  + \frac{\partial^2 M}{\partial x ^2 }   - H \frac{\partial^2 y}{\partial x ^2 } = F(x,t) \\
    &M(y) = EI_{min}\chi(y) + (EI_{max} - EI_{min})\chi_0 \eta \\
    & \chi_0 \frac{\partial \eta}{\partial t} = \frac{\partial \chi}{\partial t} - \frac{1}{2} (\frac{\partial \chi}{\partial t} |\eta| + |\frac{\partial \chi}{\partial t}| \eta) 


For the case of the approximated curvature, we choose a Crank-Nicolson scheme for the velocity and the displacement and an Euler implicit scheme for the hysteresis variable:

.. math::
    &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
    & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
    & \hspace{15em} + \frac{\Delta t}{2}(EI_{max} - EI_{min}) \chi_0 D_2 (\eta^{n+1} + \eta^n)\\
    & \hspace{15em} -\Delta t K y^n \\
    &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n) \\
    &\eta^{n+1}( \chi_0 + \frac{\Delta t}{2} |D_2 v^{n+1}|) = \chi_0 \eta^n + \Delta t D_2 v^{n+1} - \frac{\Delta t}{2} D_2 v^{n+1} |\eta^{n+1|}

with :math:`K = EI_{min} D_4 - H D_2`.

It is quite similar to :eq:`eq:beam_dynamic_const_cn_approx` except that there is an additional term due to the hysteresis variable :math:`\eta`.
The term :math:`|\eta^{n+1}|` makes the problem non linear thus we use Picard iterations to solve the problem at each time step. Here is the process:

* Initialize :math:`\eta^p = \eta^n`
* Iterate until convergence:
  
  * Compute :math:`v^{n+1}` and :math:`\eta^{n+1}`:
  
  .. math::
    &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
    & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
    & \hspace{15em} + \frac{\Delta t}{2}(EI_{max} - EI_{min}) \chi_0 D_2 (\eta^p + \eta^n)\\
    & \hspace{15em} -\Delta t K y^n \\  
    &\eta^{n+1}( \chi_0 + \frac{\Delta t}{2} |D_2 v^{n+1}|) = (\chi_0 \eta^n + \Delta t D_2 v^{n+1} - \frac{\Delta t}{2} D_2 v^{n+1} |\eta^p|) 

  * Update :math:`\eta^p = \eta^{n+1}`

* Update :math:`y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n)`


For the case of the exact curvature formula, we choose a Crank-Nicolson scheme for the velocity and the displacement and an Euler implicit scheme for the hysteresis variable:

.. math::
    &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
    & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
    & \hspace{15em} - \frac{\Delta t}{2} D_2 (M^{n+1} + M^n) \\
    & \hspace{15em} -\Delta t K y^n \\
    &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n) \\
    &\eta^{n+1}( \chi_0 + \frac{1}{2} |\chi^{n+1} - \chi^n|) = \chi_0 \eta^n + \chi^{n+1} - \chi^n - \frac{1}{2} (\chi^{n+1} - \chi^n) |\eta^{n+1}|

with :math:`K = - H D_2`.

Again the term :math:`D_2 M^{n+1}` is non-linear. We thus use Picard iterations to solve the non-linear problem at each time step. Here is the process:

* Initialize :math:`y^p = y^n, \eta^p = \eta^n`
* Iterate until convergence:
  
  * Compute :math:`M^p = EI_{min} \chi(y^p) + (EI_{max} - EI_{min})\chi_0 \eta_p` 
  * Compute :math:`v^{n+1}` and :math:`y^{n+1}`:
  
  .. math::
      &v^{n+1}(I_d (m + m\omega_0 \zeta \Delta t) + \frac{\Delta t^2}{4}K) = v^n(I_d (m - m\omega_0 \zeta \Delta t) - \frac{\Delta t^2}{4}K) \\
      & \hspace{15em} + \frac{\Delta t}{2}(F^{n+1} + F^n) \\
      & \hspace{15em} - \frac{\Delta t}{2} D_2 (M^p + M^n) \\
      & \hspace{15em} -\Delta t K y^n \\
      &y^{n+1} = y^n + \frac{\Delta t}{2} (v^{n+1} + v^n)
  
  * Update :math:`y^p = y^{n+1}`

  * Compute :math:`\chi^p` from :math:`y^p`
  * Compute :math:`\eta^{n+1}`
  .. math::
      \eta^{n+1}( \chi_0 + \frac{1}{2} |\chi^p - \chi^n|) = \chi_0 \eta^n + \chi^p - \chi^n - \frac{1}{2} (\chi^p - \chi^n) |\eta^p|

  * Update :math:`\eta^p = \eta^{n+1}`


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

Thus when using the method :code:`solve_dynamic` the user should set the attribute :code:`dynamic_values` with :math:`\frac{\partial d_i}{\partial t} \forall i \in \left\{1,2,3,4\right\}` 
in the :class:`~slenderpy.future.beam.fd_utils.BoundaryCondition` constructor. 