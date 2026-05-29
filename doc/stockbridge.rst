This document describes the stockbridge solvers available in SlenderPy. More precisely the different numerical schemes implemented to solve the related equation.

The stockbridge model is composed of a clamp and two inertial masses connected by a messenger cable. 
These two cables are modeled as uncoupled planar cantilever. 
The cable is characterized by its length, and the following parameters:

* :math:`EI_{max}`: the maximum bending stiffness
* :math:`EI_{min}`: the minimum bending stiffness
* :math:`\chi_0`: the critical curvature below which the bending moment is close to :math:`EI_{max}` and above which it is close to :math:`EI_{min}`
  
These three parameters are not constant along the cable, we distinguish 3 different zones. 
The boundary zones (near the clamp and near the mass) that are characterized by the same parameters, and the middle zone that is characterized by different parameters.
The user can set the length of the boundary zones thanks to the argument :code:`ratio_boundary1` and :code:`ratio_boundary2` of the class :class:`~slenderpy.future.stockbridge.core.Parameters.MessengerCableParameters`.
The ratio being the length of the boundary zone divided by the total length of the messenger cable.

The motion of the mass :math:`i` is fully described by its vertical displacement  acement at tip :math:`x_i` and its rotation :math:`\varphi_i`.
The motion of the clamp is :math:`w_c` and its rotation is :math:`\varphi_c`.

Beware that in this model the angle :math:`\varphi_1` and :math:`\varphi_2` are counted positive in opposite direction of rotation. 
Meaning that for a symetrical stockbridge, these quantities will be exactly the same. 

For a mass :math:`i`, we consider the following quantities :

* :math:`x_i`: the vertical position of the mass
* :math:`\varphi_i`: the rotation of the mass
* :math:`F_i`: the vertical force applied on the mass
* :math:`M_i`: the moment applied on the mass
* :math:`e_{G_i}`: the distance between the center of gravity of the mass and the attachment point to the clamp
* :math:`I_{G_i}`: the moment of inertia about the centroid
* :math:`m_i`: the mass of the mass
* :math:`l_i`: the length of the messenger cable between the mass and the clamp
* :math:`\chi_i`: the curvature of the messenger cable 
* :math:`\eta_i`: the hysteretical variable of the messenger cable
* :math:`\varepsilon_i = (-1)^{i-1}`: the sign factor for the mass

For the clamp, we consider the following quantities :

* :math:`w_c`: the vertical position of the clamp
* :math:`\varphi_c`: the rotation of the clamp
* :math:`F_c`: the vertical force applied on the clamp
* :math:`M_c`: the moment applied on the clamp
* :math:`b_c`: the half-length of the clamp along the main cable
* :math:`I_{G_c}`: moment of inertia about the centroid of the clamp
* :math:`m_c`: the mass of the clamp

The equations of motion of each mass are given by:

.. math::
    \underbrace{\begin{pmatrix}
    m_i & -m_ie_{G_i} \\
     -m_ie_{G_i} & I_{G_i} + m_i e_{G_i}^2 
    \end{pmatrix}}_{\mathcal{M}_i}
    \begin{pmatrix}
        \ddot x_i \\ 
        \ddot \varphi_i 
    \end{pmatrix} + 
    \begin{pmatrix}
        F_i \\
        M_i
    \end{pmatrix} = 
    \begin{pmatrix}
        f_i^{ext} \\
        m_i^{ext}
    \end{pmatrix}

where :math:`f_i^{ext} = -m_i\ddot w_c(t) - m_i\varepsilon_i b_c \ddot \varphi_c(t), ~m_i^{ext} = m_i e_{G_i}\ddot w_c(t) + m_i e_{G_i} \varepsilon_i b_c \ddot \varphi_c (t)`

The equilibrium equations of the clamp are given by:

.. math::
    & F_c = m_c \ddot w_c - (F_1 + F_2) \\
    & M_c = I_{G_c} \ddot \varphi_c + M_2 - M_1 + F_2 (l_2 + b_c) - F_1 (l_1 + b_c)

Which is equivalent to:

.. math::
    F_c &= (m_1 + m_2 + m_c)\ddot w_c + (m_1 - m_2) b_c \ddot \varphi_c + m_1 \ddot x_1 +  m_2 \ddot x_2 - m_1 e_{G_1} \ddot \varphi_1 - m_2 e_{G_2} \ddot \varphi_2\\
    M_c &= (m_1l_1^* - m_2l_2^*) \ddot w_c + (I_{G_c} + m_1 b_c l^*_1 + m_2 b_c l^*_2) \ddot \varphi_c + m_1l^*_1\ddot x_1 - m_2l^*_2\ddot x_2 + \\ 
    & \quad  (I_{G_1} - m_1 e_{G_1} l^*_1)\ddot \varphi_1 
    - (I_{G_2} - m_2 e_{G_2} l^*_2)\ddot \varphi_2 
    
with :math:`l^*_i = l_i + b_c - e_{G_i}`



Linearized case
================

As a first approximation of  the model, we can consider that the force and moment applied on each mass are linearly related to the displacement and rotation of the mass.

.. math::
    & F_i = k_i x_i + \tilde k_i \varphi_i + c_i \dot x_i + \tilde c_i \dot \varphi_i \\ 
    & M_i = q_i \varphi_i + \tilde k_i x_i + \beta_i \dot \varphi_i + \tilde c_i \dot x_i

Thus, the system to solve is:

.. math::
    M \ddot X + C \dot X + K X = R 

.. math::
    & M = \begin{pmatrix}
        m_1 & -m_1e_{G_1} & 0 & 0 & m_1 & m_1\varepsilon_1 b_c\\
        -m_1e_{G_1} & I_{G_1} + m_1 e_{G_1}^2 & 0 & 0 & - m_1e_{G_1} & - m_1e_{G_1}\varepsilon_1 b_c \\ 
        0 & 0 & m_2 & -m_2e_{G_2} & m_2 & m_2\varepsilon_2 b_c \\
        0 & 0 & -m_2e_{G_2} & I_{G_2} + m_2 e_{G_2}^2 & - m_2e_{G_2} & - m_2e_{G_2}\varepsilon_2 b_c \\ 
        m_1 & - m_1e_{G_1} & m_2 & - m_2e_{G_2} & m_1 + m_2 + m_c &  (m_1 - m_2)b_c \\
        m_1 l_1^* & -m_2 l_2^* & (I_{G_1} - m_1 e_{G_1} l^*_1) & - (I_{G_2} - m_2 e_{G_2} l^*_2) & (m_1l_1^* - m_2l_2^*) & (I_{G_c} + m_1 b_c l^*_1 + m_2 b_c l^*_2)
    \end{pmatrix} \\ 
    & X = \begin{pmatrix}
        x_1 \\
        \varphi_1 \\
        x_2 \\
        \varphi_2 \\
        w_c \\
        \varphi_c 
    \end{pmatrix} \quad 
     R = \begin{pmatrix}
        0 \\
        0 \\
        0 \\
        0 \\
        F_c \\
        M_c 
    \end{pmatrix} \quad 
        C = \begin{pmatrix}
        c_1 &\tilde c_1 & 0 & 0 & 0 & 0 \\
        \tilde c_1 & \beta_1 & 0 & 0 & 0 & 0 \\
        0 & 0 & c_2 &\tilde c_2 & 0 & 0 \\
        0 & 0 & \tilde c_2 & \beta_2 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0
    \end{pmatrix} \quad 
    K = \begin{pmatrix}
        k_1 & \tilde k_1 & 0 & 0 & 0 & 0 \\
        \tilde k_1 & q_1 & 0 & 0 & 0 & 0 \\
        0 & 0 & k_2 & \tilde k_2 & 0 & 0\\
        0 & 0 & \tilde k_2 & q_2 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0
        \end{pmatrix} \quad 




Acceleration imposed at the clamp
=================================

To model the nonlinear behavior, each section of the messenger cable is modeled thanks to the Bouc-Wen model.

.. math::
    & \mathbb{M}_i(\chi_i(s),t) = EI_{\min} \chi_i(s,t) + (EI_{\max} - EI_{min} ) \chi_0 \eta_i(t) \\
    & \dot \eta_i(t) = \frac{1}{\chi_0} (\dot \chi_i(s,t) - |\dot \chi_i(s, t)| \eta_i(t) )

And the moment at any point of the messenger cable is given by:

.. math::
    \mathbb{M}_i(s,t) = M_i(t) + F_i(t)(l_i - s), \quad 0\leq s \leq l_i 


The relation between displacement, rotation, force and moment is:

.. math::
    \nu_i (t) &= l_i \int_0^{l_i} (1 - \frac{s}{l_i}) \chi_i(s,t) ds \\
    \varphi_i (t) &= \int_0^{l_i} \chi_i(s, t)\ ds


Finally, the system to solve is given by:

.. math:: 
    \mathcal{M}_i \begin{pmatrix}
               \ddot \nu_i \\
               \ddot \varphi_i 
           \end{pmatrix} + \begin{pmatrix}
               F_i \\
               M_i 
           \end{pmatrix} &= \begin{pmatrix}
            f_{ext} \\
            m_{ext} 
        \end{pmatrix}\\
    \nu_i (t) - \int_0^{l_i} (l_i - s) \chi_i(s,t) ds &= 0 \\
    \varphi_i (t) - \int_0^{l_i} \chi_i(s, t)\ ds &=0 \\ 
        M_i(t) + F_i(t)(l_i- s) - EI_{\min} \chi_i(s,t) -(EI_{\max} - EI_{\min} ) \chi_0 \eta_i(s, t) &= 0  \\
    \dot \eta_i(s, t) - \frac{1}{\chi_0} (\dot \chi_i(s,t) - |\dot \chi_i(s, t)| \eta_i(s,t) ) &= 0  \\
        m_c \ddot w_c - (F_1 + F_2) &= F_c \\
        I_{G_c} \ddot \varphi_c + M_2 - M_1 + F_2 (l_2 + b_c) - F_1 (l_1 + b_c) &= M_c


When imposing the acceleration at the clamp, we are able to solve each system independently, and then deduce the force and moment at the clamp.

We do a Crank Nicolson scheme to get the positions from the accelerations. 

.. math:: 
    \begin{pmatrix}
        \ddot \nu_i \\
        \ddot \varphi_i
    \end{pmatrix} = \mathcal{M}_i^{-1} \begin{pmatrix}
        f_{ext} - F_i\\
        m_{ext} - M_i
    \end{pmatrix}

Which rewrites as:

.. math::
    \dot X_i = \begin{pmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 
    \end{pmatrix} X + \begin{pmatrix}
    0 \\
    0 \\ 
    \mathcal{M}_i^{-1} \begin{pmatrix}
        f_{ext} - F_i\\
        m_{ext} - M_i
    \end{pmatrix}
    \end{pmatrix} \quad 
    X_i = \begin{pmatrix}
    \nu_i \\
    \varphi_i \\
    \dot \nu_i \\
    \dot \varphi_i 
    \end{pmatrix}

.. math:: 
        \frac{X_i^{n+1} - X_i^n}{\Delta t} = \begin{pmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 
    \end{pmatrix} \frac{X_i^{n+1} + X_i^n}{2} + \begin{pmatrix}
    0 \\
    0 \\ 
    \frac{\mathcal{M}_i^{-1}}{2} \begin{pmatrix}
        f_{ext}^{n+1} + f_{ext}^n - F_i^{n+1} - F_i^n\\
        m_{ext}^{n+1} + m_{ext}^n - M_i^{n+1} - M_i^n 
    \end{pmatrix}
    \end{pmatrix}

where :math:`n` is the time step index and :math:`\Delta t` is the time step.

For the integral, we do a trapezoidal rule:

.. math::
    \nu_i (t) - \int_0^{l_i} (l_i - s) \chi_i(s,t) ds \\
    = \nu_i ^n - \sum_{k=1}^{k=N_x - 1} \frac{(l_i - s_k)\chi_i(s_k)^n + (l_i - s_{k+1})\chi_i(s_{k+1})^n}{2}(s_{k+1} - s_k)

For the nonlinear term, we explicit the nonlinear term at time step :math:`n`:

.. math::
    \chi_0\frac{\eta_i^{n+1} - \eta_i^n}{\Delta t} - \frac{\chi_i^{n+1} - \chi_i^n}{\Delta t} + \frac{|\chi_i^n - \chi_i^{n-1}|}{\Delta t}\eta_i^{n+1} = 0 

We finally get the following unknowns vector of size :math:`6  + 2N_x` with :math:`N_x` the number of points of the spatial discretization of the messenger cable:

.. math::
    \begin{pmatrix}
    \nu_i^n & \varphi_i^n & \dot \nu_i^n & \dot \varphi_i^n & F_i^n & M_i^n  & \chi_{i,k}^n & \eta_{i,k}^n
    \end{pmatrix}

Finally the system rewrites: 

.. math::
    \nu_i^{n+1} - \frac{\Delta t}{2}\dot \nu_i^{n+1} &= \nu_i^n + \frac{\Delta t}{2} \dot \nu_i^n  \\
        \varphi_i^{n+1} - \frac{\Delta t}{2}\dot \varphi_i^{n+1} &= \varphi_i^n + \frac{\Delta t}{2} \dot \varphi_i^n  \\
        \dot \nu_i^{n+1} + \frac{\Delta t}{2}\mathcal{M}_i^{-1}[0,0] F_i^{n+1} + \frac{\Delta t}{2}\mathcal{M}_i^{-1}[0,1] M_i^{n+1} &= \dot \nu_i^n +  \frac{\Delta t}{2}\mathcal{M}_i^{-1}[0,0] (f_{ext}^{n+1} + f_{ext}^n - F_i^n) \\ 
        & \quad \quad ~ + \frac{\Delta t}{2}\mathcal{M}_i^{-1}[0,1] (m_{ext}^{n+1} + m_{ext}^n - M_i^n)\\
        \dot \varphi_i^{n+1} + \frac{\Delta t}{2}\mathcal{M}_i^{-1}[1,0] F_i^{n+1} + \frac{\Delta t}{2}\mathcal{M}_i^{-1}[1,1] M_i^{n+1} &= \dot \varphi_i^n +  \frac{\Delta t}{2}\mathcal{M}_i^{-1}[1,0] (f_{ext}^{n+1} + f_{ext}^n - F_i^n) \\ 
        & \quad \quad ~ + \frac{\Delta t}{2}\mathcal{M}_i^{-1}[1,1] (m_{ext}^{n+1} + m_{ext}^n - M_i^n)\\
        \nu_i^{n+1} - \sum_{k=1}^{k=N_x-1} \frac{(l_i - s_k)\chi_{i,k}^n + (l_i - s_{k+1})\chi_{i,k+1}^n}{2}(s_{k+1} - s_k) &= 0 \\
        \varphi_i^{n+1} - \sum_{k=1}^{k=N_x - 1} \frac{\chi_{i,k}^n + \chi_{i,k+1}^n}{2} (s_{k+1} - s_{k}) &=0 \\ 
         M_i^{n+1} + F_i^{n+1}(l_i- s_k) - EI_{\min} \chi_{i,k}^{n+1} -(EI_{\max} - EI_{min} ) \chi_0 \eta_{i,k}^{n+1} &= 0 \\
    \eta_{i,k}^{n+1} (\chi_0 + |\chi_{i,k}^n - \chi_{i,k}^{n-1}|) - \chi_{i,k}^{n+1} &= -\chi_{i,k}^n + \chi_0 \eta^n_{i,k} 




Force imposed at the clamp
==========================

This model is simply a reformulation of the "acceleration imposed at the clamp" model. 
When imposing the force at the clamp, we have to solve the system of each mass and the clamp at the same time.

Here is how we rewrite the equations of motions of the whole system:

.. math::
    \underbrace{
    \begin{pmatrix}
    m_1 & -m_1e_{G_1} & 0 & 0 & m_1 & m_1 \varepsilon_1 b_c \\
    -m_1e_{G_1} & I_{G_1} + m_1 e_{G_1}^2 & 0 & 0 & -m_1e_{G_1} & -m_1 e_{G_1} \varepsilon_1 b_c  \\ 
    0 & 0 & m_2 & -m_2e_{G_2} & m_2 & m_2 \varepsilon_2 b_c \\
    0 & 0 & -m_2e_{G_2} & I_{G_2} + m_2 e_{G_2}^2 & -m_2e_{G_2} & -m_2 e_{G_2} \varepsilon_1 b_c  \\
    m_1 & -m_1e_{G_1} & m_2 & -m_2e_{G_2} & m_c + m_1 + m_2 & (m_1 - m_2)b_c \\
        m_1l_1^* & I_1  - m_1e_1l_1^* & -m_2l_2^* & -(I_2 - m_2e_2l_2^*) & m_1l_1^*-m_2l_2^* & I_{G_c} + m_1b_cl_1^* + m_2b_cl_2^*
    \end{pmatrix}}_{M}
    \underbrace{
    \begin{pmatrix}
    \ddot x_1 \\ 
    \ddot \varphi_1 \\ 
    \ddot x_2 \\ 
    \ddot \varphi_2 \\ 
    \ddot w_c \\ 
    \ddot \varphi_c 
    \end{pmatrix}}_{Y} + 
    \underbrace{
    \begin{pmatrix}
    F_1 \\
    M_1 \\
    F_2 \\ 
    M_2 \\ 
    0 \\ 
    0
    \end{pmatrix}}_{F} = 
    \underbrace{
    \begin{pmatrix}
    0 \\
    0 \\
    0 \\ 
    0 \\
    F_c \\
    M_c 
    \end{pmatrix}}_{R}

We also do a Crank Nicolson scheme to get the positions from the accelerations, only with :math:`\ddot x_1, \ddot \varphi_1, \ddot x_2, \ddot \varphi_2`
and we have an unknowns vector of size :math:`14 + 4N_x` with :math:`N_x` the number of points of the spatial discretization of the messenger cable:

.. math::
    \begin{pmatrix}
    x_1^n ~
    \varphi_1^n ~
    x_2^n ~
    \varphi_2^n ~
    \dot x_1^n ~
    \dot \varphi_1^n ~
    \dot x_2^n ~
    \dot \varphi_2^n ~
    \ddot w_c^n ~
    \ddot \varphi_c^n ~
    F_1^n ~
    M_1^n ~
    F_2^n ~
    M_2^n ~ 
    \chi_{1,k}^n ~
    \eta_{1,k}^n ~
    \chi_{2,k}^n ~
    \eta_{2,k}^n ~
    \end{pmatrix}


We use the same discretization for the integral and the nonlinear term as in the "acceleration imposed at the clamp" model. 


Coupling with the beam solver
=============================

We recall the beam equation: 

.. math::
    m\frac{\partial^2 y}{\partial t ^2 } + 2m\omega_0 \zeta \frac{\partial y}{\partial t  }  + \frac{\partial^2 M}{\partial x ^2 }   - H \frac{\partial^2 y}{\partial x ^2 } = F(x,t)

where the bending moment $M$ depends on the curvature. 

We denote by the index :math:`s` the stockbridge position. 

The coupling resolution is the following:

* Transfer of the stockbridge force to the cable force: :math:`F_s = -F_c`
* Resolution of the beam equations 
* Computation of :math:`\ddot y_s` 
* Resolution of the stockbridges equations by imposing :math:`\ddot y_s`
* Computation of :math:`F_c` 