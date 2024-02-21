Install
=======

If you have access to eurobios gitlab, just type ``pip install slenderpy@git+https://github.com/rte-france/pallas-slenderpy``
in a terminal. If you got
this package from the sources, type ``pip install .`` at
the root of the repository.

Basic usage
===========

Cable Example
-------------

.. code-block:: python

   from slenderpy import cable
   from slenderpy import simtools
   from slenderpy import wind

   # cable parameters
   cp = dict(mass=1.57, diameter=0.031, EA=3.76E+07, length=400., tension=3.7E+04, h=0.)

   # simulation parameters
   sp = dict(ns=501, t0=0., tf=30., dt=2.0E-03, dr=1.0E-02, los=[0.1,0.5], pp=True)

   # force parameters (bishop & hassan)
   fp = dict(u=3.0, st=0.2, cd=1.1, cd0=0., cl=0., cl0=0.6, d=cp['diameter'])

   # cable, parameters and force construction
   cb = cable.SCable(**cp)
   pm = simtools.Parameters(**sp)
   fr = wind.BHRel(**fp)

   # run solver (zt is the viscous damping)
   dat = cable.solve(cb, pm, force=fr, zt=0.0)

   # plot results overview
   simtools.multiplot(dat, lb=['test'], Lref=cb.Lp)


Beam Example
------------

.. code-block:: python

   from slenderpy import beam
   from slenderpy import fdm_utils as fdu
   from slenderpy import force
   from slenderpy import simtools

   # beam parameters
   bp = dict(mass=1.57, ei=[2155., 28.], kp=[0.018], length=50., tension=2.8E+04)

   # simulation parameters
   pp = dict(ns=501, t0=0., tf=5., dt=2.0E-03, dr=1.0E-02, los=[0.1, 0.5], pp=True)

   # force parameters
   fp = dict(f=1.0, a=+1.0E+03, s=0.5*bp['length'], m=bp['mass'], L=bp['length'])

   # create boundary conditions
   bl, br = fdu.rot_free('left', y=0., d2y=0.), fdu.rot_free('right', y=0., d2y=0.)

   # beam, simulation and force construction
   cb = beam.Beam(**bp)
   pm = simtools.Parameters(**pp)
   fr = force.Excitation(**fp)

   # run solver
   dat = beam.solve_cst(cb, pm, force=fr, c0=0., bcl=bl, bcr=br)

   # plot results overview
   simtools.multiplot(dat, lb=['test'], Lref=cb.Lp)
