.. slenderpy documentation master file, created by sphinx-quickstart
on Thu Mar 18 09:09:09 2021.  You can adapt this file completely to
your liking, but it should at least contain the root `toctree`
directive.

Welcome to slenderpy's documentation!
=====================================

**slenderpy** is a python package for simulating the motion of long-span
structures (cables and beams) using Python.

Cables and beams have many structural and engineering applications:
cable-stayed bridges, guyed masts, mooring lines, overhead
lines. slenderpy can compute the response of the cable or beam
structure to a dynamic excitation. The response can then be used for
modal analysis, e.g. to find natural frequencies or mode shapes.

Our objective is to run fast simulations in order perform large
parametric studies which cannot be accessed using full 3D
codes. slenderpy has been developed within RTE's research project
OLLA. RTE (Réseau de Transport d’Électricité) is the electricity
transmission system operator of France.


Related Work
------------

- `On the validation and use of a simplified model of aeolian vibration of overhead lines for parametric studies, Eurodyn 2020 <https://doi.org/10.47964/1120.9163.19200>`_
- Use of simplified models and wind data for prediction of overhead power cable fatigue, Submitted, 2021.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting started

   getting-started

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Philosophy

   contents

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Package Reference

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
