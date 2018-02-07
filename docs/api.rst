Stingray API
************

Library of Time Series Methods For Astronomical X-ray Data.

Data Classes
============

Lightcurve
----------

.. autoclass:: stingray.Lightcurve
   :members:
   :private-members:
   :special-members:

----

EventList
---------

.. autoclass:: stingray.events.EventList
   :members:
   :private-members:
   :special-members:

----


Fourier Products
================

Crossspectrum
-------------

.. autoclass:: stingray.Crossspectrum
   :members:
   :private-members:
   :special-members:

----

Powerspectrum
-------------

.. autoclass:: stingray.Powerspectrum
   :members:
   :private-members:
   :special-members:

----

AveragedCrossspectrum
---------------------

.. autoclass:: stingray.AveragedCrossspectrum
   :members:
   :private-members:
   :special-members:

----


AveragedPowerspectrum
---------------------

.. autoclass:: stingray.AveragedPowerspectrum
   :members:
   :private-members:
   :special-members:


----

CrossCorrelation
----------------

.. autoclass:: stingray.CrossCorrelation
   :members:
   :private-members:
   :special-members:

----

AutoCorrelation
---------------

.. autoclass:: stingray.AutoCorrelation
   :members:
   :private-members:
   :special-members:

----


Higher-Order Fourier and Spectral Timing Products
=================================================

Bispectrum
----------

.. autoclass:: stingray.bispectrum.Bispectrum
   :members:
   :private-members:
   :special-members:

----


Covariancespectrum
------------------

.. autoclass:: stingray.Covariancespectrum
   :members:
   :private-members:
   :special-members:

----

AveragedCovariancespectrum
--------------------------

.. autoclass:: stingray.AveragedCovariancespectrum
   :members:
   :private-members:
   :special-members:

----

VarEnergySpectrum
------------------
Abstract base class for spectral timing products including
both variability and spectral information.

.. autoclass:: stingray.varenergyspectrum.VarEnergySpectrum
   :members:
   :private-members:
   :special-members:

----

RmsEnergySpectrum
-----------------

.. autoclass:: stingray.varenergyspectrum.RmsEnergySpectrum
   :members:
   :private-members:
   :special-members:

----

LagEnergySpectrum
-----------------

.. autoclass:: stingray.varenergyspectrum.LagEnergySpectrum
   :members:
   :private-members:
   :special-members:

----

ExcessVarianceSpectrum
----------------------

.. autoclass:: stingray.varenergyspectrum.ExcessVarianceSpectrum
   :members:
   :private-members:
   :special-members:

----


Utilities
=========

GTI Functionality
-----------------
.. automodule:: stingray.gti
   :members:
   :imported-members:

IO Functionality
----------------

.. automodule:: stingray.io
   :members:
   :imported-members:

Other Utility Functions
-----------------------

.. automodule:: stingray.utils
   :members:
   :imported-members:

Modeling
==========

Log-Likelihood Classes
----------------------

.. autoclass:: stingray.modeling.LogLikelihood
   :members:
   :private-members:
   :special-members:


.. autoclass:: stingray.modeling.GaussianLogLikelihood
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.PoissonLogLikelihood
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.PSDLogLikelihood
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.LaplaceLogLikelihood
   :members:
   :private-members:
   :special-members:

----

Posterior Classes
-----------------

.. autoclass:: stingray.modeling.Posterior
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.GaussianPosterior
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.PoissonPosterior
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.PSDPosterior
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.LaplacePosterior
   :members:
   :private-members:
   :special-members:

----

Parameter Estimation Classes
----------------------------

.. autoclass:: stingray.modeling.ParameterEstimation
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.PSDParEst
   :members:
   :private-members:
   :special-members:

----

Auxiliary Classes
-----------------

.. autoclass:: stingray.modeling.OptimizationResults
   :members:
   :private-members:
   :special-members:

.. autoclass:: stingray.modeling.SamplingResults
   :members:
   :private-members:
   :special-members:

----

Convenience Functions
---------------------

.. automodule:: stingray.modeling.scripts
   :members:
   :imported-members:

----

Pulsar
======
.. automodule:: stingray.pulse
   :members:
   :imported-members:

Simulator
=========
.. autoclass:: stingray.simulator.simulator.Simulator
   :members:
   :undoc-members:

Exceptions
==========

.. autoclass:: stingray.exceptions.StingrayError
   :members:
   :undoc-members:
