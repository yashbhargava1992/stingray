# Licensed under MIT license - see LICENSE.rst

"""
Library of Time Series Methods For Astronomical X-ray Data.

*******
Modules
*******

covariancespectrum
==================

.. automodule:: stingray.covariancespectrum

AveragedCovariancespectrum
--------------------------

.. autoclass:: stingray.AveragedCovariancespectrum

Covariancespectrum
------------------

.. autoclass:: stingray.Covariancespectrum

crossspectrum
=============

.. automodule:: stingray.crossspectrum

AveragedCrossspectrum
---------------------

.. autoclass:: stingray.AveragedCrossspectrum

Crossspectrum
-------------

.. autoclass:: stingray.Crossspectrum

lightcurve
==========

.. automodule:: stingray.lightcurve

Lightcurve
----------

.. autoclass:: stingray.Lightcurve

powerspectrum
=============

.. automodule:: stingray.powerspectrum

AveragedPowerspectrum
---------------------

.. autoclass:: stingray.AveragedPowerspectrum

Powerspectrum
-------------

.. autoclass:: stingray.Powerspectrum

***************
All definitions
***************

"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from stingray.lightcurve import *
    from stingray.utils import *
    from stingray.powerspectrum import *
    from stingray.crossspectrum import *
    from stingray import *
    from stingray.exceptions import *
    from stingray.covariancespectrum import *
