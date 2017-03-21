.. _stingray-simulator:

***************************************
Simulations (`stingray.simulator`)
***************************************

Introduction
============

`stingray.simulator` provides a framework to simulate light curves with given variability distributions. In time series experiments, understanding the certainty is crucial to interpret the derived results in context of physical models. The simulator module provides tools to assess this uncertainty by simulating time series and spectral data. 

Stingray simulator supports multiple methods to carry out these simulation. Light curves can be simulated through power-law spectrum, through a user-defined defined or pre-defined model, or through impulse responses. The module is designed in a way such that all these methods can be accessed using similar set of commands.

.. note::

    `stingray.simulator` is currently a work-in-progress, and thus it is likely
    there will still be API changes in later versions of Astropy.  Backwards
    compatibility support between versions will still be maintained as much as
    possible, but new features and enhancements are coming in future versions.

.. _stingray-getting-started:

Getting started
===============

The examples here assume that the following libraries and modules have been imported::
	
	>>> import numpy as np
	>>> from stingray import Lightcurve, sampledata
	>>> from stingray.simulator import simulator, models

Creating a Simulator Object
---------------------------

Stingray has a simulator class which can be used to instantiate a simulator
object and subsequently, perform simulations. We can pass on arguments to
the simulator class to set the properties of desired light curve.

The simulator object can be instantiated as::

	>>> sim = simulator.Simulator(N=1024, mean=0.5, dt=0.125)

Here, `N` specifies the bins count of the simulated light curve, `mean` specifies
the mean value, and `dt` is the time resolution. Additional arguments can be
provided to specify the `rms` of the simulated light curve, or to account for the
effect of red noise leakage.

Simulate Method
---------------

Stingray provides multiple ways to simulate a light curve. However, all these
techniques can be accessed using a single simulate method. When only an
integer argument is provided, that integer is interpreted as the decay value of
the power-law spectrum. When an integer array is the input, that array is
treated as a user-defined spectral model. In contrast, a pre-defined keyword
can be specified to select a pre-defined model. Finally, by providing an original
light curve object and an impulse response array, a simulated light curve can be
modelled.

Using Power-Law Spectrum
------------------------

When only an integer argument ( beta) is provided, that integer defines the
shape of the power law spectrum. Passing beta as 1 gives a flicker-noise distribution,
while a beta of 2 generates a random-walk distribution.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from stingray.simulator import simulator

   # Instantiate simulator object
   sim = simulator.Simulator(N=1024, mean=0.5, dt=0.125)
   # Specify beta values
   lc1 = sim.simulate(1)
   lc2 = sim.simulate(2)

   plt.figure(figsize=(8, 10))
   plt.subplot(2,1,1)
   plt.plot(lc1.counts, 'k')
   plt.title('Flicker-noise distribution simulation')
   plt.subplot(2,1,2)
   plt.title('Random-Walk distribution simulation')
   plt.plot(lc2.counts, 'k')
   plt.show()

Using User-defined Model
------------------------

Light curve can also be simulated using a user-defined spectrum, which can be
passed on as a numpy array.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   from stingray.simulator import simulator

   # Instantiate simulator object
   sim = simulator.Simulator(N=1024, mean=0.5, dt=0.125)
   # Define a spectrum and simulate
   w = np.fft.rfftfreq(sim.N, d=sim.dt)[1:]
   spectrum = np.power((1/w),2/2)
   lc = sim.simulate(spectrum)

   plt.plot(lc.counts, 'k')
   plt.show()

Using Pre-defined Models
------------------------

One of the pre-defined spectrum models can be used to simulate a light curve.
In this case, model name and model parameters (as list iterable) need to be
passed on as function arguments.

Using Impulse Response
----------------------

In order to simulate a light curve using impulse response, first we need to generate
the impulse response itself and the original light curve.

Here, we import a sample light curve from stingray sampledata module.

Channel Simulation
==================

The simulator class provides the functionality to simulate light curves independently for each channel. This is useful, for example, when dealing with energy dependent impulse responses where we can create a di↵erent simulation channel for each energy range. The module provides options to count, retrieve and delete channels.

Reference/API
=============

