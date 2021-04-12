##################################################
Stingray and Spectral Timing: A Brief Introduction
##################################################

Stingray is a Python library designed to perform times series analysis and related tasks on astronomical light curves.
It supports a range of commonly-used Fourier analysis techniques, as well as extensions for analyzing pulsar data, simulating data sets, and statistical modelling.
Stingray is designed to be easy to extend, and easy to incorporate into data analysis workflows and pipelines.

For a brief overview of the history and state-of-the-art in spectral timing, and for more information about the design and capabilities of Stingray, please refer to `Huppenkothen et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019ApJ...881...39H/abstract>`_.

Current Capabilities
====================

Currently implemented functionality in this library comprises:

* loading event lists from fits files of a few missions (RXTE/PCA, NuSTAR/FPM, XMM-Newton/EPIC)
* constructing light curves from event data, various operations on light curves (e.g. addition, subtraction, joining, and truncation)
* Good Time Interval operations
* power spectra in Leahy, rms normalization, absolute rms and no normalization
* averaged power spectra
* dynamical power spectra
* maximum likelihood fitting of periodograms/parametric models
* (averaged) cross spectra
* coherence, time lags
* cross correlation functions
* RMS spectra and lags (time vs energy, time vs frequency); *needs testing*
* covariance spectra; *needs testing*
* bispectra; *needs testing*
* (Bayesian) quasi-periodic oscillation searches
* simulating a light curve with a given power spectrum
* simulating a light curve from another light curve and a 1-d (time) or 2-d (time-energy) impulse response
* simulating an event list from a given light curve _and_ with a given energy spectrum
* pulsar searches with Epoch Folding, :math:`Z^2_n` test

Future Additions
================

We welcome feature requests: if you need a particular tool that's currently not available or have a new method you think might be usefully implemented in Stingray, please :doc:`get in touch <help>`!

Other future additions we are currently implementing are:

* bicoherence
* phase-resolved spectroscopy of quasi-periodic oscillations
* Fourier-frequency-resolved spectroscopy
* power colours
* full HEASARC-compatible mission support
* pulsar searches with :math:`H`-test
* binary pulsar searches
