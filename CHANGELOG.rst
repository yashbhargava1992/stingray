Changelog
=========

v1.0 (2022-03-29)
---------------------
TL,DR: these things will break your code with v1.0:

- Python version < 3.8
- The ``gtis`` keyword in `pulse/pulsar.py` (it is now ``gti``, without the 's')

New
^^^
- Dropped support to Python < 3.8
- Multi-taper periodogram, including a Lomb-Scargle implementation for non-uniformly sampled data
- Create count-rate spectrum when calculating spectral-timing products
- Make modlation upper limit in ``(Averaged)Powerspectrum`` work with any normalization (internally converts to Leahy for the calculation)
- Implement Gardner-Done normalization (1 for perfect correlation, -1 for perfect anticorrelation) for ``Auto/Crosscorrelation``
- New infrastructure for converting ``EventList`` and ``LightCurve`` objects into Astropy ``TimeSeries``
- New infrastructure for converting most Stingray classes into Astropy ``Table`` objects, Xarray and Pandas data frames.
- Save and load of most Stingray classes to/from many different file formats (``pickle``, ``ECSV``, ``HDF5``, ``FITS``, and all formats compatible with Astropy Table)
- Accept input ``EventList`` in ``DynamicalPowerSpectrum``
- New ``stingray.fourier`` module containing the basic timing products, usable on ``numpy`` arrays, and centralizes fft import
- New methods in ``Crossspectrum`` and ``Powerspectrum`` to load data from specific inputs: ``from_events``, ``from_lightcurve``, ``from_time_array``, ``from_lc_list`` (``from_time_array`` was also tested using memory-mapped event lists as inputs: useful in very large datasets)
- New and improved spectral timing methods: ``ComplexCovarianceSpectrum``, ``CovarianceSpectrum``, ``LagSpectrum``, ``RmsSpectrum``
- Some deprecated features are now removed
- ``PSDLogLikelihood`` now also works with a log-rebinned PDS

Improvements
^^^^^^^^^^^^
- Performance on large data sets is VASTLY improved
- Lots of performance improvements in the ``AveragedCrossspectrum`` and ``AveragedPowerspectrum`` classes
- Standardized use of new fast psd/cs algorithm, with ``legacy`` still available as an alternative option to specify
- Reading calibrated photon energy from event files by default
- In ``pulse/pulsar.py``, methods use the keyword ``gti`` instead of ``gtis`` (for consistency with the rest of Stingray)
- Moved ``CovarianceSpectrum` to ``VarEnergySpectrum`` and reuse part of the machinery
- Improved error bars on cross-spectral and spectral timing methods
- Measure absolute rms in ``RmsEnergySpectrum``
- Friendlier ``pyfftw`` warnings
- Streamline PDS/CrossSp production, adding ``from_events``, ``from_lc``, ``from_lc_iterable``, and ``from_time_array`` (to input a numpy array) methods
- PDS/CrossSp initially store the unnormalized power, and convert it on the fly when requested, to any normalization

Bug fixes
^^^^^^^^^
- Fixed error bars and ``err_dist`` for sliced (iterated) light curves and power spectra
- Fixed a bug in how the start time when applying GTIs (now using the minimum value of the GTI array, instead of half a time bin below the minimum value)
- Fixed a bug in which all simulator errors were incorrectly non-zero
- Fixed coherence uncertainty
- Documented a Windows-specific issue when large count rate light curves are defined as integer arrays (Windows users should use ``float`` or specify ``int-64``)
- If the variance of the lightcurve is zero, the code will fail to implement Leahy normalization
- The value of the ``PLEPHEM`` header keyword is forced to be a string, in the rare cases that it's a number
- and more!

`Full list of changes`__

__ https://github.com/StingraySoftware/stingray/compare/v0.3...v1.0

v1.0beta was released on 2022-02-25.

v0.3 (2021-05-31)
-----------------

- Lots of performance improvements
- Faster simulations
- Averaged Power spectra and Cross spectra now handle Gaussian light curves correctly
- Fixes in rebin functions
- New statistical functions for signal detection in power spectra and pulsar search periodograms
- Much improved FTOOL-compatible mission support
- New implementation of the FFTFIT method to calculate pulsar times of arrival
- H-test for pulsar searches
- Z^2_n search adapted to binned and normally distribute pulse profiles
- Large data processing (e.g. from NICER) allowed
- Rebinning function now accepts unevenly sampled data
- New saving and loading from/to Astropy Tables and Timeseries
- Improved I/O to ascii, hdf5 and other formats
- Rehaul of documentation

`Full list of changes`__

__ https://github.com/StingraySoftware/stingray/compare/v0.2...v0.3

v0.2 (2020-06-17)
-----------------

- Added Citation info
- Fixed various normalization bugs in Powerspectrum
- Speedup of lightcurve creation and handling
- Made code compatible with Python 3.6, and dropped support to Python 2.7
- Test speedups
- Dead time models and Fourier Amplitude Difference correction
- Roundtrip of LightCurve to lightkurve objects
- Fourier-domain accelerated search for pulsars
- Adapt package to APE-17
- Periodograms now also accept event lists (instead of just light curves)
- Allow transparent MJDREF change in event lists and light curves

`Full list of changes`__

__ https://github.com/StingraySoftware/stingray/compare/v0.1.3...v0.2

v0.1.3 (2019-06-11)
-------------------

- Bug fixes

v0.1.2
------

- Bug fixes

v0.1.1
------

- Bug fixes

v0.1 (2019-05-29)
-----------------

- Initial release.
