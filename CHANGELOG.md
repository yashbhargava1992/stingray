# Changelog

## Unreleased (2021-01-18)

+ Lots of performance improvements
+ Fixes in rebin functions
+ New statistical functions for signal detection in power spectra and pulsar search periodograms
+ Improved data loading from X-ray satellites (XMM, NICER, NuSTAR, XTE)
+ New implementation of the FFTFIT method to calculate pulsar times of arrival
+ H-test for pulsar searches
+ Z^2_n search adapted to binned and normally distribute pulse profiles

Full set of changes: [`v0.2...HEAD`](git@github.com:stingraysoftware/stingray/compare/v0.2...HEAD)

## v0.2 (2020-06-17)

+ Added Citation info
+ Fixed various normalization bugs in Powerspectrum
+ Speedup of lightcurve creation and handling
+ Made code compatible with Python 3.8, and dropped support to Python 2.7
+ Test speedups
+ Dead time models and Fourier Amplitude Difference correction
+ Roundtrip of LightCurve to lightkurve objects
+ Fourier-domain accelerated search for pulsars
+ Adapt package to APE-17
+ Periodograms now also accept event lists (instead of just light curves)
+ Allow transparent MJDREF change in event lists and light curves

Full set of changes: [`v0.1.3...v0.2`](git@github.com:stingraysoftware/stingray/compare/v0.1.3...v0.2)

## v0.1.3 (2019-06-11)

+ Bug fixes

## v0.1.2

+ Bug fixes

## v0.1.1

+ Bug fixes

## v0.1 (2019-05-29)

