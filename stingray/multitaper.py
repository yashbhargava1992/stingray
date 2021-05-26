import copy
from stingray.gti import check_gtis, cross_two_gtis
from stingray.crossspectrum import Crossspectrum
from stingray.powerspectrum import Powerspectrum
import warnings
from collections.abc import Iterable, Iterator

import numpy as np
import scipy
import scipy.optimize
import scipy.stats
from scipy import signal
from scipy import fft
from scipy.signal import windows

from .events import EventList
from .lightcurve import Lightcurve
from .utils import show_progress, simon


class Multitaper(Powerspectrum):
    """
    Class to calculate the multitaper periodogram from a lightcurve data.
    Parameters
    ----------
    data: :class:`stingray.Lightcurve` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }, optional, default ``frac``
        The normaliation of the power spectrum to be used. Options are
        ``leahy``, ``frac``, ``abs`` and ``none``, default is ``frac``.

    NW: float, optional, default ``None``
        The normalized half-bandwidth of the data tapers, indicating a
        multiple of the fundamental frequency of the DFT (Fs/N).
        Common choices are n/2, for n >= 4.

    adaptive: boolean, optional, default ``False``
        Use an adaptive weighting routine to combine the PSD estimates of
        different tapers.

    jackknife: boolean, optional, default ``True``
        Use the jackknife method to make an estimate of the PSD variance
        at each point.

    low_bias: boolean, optional, default ``True``
        Rather than use 2NW tapers, only use the tapers that have better than
        90% spectral concentration within the bandwidth (still using
        a maximum of 2NW tapers)

    Fs: float, optional, default ``1``
        Sampling rate of the signal

    Other Parameters
    ----------------
    gti: 2-d float array
        ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!

    Attributes
    ----------
    norm: {``leahy`` | ``frac`` | ``abs`` | ``none`` }
        the normalization of the power spectrun

    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectrum). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged powers in each bin

    n: int
        The number of data points in the light curve

    nphots: float
        The total number of photons in the light curve

    jk_var_deg_freedom: numpy.ndarray
        Array differs depending on whether
        the jackknife was used. It is either
        * The jackknife estimated variance of the log-psd, OR
        * The degrees of freedom in a chi2 model of how the estimated
          PSD is distributed about the true log-PSD (this is either
          2*floor(2*NW), or calculated from adaptive weights)

    Notes
    -----
    The bandwidth of the windowing function will determine the number of
    tapers to use. This parameters represents trade-off between frequency
    resolution (lower main lobe BW for the taper) and variance reduction
    (higher BW and number of averaged estimates). Typically, the number of
    tapers is calculated as 2x the bandwidth-to-fundamental-frequency
    ratio, as these eigenfunctions have the best energy concentration.

    """

    def __init__(self, data=None, norm="frac", gti=None, dt=None, lc=None,
                 NW=None, adaptive=False, jackknife=True, low_bias=True):

        if lc is not None:
            warnings.warn("The lc keyword is now deprecated. Use data "
                          "instead", DeprecationWarning)
        if data is None:
            data = lc

        if isinstance(norm, str) is False:
            raise TypeError("norm must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be 'frac', 'abs', 'leahy', or 'none'!")

        self.norm = norm.lower()

        if isinstance(data, EventList) and dt is None:
            raise ValueError(
                "If using event lists, please specify "
                "the bin time to generate lightcurves.")

        if data is None:
            self.freq = None
            self.power = None
            self.power_err = None
            self.df = None
            self.m = 1
            self.n = None
            self.nphots = None
            self.jk_var_deg_freedom = None
            return
        elif not isinstance(data, EventList):
            lc = data
        else:
            lc = data.to_lc(dt)

        self.gti = gti
        self.lc = lc
        self.power_type = 'real'
        self.fullspec = False

    def _make_multitaper_periodogram(self, lc, NW=4, adaptive=False,
                                     jackknife=True, low_bias=True):

        Kmax = int(2 * NW)

        if not isinstance(lc, Lightcurve):
            raise TypeError("lc must be a lightcurve.Lightcurve object")

        if self.gti is None:
            self.gti = cross_two_gtis(lc.gti, lc.gti)

        check_gtis(self.gti)

        if self.gti.shape[0] != 1:
            raise TypeError("Non-averaged Spectra need "
                            "a single Good Time Interval")

        lc = lc.split_by_gti()[0]

        self.meancounts = lc.meancounts
        self.nphots = np.float64(np.sum(lc.counts))

        self.err_dist = 'poisson'
        if lc.err_dist == 'poisson':
            self.var = lc.meancounts
        else:
            self.var = np.mean(lc.counts_err) ** 2
            self.err_dist = 'gauss'

        # the frequency resolution
        self.df = 1.0 / lc.tseg

        # the number of averaged periodograms in the final output
        # This should *always* be 1 here
        self.m = 1

        self.freq, self.unnorm_power = \
            self._fourier_multitaper(lc, NW=NW, adaptive=adaptive,
                                     jackknife=jackknife, low_bias=low_bias)

        self.power = self._normalize_multitaper(self.unnorm_power, lc.tseg)

        if lc.err_dist.lower() != "poisson":
            simon("Looks like your lightcurve statistic is not poisson."
                  "The errors in the Powerspectrum will be incorrect.")

        self.power_err = self.power / np.sqrt(self.m)

    def _fourier_nultitaper(self, lc, NW=4, adaptive=False,
                            jackknife=True, low_bias=True):
        pass

    def _normalize_multitaper(self, unnorm_power, tseg):
        pass
