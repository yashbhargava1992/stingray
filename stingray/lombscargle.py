import copy
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from astropy.timeseries.periodograms import LombScargle

from .crossspectrum import Crossspectrum
from .events import EventList
from .exceptions import StingrayError
from .fourier import lsft_fast, lsft_slow, impose_symmetry_lsft
from .lightcurve import Lightcurve
from .utils import simon


class LombScargleCrossspectrum(Crossspectrum):
    main_array_attr = "freq"
    type = "crossspectrum"
    """
    Make a cross spectrum from an unevenly sampled (binned) light curve.
    You can also make an empty :class:`LombScargleCrossspectrum` object to populate with your
    own Fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data1: :class:`stingray.lightcurve.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The dataset for the first channel/band of interest.

    data2: :class:`stingray.lightcurve.Lightcurve` or :class:`stingray.events.EventList`, optional, default ``None``
        The dataset for the second, or "reference", band.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, string, optional, default ``none``
        The normalization of the cross spectrum.

    power_type: {``real``, ``absolute``, ``all`}, string, optional, default ``all``
        Parameter to choose among complete, real part and magnitude of the cross spectrum.

    fullspec: boolean, optional, default ``False``
        If False, keep only the positive frequencies, or if True, keep all of them .

    Other Parameters
    ----------------
    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where ``data1``, ``data2`` are
        :class:`EventList` objects

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!)

    min_freq : float
        Minimum frequency to take the Lomb-Scargle Fourier Transform

    max_freq: float
        Maximum frequency to take the Lomb-Scargle Fourier Transform

    df : float
        The time resolution of the light curve. Only needed where ``data1``, ``data2`` are

    method : str
        The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
        and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
        and Rybicki O(n*log(n))
    
    oversampling : float, optional, default: 5
        Interpolation Oversampling Factor (for the fast algorithm)

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples

    power: numpy.ndarray
        The array of cross spectra (complex numbers)

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging more than one spectra). Note that for a single
        realization (``m=1``) the error is equal to the power.

    df: float
        The frequency resolution

    m: int
        The number of averaged cross-spectra amplitudes in each bin.

    n: int
        The number of data points/time bins in one segment of the light
        curves.

    k: array of int
        The rebinning scheme if the object has been rebinned otherwise is set to 1.

    nphots1: float
        The total number of photons in light curve 1

    nphots2: float
        The total number of photons in light curve 2

    References
    ----------
    .. [1] Scargle, J. D. , "Studies in astronomical time series analysis. III - Fourier
        transforms, autocorrelation functions, and cross-correlation
        functions of unevenly spaced data". ApJ 1:343, p874-887, 1989
    .. [2] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
        of unevenly sampled data". ApJ 1:338, p277, 1989
    """

    def __init__(
        self,
        data1: Optional[Union[EventList, Lightcurve]] = None,
        data2: Optional[Union[EventList, Lightcurve]] = None,
        norm: Optional[str] = "none",
        power_type: Optional[str] = "all",
        dt: Optional[float] = None,
        fullspec: Optional[bool] = False,
        skip_checks: bool = False,
        min_freq: float = 0,
        max_freq: float = None,
        df: float = None,
        method: str = "fast",
        oversampling: int = 5,
    ):
        self._type = None

        if data1 is None and data2 is None:
            self._initialize_empty()
            return

        if dt is None:
            if isinstance(data1, Lightcurve) or isinstance(data2, EventList):
                dt = data1.dt
            elif isinstance(data2, Lightcurve) or isinstance(data2, EventList):
                dt = data2.dt
            if dt is None:
                raise ValueError("dt must be provided for EventLists")

        if not skip_checks:
            good_input = self.initial_checks(
                data1=data1,
                data2=data2,
                norm=norm,
                power_type=power_type,
                dt=dt,
                fullspec=fullspec,
                min_freq=min_freq,
                max_freq=max_freq,
                df=df,
                method=method,
                oversampling=oversampling,
            )

            if not good_input:
                self._initialize_empty()
                return

        self.dt = dt
        norm = norm.lower()
        self.norm = norm
        self.k = 1
        self.df = df

        if isinstance(data1, EventList):
            self.lc1 = data1.to_lc(self.dt)
        else:
            self.lc1 = data1
        if isinstance(data2, EventList):
            self.lc2 = data2.to_lc(self.dt)
        else:
            self.lc2 = data2
        self.power_type = power_type
        self.fullspec = fullspec
        self.norm = norm

        self.nphots1 = self.lc1.counts.sum()
        self.nphots2 = self.lc2.counts.sum()

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.method = method
        self.oversampling = oversampling
        self._make_crossspectrum(
            self.lc1, self.lc2, fullspec, method=method, oversampling=oversampling
        )
        if self.power_type == "absolute":
            self.power = np.abs(self.power)
            self.power_err = np.abs(self.power_err)
            self.unnorm_power = np.abs(self.unnorm_power)
            self.unnorm_power_err = np.abs(self.unnorm_power_err)
        if self.power_type == "real":
            self.power = np.real(self.power)
            self.power_err = np.real(self.power)
            self.unnorm_power = np.real(self.unnorm_power)
            self.unnorm_power_err = np.real(self.unnorm_power_err)
        self._make_auxil_pds(self.lc1, self.lc2)

    def initial_checks(
        self,
        data1,
        data2,
        norm,
        power_type,
        dt,
        min_freq,
        max_freq,
        fullspec,
        df,
        method,
        oversampling,
    ):
        if not isinstance(norm, str):
            raise TypeError("norm must be a string")

        if not isinstance(power_type, str):
            raise TypeError("power_type must be a string")

        if norm.lower() not in ["frac", "abs", "leahy", "none"]:
            raise ValueError("norm must be one of ['frac','abs','leahy','none']")

        if power_type not in ["all", "absolute", "real"]:
            raise ValueError("power_type must be one of ['all','absolute','real']")

        if np.logical_xor(data1 is None, data2 is None):
            raise ValueError("You can't do a cross spectrum with just one lightcurve")

        if min_freq < 0:
            raise ValueError("min_freq must be non-negative")

        if max_freq is not None:
            if max_freq < min_freq or max_freq < 0:
                raise ValueError("max_freq must be non-negative and greater than min_freq")

        if method not in ["fast", "slow"]:
            raise ValueError("method must be one of ['fast','slow']")

        if not isinstance(oversampling, int):
            raise TypeError("oversampling must be an integer")

        if not isinstance(fullspec, bool):
            raise TypeError("fullspec must be a boolean")

        if np.logical_xor(
            not (isinstance(data1, EventList) or isinstance(data1, Lightcurve) or data1 is None),
            not (isinstance(data2, EventList) or isinstance(data2, Lightcurve) or data2 is None),
        ):
            raise TypeError("One of the arguments is not of type Eventlist or Lightcurve or None")

        if not (
            isinstance(data1, EventList) or isinstance(data1, Lightcurve) or data1 is None
        ) and (
            not (isinstance(data2, EventList) or isinstance(data2, Lightcurve) or data2 is None),
        ):
            raise TypeError("Both the events are not of type Eventlist or Lightcurve or None")

        if type(data1) == type(data2):
            if data1 is not None:
                if len(data1.time) != len(data2.time):
                    raise ValueError("data1 and data2 must have the same length")
        else:
            if (isinstance(data1, EventList) or isinstance(data2, EventList)) and (
                isinstance(data1, Lightcurve) or isinstance(data2, Lightcurve)
            ):
                if len(data1.time) != len(data2.time):
                    raise ValueError("data1 and data2 must have the same length")

        return True

    def _make_crossspectrum(self, lc1, lc2, fullspec, method, oversampling):
        """
        Auxiliary method computing the normalized cross spectrum from two
        light curves. This includes checking for the presence of and
        applying Good Time Intervals, computing the unnormalized Fourier
        cross-amplitude, and then renormalizing using the required
        normalization. Also computes an uncertainty estimate on the cross
        spectral powers.

        Parameters
        ----------
        lc1, lc2 : :class:`stingray.lightcurve.Lightcurve` objects
            Two light curves used for computing the cross spectrum.

        fullspec: boolean, default ``False``
            Return full frequency array (True) or just positive frequencies (False)

        method : str
            The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
            and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
            and Rybicki O(n*log(n))

        """
        self.meancounts1 = lc1.meancounts
        self.meancounts2 = lc2.meancounts

        self.err_dist = "poisson"
        if lc1.err_dist == "poisson":
            self.variance1 = lc1.meancounts
        else:
            self.variance1 = np.mean(lc1.meancounts) ** 2
            self.err_dist = "gauss"

        if lc2.err_dist == "poisson":
            self.variance2 = lc2.meancounts
        else:
            self.variance2 = np.mean(lc2.meancounts) ** 2
            self.err_dist = "gauss"

        lc1.dt = lc2.dt
        self.dt = lc1.dt
        self.n = lc1.n

        self.df = 1.0 / lc1.tseg

        self.m = 1

        self.freq, self.unnorm_power = self._ls_cross(
            self.lc1,
            self.lc2,
            fullspec=fullspec,
            method=method,
            oversampling=oversampling,
        )

        self.power = self._normalize_crossspectrum(self.unnorm_power)
        if lc1.err_dist.lower() != lc2.err_dist.lower():
            simon(
                "Your lightcurves have different statistics."
                "The errors in the Crossspectrum will be incorrect."
            )

        elif lc1.err_dist.lower() != "poisson":
            simon(
                "Looks like your lightcurve statistic is not poisson."
                "The errors in the Crossspectrum will be incorrect."
            )

        if self.__class__.__name__ == "LombScarglePowerspectrum":
            self.power_err = self.unnorm_power_err = self.power / np.sqrt(self.m)
        elif self.__class__.__name__ == "LombScargleCrossspectrum":
            simon(
                "Errorbars on cross spectra are not thoroughly tested."
                "Please report any inconsistencies."
            )
            self.unnorm_power_err = np.sqrt(2) / np.sqrt(self.m)
            self.unnorm_power_err /= np.divide(2, np.sqrt(np.abs(self.nphots1 * self.nphots2)))
            self.unnorm_power_err += np.zeros_like(self.unnorm_power)
            self.power_err = self._normalize_crossspectrum(self.unnorm_power_err)

    def _make_auxil_pds(self, lc1, lc2):
        __doc__ = super()._make_auxil_pds.__doc__
        if lc1 is not lc2 and isinstance(lc1, Lightcurve):
            self.pds1 = LombScargleCrossspectrum(
                lc1,
                lc1,
                power_type=self.power_type,
                norm=self.norm,
                dt=self.dt,
                fullspec=self.fullspec,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                df=self.df,
                method=self.method,
                oversampling=self.oversampling,
            )
            self.pds2 = LombScargleCrossspectrum(
                lc2,
                lc2,
                power_type=self.power_type,
                norm=self.norm,
                dt=self.dt,
                fullspec=self.fullspec,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                df=self.df,
                method=self.method,
                oversampling=self.oversampling,
            )

    def _ls_cross(self, lc1, lc2, freq=None, fullspec=False, method="fast", oversampling=5):
        """
        Lomb-Scargle Fourier transform the two light curves, then compute the cross spectrum.
        Computed as CS = lc1 x lc2* (where lc2 is the one that gets
        complex-conjugated). The user has the option to either get just the
        positive frequencies or the full spectrum.

        Parameters
        ----------
        lc1: :class:`stingray.lightcurve.Lightcurve` object
            One light curve to be Lomb-Scargle Fourier transformed. Ths is the band of
            interest or channel of interest.

        lc2: :class:`stingray.lightcurve.Lightcurve` object
            Another light curve to be Fourier transformed.
            This is the reference band.

        fullspec: boolean. Default is False.
            If True, return the whole array of frequencies, or only positive frequencies (False).

        method : str
            The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
            and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
            and Rybicki O(n*log(n))

        Returns
        -------
        freq: numpy.ndarray
            The frequency grid at which the LSFT was evaluated

        cross: numpy.ndarray
            The cross spectrum value at each frequency.

        """
        if not freq:
            freq = (
                LombScargle(
                    lc1.time,
                    lc1.counts,
                    fit_mean=False,
                    center_data=False,
                    normalization="psd",
                ).autofrequency(
                    minimum_frequency=max(self.min_freq, 0), maximum_frequency=self.max_freq
                ),
            )[0]
            freqs2 = (
                LombScargle(
                    lc2.time,
                    lc2.counts,
                    fit_mean=False,
                    center_data=False,
                    normalization="psd",
                ).autofrequency(
                    minimum_frequency=max(self.min_freq, 0), maximum_frequency=self.max_freq
                ),
            )[0]
            if max(freqs2) > max(freq):
                freq = freqs2

        if method == "slow":
            lsft1 = lsft_slow(lc1.counts, lc1.time, freq)
            lsft2 = lsft_slow(lc2.counts, lc2.time, freq)
        elif method == "fast":
            lsft1 = lsft_fast(lc1.counts, lc1.time, freq, oversampling=oversampling)
            lsft2 = lsft_fast(lc2.counts, lc2.time, freq, oversampling=oversampling)
        if fullspec:
            lsft1, _ = impose_symmetry_lsft(lsft1, np.sum((lc1.counts)), lc1.n, freq)
            lsft2, freq = impose_symmetry_lsft(lsft2, np.sum(lc2.counts), lc2.n, freq)
        cross = np.multiply(lsft1, np.conjugate(lsft2))
        return freq, cross

    def _initialize_empty(self):
        self.freq = None
        self.power = None
        self.power_err = None
        self.unnorm_power = None
        self.unnorm_power_err = None
        self.df = None
        self.dt = None
        self.nphots1 = None
        self.nphots2 = None
        self.m = 1
        self.n = None
        self.fullspec = None
        self.k = 1
        self.err_dist = None
        self.method = None
        self.meancounts1 = None
        self.meancounts2 = None
        self.oversampling = None
        self.variance1 = None
        self.variance2 = None
        return

    def time_lag(self):
        super().__doc__
        return self.phase_lag() / (2 * np.pi * self.freq)

    def classical_significances(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'classical_significances' ! Not applicable for unevenly sampled data"
        )

    def from_time_array(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_time_array' ! Not applicable for unevenly sampled data"
        )

    def from_events(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_events' ! Not applicable for unevenly sampled data"
        )

    def from_lightcurve(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_lightcurve' ! Not applicable for unevenly sampled data"
        )

    def from_lc_iterable(self):
        """Not applicable for unevenly sampled data"""
        raise AttributeError(
            "Object has no attribute named 'from_lc_iterable' ! Not applicable for unevenly sampled data"
        )

    def _initialize_from_any_input(self):
        """Not required for unevenly sampled data"""
        raise AttributeError("Object has no attribute named '_initialize_from_any_input' !")


class LombScarglePowerspectrum(LombScargleCrossspectrum):
    type = "powerspectrum"
    """
    Make a :class:`LombScarglePowerspectrum` (also called periodogram) from a unevenly sampled (binned)
    light curve. Periodograms can be normalized by either Leahy normalization,
    fractional rms normalization, absolute rms normalization, or not at all.

    You can also make an empty :class:`LombScarglePowerspectrum` object to populate with
    your own fourier-transformed data (this can sometimes be useful when making
    binned power spectra).

    Parameters
    ----------
    data: :class:`stingray.lightcurve.Lightcurve` or :class:`stingray.events.EventList` object, optional, default ``None``
        The light curve data to be Fourier-transformed.

    norm: {``frac``, ``abs``, ``leahy``, ``none``}, string, optional, default ``none``
        The normalization of the power spectrum.

    power_type: {``real``, ``absolute``, ``all`}, string, optional, default ``all``
        Parameter to choose among complete, real part and magnitude of the power spectrum.

    fullspec: boolean, optional, default ``False``
        If False, keep only the positive frequencies, or if True, keep all of them .

    Other Parameters
    ----------------
    dt: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where ``data`` is a
        :class:`EventList` object

    skip_checks: bool
        Skip initial checks, for speed or other reasons (you need to trust your
        inputs!).
    
    min_freq : float
        Minimum frequency to take the Lomb-Scargle Fourier Transform

    max_freq: float
        Maximum frequency to take the Lomb-Scargle Fourier Transform

    df : float
        The time resolution of the light curve. Only needed where ``data`` is a :class`stingray.Eventlist` object

    method : str
        The method to be used by the Lomb-Scargle Fourier Transformation function. `fast`
        and `slow` are the allowed values. Default is `fast`. fast uses the optimized Press
        and Rybicki O(n*log(n))

    oversampling : float, optional, default: 5
        Interpolation Oversampling Factor (for the fast algorithm)

    Attributes
    ----------
    freq: numpy.ndarray
        The array of mid-bin frequencies that the Fourier transform samples.

    power: numpy.ndarray
        The array of normalized squared absolute values of Fourier
        amplitudes.

    power_err: numpy.ndarray
        The uncertainties of ``power``.
        An approximation for each bin given by ``power_err= power/sqrt(m)``.
        Where ``m`` is the number of power averaged in each bin (by frequency
        binning, or averaging power spectra of segments of a light curve).
        Note that for a single realization (``m=1``) the error is equal to the
        power.

    df: float
        The frequency resolution.

    m: int
        The number of averaged powers in each bin.

    n: int
        The number of data points in the light curve.

    nphots: float
        The total number of photons in the light curve.
    """

    def __init__(
        self,
        data: Optional[Union[Lightcurve, EventList]] = None,
        norm: Optional[str] = "frac",
        power_type: Optional[str] = "all",
        dt: Optional[float] = None,
        fullspec: Optional[bool] = False,
        skip_checks: Optional[bool] = False,
        min_freq: Optional[float] = 0,
        max_freq: Optional[float] = None,
        df: Optional[float] = None,
        method: Optional[str] = "fast",
        oversampling: Optional[int] = 5,
    ):
        self._type = None
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)

        LombScargleCrossspectrum.__init__(
            self,
            data1=data1,
            data2=data2,
            norm=norm,
            power_type=power_type,
            dt=dt,
            skip_checks=skip_checks,
            min_freq=min_freq,
            max_freq=max_freq,
            df=df,
            method=method,
            oversampling=oversampling,
        )

        self.nphots = self.nphots1
        self.dt = dt
