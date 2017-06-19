from __future__ import division
import numpy as np
from stingray.gti import check_separate, cross_two_gtis
from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none, simon
from stingray.crossspectrum import AveragedCrossspectrum
from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class VarEnergySpectrum(object):
    def __init__(self, events, freq_interval, energy_spec, ref_band=None,
                 bin_time=1, use_pi=False, segment_size=None, events2=None):

        """Base variability-energy spectrum.

        This class is only a base for the various variability spectra, and it's
        not to be instantiated by itself.

        Parameters
        ----------
        events : stingray.events.EventList object
            event list
        freq_interval : [f0, f1], floats
            the frequency range over which calculating the variability quantity
        energy_spec : list or tuple (emin, emax, N, type)
            if a list is specified, this is interpreted as a list of bin edges;
            if a tuple is provided, this will encode the minimum and maximum
            energies, the number of intervals, and "lin" or "log".

        Other Parameters
        ----------------
        ref_band : [emin, emax], floats; default None
            minimum and maximum energy of the reference band. If None, the
            full band is used.
        use_pi : boolean, default False
            Use channel instead of energy
        events2 : stingray.events.EventList object
            event list for the second channel, if not the same. Useful if the
            reference band has to be taken from another detector.
            
        Attributes
        ----------
        events1 : array-like
            list of events used to produce the spectrum
        events2 : array-like
            if the spectrum requires it, second list of events
        freq_interval : array-like
            interval of frequencies used to calculate the spectrum
        energy_intervals : [[e00, e01], [e10, e11], ...]
            energy intervals used for the spectrum
        spectrum : array-like
            the spectral values, corresponding to each energy interval
        spectrum_error : array-like
            the errorbars corresponding to spectrum
        
        """
        self.events1 = events
        self.events2 = assign_value_if_none(events2, events)
        self.freq_interval = freq_interval
        self.use_pi = use_pi
        self.bin_time = bin_time
        if isinstance(energy_spec, tuple):
            if energy_spec[-1].lower() not in ["lin", "log"]:
                raise ValueError("Incorrect energy specification")

            log_distr = True if energy_spec[-1].lower() == "log" else False

            if log_distr:
                energies = np.logspace(np.log10(energy_spec[0]),
                                       np.log10(energy_spec[1]),
                                       energy_spec[2] + 1)
            else:
                energies = np.linspace(energy_spec[0], energy_spec[1],
                                       energy_spec[2] + 1)
        else:
            energies = np.asarray(energy_spec)

        self.energy_intervals = list(zip(energies[0: -1], energies[1:]))

        self.ref_band = assign_value_if_none(ref_band, [0, np.inf])

        self.segment_size = segment_size
        self.spectrum, self.spectrum_error = self._spectrum_function()

    def _decide_ref_intervals(self, channel_band, ref_band):
        """Eliminate channel_band from ref_band."""
        if check_separate([ref_band], [channel_band]):
            return np.asarray([ref_band])
        not_channel_band = [[0, channel_band[0]],
                            [channel_band[1], np.max([ref_band[-1],
                                                      channel_band[1] + 1])]]
        return cross_two_gtis([ref_band], not_channel_band)

    def _construct_lightcurves(self, channel_band, tstart=None, tstop=None,
                               exclude=True):
        if self.use_pi:
            energies1 = self.events1.pi
            energies2 = self.events2.pi
        else:
            energies2 = self.events2.pha
            energies1 = self.events1.pha

        gti = cross_two_gtis(self.events1.gti, self.events2.gti)

        tstart = assign_value_if_none(tstart,
                                      np.max([self.events1.time[0],
                                              self.events2.time[0]]))
        tstop = assign_value_if_none(tstop,
                                     np.min([self.events1.time[-1],
                                             self.events2.time[-1]]))

        good = (energies1 >= channel_band[0]) & (energies1 < channel_band[1])
        base_lc = Lightcurve.make_lightcurve(self.events1.time[good],
                                             self.bin_time,
                                             tstart=tstart,
                                             tseg=tstop - tstart,
                                             gti=gti,
                                             mjdref=self.events1.mjdref)

        if exclude:
            ref_intervals = self._decide_ref_intervals(channel_band,
                                                       self.ref_band)
        else:
            ref_intervals = [self.ref_band]

        ref_lc = Lightcurve(base_lc.time, np.zeros_like(base_lc.counts),
                            gti=base_lc.gti, mjdref=base_lc.mjdref)

        for i in ref_intervals:
            good = (energies2 >= i[0]) & (energies2 < i[1])
            new_lc = Lightcurve.make_lightcurve(self.events2.time[good],
                                                self.bin_time,
                                                tstart=tstart,
                                                tseg=tstop - tstart,
                                                gti=gti,
                                                mjdref=self.events2.mjdref)
            ref_lc = ref_lc + new_lc

        return base_lc, ref_lc

    @abstractmethod
    def _spectrum_function(self):
        pass


class RmsEnergySpectrum(VarEnergySpectrum):

    def _spectrum_function(self):

        rms_spec = np.zeros(len(self.energy_intervals))
        rms_spec_err = np.zeros_like(rms_spec)
        for i, eint in enumerate(self.energy_intervals):
            base_lc, ref_lc = self._construct_lightcurves(eint,
                                                          exclude=False)
            xspect = AveragedCrossspectrum(base_lc, ref_lc,
                                           segment_size=self.segment_size,
                                           norm='frac')
            good = (xspect.freq >= self.freq_interval[0]) & \
                   (xspect.freq < self.freq_interval[1])
            rms_spec[i] = np.sqrt(np.sum(xspect.power[good]*xspect.df))

            # Root squared sum of errors of the spectrum
            root_sq_err_sum = np.sqrt(np.sum(xspect.power[good]**2))*xspect.df
            # But the rms is the squared root. So,
            # Error propagation
            rms_spec_err[i] = 1 / (2 * rms_spec[i]) * root_sq_err_sum

        return rms_spec, rms_spec_err


class LagEnergySpectrum(VarEnergySpectrum):

    def _spectrum_function(self):

        lag_spec = np.zeros(len(self.energy_intervals))
        lag_spec_err = np.zeros_like(lag_spec)
        for i, eint in enumerate(self.energy_intervals):
            base_lc, ref_lc = self._construct_lightcurves(eint)
            xspect = AveragedCrossspectrum(base_lc, ref_lc,
                                           segment_size=self.segment_size)
            good = (xspect.freq >= self.freq_interval[0]) & \
                   (xspect.freq < self.freq_interval[1])
            lag, lag_err = xspect.time_lag()
            good_lag, good_lag_err = lag[good], lag_err[good]
            coh, coh_err = xspect.coherence()
            lag_spec[i] = np.mean(good_lag)
            coh_check = coh > 1.2 / (1 + 0.2 * xspect.m)
            if not np.all(coh_check[good]):
                simon("Coherence is not ideal over the specified energy range."
                      " Lag values and uncertainties might be underestimated. "
                      "See Epitropakis and Papadakis, A\&A 591, 1113, 2016")

            # Root squared sum of errors of the spectrum
            # Verified!
            lag_spec_err[i] = np.sqrt(np.sum(good_lag_err**2) / len(good_lag))

        return lag_spec, lag_spec_err
