from __future__ import division
import numpy as np
from stingray.gti import check_separate, cross_two_gtis
from stingray.lightcurve import Lightcurve
from stingray.utils import assign_value_if_none
from stingray.crossspectrum import AveragedCrossspectrum


class VarEnergySpectrum(object):
    def __init__(self, events, freq_interval, energy_spec, ref_band,
                 bin_time=1, use_pi=False, log_distr=False,
                 segment_size=None):
        """Generic variability-energy spectrum.
        
        Parameters
        ----------
        events : stingray.events.EventList object 
            event list
        freq_interval : [f0, f1], floats
            the frequency range over which calculating the variability quantity
        energies : [emin, emax, N]
            minimum and maximum energy, and number of intervals, of the final 
            spectrum
        ref_band : [emin, emax]
            minimum and maximum energy of the reference band
            
        Other Parameters
        ----------------
        use_pi : boolean
            Use channel instead of energy
        log_distr : boolean
            distribute the energy interval logarithmically
        """
        self.events = events
        self.freq_interval = freq_interval
        self.use_pi = use_pi
        self.bin_time = bin_time
        if log_distr:
            energies = np.logspace(np.log10(energy_spec[0]),
                                   np.log10(energy_spec[1]),
                                   energy_spec[2] + 1)
        else:
            energies = np.linspace(energy_spec[0], energy_spec[1],
                                   energy_spec[2] + 1)

        self.energy_intervals = list(zip(energies[0: -1], energies[1:]))
        self.ref_band = ref_band

        self.segment_size = segment_size
        self.spectrum, self.spectrum_error = self._spectrum_function()


    def _decide_ref_intervals(self, base_band, ref_band):
        """Eliminate base_band from ref_band."""
        if check_separate([ref_band], [base_band]):
            return np.asarray([ref_band])
        not_base_band = [[0, base_band[0]],
                         [base_band[1], np.max([ref_band[-1],
                                                base_band[1] + 1])]]
        return cross_two_gtis([ref_band], not_base_band)

    def _construct_lightcurves(self, base_band, tstart=None, tstop=None,
                               exclude=True):
        if self.use_pi:
            energies = self.events.pi
        else:
            energies = self.events.pha

        tstart = assign_value_if_none(tstart, self.events.time[0])
        tstop = assign_value_if_none(tstop, self.events.time[-1])

        good = (energies >= base_band[0]) & (energies < base_band[1])
        base_lc = Lightcurve.make_lightcurve(self.events.time[good],
                                             self.bin_time,
                                             tstart=tstart,
                                             tseg=tstop - tstart,
                                             gti=self.events.gti)

        if exclude:
            ref_intervals = self._decide_ref_intervals(base_band, ref_band)
        else:
            ref_intervals = [ref_band]

        ref_lc = Lightcurve(base_lc.time, np.zeros_like(base_lc.counts),
                            gti=base_lc.gti)
        for i in ref_intervals:
            good = (energies >= i[0]) & (energies < i[1])
            new_lc = Lightcurve.make_lightcurve(self.events.time[good],
                                                self.bin_time,
                                                tstart=tstart,
                                                tseg=tstop - tstart,
                                                gti=self.events.gti)
            ref_lc = ref_lc + new_lc

        return base_lc, ref_lc

    def _spectrum_function(self):
        return None

    def _calculate_stat(self):
        return None


class RmsEnergySpectrum(VarEnergySpectrum):
    def _spectrum_function(self):
        rms_spec = np.zeros(len(self.energy_intervals))
        for i, eint in enumerate(self.energy_intervals):
            base_lc, ref_lc = self._construct_lightcurves(eint)
            xspect = AveragedCrossspectrum(base_lc, ref_lc,
                                           segment_size=self.segment_size,
                                           norm='frac')
            good = (xspect.freq >= self.freq_interval[0]) & \
                   (xspect.freq < self.freq_interval[1])
            rms_spec[i] = np.sqrt(np.sum(xspect.power[good]))

        return rms_spec, None
