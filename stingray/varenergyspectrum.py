from __future__ import division
import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize
import logging

import stingray.lightcurve as lightcurve
import stingray.utils as utils
from stingray.gti import check_separate, cross_two_gtis
from stingray.utils import simon
from stingray.crossspectrum import Crossspectrum, AveragedCrossspectrum
from stingray.events import EventList
from stingray.lightcurve import Lightcurve


class VarEnergySpectrum(object):
    def __init__(self, events, freq_interval, energies, ref_band,
                 use_pi=False, log_distr=False):
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
        if self.log_distr:
            self.energies = np.logspace(np.log10(energies[0]),
                                        np.log10(energies[1]),
                                        energies[2])
        else:
            self.energies = np.linspace(energies[0], energies[1], energies[2])
        self.ref_band = ref_band

    def _decide_ref_intervals(self, ref_band, base_band):
        """Eliminate base_band from ref_band."""
        if check_separate(ref_band, base_band):
            return np.asarray([ref_band])
        not_base_band = [[0, base_band[0]],
                         [base_band[1], np.max([ref_band[-1],
                                                base_band[1] + 1])]]
        return cross_two_gtis([ref_band], not_base_band)

    def _construct_lightcurves(self):
        pass



