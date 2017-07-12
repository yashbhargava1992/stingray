from __future__ import division
import numpy as np

from stingray import lightcurve

class Bispectrum(object):
    def __init__(self, lc, maxlag, scale=None):
    	    def __init__(self, lc, maxlag, scale=None):

        self._make_bispetrum(lc, maxlag, scale)

    def _make_bispetrum(self, lc, maxlag, scale):
        if not isinstance(lc, lightcurve.Lightcurve):
            raise TypeError('lc must be a lightcurve.ightcurve object')

        self.lc = lc
        self.fs = 1 / lc.dt
        self.n = self.lc.n

        if not isinstance(maxlag, int):
            raise ValueError('maxlag must be an integer')

        # if negative maxlag is entered, convert it to +ve
        if maxlag < 0:
            self.maxlag = -maxlag
        else:
            self.maxlag = maxlag

        if isinstance(scale, str) is False:
            raise TypeError("scale must be a string")

        if scale.lower() not in ["biased", "unbiased"]:
            raise ValueError("scale can only be either 'biased' or 'unbiased'.")
        self.scale = scale.lower()

        # Other Atributes
        self.lags = None
        self.cum3 = None
        self.freq = None
        self.bispec = None
        self.bispec_mag = None
        self.bispec_phase = None

        # converting to a row vector to apply matrix operations
        self.signal = np.reshape(lc, (1, len(self.lc.counts)))
        
        # Mean subtraction before bispecrum calculation
        self.signal = self.signal - np.mean(lc.counts)