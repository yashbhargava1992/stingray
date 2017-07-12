from __future__ import division
import numpy as np

from stingray import lightcurve

class Bispectrum(object):
    def __init__(self, lc, maxlag, scale=None):
    	self.lc = lc
        # change to fs = 1/lc.dt
        self.maxlag = maxlag
        self.scale = scale
        self.fs = None
        
        # Outputs
        self.bispec = None
        self.freq = None
        self.cum3 = None
        self.lag = None
