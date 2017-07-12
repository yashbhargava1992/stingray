from __future__ import division
import numpy as np

from stingray import lightcurve

class Bispectrum(object):
    def __init__(self, lc, maxlag, scale=None):
    	"""
            Makes a :class:`Bispectrum` object from a given :class:`Lightcurve`.
            
            Bispectrum is a higher order time series analysis method and is calculated by indirect method as
            fourier transform of triple auto-correlation function also called as 3rd Order cumulant.
            
            Parameters
            ----------
            lc: lightcurve.Lightcurve object
                The light curve data for bispectrum calculation.
            maxlag: int, optional, default None
                Maximum lag on both positive and negative sides of 
                3rd order cumulant (Similar to lags in correlation).
                if None, max lag is set to one-half of length of lightcurve.
            scale: {'biased', 'unbiased'}, optional, default 'biased'
                Flag to decide biased or unbiased normalization for 3rd order cumulant function.

            Attributes
            ----------
            lc: lightcurve.Lightcurve 
                The light curve data for bispectrum.
            fs: float
                Sampling freq of light curve.        
            n: int
                Total Number of samples of light curve observations.
            maxlag: int
                Maximum lag on both positive and negative sides of 
                3rd order cumulant (Similar to lags in correlation)
            scale: {'biased', 'unbiased'}
                Flag to decide biased or unbiased normalization for 3rd order cumulant function. 
            lags: numpy.ndarray
                An array of time lags for which 3rd order cumulant is calculated
            freq: numpy.ndarray
                An array of freq values for bispectrum.
            cum3: numpy.ndarray
                A maxlag*2+1 x maxlag*2+1 matrix containing 3rd order cumulant data for different lags.
            bispec: numpy.ndarray
                A maxlag*2+1 x maxlag*2+1 matrix containing bispectrum data for different frequencies.
            bispec_mag: numpy.ndarray
                Magnitude of Bispectrum
            bispec: numpy.ndarray
                Phase of Bispectrum
            References
            ----------     
            [1] The biphase explained: understanding the asymmetries invcoupled Fourier components of astronomical timeseries
            by Thomas J. Maccarone Department of Physics, Box 41051, Science Building, Texas Tech University, Lubbock TX 79409-1051
            School of Physics and Astronomy, University of Southampton, SO16 4ES
                
            [2] T. S. Rao, M. M. Gabr, An Introduction to Bispectral Analysis and Bilinear Time
            Series Models, Lecture Notes in Statistics, Volume 24, D. Brillinger, S. Fienberg,
            J. Gani, J. Hartigan, K. Krickeberg, Editors, Springer-Verlag, New York, NY, 1984.
            
            [3] Matlab version of bispectrum under following link. 
            https://www.mathworks.com/matlabcentral/fileexchange/60-bisp3cum
                                 
        """

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