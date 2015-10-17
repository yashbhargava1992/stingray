import numpy as np
import scipy
import scipy.optimize

from . import lightcurve
from . import utils

class Periodogram(object):

    def __init__(self, lc = None, time = None, counts = None, nphot=None, norm='leahy', m=1):
        """
        Make a Periodogram from a (binned) light curve. Periodograms can be Leahy normalized,
        fractional rms normalized or variance normalized. You can also make an empty
        Periodogram object to populate with your own fourier-transformed data (this can
        sometimes be useful when making binned periodograms).

        """
        self.norm = norm

        if lc is None and time and counts:
            lc = lightcurve.Lightcurve(lc, counts)

        elif lc is None and time is None and counts is None:
            self.freq = None
            self.ps = None
            self.df = None
            return

        if nphot is None:
            self.nphots = np.sum(lc.counts)
        else:
            self.nphots = nphot

        self._fourier_transform(lc, m)

    def _fourier_transform(self, lc, m)   :
        nel = np.round(lc.tseg/lc.res)

        df = 1.0/lc.tseg

        fourier= scipy.fft(lc.counts) ### do Fourier transform
        ff = fourier.conjugate()*fourier   ### multiply both together
        fr = np.abs(ff)#np.array([x.real for x in ff]) ### get out the real part of ff
        self.ps = 2.0*fr[0: int(nel/2)]/self.nphots



        self.df = df
        self.freq = np.arange(self.ps.shape[0])*self.df + self.df/2.
        self.nphots = self.nphots
        self.n = lc.counts.shape[0]
        self.m = m

    def _normalize_periodogram(self, ps, lc, df):
        """
        TO DO: Actually move code from normalization here! This doesn't make sense yet!

        """

        if self.norm.lower() in ['leahy']:
            pass

        elif self.norm.lower() in ['rms']:
            self.ps /= (df*self.nphots)

        elif self.norm.lower() in ['variance', 'var']:
            self.ps *= (self.nphots/lc.counts.shape[0]**2.0)

        return


    def rebinps(self, df, verbose=False):
        ### frequency range of power spectrum
        flen = (self.freq[-1] - self.freq[0])
        ### calculate number of new bins in rebinned spectrum
        bins = np.floor(flen/df)
        ### rebin power spectrum to new resolution
        binfreq, binps = utils.rebin_data(self.freq, self.ps, df, method='mean')
        newps = Periodogram()
        newps.freq = binfreq
        newps.ps = binps
        newps.df = df
        newps.nphots = binps[0]
        newps.n = 2*len(binps)
        return newps


    def rebin_log(self, f=0.01):
        """
        Logarithmic rebin of the periodogram.
        The new frequency depends on the previous frequency
        modified by a factor f:

        dnu_j = dnu_{j-1}*(1+f)

        Parameters:
        -----------
        f: float, optional, default 0.01
            parameter that steers the frequency resolution


        Returns:
            binfreq: numpy.ndarray
                the binned frequencies
            binps: numpy.ndarray
                the binned powers
        """

        minfreq = self.freq[1]*0.5
        maxfreq = self.freq[-1]
        binfreq = [minfreq]
        df = self.freq[1]
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df*(1.+f))
            df = binfreq[-1]-binfreq[-2]
        binps, bin_edges, binno = scipy.stats.binned_statistic(self.freq, self.ps, statistic="mean", bins=binfreq)

        nsamples = np.array([len(binno[np.where(binno == i)[0]]) for i in xrange(np.max(binno))])
        df = np.diff(binfreq)
        binfreq = binfreq[:-1]+df/2.
        return binfreq, binps, nsamples


class AveragedPeriodogram(object):

    def __init__(self, lc=None, time=None, counts=None, segment_size=10.0, overlap=0.1, norm="leahy"):
        """
        Make an averaged periodogram from a light curve by segmenting the light curve, Fourier-
        transforming each segment and then averaging the resulting periodograms.

        """
        return



