import numpy as np
import scipy
import scipy.stats
import scipy.fftpack
import scipy.optimize

from . import lightcurve
from . import utils

class Periodogram(object):

    def __init__(self, lc = None, time = None, counts = None, norm='rms'):
        """
        Make a Periodogram from a (binned) light curve. Periodograms can be
        Leahy normalized or fractional rms normalized.
        You can also make an empty Periodogram object to populate with your
        own fourier-transformed data (this can sometimes be useful when making
        binned periodograms).

        Parameters:
        -----------
        lc: lightcurve.Lightcurve object, optional, default None
            The light curve data to be Fourier-transformed.

        time: iterable, optional, default None
            If lc is None, then this argument should contain a list of time
            stamps of light curve bins

        counts: iterable, optional, default None
            If lc is None, this argument should contain a list of **counts per
            bin** corresponding to the timestamps in *time*.

        norm: {"leahy" | "rms"}, optional, default "rms"
            The normaliation of the periodogram to be used. Options are
            "leahy" or "rms", default is "rms".


        Attributes
        -----------
        norm: {"leahy" | "rms"}
            the normalization of the periodogram

        freq: numpy.ndarray
            The array of mid-bin frequencies that the Fourier transform samples

        ps: numpy.ndarray
            The array of normalized squared absolute values of Fourier
            amplitudes

        df: float
            The frequency resolution

        m: int
            The number of averaged periodograms

        n: int
            The number of data points in the light curve

        nphots: float
            The total number of photons in the light curve


        """
        self.norm = norm

        ## check if input data is a Lightcurve object, if not make one or
        ## make an empty Periodogram object if lc == time == counts == None
        if lc is None and time is not None and counts is not None:
            lc = lightcurve.Lightcurve(time, counts=counts)
        elif lc is not None:
            pass
        else:
            self.freq = None
            self.ps = None
            self.df = None
            self.nphots = None
            return

        ## total number of photons is the sum of the
        ## counts in the light curve
        self.nphots = np.sum(lc.counts)

        ## the number of data points in the light curve
        self.n = lc.counts.shape[0]

        ## the frequency resolution
        self.df = 1.0/lc.tseg

        ## the number of averaged periodograms in the final output
        ## This should *always* be 1 here
        self.m = 1

        ## make the actual Fourier transform
        fr = self._fourier_transform(lc)

        ## normalize to either Leahy or rms normalization
        self.ps = self._normalize_periodogram(fr, lc)

        ## make a list of frequencies to go with the powers
        self.freq = np.arange(self.ps.shape[0])*self.df + self.df/2.


    def _fourier_transform(self, lc):
        """
        Fourier transform the light curve, then square the
        absolute value of the Fourier amplitudes.

        Parameters:
        -----------
        lc: lightcurve.Lightcurve object
            The light curve to be Fourier transformed

        Returns:
        --------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        """
        fourier= scipy.fftpack.fft(lc.counts) ### do Fourier transform
        fr = np.abs(fourier)**2.
        return fr

    def _normalize_periodogram(self, fr, lc):
        """
        Normalize the periodogram to either Leahy or RMS normalization.
        In Leahy normalization, the periodogram is normalized in such a way
        that a flat light curve of Poissonian data will make a realization of
        the power spectrum in which the powers are distributed as Chi^2 with
        two degrees of freedom (with a mean of 2 and a variance of 4).

        In rms normalization, the periodogram will be normalized such that
        the integral of the periodogram will equal the total variance in the
        light curve divided by the mean of the light curve squared.

        Parameters
        ----------
        fr: numpy.ndarray
            The squared absolute value of the Fourier amplitudes

        lc: lightcurve.Lightcurve object
            The input light curve


        Returns
        ----------
        ps: numpy.nd.array
            The normalized periodogram
        """
        if self.norm.lower() == 'leahy':
            p = fr[:self.n/2+1]
            ps =  2.*p/self.nphots

        elif self.norm.lower() == 'rms':
            p = fr[:self.n/2+1]/np.float(self.n**2.)
            ps = p*2.*lc.tseg/(np.mean(lc.counts)**2.0)

        return ps

    def rebin(self, df):
        """
        Rebin the periodogram to a new frequency resolution df.
        
        Parameters:
        ------------
        df: float
            The new frequency resolution
            
        Returns:
        ---------
        bin_ps = Periodogram object
            The newly binned periodogram
        
        """
        ### frequency range of power spectrum
        flen = (self.freq[-1] - self.freq[0])
        ### calculate number of new bins in rebinned spectrum
        bins = np.floor(flen/df)
        ### rebin power spectrum to new resolution
        binfreq, binps = utils.rebin_data(self.freq, self.ps, df, method='mean')
        bin_ps = Periodogram()

        """
               norm: {"leahy" | "rms"}
                    the normalization of the periodogram

                freq: numpy.ndarray
                    The array of mid-bin frequencies that the Fourier transform samples

                ps: numpy.ndarray
                    The array of normalized squared absolute values of Fourier
                    amplitudes

                df: float
                    The frequency resolution

                m: int
                    The number of averaged periodograms

                n: int
                    The number of data points in the light curve

                nphots: float
                    The total number of photons in the light curve

        """
        bin_ps.norm = self.norm
        bin_ps.freq = binfreq
        bin_ps.ps = binps
        bin_ps.df = df
        bin_ps.n = self.n
        bin_ps.nphots = self.nphots

        return bin_ps


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
        --------
        binfreq: numpy.ndarray
            the binned frequencies

        binps: numpy.ndarray
            the binned powers

        nsamples: numpy.ndarray
            the samples of the original periodogramincluded in each
            frequency bin
        """

        minfreq = self.freq[1]*0.5 ## frequency to start from
        maxfreq = self.freq[-1] ## maximum frequency to end
        binfreq = [minfreq, minfreq + self.df] ## first
        df = self.freq[1] ## the frequency resolution of the first bin

        ## until we reach the maximum frequency, increase the width of each
        ## frequency bin by f
        while binfreq[-1] <= maxfreq:
            binfreq.append(binfreq[-1] + df*(1.+f))
            df = binfreq[-1]-binfreq[-2]

        ## compute the mean of the powers that fall into each new frequency
        ## bin
        binps, bin_edges, binno = scipy.stats.binned_statistic(self.freq,
                                                               self.ps,
                                                               statistic="mean",
                                                               bins=binfreq)

        ## compute the number of powers in each frequency bin
        nsamples = np.array([len(binno[np.where(binno == i)[0]]) \
                             for i in xrange(np.max(binno))])

        ## the frequency resolution
        df = np.diff(binfreq)

        ## shift the lower bin edges to the middle of the bin and drop the
        ## last right bin edge
        binfreq = binfreq[:-1]+df/2.
        
        return binfreq, binps, nsamples


class AveragedPeriodogram(object):

    def __init__(self, lc=None, time=None, counts=None, segment_size=10.0,
                norm="leahy"):
        """
        Make an averaged periodogram from a light curve by segmenting the light
        curve, Fourier-transforming each segment and then averaging the
        resulting periodograms.

        """
        return



