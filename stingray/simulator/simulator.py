
import numpy as np
import logging

from ..lightcurve import Lightcurve
from ..powerspectrum import Powerspectrum

class Simulator(object):

    def __init__(self, dt=1, N=1024, seed=None):
        
        """
        Methods to simulate and visualize light curves.

        Parameters
        ----------
        dt: int, default 1
            time resolution of simulated light curve
        N: int, default 1024
            bins count of simulated light curve
        seed: int, default None
            seed value for random processes
        """

        self.dt = dt
        self.N = N
        self.time = dt*np.arange(N)
        self.lc = None
        
        if seed is not None:
            np.random.seed(seed)

    def simulate(self, *args):
        
        """
        Simulate light curve generation using power spectrum or
        impulse response.

        Examples
        --------
        - x = simulate(2)
            For generating a light curve using power law spectrum.

            Parameters
            ----------
            Beta: int
                Defines the shape of spectrum
            N: int
                Number of samples

            Returns
            -------
            lightCurve: `LightCurve` object

        - x = simulate(s)
            For generating a light curve from user-provided spectrum.

            Parameters
            ----------
            s: array-like
                power spectrum

            Returns
            -------
            lightCurve: `LightCurve` object

        - x = simulate(s,h)
            For generating a light curve using impulse response.

            Parameters
            ----------
            s: array-like
                Underlying variability signal
            h: array-like
                Impulse response

            Returns
            -------
            lightCurve: `LightCurve` object
        """

        if type(args[0]) == int:
            return  self._simulate_power_law(args[0])

        elif len(args) == 1:
            return self._simulate_power_spectrum(args[0])

        elif len(args) == 2:
            return self._simulate_impulse_response(args[0], args[1])

        else:
            raise AssertionError("Length of arguments must be 1 or 2.")


    def _simulate_power_law(self, B):

        """
        Generate LightCurve from a power law spectrum.

        Parameters
        ----------
        B: int
            Defines the shape of power law spectrum.

        Returns
        -------
        lightCurve: array-like
        """

        # Define frequencies from 0 to 2*pi
        w = np.linspace(0.001,2*np.pi,self.N)

        # Draw two set of 'N' guassian distributed numbers
        a1 = np.random.normal(size=self.N)
        a2 = np.random.normal(size=self.N)

        # Multiply by (1/w)^B to get real and imaginary parts
        real = a1 * np.power((1/w),B/2)
        imaginary = a2 * np.power((1/w),B/2)

        # Obtain time series
        rate = self._find_inverse(real, imaginary)
        counts = self._scale(rate)

        self.lc = Lightcurve(self.time, counts)

        return self.lc

    def _simulate_power_spectrum(self, s):
        """
        Generate a light curve from user-provided spectrum.

        Parameters
        ----------
        s: array-like
            power spectrum

        Returns
        -------
        lightCurve: `LightCurve` object
        """
        # Cast spectrum as numpy array
        s = np.array(s)

        # Draw two set of 'N' guassian distributed numbers
        a1 = np.random.normal(size=self.N)
        a2 = np.random.normal(size=self.N)

        rate = self._find_inverse(a1*s, a2*s)
        self.lc = Lightcurve(self.time, self._scale(rate))

        return self.lc

    def _simulate_impulse_response(self, s, h):
        
        """
        Generate LightCurve from a power law spectrum.

        Parameters
        ----------
        s: array-like
                Underlying variability signal
        h: array-like
                Impulse response

        Returns
        -------
        lightCurve: array-like
        """
        pass

    def _find_inverse(self, real, imaginary):
    
        """
        Forms complex numbers corresponding to real and imaginary
        parts and finds inverse series.
        """

        # Form complex numbers corresponding to each frequency
        f = [complex(r, i) for r,i in zip(real,imaginary)]
        
        # Obtain real valued time series
        f_conj = np.conjugate(np.array(f))

        # Obtain time series
        return np.real(np.fft.ifft(f_conj))

    def _scale(self, rate):
        
        """
        Rescale light curve with zero mean and unit standard
        deviation.
        """

        avg = np.mean(rate)
        std = np.std(rate)

        return (rate-avg)/std

    def periodogram(self):
        """
        Make a periodogram of the simulated light curve.
        """
        if self.lc is None:
            logging.warn("Drawing periodogram without a simulated light curve.")
        else:
            return Powerspectrum(self.lc)