
import numpy as np

class Simulator(object):

    """
    Methods to simulate and visualize light curves.
    """

    def simulate(self, *args):
        """
        Simulate light curve generation given power spectrum or
        impulse response.

        Examples:
        --------
        - x = simulate(2)
            For generating a light curve given a power spectrum.

            Parameters:
            -----------
            Beta: int
                Defines the shape of spectrum.

            Returns:
            --------
            lightCurve: `LightCurve` object

        - x = simulate(s,h)
            For generating a light curve given the impulse response.

            Parameters:
            -----------
            s: array-like
                Underlying variability signal
            h: array-like
                Impulse response

            Returns:
            -------
            lightCurve: `LightCurve` object
        """

        if (len(args) == 1):
            return  self._simulate_power_law(args[0])

        elif (len(args) == 2):
            return self._simulate_impulse_response(args[0], args[1])

        else:
            raise AssertionError("Length of arguments must be 1 or 2.")


    def _simulate_power_law(self, B):

        """
        Generate LightCurve given a power spectrum.

        Parameters:
        ----------
        B: int
            Defines the shape of spectrum.

        Returns
        -------
        lightCurve: array-like
        """

        N = 1024

        # Define frequencies from 0 to 2*pi
        w = np.linspace(0.001,2*np.pi,N)

        # Draw two set of 'N' guassian distributed numbers
        a1 = np.random.normal(size=N)
        a2 = np.random.normal(size=N)

        # Multiply by (1/w)^B to get real and imaginary parts
        real = a1 * np.power((1/w),B/2)
        imaginary = a2 * np.power((1/w),B/2)

        # Form complex numbers corresponding to each frequency
        f = [complex(r, i) for r,i in zip(real,imaginary)]

        # Obtain real valued time series
        f_conj = np.conjugate(np.array(f))

        # Obtain time series
        return np.fft.ifft(f_conj)

    def _simulate_impulse_response(self):
        pass