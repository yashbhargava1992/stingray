
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
            self._simulate_power_law()

        elif (len(args) == 2):
            self._simulate_impulse_response()

        else:
            raise AssertionError("Length of arguments must be 1 or 2.")

    def _simulate_power_law(self):
        pass

    def _simulate_impulse_response(self):
        pass