import numpy as np

from astropy.tests.helper import pytest
from stingray.simulator import simulator

class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.simulator = simulator.Simulator(N=1024)

    def test_simulate_powerlaw(self):
        """
        Simulate light curve from power law spectrum.
        """
        assert len(self.simulator.simulate(2).counts), 1024

    def test_simulate_powerspectrum(self):
        """
        Simulate light curve from any power spectrum.
        """
        s = np.random.rand(1024)
        assert len(self.simulator.simulate(s)), 1024

    def test_simulate_impulse(self):
        """
        Simulate light curve from impulse response.
        """
        self.simulator.simulate([],[])

    def test_periodogram_with_lc(self):
        """
        Create a periodogram from light curve.
        """
        self.simulator.simulate(2)
        self.simulator.periodogram()

    def test_periodogram_without_lc(self):
        """
        Create a periodogram without light curve.
        """
        self.simulator.periodogram()   
            

