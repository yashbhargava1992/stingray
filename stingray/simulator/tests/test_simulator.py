import numpy as np

from astropy.tests.helper import pytest
from stingray.simulator import simulator
from matplotlib import pyplot as plt
from scipy import signal

class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.simulator = simulator.Simulator(N=1024)
        self.w = np.fft.rfftfreq(self.simulator.N, d=self.simulator.dt)[1:]
        self.B = 2

    def test_simulate_powerlaw(self):
        """
        Simulate light curve from power law spectrum.
        """
        assert len(self.simulator.simulate(2).counts), 1024
        
    def test_compare_powerlaw(self):
        """
        Compare simulated power spectrum with actual one.
        """
        actual = np.power((1/self.w),self.B/2)
        a_mean = np.mean(actual)
        
        lc = [self.simulator.simulate(self.B) for i in xrange(1,30)] 
        simulated = self.simulator.periodogram(lc, lc[0].tseg)
        
        # plt.figure()
        # plt.plot(simulated, label='Simulated')
        # plt.plot(actual, label='Actual')
        # plt.title('Comparison of Actual and Simulated Power \n Spectrums [Power Law with B=2]')
        # plt.legend()
        # plt.show()

    def test_simulate_powerspectrum(self):
        """
        Simulate light curve from any power spectrum.
        """
        s = np.random.rand(1024)
        assert len(self.simulator.simulate(s)), 1024

    def test_simulate_lorenzian(self):
        """
        Simulate light curve using lorenzian model.
        """
        assert len(self.simulator.simulate('lorenzian',[1,2,3,4])), 1024

    def test_simulate_smoothbknpo(self):
        """
        Simulate light curve using smooth broken power law model.
        """
        assert len(self.simulator.simulate('smoothbknpo',[1,2,3,4])), 1024

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
        self.simulator.periodogram(self.simulator.lc)

    def test_periodogram_without_lc(self):
        """
        Create a periodogram without light curve.
        """
        self.simulator.periodogram(self.simulator.lc)   
            

