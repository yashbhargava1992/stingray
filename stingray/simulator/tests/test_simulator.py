import pytest
from stingray.simulator import simulator
class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.simulator = simulator.Simulator(N=1024)

    def test_simulate_create(self):
        """
        Simulate from power law spectrum.
        """
        assert len(self.simulator.simulate(2)), 1024

