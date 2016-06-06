import numpy as np

from stingray.simulator import simulator
class TestSimulator(object):

    @classmethod
    def setup_class(self):
        self.simulator = simulator.Simulator()

    def test_simulate_create(self):
        """
        Simulate an event list from fake times and counts.
        """
        self.simulator.simulate(2,3)
