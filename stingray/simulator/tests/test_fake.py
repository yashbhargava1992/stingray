from stingray.simulator import fake
from stingray.sampledata import sample_data

class TestFakeSimulator(object):

    @classmethod
    def setup_class(self):
        self.times = [1, 2, 3, 4]
        self.counts = [2, 2, 2, 2]

    def test_simple_create(self):
        """
        Simulate an event list from simple times and counts.
        """
        fake.fake_events_from_lc(self.times, self.counts)

    def test_actual_create(self):
        """
        Simulate an event list from actual light curve.
        """
        lc = sample_data()
        fake.fake_events_from_lc(lc.time, lc.counts)
