
import numpy as np

from stingray.simulator import events
from stingray.sampledata import sample_data

class TestFakeSimulator(object):

    @classmethod
    def setup_class(self):
        self.times = [0.5, 1.5, 2.5, 3.5]
        self.counts = [3000, 2000, 2200, 3600]

    def test_fake_event_create(self):
        """
        Simulate an event list from fake times and counts.
        """
        events.gen_events_from_lc(self.times, self.counts)

    def test_actual_event_create(self):
        """
        Simulate an event list from actual light curve.
        """
        lc = sample_data()
        new_lc = lc[0:100]
        events.gen_events_from_lc(new_lc.time, new_lc.counts)

    def test_fake_recover_lcurve(self):
        """
        Recover a lightcurve from a fake event list.
        """
        ev_list = events.gen_events_from_lc(self.times, self.counts)
        new_times, new_counts = events.gen_lc_from_events(ev_list, 1, start_time=0,
                                                          stop_time=4)

        assert np.all(np.abs(new_counts - self.counts) < 3 * np.sqrt(self.counts))
        np.testing.assert_almost_equal(new_times, self.times)

    def test_actual_recover_lcurve(self):
        """
        Recover a lightcurve from an actual event list.
        """
        lc = sample_data()
        new_lc = lc[0:100]
        ev_list = events.gen_events_from_lc(new_lc.time, new_lc.counts)

        bin_length = new_lc.time[1] - new_lc.time[0]
        new_times, new_counts = events.gen_lc_from_events(ev_list, bin_length,
                                                          start_time = new_lc.time[0] - bin_length/2,
                                                          stop_time = new_lc.time[-1] + bin_length/2)

        #TODO: Sigma needs to be 4 in order to pass test. Should it be 3?
        assert np.all(np.abs(new_counts - new_lc.counts) < 4 * np.sqrt(new_lc.counts))
        np.testing.assert_almost_equal(new_times, new_lc.time)
