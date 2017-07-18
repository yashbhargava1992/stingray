from __future__ import division, print_function
from stingray.pulse.search import epoch_folding_search, z_n_search, _profile_fast
import numpy as np
from stingray import Lightcurve
from stingray.events import EventList

np.random.seed(20150907)

class TestAll(object):
    """Unit tests for the stingray.pulse.search module."""
    @classmethod
    def setup_class(cls):
        cls.pulse_frequency = 1/0.101
        cls.tstart = 0
        cls.tend = 25.25
        cls.tseg = cls.tend - cls.tstart
        cls.dt = 0.0202
        cls.times = np.arange(cls.tstart, cls.tend, cls.dt) + cls.dt / 2
        cls.counts = \
            100 + 20 * np.cos(2 * np.pi * cls.times * cls.pulse_frequency)
        lc = Lightcurve(cls.times, cls.counts, gti=[[cls.tstart, cls.tend]])
        events = EventList()
        events.simulate_times(lc)
        cls.event_times = events.time

    def test_prepare(self):
        pass

    def test_profile_fast(self):
        test_phase = np.arange(0, 1, 1/16)
        prof = _profile_fast(test_phase, nbin=16)
        assert np.all(prof == np.ones(16))

    def test_epoch_folding_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.85, 9.95, 0.1/self.tseg)
        freq, stat = epoch_folding_search(self.event_times, frequencies,
                                          nbin=16)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

    def test_epoch_folding_search_expocorr(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.89, 9.91, 0.1/self.tseg)
        freq, stat = epoch_folding_search(self.event_times, frequencies,
                                          nbin=16, expocorr=True)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

    def test_z_n_search(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.85, 9.95, 0.3/self.tseg)
        freq, stat = z_n_search(self.event_times, frequencies, nbin=16, n=1)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

    def test_z_n_search_expocorr(self):
        """Test pulse phase calculation, frequency only."""
        frequencies = np.arange(9.89, 9.91, 0.1/self.tseg)
        freq, stat = z_n_search(self.event_times, frequencies, nbin=16, n=1,
                                expocorr=True)

        minbin = np.argmin(np.abs(frequencies - self.pulse_frequency))
        maxstatbin = freq[np.argmax(stat)]
        assert maxstatbin == frequencies[minbin]

