
import numpy as np
from stingray import Lightcurve
from stingray import Periodogram
np.random.seed(20150907)

class TestPeriodgram(object):

    def setUp(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.1

        time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 100.0
        mean_counts = mean_count_rate*dt

        poisson_counts = np.random.poisson(mean_counts,
                                           size=time.shape[0])

        self.time = time
        self.counts = poisson_counts
        self.lc = Lightcurve(time, counts=poisson_counts)


    def test_make_empty_periodogram(self):
        ps = Periodogram()
        assert ps.norm == "rms"
        assert ps.freq is None
        assert ps.ps is None
        assert ps.df is None
        assert ps.m == 1
        assert ps.n is None

    def test_make_periodgram_from_arrays(self):
        ps = Periodogram(time=self.time, counts=self.counts)
        assert ps.freq is not None
        assert ps.ps is not None
        assert ps.df == 1.0/self.lc.tseg
        assert ps.norm == "rms"
        assert ps.m == 1
        assert ps.n == self.lc.time.shape[0]
        assert ps.nphots == np.sum(self.lc.counts)

    def test_make_periodogram_from_lightcurve(self):
        ps = Periodogram(lc=self.lc)
        assert ps.freq is not None
        assert ps.ps is not None
        assert ps.df == 1.0/self.lc.tseg
        assert ps.norm == "rms"
        assert ps.m == 1
        assert ps.n == self.lc.time.shape[0]
        assert ps.nphots == np.sum(self.lc.counts)

    def test_periodogram_types(self):
        ps = Periodogram(lc=self.lc)
        assert isinstance(ps.freq, np.ndarray)
        assert isinstance(ps.ps, np.ndarray)

