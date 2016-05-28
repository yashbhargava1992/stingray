from __future__ import division
import numpy as np
from nose.tools import raises
from stingray import Lightcurve
from stingray import Crossspectrum, AveragedCrossspectrum

np.random.seed(20160528)

class TestCrossspectrum(object):

    def setUp(self):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.linspace(tstart, tend, int((tend - tstart)/dt))

        counts1 = np.random.poisson(0.01, size=time.shape[0])
        counts2 = np.random.negative_binomial(1, 0.09, size=time.shape[0])

        self.lc1 = Lightcurve(time, counts1)
        self.lc2 = Lightcurve(time, counts2)

        self.cs = Crossspectrum(self.lc1, self.lc2)

    def test_make_empty_crossspectrum(self):
        cs = Crossspectrum()
        assert cs.freq is None
        assert cs.cs is None
        assert cs.df is None
        assert cs.nphots1 is None
        assert cs.nphots2 is None
        assert cs.m == 1
        assert cs.n is None

    @raises(TypeError)
    def test_make_crossspectrum_with_one_lc_none(self):
        cs = Crossspectrum(self.lc1)
