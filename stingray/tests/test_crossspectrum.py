from __future__ import division
import numpy as np
from nose.tools import raises
from stingray import Lightcurve
from stingray import Crossspectrum, AveragedCrossspectrum

np.random.seed(20160528)

class TestCrossspectrum(object):

    def setup_class(self):
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
    def test_init_with_one_lc_none(self):
        cs = Crossspectrum(self.lc1)

    @raises(AssertionError)
    def test_init_with_norm_not_str(self):
        cs = Crossspectrum(norm=1)

    @raises(AssertionError)
    def test_init_with_invalid_norm(self):
        cs = Crossspectrum(norm='frabs')

    @raises(AssertionError)
    def test_init_with_wrong_lc1_instance(self):
        lc_ = Crossspectrum()
        cs = Crossspectrum(lc_, self.lc2)

    @raises(AssertionError)
    def test_init_with_wrong_lc2_instance(self):
        lc_ = Crossspectrum()
        cs = Crossspectrum(self.lc1, lc_)

    @raises(AssertionError)
    def test_make_crossspectrum_diff_lc_counts_shape(self):
        counts = np.array([1]*10001)
        time = np.linspace(0.0, 1.0001, 10001)
        lc_ = Lightcurve(time, counts)
        cs = Crossspectrum(self.lc1, lc_)

    @raises(AssertionError)
    def test_make_crossspectrum_diff_dt(self):
        counts = np.array([1]*10000)
        time = np.linspace(0.0, 2.0, 10000)
        lc_ = Lightcurve(time, counts)
        cs = Crossspectrum(self.lc1, lc_)

    @raises(AssertionError)
    def test_rebin_smaller_resolution(self):
        # Original df is between 0.9 and 1.0
        new_cs = self.cs.rebin(df=0.1)

    def test_rebin(self):
        new_cs = self.cs.rebin(df=1.5)
        assert new_cs.df == 1.5
