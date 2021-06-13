import numpy as np
import copy
import warnings

from astropy.tests.helper import pytest
from numpy.random import poisson, standard_cauchy
from scipy.signal.ltisys import TransferFunction

from stingray import Lightcurve
from stingray.events import EventList
from stingray import Multitaper, Powerspectrum

np.random.seed(1)


class TestMultitaper(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 1.0
        dt = 0.0001

        time = np.arange(tstart + 0.5*dt, tend + 0.5*dt, dt)

        mean_count_rate = 100.0
        mean_counts = mean_count_rate * dt

        poisson_counts = np.random.poisson(mean_counts, size=time.shape[0])

        cls.lc = Lightcurve(time, counts=poisson_counts, dt=dt,
                            gti=[[tstart, tend]])

        mean = 0
        standard_deviation = 0.1
        gauss_counts = \
            np.random.normal(mean, standard_deviation, size=time.shape[0])

        cls.lc_gauss = Lightcurve(time, counts=gauss_counts, dt=dt,
                                  gti=[[tstart, tend]], err_dist='gauss')

    def test_lc_keyword_deprecation(self):
        mtp1 = Multitaper(self.lc)
        with pytest.warns(DeprecationWarning) as record:
            mtp2 = Multitaper(lc=self.lc)
        assert np.any(['lc keyword' in r.message.args[0]
                       for r in record])
        assert np.allclose(mtp1.power, mtp2.power)
        assert np.allclose(mtp1.freq, mtp2.freq)

    def test_make_empty_multitaper(self):
        mtp = Multitaper()
        assert mtp.norm == 'frac'
        assert mtp.freq is None
        assert mtp.power is None
        assert mtp.multitaper_norm_power is None
        assert mtp.eigvals is None
        assert mtp.power_err is None
        assert mtp.df is None
        assert mtp.m == 1
        assert mtp.nphots is None
        assert mtp.jk_var_deg_freedom is None

    @pytest.mark.parametrize("lightcurve", ['lc', 'lc_gauss'])
    def test_make_multitaper_from_lightcurve(self, lightcurve):
        mtp = Multitaper(getattr(self, lightcurve))
        assert mtp.norm == "frac"
        assert mtp.fullspec is False
        assert mtp.meancounts == getattr(self, lightcurve).meancounts
        assert mtp.nphots == np.float64(np.sum(getattr(self, lightcurve).counts))
        assert mtp.err_dist == getattr(self, lightcurve).err_dist
        assert mtp.dt == getattr(self, lightcurve).dt
        assert mtp.n == getattr(self, lightcurve).time.shape[0]
        assert mtp.df == 1.0 / getattr(self, lightcurve).tseg
        assert mtp.m == 1
        assert mtp.freq is not None
        assert mtp.multitaper_norm_power is not None
        assert mtp.power is not None
        assert mtp.power_err is not None
        assert mtp.jk_var_deg_freedom is not None

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            mpt = Multitaper(norm=1)

    def test_init_with_lightcurve(self):
        assert Multitaper(self.lc)

    def test_init_without_lightcurve(self):
        with pytest.raises(TypeError):
            assert Multitaper(self.lc.counts)

    def test_init_with_nonsense_data(self):
        nonsense_data = [None for i in range(100)]
        with pytest.raises(TypeError):
            assert Multitaper(nonsense_data)

    def test_init_with_nonsense_norm(self):
        nonsense_norm = "bla"
        with pytest.raises(ValueError):
            assert Multitaper(self.lc, norm=nonsense_norm)

    def test_init_with_wrong_norm_type(self):
        nonsense_norm = 1.0
        with pytest.raises(TypeError):
            assert Multitaper(self.lc, norm=nonsense_norm)

    @pytest.mark.parametrize('low_bias', [False, True])
    def test_make_multitaper_adaptive_and_low_bias(self, low_bias):
        mtp = Multitaper(self.lc, low_bias=low_bias, adaptive=True)

        if low_bias:
            assert np.min(mtp.eigvals) >= 0.9
        assert mtp.jk_var_deg_freedom is not None
        assert mtp.freq is not None
        assert mtp.multitaper_norm_power is not None

    @pytest.mark.parametrize('lightcurve', ['lc', 'lc_gauss'])
    def test_make_multitaper_var(self, lightcurve):

        if getattr(self, lightcurve).err_dist == "poisson":
            mtp = Multitaper(getattr(self, lightcurve))
            assert mtp.err_dist == "poisson"
            assert mtp.var == getattr(self, lightcurve).meancounts
        else:
            with pytest.warns(UserWarning) as record:
                mtp = Multitaper(getattr(self, lightcurve))
            assert mtp.err_dist == "gauss"
            assert mtp.var == \
                np.mean(getattr(self, lightcurve).counts_err) ** 2
            assert np.any(["not poisson" in r.message.args[0]
                           for r in record])

    def test_fourier_multitaper_with_invalid_NW(self):
        with pytest.raises(ValueError):
            mtp = Multitaper(self.lc, NW=0.1)

    @pytest.mark.parametrize("adaptive, jackknife",
                             [(a, j) for a in (True, False) for j in (True, False)])
    def test_fourier_multitaper_with_adaptive_jackknife_combos(self, adaptive, jackknife):
        mtp = Multitaper(self.lc, adaptive=adaptive, jackknife=jackknife)
        assert mtp.multitaper_norm_power is not None
        assert mtp.jk_var_deg_freedom is not None
