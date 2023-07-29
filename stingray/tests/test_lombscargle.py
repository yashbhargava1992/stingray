import numpy as np
import pytest
import copy
from stingray.lombscargle import LombScargleCrossspectrum, LombScarglePowerspectrum
from stingray.lightcurve import Lightcurve
from stingray.events import EventList
from stingray.simulator import Simulator
from stingray.exceptions import StingrayError
from scipy.interpolate import interp1d


class TestLombScargleCrossspectrum:
    def setup_class(self):
        sim = Simulator(0.0001, 10000, 100, 1, random_state=42, tstart=0)
        lc1 = sim.simulate(0)
        lc2 = sim.simulate(0)
        self.rate1 = lc1.countrate
        self.rate2 = lc2.countrate
        low, high = lc1.time.min(), lc1.time.max()
        s1 = lc1.counts
        s2 = lc2.counts
        t = lc1.time
        t_new = t.copy()
        t_new[1:-1] = t[1:-1] + (np.random.rand(len(t) - 2) / (high - low))
        s1_new = interp1d(t, s1, fill_value="extrapolate")(t_new)
        s2_new = interp1d(t, s2, fill_value="extrapolate")(t_new)
        self.lc1 = Lightcurve(t, s1_new, dt=lc1.dt)
        self.lc2 = Lightcurve(t, s2_new, dt=lc2.dt)
        with pytest.warns(UserWarning) as record:
            self.lscs = LombScargleCrossspectrum(lc1, lc2)

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_initialize_empty(self, skip_checks):
        lscs = LombScargleCrossspectrum(skip_checks=skip_checks)
        assert lscs.freq is None
        assert lscs.power is None

    def test_make_empty_crossspectrum(self):
        lscs = LombScargleCrossspectrum()
        assert lscs.freq is None
        assert lscs.power is None
        assert lscs.df is None
        assert lscs.nphots1 is None
        assert lscs.nphots2 is None
        assert lscs.m == 1
        assert lscs.n is None
        assert lscs.power_err is None
        assert lscs.dt is None
        assert lscs.err_dist is None
        assert lscs.variance1 is None
        assert lscs.variance2 is None
        assert lscs.meancounts1 is None
        assert lscs.meancounts2 is None
        assert lscs.oversampling is None
        assert lscs.method is None

    def test_init_with_one_lc_none(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(norm="frabs")

    def test_init_with_power_type_not_str(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(power_type=3)

    def test_init_with_invalid_power_type(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(power_type="reel")

    def test_init_with_wrong_lc_instance(self):
        lc1_ = {"a": 1, "b": 2}
        lc2_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(lc1_, lc2_)

    def test_init_with_wrong_lc2_instance(self):
        lc_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(self.lc1, lc_)

    def test_init_with_wrong_lc1_instance(self):
        lc_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(lc_, self.lc2)

    def test_init_with_invalid_min_freq(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, min_freq=-1)

    def test_init_with_invalid_max_freq(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, max_freq=1, min_freq=3)

    def test_make_crossspectrum_diff_lc_counts_shape(self):
        lc_ = Simulator(0.0001, 10423, 100, 1, random_state=42, tstart=0).simulate(0)
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_stat(self):
        lc_ = copy.deepcopy(self.lc1)
        lc_.err_dist = "gauss"
        with pytest.warns(UserWarning) as record:
            cs = LombScargleCrossspectrum(self.lc1, lc_)
        assert np.any(["different statistics" in r.message.args[0] for r in record])

    def test_make_crossspectrum_diff_dt(self):
        lc_ = Simulator(0.0002, 10000, 100, 1, random_state=42, tstart=0).simulate(0)
        with pytest.raises(
            StingrayError, match="Lightcurves do not have the same time binning dt."
        ):
            lscs = LombScargleCrossspectrum(self.lc1, lc_)
