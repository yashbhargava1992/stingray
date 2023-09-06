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
        sim = Simulator(0.0001, 100, 100, 1, random_state=42, tstart=0)
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
        lscs.freq is None
        lscs.power is None

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

    def test_bad_input(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(1, self.lc1)

    def test_init_with_one_lc_none(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, None)

    def test_init_with_norm_not_str(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, norm=1)

    def test_init_with_invalid_norm(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, norm="frabs")

    def test_init_with_power_type_not_str(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, power_type=3)

    def test_init_with_invalid_power_type(self):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, power_type="reel")

    def test_init_with_wrong_lc_instance(self):
        lc1_ = {"a": 1, "b": 2}
        lc2_ = {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(lc1_, lc2_, dt=1)

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
        lc_ = Simulator(0.0001, 103, 100, 1, random_state=42, tstart=0).simulate(0)
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, lc_)

    def test_make_crossspectrum_diff_lc_stat(self):
        lc_ = copy.deepcopy(self.lc1)
        lc_.err_dist = "gauss"
        with pytest.warns(UserWarning) as record:
            cs = LombScargleCrossspectrum(self.lc1, lc_)
        assert np.any(["different statistics" in r.message.args[0] for r in record])

    @pytest.mark.parametrize("power_type", ["real", "absolute", "all"])
    def test_power_type(self, power_type):
        lscs = LombScargleCrossspectrum(self.lc1, self.lc2, power_type=power_type)
        assert lscs.power_type == power_type

    @pytest.mark.parametrize("method", ["fft", "randommethod"])
    def test_init_with_invalid_method(self, method):
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, method=method)

    def test_with_invalid_fullspec(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, fullspec=1)

    def test_with_invalid_oversampling(self):
        with pytest.raises(TypeError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2, oversampling="invalid")

    def test_invalid_mixed_data(self):
        data2 = EventList(self.lc2.time[3:], np.ones_like(self.lc2.time[3:]))
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(self.lc1, data2)
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(data2, self.lc1)

    def test_valid_mixed_data(self):
        data2 = EventList(self.lc2.time, np.ones_like(self.lc2.time))
        lscs = LombScargleCrossspectrum(self.lc1, data2)
        assert lscs.power is not None
        lscs2 = LombScargleCrossspectrum(data2, self.lc1)
        assert lscs2.power is not None

    def test_fullspec(self):
        lscs = LombScargleCrossspectrum(self.lc1, self.lc2, fullspec=True)
        assert lscs.fullspec

    @pytest.mark.parametrize("method", ["slow", "fast"])
    def test_valid_method(self, method):
        lscs = LombScargleCrossspectrum(self.lc1, self.lc2, method=method)
        assert lscs.method == method

    @pytest.mark.parametrize(
        "func_name",
        [
            "classical_significances",
            "from_time_array",
            "from_events",
            "from_lightcurve",
            "from_lc_iterable",
            "_initialize_from_any_input ",
        ],
    )
    def test_raise_on_invalid_function(self, func_name):
        with pytest.raises(AttributeError):
            lscs = LombScargleCrossspectrum(self.lc1, self.lc2)
            func = getattr(lscs, func_name)
            func()

    def test_no_dt(self):
        el1 = EventList(self.lc1.counts, self.lc1.time, dt=None)
        el2 = EventList(self.lc2.counts, self.lc2.time, dt=None)
        with pytest.raises(ValueError):
            lscs = LombScargleCrossspectrum(el1, el2)


class TestLombScarglePowerspectrum:
    def setup_class(self):
        sim = Simulator(0.0001, 100, 100, 1, random_state=42, tstart=0)
        lc = sim.simulate(0)
        self.rate = lc.countrate
        low, high = lc.time.min(), lc.time.max()
        s1 = lc.counts
        t = lc.time
        t_new = t.copy()
        t_new[1:-1] = t[1:-1] + (np.random.rand(len(t) - 2) / (high - low))
        s_new = interp1d(t, s1, fill_value="extrapolate")(t_new)
        self.lc = Lightcurve(t, s_new, dt=lc.dt)

    @pytest.mark.parametrize("norm", ["leahy", "frac", "abs", "none"])
    def test_normalize_powerspectrum(self, norm):
        lps = LombScarglePowerspectrum(self.lc, norm=norm)
        assert lps.norm == norm

    @pytest.mark.parametrize("skip_checks", [True, False])
    def test_init_empty(self, skip_checks):
        ps = LombScarglePowerspectrum(skip_checks=skip_checks)
        assert ps.freq is None
        assert ps.power is None
        assert ps.power_err is None
        assert ps.df is None
        assert ps.m == 1

    def test_make_empty_powerspectrum(self):
        ps = LombScarglePowerspectrum()
        assert ps.freq is None
        assert ps.power is None
        assert ps.power_err is None
        assert ps.df is None
        assert ps.m == 1
        assert ps.nphots1 is None
        assert ps.nphots2 is None
        assert ps.method is None

    def test_ps_real(self):
        ps = LombScarglePowerspectrum(self.lc)
        assert np.allclose(ps.power.imag, [0])
