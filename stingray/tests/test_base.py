import os
import copy
import pytest
import numpy as np
from stingray.base import StingrayObject, StingrayTimeseries

_HAS_XARRAY = _HAS_PANDAS = _HAS_H5PY = True

try:
    import xarray
    from xarray import Dataset
except ImportError:
    _HAS_XARRAY = False

try:
    import pandas
    from pandas import DataFrame
except ImportError:
    _HAS_PANDAS = False

try:
    import h5py
except ImportError:
    _HAS_H5PY = False


class DummyStingrayObj(StingrayObject):
    main_array_attr = "guefus"

    def __init__(self, dummy=None):
        self.guefus = dummy
        self._mask = None
        # StingrayObject.__init__(self)


class DummyStingrayTs(StingrayTimeseries):
    main_array_attr = "time"

    def __init__(self, dummy=None):
        self.guefus = dummy
        # StingrayObject.__init__(self)


# A StingrayObject with no main_array_attr
class BadStingrayObj(StingrayObject):
    def __init__(self, dummy=None):
        self.guefus = dummy
        StingrayObject.__init__(self)


class TestStingrayObject:
    @classmethod
    def setup_class(cls):
        cls.arr = [4, 5, 2]
        sting_obj = DummyStingrayObj(cls.arr)
        sting_obj.pardulas = [3.0 + 1.0j, 2.0j, 1.0 + 0.0j]
        sting_obj.pirichitus = 4
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = [[41, 25], [98, 3]]
        cls.sting_obj = sting_obj

    def test_preliminary(self):
        assert np.allclose(self.sting_obj.guefus, self.arr)

    def test_instantiate_without_main_array_attr(self):
        with pytest.raises(RuntimeError):
            BadStingrayObj(self.arr)

    def test_apply_mask(self):
        ts = copy.deepcopy(self.sting_obj)
        newts0 = ts.apply_mask([True, True, False], inplace=False)
        newts1 = ts.apply_mask([True, True, False], inplace=True)
        assert newts0.parafritus == "bonus!"
        assert newts1.parafritus == "bonus!"
        for obj in [newts1, newts0]:
            assert obj.parafritus == "bonus!"
            assert np.array_equal(obj.guefus, [4, 5])
            assert np.array_equal(obj.panesapa, ts.panesapa)
            assert np.array_equal(obj.pardulas, [3.0 + 1.0j, 2.0j])
        assert ts is newts1
        assert ts is not newts0

    def test_operations(self):
        guefus = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = DummyStingrayObj(guefus)
        ts2 = DummyStingrayObj(guefus)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1 + ts2  # Test __add__
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.guefus, guefus)
        lc = ts1 - ts2  # Test __sub__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.guefus, guefus)
        lc = -ts2 + ts1  # Test __neg__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.guefus, guefus)

    def test_len(self):
        assert len(self.sting_obj) == 3

    def test_slice(self):
        ts1 = self.sting_obj
        ts_filt = ts1[1]
        assert np.array_equal(ts_filt.guefus, [5])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)
        assert np.array_equal(ts_filt.pardulas, [2.0j])

        ts_filt = ts1[:2]
        assert np.array_equal(ts_filt.guefus, [4, 5])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)
        assert np.array_equal(ts_filt.pardulas, [3.0 + 1.0j, 2.0j])

        with pytest.raises(IndexError, match="The index must be either an integer or a slice"):
            ts1[1.0]

    def test_side_effects(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        new_so = DummyStingrayObj([4, 5, 6])
        assert not hasattr(new_so, "panesapa")

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so.stingattr = DummyStingrayObj([3, 4, 5])
        ts = so.to_astropy_table()
        new_so = DummyStingrayObj.from_astropy_table(ts)
        assert so == new_so
        assert not hasattr(new_so, "stingattr")

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_xarray()
        new_so = DummyStingrayObj.from_xarray(ts)

        assert so == new_so

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        assert np.allclose(so.guefus, [4, 5, 2])
        so.guefus = np.random.randint(0, 4, 3)
        ts = so.to_pandas()
        new_so = DummyStingrayObj.from_pandas(ts)

        assert so == new_so

    def test_astropy_roundtrip_empty(self):
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so = DummyStingrayObj([])
        ts = so.to_astropy_table()
        new_so = DummyStingrayObj.from_astropy_table(ts)
        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_XARRAY")
    def test_xarray_roundtrip_empty(self):
        so = DummyStingrayObj([])
        ts = so.to_xarray()
        new_so = DummyStingrayObj.from_xarray(ts)

        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_PANDAS")
    def test_pandas_roundtrip_empty(self):
        so = DummyStingrayObj([])
        ts = so.to_pandas()
        new_so = DummyStingrayObj.from_pandas(ts)

        assert new_so.guefus is None or len(new_so.guefus) == 0

    @pytest.mark.skipif("not _HAS_H5PY")
    def test_hdf_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        assert so == new_so

    def test_file_roundtrip_fits(self):
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        so.write("dummy.fits")
        new_so = DummyStingrayObj.read("dummy.fits")
        os.unlink("dummy.fits")
        # panesapa is invalid for FITS header and got lost
        assert not hasattr(new_so, "panesapa")
        new_so.panesapa = so.panesapa
        assert so == new_so

    @pytest.mark.parametrize("fmt", ["pickle", "ascii", "ascii.ecsv"])
    def test_file_roundtrip(self, fmt):
        so = copy.deepcopy(self.sting_obj)
        so.guefus = np.random.randint(0, 4, 3)
        so.panesapa = np.random.randint(5, 9, (6, 2))
        so.write(f"dummy.{fmt}", fmt=fmt)
        new_so = DummyStingrayObj.read(f"dummy.{fmt}", fmt=fmt)
        os.unlink(f"dummy.{fmt}")

        assert so == new_so


class TestStingrayTimeseries:
    @classmethod
    def setup_class(cls):
        cls.time = np.arange(0, 10, 1)
        cls.arr = cls.time + 2
        sting_obj = StingrayTimeseries(
            time=cls.time,
            mjdref=59777.000,
            array_attrs=dict(guefus=cls.arr),
            parafritus="bonus!",
            panesapa=np.asarray([[41, 25], [98, 3]]),
            gti=np.asarray([[-0.5, 10.5]]),
        )
        cls.sting_obj = sting_obj

    def test_apply_mask(self):
        ts = copy.deepcopy(self.sting_obj)
        mask = [True, True] + 8 * [False]
        newts0 = ts.apply_mask(mask, inplace=False)
        newts1 = ts.apply_mask(mask, inplace=True)
        for obj in [newts1, newts0]:
            for attr in ["parafritus", "mjdref"]:
                assert getattr(obj, attr) == getattr(ts, attr)
            for attr in ["panesapa", "gti"]:
                assert np.array_equal(getattr(obj, attr), getattr(ts, attr))

            assert np.array_equal(obj.guefus, [2, 3])
            assert np.array_equal(obj.time, [0, 1])
        assert ts is newts1
        assert ts is not newts0

    def test_operations(self):
        time = [5, 10, 15]
        count1 = [300, 100, 400]
        count2 = [600, 1200, 800]
        ts1 = StingrayTimeseries(time=time)
        ts2 = StingrayTimeseries(time=time)
        ts1.counts = count1
        ts2.counts = count2
        lc = ts1 + ts2  # Test __add__
        assert np.allclose(lc.counts, [900, 1300, 1200])
        assert np.array_equal(lc.time, time)
        lc = ts1 - ts2  # Test __sub__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.time, time)
        lc = -ts2 + ts1  # Test __neg__
        assert np.allclose(lc.counts, [-300, -1100, -400])
        assert np.array_equal(lc.time, time)

    def test_sub_with_gti(self):
        time = [10, 20, 30]
        count1 = [600, 1200, 800]
        count2 = [300, 100, 400]
        gti1 = [[0, 35]]
        gti2 = [[5, 40]]
        ts1 = StingrayTimeseries(time, array_attrs=dict(counts=count1), gti=gti1, dt=10)
        ts2 = StingrayTimeseries(time, array_attrs=dict(counts=count2), gti=gti2, dt=10)
        lc = ts1 - ts2
        assert np.allclose(lc.counts, [300, 1100, 400])

    def test_len(self):
        assert len(self.sting_obj) == 10

    def test_slice(self):
        ts1 = self.sting_obj
        ts_filt = ts1[1]
        assert np.array_equal(ts_filt.guefus, [3])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)

        ts_filt = ts1[:2]
        assert np.array_equal(ts_filt.guefus, [2, 3])
        assert ts_filt.parafritus == "bonus!"
        assert np.array_equal(ts_filt.panesapa, ts1.panesapa)

        with pytest.raises(IndexError, match="The index must be either an integer or a slice"):
            ts1[1.0]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_apply_gti(self, inplace):
        so = copy.deepcopy(self.sting_obj)
        so.gti = np.asarray([[-0.1, 2.1]])
        so2 = so.apply_gtis()
        if inplace:
            assert so2 is so

        assert np.allclose(so2.time, [0, 1, 2])
        assert np.allclose(so2.guefus, [2, 3, 4])
        assert np.allclose(so2.gti, [[-0.1, 2.1]])
        assert np.allclose(so2.mjdref, 59777.000)

    def test_split_ts_by_gtis(self):
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        counts = [1, 1, 1, 1, 2, 3, 3, 2, 3, 3]
        bg_counts = [0, 0, 0, 1, 0, 1, 2, 0, 0, 1]
        bg_ratio = [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2]
        frac_exp = [1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 1]
        gti = [[0.5, 4.5], [5.5, 7.5], [8.5, 9.5]]

        ts = StingrayTimeseries(
            times,
            array_attrs=dict(
                counts=counts, bg_counts=bg_counts, bg_ratio=bg_ratio, frac_exp=frac_exp
            ),
            gti=gti,
        )
        list_of_tss = ts.split_by_gti(min_points=0)
        assert len(list_of_tss) == 3

        ts0 = list_of_tss[0]
        ts1 = list_of_tss[1]
        ts2 = list_of_tss[2]
        assert np.allclose(ts0.time, [1, 2, 3, 4])
        assert np.allclose(ts1.time, [6, 7])
        assert np.allclose(ts2.time, [9])
        assert np.allclose(ts0.counts, [1, 1, 1, 1])
        assert np.allclose(ts1.counts, [3, 3])
        assert np.allclose(ts1.counts, [3])
        assert np.allclose(ts0.gti, [[0.5, 4.5]])
        assert np.allclose(ts1.gti, [[5.5, 7.5]])
        assert np.allclose(ts2.gti, [[8.5, 9.5]])
        # Check if new attributes are also splited accordingly
        assert np.allclose(ts0.bg_counts, [0, 0, 0, 1])
        assert np.allclose(ts1.bg_counts, [1, 2])
        assert np.allclose(ts0.bg_ratio, [0.1, 0.1, 0.1, 0.2])
        assert np.allclose(ts1.bg_ratio, [0.2, 0.2])
        assert np.allclose(ts0.frac_exp, [1, 0.5, 1, 1])
        assert np.allclose(ts1.frac_exp, [0.5, 0.5])

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_table()
        new_so = StingrayTimeseries.from_astropy_table(ts)
        assert so == new_so

    def test_astropy_ts_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_timeseries()
        new_so = StingrayTimeseries.from_astropy_timeseries(ts)
        assert so == new_so

    def test_shift_time(self):
        new_so = self.sting_obj.shift(1)
        assert np.allclose(new_so.time - 1, self.sting_obj.time)
        assert np.allclose(new_so.gti - 1, self.sting_obj.gti)

    def test_change_mjdref(self):
        new_so = self.sting_obj.change_mjdref(59776.5)
        assert new_so.mjdref == 59776.5
        assert np.allclose(new_so.time - 43200, self.sting_obj.time)
        assert np.allclose(new_so.gti - 43200, self.sting_obj.gti)


class TestStingrayTimeseriesSubclass:
    @classmethod
    def setup_class(cls):
        cls.arr = [4, 5, 2]
        sting_obj = DummyStingrayTs(cls.arr)
        sting_obj.time = np.asarray([0, 1, 2])
        sting_obj.mjdref = 59777.000
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = np.asarray([[41, 25], [98, 3]])
        sting_obj.gti = np.asarray([[-0.5, 2.5]])
        cls.sting_obj = sting_obj

    def test_astropy_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        ts = so.to_astropy_table()
        new_so = DummyStingrayTs.from_astropy_table(ts)
        assert so == new_so

    def test_astropy_ts_roundtrip(self):
        so = copy.deepcopy(self.sting_obj)
        ts = so.to_astropy_timeseries()
        new_so = DummyStingrayTs.from_astropy_timeseries(ts)
        assert so == new_so

    def test_shift_time(self):
        new_so = self.sting_obj.shift(1)
        assert np.allclose(new_so.time - 1, self.sting_obj.time)
        assert np.allclose(new_so.gti - 1, self.sting_obj.gti)

    def test_change_mjdref(self):
        new_so = self.sting_obj.change_mjdref(59776.5)
        assert new_so.mjdref == 59776.5
        assert np.allclose(new_so.time - 43200, self.sting_obj.time)
        assert np.allclose(new_so.gti - 43200, self.sting_obj.gti)
