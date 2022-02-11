import os
import pytest
import numpy as np
from stingray.base import StingrayObject

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
        StingrayObject.__init__(self)


# A StingrayObject with no main_array_attr
class BadStingrayObj(StingrayObject):
    def __init__(self, dummy=None):
        self.guefus = dummy
        StingrayObject.__init__(self)


class TestStingrayObject():
    @classmethod
    def setup_class(cls):
        cls.arr = [4, 5, 2]
        sting_obj = DummyStingrayObj(cls.arr)
        sting_obj.pardulas = [3. + 1.j, 2.j, 1. + 0.j]
        sting_obj.pirichitus = 4
        sting_obj.parafritus = "bonus!"
        sting_obj.panesapa = [[41, 25], [98, 3]]
        cls.sting_obj = sting_obj

    def _check_equal(self, so, new_so):
        for attr in ["guefus", "pardulas", "panesapa"]:
            print(attr, getattr(so, attr), getattr(new_so, attr))
            assert np.allclose(getattr(so, attr), getattr(new_so, attr))

        for attr in ["pirichitus", "parafritus"]:
            print(attr, getattr(so, attr), getattr(new_so, attr))
            assert getattr(so, attr) == getattr(new_so, attr)

    def test_preliminary(self):
        assert np.allclose(self.sting_obj.guefus, self.arr)

    def test_instantiate_without_main_array_attr(self):
        with pytest.raises(RuntimeError):
            BadStingrayObj(self.arr)

    def test_astropy_roundtrip(self):
        so = self.sting_obj
        # Set an attribute to a DummyStingrayObj. It will *not* be saved
        so.stingattr = DummyStingrayObj(self.arr)
        ts = so.to_astropy_table()
        new_so = so.from_astropy_table(ts)
        self._check_equal(so, new_so)
        assert not hasattr(new_so, "stingattr")

    @pytest.mark.skipif('not _HAS_XARRAY')
    def test_xarray_roundtrip(self):
        so = self.sting_obj
        ts = so.to_xarray()
        new_so = so.from_xarray(ts)

        self._check_equal(so, new_so)

    @pytest.mark.skipif('not _HAS_PANDAS')
    def test_pandas_roundtrip(self):
        so = self.sting_obj
        ts = so.to_pandas()
        new_so = so.from_pandas(ts)

        self._check_equal(so, new_so)

    @pytest.mark.skipif('not _HAS_H5PY')
    def test_hdf_roundtrip(self):
        so = self.sting_obj
        so.write("dummy.hdf5")
        new_so = so.read("dummy.hdf5")
        os.unlink("dummy.hdf5")

        self._check_equal(so, new_so)

    @pytest.mark.parametrize("fmt", ["pickle", "ascii", "ascii.ecsv", "fits"])
    def test_file_roundtrip(self, fmt):
        so = self.sting_obj
        so.write("dummy", fmt=fmt)
        new_so = so.read("dummy", fmt=fmt)
        os.unlink("dummy")

        self._check_equal(so, new_so)
