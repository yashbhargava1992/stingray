import tempfile

import pytest
import numpy as np
from stingray.lightcurve import Lightcurve
from stingray.events import EventList
from stingray.largememory import createChunkedSpectra, saveData
from stingray.crossspectrum import AveragedCrossspectrum
from stingray.powerspectrum import AveragedPowerspectrum


HAS_ZARR = False
try:
    import zarr

    HAS_ZARR = True
    from numcodecs import Blosc
except ImportError:
    pass


@pytest.mark.skipif('not HAS_ZARR')
class TestSaveAndRetrieve(object):
    @classmethod
    def setup_class(cls):
        time = np.arange(0, 1e7)
        counts = np.random.poisson(10, time.size)
        cls.lc = Lightcurve(time, counts, skip_checks=True)
        evtimes = np.sort(np.random.uniform(0, 1e7, 10**7))
        cls.ev = EventList(time=evtimes)

    def test_save_wrong_data(self):
        with pytest.raises(ValueError) as excinfo:
            saveData("A string", 'bububu')
        assert 'Cannot save data of type str' in str(excinfo.value)

    def test_save_lc(self):
        file = tempfile.mkdtemp()
        saveData(self.lc, file)
        newfile = zarr.open(file, mode='r')
        # TODO: Retrieve the data and test that they are identical to the
        #  original ones

    def test_save_ev(self):
        file = tempfile.mkdtemp()
        saveData(self.ev, file)
        newfile = zarr.open(file, mode='r')
        # TODO: Retrieve the data and test that they are identical to the
        #  original ones

class TestChunkPS(object):
    @classmethod
    def setup_class(cls):
        time = np.arange(0, 2048000)
        counts1 = np.random.poisson(10, time.size)
        counts2 = np.random.poisson(10, time.size)
        cls.lc1 = Lightcurve(time, counts1, skip_checks=True)
        cls.file1 = tempfile.mkdtemp(suffix='.zarray')
        cls.lc2 = Lightcurve(time, counts2, skip_checks=True)
        cls.file2 = tempfile.mkdtemp(suffix='.zarray')
        saveData(cls.lc1, cls.file1)
        saveData(cls.lc2, cls.file2)

    def test_invalid_data_to_pds(self):
        with pytest.raises(ValueError) as excinfo:
            AveragedPowerspectrum("sdfasfsa", segment_size=2048,
                                  large_data=True)
        assert 'Invalid input data type: str' in str(excinfo.value)

    def test_invalid_data_to_cpds(self):
        with pytest.raises(ValueError) as excinfo:
            AveragedCrossspectrum("sdfasfsa", "sdfasfsa", segment_size=2048,
                                  large_data=True)
        assert 'Invalid input data type: str' in str(excinfo.value)

    def test_calc_pds(self):

        ps_normal = AveragedPowerspectrum(self.lc1, segment_size=2048)
        ps_large = AveragedPowerspectrum(self.lc1, segment_size=2048,
                                         large_data=True)
        for attr in ['freq', 'power']:
            assert np.all(getattr(ps_normal, attr) == getattr(ps_large, attr))
        # TODO: Add more attributes

    def test_calc_cpds(self):
        cs_normal = AveragedCrossspectrum(
            self.lc1, self.lc2, segment_size=2048)
        cs_large = AveragedCrossspectrum(
            self.lc1,  self.lc2, segment_size=2048, large_data=True)
        for attr in ['freq', 'power']:
            assert np.all(getattr(cs_normal, attr) == getattr(cs_large, attr))
        # TODO: Add more attributes
