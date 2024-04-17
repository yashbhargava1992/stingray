import pytest
import os
from astropy.io import fits
from stingray.mission_support import rxte_pca_event_file_interpretation

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, "..", "..", "tests", "data")


class TestXTE(object):
    @classmethod
    def setup_class(cls):
        cls.xtefile = os.path.join(datadir, "xte_test.evt.gz")
        cls.wrongfile = os.path.join(datadir, "monol_testA.evt")

    def test_wrong_file_raises(self):
        with pytest.raises(ValueError, match="No XTE_SE extension found."):
            rxte_pca_event_file_interpretation(fits.open(self.wrongfile))
