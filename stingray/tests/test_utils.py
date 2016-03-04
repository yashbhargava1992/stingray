
import numpy as np
import stingray.utils as utils

np.random.seed(20150907)

class TestRebinData(object):

    @classmethod
    def setup_class(cls):
        cls.dx = 1.0
        cls.n = 10
        cls.counts = 2.0
        cls.x = np.arange(cls.dx/2, cls.dx/2+cls.n*cls.dx, cls.dx)
        cls.y = np.zeros_like(cls.x)+cls.counts

    def test_new_stepsize(self):
        dx_new = 2.0
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert step_size == dx_new/self.dx

    def test_arrays(self):
        dx_new = 2.0
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert isinstance(xbin, np.ndarray)
        assert isinstance(ybin, np.ndarray)

    def test_length_matches(self):
        dx_new = 2.0
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert xbin.shape[0] == ybin.shape[0]


    def test_binned_counts(self):
        dx_new = 2.0

        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)

        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin, ybin_test)

    def test_uneven_bins(self):
        dx_new = 1.5
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        assert np.isclose(xbin[1]-xbin[0], dx_new)

    def test_uneven_binned_counts(self):
        dx_new = 1.5
        xbin, ybin, step_size = utils.rebin_data(self.x, self.y, dx_new)
        print(xbin)
        print(ybin)
        ybin_test = np.zeros_like(xbin) + self.counts*dx_new/self.dx
        assert np.allclose(ybin_test, ybin)

class TestUtils(object):

    def test_optimal_bin_time(self):
        assert utils.optimal_bin_time(512, 2.1) == 2
        assert utils.optimal_bin_time(512, 3.9) == 2
        assert utils.optimal_bin_time(512, 4.1) == 4

