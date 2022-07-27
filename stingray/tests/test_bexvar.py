import warnings
import pytest
import os
import numpy as np
from stingray import bexvar
from astropy.table import Table


_HAS_ULTRANEST = True

try:
    import ultranest
except ImportError:
    _HAS_ULTRANEST = False

curdir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(curdir, 'data')


class TestBexvarResult(object):
    @classmethod
    def setup_class(cls):
        fname_data = os.path.join(datadir,"LightCurve_bexvar.fits")
        lightcurve = Table.read(fname_data, hdu='RATE', format='fits')
        band = 0

        cls.time = lightcurve['TIME'] - lightcurve['TIME'][0]
        cls.time_delta = lightcurve["TIMEDEL"]
        cls.bg_counts = lightcurve["BACK_COUNTS"][:,band]
        cls.src_counts = lightcurve["COUNTS"][:,band]
        cls.bg_ratio = lightcurve["BACKRATIO"]
        cls.frac_exp = lightcurve["FRACEXP"][:,band]

        cls.fname_result = os.path.join(datadir,"bexvar_results_band_0.npy")

    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_bexvar(self):
        log_cr_sigma_from_function = bexvar.bexvar(self.time, self.time_delta, self.src_counts, self.bg_counts, self.bg_ratio, self.frac_exp)
        log_cr_sigma_result = np.load(self.fname_result,allow_pickle=True)[1]
        assert np.allclose(log_cr_sigma_from_function, log_cr_sigma_result)


class TestInternalFunctions(object):
    @classmethod
    def setup_class(cls):
        fname_data = os.path.join(datadir,"LightCurve_bexvar.fits")
        lightcurve = Table.read(fname_data, hdu='RATE', format='fits')
        band = 0

        lightcurve = lightcurve[lightcurve['FRACEXP'][:,band] > 0.1]
        cls.time_delta = lightcurve["TIMEDEL"]
        cls.bg_counts = lightcurve["BACK_COUNTS"][:,band]
        cls.src_counts = lightcurve["COUNTS"][:,band]
        cls.bg_ratio = lightcurve["BACKRATIO"]
        cls.frac_exp = lightcurve["FRACEXP"][:,band]
        cls.bg_area = 1. / cls.bg_ratio
        cls.rate_conversion = cls.frac_exp * cls.time_delta

        fname_result = os.path.join(datadir,"bexvar_results_band_0.npy")
        cls.function_result = np.load(fname_result,allow_pickle=True)

    def test_lscg_gen(self):

        log_src_crs_grid_from_function = bexvar._lscg_gen(self.src_counts , self.bg_counts, \
            self.bg_area, self.rate_conversion, 100)       
        log_src_crs_grid_result = self.function_result[0]

        assert np.allclose(log_src_crs_grid_from_function, log_src_crs_grid_result)

    def test_estimate_source_cr_marginalised(self):
        
        log_src_crs_grid = self.function_result[0]
        weights_from_function = bexvar._estimate_source_cr_marginalised(log_src_crs_grid,\
            self.src_counts[0], self.bg_counts[0], self.bg_area[0], self.rate_conversion[0])
        weights_from_results = self.function_result[2][0]

        assert np.allclose(weights_from_function, weights_from_results)
    
    @pytest.mark.skipif("not _HAS_ULTRANEST")
    def test_calculate_bexvar(self):
        log_src_crs_grid = self.function_result[0]
        pdfs = self.function_result[2]

        posterior_log_sigma_src_cr_from_function = bexvar._calculate_bexvar(log_src_crs_grid, pdfs)
        posterior_log_sigma_src_cr_results = self.function_result[1]

        assert np.allclose(posterior_log_sigma_src_cr_from_function, posterior_log_sigma_src_cr_results)
