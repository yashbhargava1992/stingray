from __future__ import division
import numpy as np

import pytest
import warnings

from stingray import AveragedCovariancespectrum, Covariancespectrum, Lightcurve


class TestCovariancespectrum(object):

    def setup_class(self):
        event_list = np.array([[1, 2], [1, 4], [1, 8], [3, 6], [3, 1], [4, 2],
                               [4, 4], [4, 6], [4, 7], [4, 5], [4, 9], [5, 1],
                               [5, 6], [5, 7], [5, 9], [5, 3], [5, 8], [6, 1],
                               [6, 3], [7, 3], [7, 4], [7, 5], [7, 7], [8, 8],
                               [9, 2], [9, 5], [9, 9]])
        self.event_list = event_list


    def test_class_fails_if_events_is_set_but_dt_is_not(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.event_list)

    def test_set_lc_flag_correctly(self):
        c = Covariancespectrum(self.event_list, dt=1)
        assert c.use_lc is False

    def test_covar_without_any_band_interest(self):
        c = Covariancespectrum(self.event_list, dt=1)

        # this should be 9???
        assert len(c.covar) == 9
        #assert len(c.covar) == 8

    def test_init_with_invalid_band_interest(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.event_list, dt=1, band_interest=(1))

    def test_init_with_invalid_ref_band_interest(self):
        with pytest.raises(ValueError):
            c = Covariancespectrum(self.event_list, dt=1, ref_band_interest=(0))

    def test_covar_with_both_bands(self):
        c = Covariancespectrum(self.event_list, dt=1,
                               band_interest=[(2, 6), (8, 9)],
                               ref_band_interest=(1, 10))
        assert len(c.covar) == 2

    def test_with_unsorted_event_list(self):
        event_list = np.array([[2, 1], [2, 3], [1, 2], [5, 2], [4, 1]])
        with warnings.catch_warnings(record=True) as w:
            c = Covariancespectrum(event_list, dt=1)
            assert "must be sorted" in str(w[0].message)

    def test_with_std_as_iterable(self):
        c = Covariancespectrum(self.event_list, dt=1, std=[1, 2])

    def test_with_std_as_a_single_float(self):
        c = Covariancespectrum(self.event_list, dt=1, std=2.55)


class TestAveragedCovariancespectrum(object):

    def setup_class(self):
        event_list = np.array([[1, 2], [1, 4], [1, 8], [3, 6], [3, 1], [4, 2],
                               [4, 4], [4, 6], [4, 7], [4, 5], [4, 9], [5, 1],
                               [5, 6], [5, 7], [5, 9], [5, 3], [5, 8], [6, 1],
                               [6, 3], [7, 3], [7, 4], [7, 5], [7, 7], [8, 8],
                               [9, 2], [9, 5], [9, 9], [10, 1], [10, 2]])
        self.event_list = event_list

    def test_with_full_segment(self):
        c = Covariancespectrum(self.event_list, 1)
        avg_c = AveragedCovariancespectrum(self.event_list, segment_size=10,
                                           dt=1)
        assert np.all(avg_c.unnorm_covar == c.unnorm_covar)

    def test_with_two_segments(self):
        avg_c = AveragedCovariancespectrum(self.event_list, segment_size=5, dt=1)
