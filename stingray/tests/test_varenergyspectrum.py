import numpy as np
from stingray.events import EventList
from stingray.varenergyspectrum import VarEnergySpectrum, RmsEnergySpectrum

from astropy.tests.helper import pytest
np.random.seed(20150907)


class TestPowerspectrum(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 100.0
        nphot = 1000
        alltimes = np.random.uniform(tstart, tend, nphot)
        alltimes.sort()
        cls.events = EventList(alltimes,
                               pha=np.random.uniform(0.3, 12, nphot),
                               gti = [[tstart, tend]])
        cls.vespec = VarEnergySpectrum(cls.events, [0., 10000],
                                       [0.5, 5, 10], [0.3, 10],
                                       bin_time=0.1)
        cls.vespeclog = \
            VarEnergySpectrum(cls.events, [0., 10000],
                              [0.5, 5, 10], [0.3, 10], log_distr=True)

    def test_intervals_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([0.5, 6], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5], [6, 10]])
        ref_int = self.vespec._decide_ref_intervals([0.5, 11], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5]])

    def test_intervals_non_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([6, 11], [0.3, 5])
        np.testing.assert_allclose(ref_int, [[0.3, 5]])

    def test_construct_lightcurves(self):
        events = EventList([0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
                           pha=[0,0,0,0,1,1],
                           gti=[[0, 0.65]])
        vespec = VarEnergySpectrum(events, [0., 10000],
                                   [0, 1, 2], [0.5, 1.1],
                                   bin_time=0.1)
        base_lc, ref_lc = \
            vespec._construct_lightcurves([0, 0.5],
                                          tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 1, 0, 1])

    def test_construct_lightcurves_no_exclude(self):
        events = EventList([0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
                           pha=[0,0,0,0,1,1],
                           gti=[[0, 0.65]])

        vespec = VarEnergySpectrum(events, [0., 10000],
                                   [0, 1, 2], [0, 0.5],
                                   bin_time=0.1)
        base_lc, ref_lc = \
            vespec._construct_lightcurves([0, 0.5],
                                          tstart=0, tstop=0.65,
                                          exclude=False)
        np.testing.assert_equal(base_lc.counts, ref_lc.counts)

    def test_construct_lightcurves_pi(self):
        events = EventList([0.09, 0.21, 0.23, 0.32, 0.4, 0.54],
                           pi=np.asarray([0, 0, 0, 0, 1, 1]),
                           gti=[[0, 0.65]])
        vespec = VarEnergySpectrum(events, [0., 10000],
                                   [0, 1, 2], [0.5, 1.1], use_pi=True,
                                   bin_time=0.1)
        base_lc, ref_lc = \
            vespec._construct_lightcurves([0, 0.5],
                                          tstart=0, tstop=0.65)
        np.testing.assert_allclose(base_lc.counts, [1, 0, 2, 1, 0, 0])
        np.testing.assert_allclose(ref_lc.counts, [0, 0, 0, 1, 0, 1])

    def test_rmsspectrum_creation(self):
        rms = RmsEnergySpectrum(self.events, [0., 10000],
                                [0.3, 12, 2], [5, 10],
                                bin_time=0.01,
                                segment_size=30)

