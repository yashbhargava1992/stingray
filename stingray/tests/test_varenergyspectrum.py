import numpy as np
from stingray.events import EventList
from stingray.varenergyspectrum import VarEnergySpectrum

from astropy.tests.helper import pytest


class TestPowerspectrum(object):

    @classmethod
    def setup_class(cls):
        tstart = 0.0
        tend = 10.0
        nphot = 100
        cls.events = EventList(np.random.uniform(tstart, tend, nphot),
                                pha=np.random.uniform(0.3, 12, nphot))
        cls.vespec = VarEnergySpectrum(cls.events, [0., 10000],
                                       [0.5, 5, 10], [0.3, 10])

    def test_intervals_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([0.5, 6], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5], [6, 10]])
        ref_int = self.vespec._decide_ref_intervals([0.5, 11], [0.3, 10])
        np.testing.assert_allclose(ref_int, [[0.3, 0.5]])

    def test_intervals_non_overlapping(self):
        ref_int = self.vespec._decide_ref_intervals([6, 11], [0.3, 5])
        np.testing.assert_allclose(ref_int, [[0.3, 5]])

    def test_construct_lightcurves(self):
        self.vespec._construct_lightcurves(0.1, [6, 11], [0.3, 5],
                                           tstart=0, tstop=10)