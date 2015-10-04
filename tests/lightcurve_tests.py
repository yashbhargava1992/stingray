
import numpy as np
from stingray import lightcurve
## TODO: Fix relative imports

np.random.seed(20150907)


class TestLightcurvefromTOA(object):

    def setUp(self):
        tstart = 0.0
        tend = 1.0
        small_dt = 1.e-8

        small_time = np.linspace(tstart, tend, int((tend-tstart)/small_dt))

        mean_count_rate = 10000.0
        mean_counts = mean_count_rate*small_dt

        poisson_counts = np.random.poisson(mean_counts, size=small_time.shape[0])

        self.toa = small_time[poisson_counts == 1]

        self.dt = 0.1
        self.lc = lightcurve.Lightcurve(self.toa, dt=self.dt)


    def test_correct_timeresolution(self):
        assert np.isclose(self.lc.res, self.dt)






