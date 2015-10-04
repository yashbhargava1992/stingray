
import numpy as np
from ..stingray import lightcurve
## TODO: Fix relative imports

class TestLightcurve(object):

    def setUp(self):
        tstart = 0.0
        tend = 1.0
        dt = 1.e-8

        small_time = np.linspace(tstart, tend, int((tend-tstart)/dt))

        mean_count_rate = 10000.0
        mean_counts = mean_count_rate*dt

        self.toa = np.random.poisson(mean_counts, size=small_time.shape[0])


    def test_correct_timeresolution(self):
        timestep = 0.1
        lc = lightcurve.Lightcurve(self.toa, timestep=timestep)
        assert np.isclose(lc.res, timestep)






