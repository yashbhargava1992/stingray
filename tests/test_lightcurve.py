from stingray.lightcurve import Lightcurve

class TestLightcurve(object):
    def test_create(self):
        """
        Demonstrate that we can create a trivial Lightcurve object.
        """
        times = [1, 2, 3, 4]
        lc = Lightcurve(times)
