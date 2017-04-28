import numpy as np

from astropy.modeling import models

from stingray import Powerspectrum

from stingray.modeling.scripts import fit_lorentzians, fit_powerspectrum_bounds


class TestFitLorentzians(object):

    @classmethod
    def setup_class(cls):

        np.random.seed(150)
        cls.nlor = 3

        cls.x_0_0 = 0.5
        cls.x_0_1 = 2.0
        cls.x_0_2 = 7.5

        cls.amplitude_0 = 200.0
        cls.amplitude_1 = 100.0
        cls.amplitude_2 = 50.0

        cls.fwhm_0 = 0.1
        cls.fwhm_1 = 1.0
        cls.fwhm_2 = 0.5

        cls.whitenoise = 2.0

        cls.model = models.Lorentz1D(cls.amplitude_0, cls.x_0_0, cls.fwhm_0) + \
                    models.Lorentz1D(cls.amplitude_1, cls.x_0_1, cls.fwhm_1) + \
                    models.Lorentz1D(cls.amplitude_2, cls.x_0_2, cls.fwhm_2) + \
                    models.Const1D(cls.whitenoise)

        freq = np.linspace(0.01, 10.0, 10.0 / 0.01)
        p = cls.model(freq)
        noise = np.random.exponential(size=len(freq))

        power = p * noise
        cls.ps = Powerspectrum()
        cls.ps.freq = freq
        cls.ps.power = power
        cls.ps.df = cls.ps.freq[1] - cls.ps.freq[0]
        cls.ps.m = 1

        cls.t0 = [200.0, 0.5, 0.1, 100.0, 2.0, 1.0, 50.0, 7.5, 0.5, 2.0]

        cls.parest, cls.res = fit_lorentzians(cls.ps, cls.nlor, cls.t0)

    def test_function_creates_right_number_of_lorentians(self):
        assert (len(self.parest.lpost.model.parameters)-1)/3 == self.nlor

    def test_correct_parameters_without_whitenoise(self):
        parest, res = fit_lorentzians(self.ps, self.nlor, self.t0[:-1],
                                      fit_whitenoise=False)

        assert len(parest.lpost.model.parameters)/3 == self.nlor

    def test_parameters_in_right_ballpark(self):

        true_pars = [self.amplitude_0, self.x_0_0, self.fwhm_0,
                     self.amplitude_1, self.x_0_1, self.fwhm_1,
                     self.amplitude_2, self.x_0_2, self.fwhm_2,
                     self.whitenoise]

        for t, p in zip(true_pars, self.res.p_opt):
            print(str(t) + "\t" + str(p))

        assert np.all(np.isclose(true_pars, self.res.p_opt, rtol=0.5))



class TestFitBoundPowerspectrum(object):
    @classmethod
    def setup_class(cls):
        np.random.seed(150)
        cls.nlor = 3

        cls.amplitude_0 = 200.0
        cls.amplitude_1 = 100.0
        cls.amplitude_2 = 50.0

        cls.x_0_0 = 2
        cls.x_0_1 = 4
        cls.x_0_2 = 6

        cls.fwhm_0 = 0.1
        cls.fwhm_1 = 0.1
        cls.fwhm_2 = 0.3

        cls.whitenoise = 2.0

        cls.model = models.Lorentz1D(cls.amplitude_0, cls.x_0_0, cls.fwhm_0) + \
                    models.Lorentz1D(cls.amplitude_1, cls.x_0_1, cls.fwhm_1) + \
                    models.Lorentz1D(cls.amplitude_2, cls.x_0_2, cls.fwhm_2) + \
                    models.Const1D(cls.whitenoise)

        freq = np.linspace(0.01, 10.0, 10.0 / 0.01)
        p = cls.model(freq)
        noise = np.random.exponential(size=len(freq))

        power = p * noise
        cls.ps = Powerspectrum()
        cls.ps.freq = freq
        cls.ps.power = power
        cls.ps.df = cls.ps.freq[1] - cls.ps.freq[0]
        cls.ps.m = 1

        cls.t0 = [200.0, 2, 0.1, 100.0, 4., 0.1, 50.0, 6, 0.3, 2.0]

    def test_fitting_with_bounds(self):
        double_f = lambda model : model.x_0_0 * 2
        triple_f = lambda model : model.x_0_0 * 3
        model = self.model.copy()
        model.x_0_1.tied = double_f
        model.x_0_2.tied = triple_f
        model.amplitude_0 = self.t0[0]
        model.amplitude_0.fixed = False
        # model.bounds = {}

        p_opt = fit_powerspectrum_bounds(self.ps, model,
                                         self.t0)

        true_pars = [self.amplitude_0,
                     self.x_0_0, self.fwhm_0,
                     self.amplitude_1, self.x_0_1, self.fwhm_1,
                     self.amplitude_2, self.x_0_2, self.fwhm_2,
                     self.whitenoise]

        for t, p in zip(true_pars, p_opt):
            print(str(t) + "\t" + str(p))

        assert np.all(np.isclose(true_pars, p_opt, rtol=0.5))




