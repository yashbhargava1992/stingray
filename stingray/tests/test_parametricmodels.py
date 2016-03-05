
from nose.tools import eq_, raises

import numpy as np

from stingray import parametricmodels
from stingray import ParametricModel, Const
from stingray import PowerLaw, BrokenPowerLaw
from stingray import Lorentzian, FixedCentroidLorentzian
from stingray import CombinedModel

logmin = parametricmodels.logmin


class TestParametricModel(object):
    def test_npar_passes_when_int(self):
        npar = int(2)
        p = ParametricModel(npar, "MyModel")

    def test_npar_passes_when_numpy_int(self):
        npar = np.int(2)
        p = ParametricModel(npar, "MyNumpyModel")

    @raises(AssertionError)
    def test_npar_fails_when_not_int(self):
        npar = float(2.0)
        p = ParametricModel(npar, "MyFailingModel")

    @raises(AssertionError)
    def test_npar_fails_when_nan(self):
        npar = np.nan
        p = ParametricModel(npar, "MyNaNModel")

    @raises(AssertionError)
    def test_npar_fails_when_inf(self):
        npar = np.inf
        p = ParametricModel(npar, "MyInfModel")

    def test_name_passes_when_string(self):
        npar = 2
        name = "MyModel"
        p = ParametricModel(npar, name)

    def test_name_fails_when_number(self):
        npar = 2
        name = 2
        p = ParametricModel(npar, name)

class TestConstModel(object):

    def setUp(self):
        self.x = np.arange(1000)
        self.const = Const()

    def test_shape(self):
        a = 2.0
        assert self.const(self.x,a).shape == self.x.shape

    def test_value(self):
        a = 2.0
        all(self.const(self.x, a)) == a

    @raises(AssertionError)
    def test_func_fails_when_nan(self):
        a = np.nan
        self.const(self.x, a)

    @raises(AssertionError)
    def test_func_fails_when_inf(self):
        a = np.inf
        self.const(self.x, a)

    @raises(AttributeError)
    def test_hyperparameters_not_set(self):
        self.logprior

    def test_hyperparamers(self):
        hyperpars = {"a_mean":2.0, "a_var":0.2}
        self.const.set_prior(hyperpars)

    def test_sensible_parameter_is_finite(self):
        hyperpars = {"a_mean":2.0, "a_var":0.2}
        self.const.set_prior(hyperpars)
        assert self.const.logprior(2.0) > 0.1
        assert self.const.logprior(2.0) < 1.0
        assert np.isfinite(self.const.logprior(2.0))

    def test_crazy_parameter_returns_logmin(self):
        hyperpars = {"a_mean":2.0, "a_var":0.2}
        self.const.set_prior(hyperpars)
        crazy_par = 200.0
        assert np.isclose(self.const.logprior(crazy_par), logmin)

    @raises(AssertionError)
    def test_inf_pars_fails_prior(self):
        hyperpars = {"a_mean":2.0, "a_var":0.2}
        self.const.set_prior(hyperpars)
        self.const.logprior(np.inf)

    @raises(AssertionError)
    def test_nan_pars_fails_prior(self):
        hyperpars =  {"a_mean":2.0, "a_var":0.2}
        self.const.set_prior(hyperpars)
        self.const.logprior(np.nan)


class TestPowerLawModel(object):

    def setUp(self):
        self.x = np.linspace(0.1, 10, 100)
        self.pl = PowerLaw()

    def test_shape(self):
        alpha = 2.0
        amplitude = 3.0
        assert self.pl(self.x, alpha, amplitude).shape == self.x.shape

    def test_value(self):
        pl_eqn = lambda x, i, a: np.exp(-i*np.log(x) + a)

        alpha = 2.0
        amplitude = 3.0

        for x in range(1,10):
            eq_(pl_eqn(x, alpha, amplitude), self.pl(x, alpha, amplitude))

    @raises(AssertionError)
    def test_func_fails_when_not_finite(self):
        x = np.linspace(0,10, 100)
        for p1 in [2.0, np.nan, np.inf]:
            for p2 in [np.inf, np.nan]:
                self.pl(x, p1, p2)

    @raises(AttributeError)
    def test_hyperparameters_not_set(self):
        self.pl.logprior

    def test_hyperparameters(self):
        #hyperparameters
        hyperpars = {"alpha_min":1.0, "alpha_max":5.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.pl.set_prior(hyperpars)
        self.pl.logprior(2.0,2.0)

    def test_prior_works(self):
        hyperpars = {"alpha_min":1.0, "alpha_max":5.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.pl.set_prior(hyperpars)
        prior_test = self.pl.logprior(2.0, 2.0)
        print("prior_test: " + str(prior_test))
        assert np.isfinite(prior_test)
        assert prior_test > logmin

        prior_test = self.pl.logprior(-1.0, 2.0)
        assert prior_test == logmin

        prior_test = self.pl.logprior(2.0, -6.0)
        assert prior_test == logmin

        prior_test = self.pl.logprior(6.0, 6.0)
        assert prior_test == logmin

    @raises(AssertionError)
    def test_nonfinite_pars_fails_prior(self):
        hyperpars = {"alpha_min":1.0, "alpha_max":5.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.pl.set_prior(hyperpars)
        for p1 in [2.0, np.inf, np.nan]:
            for p2 in [np.nan, np.inf]:
                self.pl.logprior(p1, p2)


class TestBentPowerLawModel(object):

    def setUp(self):
        self.x = np.linspace(0.1,10, 100)
        self.bpl = BrokenPowerLaw()

    def test_shape(self):
        alpha1 = 1.0
        amplitude = 3.0
        alpha2 = 3.0
        x_break = 5.0

        c = self.bpl(self.x, alpha1, amplitude, alpha2, x_break)
        assert c.shape == self.x.shape

    def test_value(self):
        ## TODO: Need to write a meaningful test for this
        pass

    @raises(AssertionError)
    def test_func_fails_when_not_finite(self):
        for alpha1 in [2.0, np.nan, np.inf]:
            for alpha2 in [np.inf, np.nan]:
                for x_break in [2.0, np.nan, np.inf]:
                    for amplitude in [np.inf, np.nan]:
                        self.bpl(self.x, alpha1, alpha2, x_break, amplitude)

    @raises(AttributeError)
    def test_hyperparameters_not_set(self):
        self.bpl.logprior

    def test_hyperparameters(self):
        #hyperparameters
        hyperpars = {"alpha1_min":1.0, "alpha1_max":5.0,
                     "alpha2_min":1.0, "alpha2_max":5.0,
                     "x_break_min": -2.0, "x_break_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.bpl.set_prior(hyperpars)
        self.bpl.logprior(2.0, 2.0, 0.0, 1.0)

    def test_prior_works(self):
        hyperpars = {"alpha1_min":1.0, "alpha1_max":5.0,
                     "alpha2_min":1.0, "alpha2_max":5.0,
                     "x_break_min": -2.0, "x_break_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.bpl.set_prior(hyperpars)
        prior_test = self.bpl.logprior(2.0, 2.0, 0.0, 1.0)
        print("prior_test: " + str(prior_test))
        assert np.isfinite(prior_test)
        assert prior_test > logmin

        prior_test = self.bpl.logprior(-1.0, 2.0, 0.0, 1.0)
        assert prior_test == logmin

        prior_test = self.bpl.logprior(2.0, 6.0, 0.0, 1.0)
        assert prior_test == logmin

        prior_test = self.bpl.logprior(2.0, 2.0, -3.0, 1.0)
        assert prior_test == logmin

        prior_test = self.bpl.logprior(2.0, 2.0, 0.0, 10.0)
        assert prior_test == logmin

    @raises(AssertionError)
    def test_nonfinite_pars_fails_prior(self):
        hyperpars = {"alpha1_min":1.0, "alpha1_max":5.0,
                     "alpha2_min":1.0, "alpha2_max":5.0,
                     "x_break_min": -2.0, "x_break_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.bpl.set_prior(hyperpars)
        for alpha1 in [2.0, np.nan, np.inf]:
            for alpha2 in [np.inf, np.nan]:
                for x_break in [2.0, np.nan, np.inf]:
                    for amplitude in [np.inf, np.nan]:
                        self.bpl.logprior(alpha1, alpha2, x_break, amplitude)


class TestLorentzianModel(object):

    def setUp(self):
        self.x = np.linspace(0.1, 10., 100)

        self.lorentzian = Lorentzian()

    def test_shape(self):
        x0 = 200.0
        gamma = 1.0
        amplitude = 2.0

        c = self.lorentzian(self.x, x0, gamma, amplitude)
        assert c.shape == self.x.shape

    def test_value(self):
        gamma = 1.0
        amplitude = 2.0
        x0 = 200.0

        qpo_func = lambda x, g, amp, cen: np.exp(amp)/(np.pi*np.exp(g))*0.5/\
                                          ((x-cen)**2.0+(0.5*np.exp(g))**2.0)

        for x in range(1, 20):
            assert np.allclose(qpo_func(x, gamma, amplitude, x0),
                               self.lorentzian(x, x0, gamma, amplitude),
                               atol=1.e-10)
    
    @raises(AssertionError)
    def test_func_fails_when_not_finite(self):
        for gamma in [2.0, np.nan, np.inf]:
            for amplitude in [np.inf, np.nan]:
                for x0 in [2.0, np.nan, np.inf]:
                    self.lorentzian(self.x, x0, gamma, amplitude)

    @raises(AttributeError)
    def test_hyperparameters_not_set(self):
        self.lorentzian.logprior

    def test_hyperparameters(self):
        #hyperparameters
        hyperpars = {"x0_min":1.0, "x0_max":5.0,
                     "gamma_min":-2.0, "gamma_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.lorentzian.set_prior(hyperpars)
        self.lorentzian.logprior(2.0, -1.0, 1.0)

    @raises(AssertionError)
    def test_func_fails_when_not_finite(self):
        for gamma in [2.0, np.nan, np.inf]:
            for amplitude in [np.inf, np.nan]:
                for x0 in [2.0, np.nan, np.inf]:
                    self.lorentzian(self.x, x0, gamma, amplitude)

    def test_prior_works(self):
        hyperpars = {"x0_min":1.0, "x0_max":5.0,
                     "gamma_min":-2.0, "gamma_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.lorentzian.set_prior(hyperpars)
        prior_test = self.lorentzian.logprior(2.0, -1.0, 1.0)
        assert np.isfinite(prior_test)
        assert prior_test > logmin

        prior_test = self.lorentzian.logprior(-1.0, -1.0, 1.0)
        assert prior_test == logmin

        prior_test = self.lorentzian.logprior(2.0, 3.0, 1.0)
        assert prior_test == logmin

        prior_test = self.lorentzian.logprior(2.0, -1.0, -10.0)
        assert prior_test == logmin

    @raises(AssertionError)
    def test_nonfinite_pars_fails_prior(self):
        hyperpars = {"x0_min":1.0, "x0_max":5.0,
                     "gamma_min":-2.0, "gamma_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.lorentzian.set_prior(hyperpars)
        for x0 in [2.0, np.nan, np.inf]:
            for gamma in [np.inf, np.nan]:
                for amplitude in [np.inf, np.nan]:
                    self.lorentzian.logprior(x0, gamma, amplitude)


class TestFixedCentroidLorentzianModel(object):

    def setUp(self):
        self.x = np.linspace(0.1, 10., 100)
        self.x0 = 10.0
        self.fcl = FixedCentroidLorentzian(x0=self.x0)

    @raises(AssertionError)
    def test_x0_is_finite(self):
        for x0 in [np.nan, np.inf, -np.inf]:
            fcl = FixedCentroidLorentzian(x0)

    def test_shape(self):
        gamma = 1.0
        amplitude = 2.0

        c = self.fcl(self.x, gamma, amplitude)
        assert c.shape == self.x.shape

    def test_value(self):
        gamma = 1.0
        amplitude = 2.0

        qpo_func = lambda x, g, amp, cen: np.exp(amp)/(np.pi*np.exp(g))*0.5/\
                                          ((x-cen)**2.0+(0.5*np.exp(g))**2.0)
        for x in range(1, 20):
            print(qpo_func(x, gamma, amplitude, self.x0))
            print(self.fcl(x, gamma, amplitude))
            assert np.allclose(qpo_func(x, gamma, amplitude, self.x0),
                               self.fcl(x, gamma, amplitude),
                               atol=1.e-6)

    @raises(AssertionError)
    def test_func_fails_when_not_finite(self):
        for gamma in [2.0, np.nan, np.inf]:
            for amplitude in [np.inf, np.nan]:
                    self.fcl(self.x, gamma, amplitude)

    @raises(AttributeError)
    def test_hyperparameters_not_set(self):
        self.fcl.logprior()

    def test_hyperparameters(self):
        #hyperparameters
        hyperpars = {"gamma_min":-2.0, "gamma_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.fcl.set_prior(hyperpars)
        self.fcl.logprior(-1.0, 1.0)

    def test_prior_works(self):
        hyperpars = {"gamma_min":-2.0, "gamma_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.fcl.set_prior(hyperpars)

        prior_test = self.fcl.logprior(-1.0, 1.0)
        assert np.isfinite(prior_test)
        assert prior_test > logmin

        prior_test = self.fcl.logprior(-3.0, 1.0)
        assert prior_test == logmin

        prior_test = self.fcl.logprior(1.0, 10.0)
        assert prior_test == logmin

    @raises(AssertionError)
    def test_nonfinite_pars_fails_prior(self):
        hyperpars = {"gamma_min":-2.0, "gamma_max":2.0,
                     "amplitude_min":-5.0, "amplitude_max":5.0}
        self.fcl.set_prior(hyperpars)
        for gamma in [np.inf, np.nan]:
            for amplitude in [np.inf, np.nan]:
                self.fcl.logprior(gamma, amplitude)


## TODO: Need to write tests for PowerLawConst and BrokenPowerLawConst

class TestCombinedModels(object):

    def setUp(self):
        self.x = np.arange(1000)
        ## number of parameters for the different models
        self.npar_const = 1
        self.npar_powerlaw = 2
        self.npar_bentpowerlaw = 4
        self.npar_lorentzian = 3

    def npar_equal(self, model1, model2):
        mod = CombinedModel([model1, model2])
        npar_model1 = self.__getattribute__("npar_"+mod.name[0])
        npar_model2 = self.__getattribute__("npar_"+mod.name[1])
        eq_(mod.npar, npar_model1+npar_model2)

    def test_model(self):
        models = [Const,
                PowerLaw,
                BrokenPowerLaw,
                Lorentzian]

        for m1 in models:
            for m2 in models:
                self.npar_equal(m1, m2)


class TestConstPrior(object):

    def setUp(self):
        self.hyperpars = {"a_mean": 2.0, "a_var": 0.1}
        self.const = Const(self.hyperpars)

    def test_prior_nonzero(self):
        a = 2.0
        assert self.const.logprior(a) > logmin

    def test_prior_zero(self):
        a = 100.0
        assert self.const.logprior(a) == logmin


class TestPowerlawPrior(object):
    def setUp(self):
        self.hyperpars = {"alpha_min":-8.0, "alpha_max":5.0,
                          "amplitude_min": -10.0, "amplitude_max":10.0}

        alpha_norm = 1.0/(self.hyperpars["alpha_max"]-
                          self.hyperpars["alpha_min"])
        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-
                              self.hyperpars["amplitude_min"])
        self.prior_norm = np.log(alpha_norm*amplitude_norm)

        self.pl = parametricmodels.PowerLaw(self.hyperpars)

    def test_prior_nonzero(self):
        alpha = 1.0
        amplitude = 2.0
        print(self.pl)
        assert self.pl.logprior(alpha, amplitude) == self.prior_norm

    def prior_zero(self, alpha, amplitude):
        assert self.pl.logprior(alpha, amplitude) == logmin

    def generate_prior_zero_tests(self):
        alpha_all = [1.0, 10.0]
        amplitude_all = [-20.0, 2.0]
        for alpha, amplitude in zip(alpha_all, amplitude_all):
            yield self.prior_zero, alpha, amplitude


class TestBentPowerLawPrior(object):

    def setUp(self):

        self.hyperpars = {"alpha1_min": -8.0, "alpha1_max":5.0,
                 "amplitude_min": -10., "amplitude_max":10.0,
                 "alpha2_min":-8.0, "alpha2_max":4.0,
                 "x_break_min":np.log(0.1), "x_break_max":np.log(500)}

        alpha1_norm = 1.0/(self.hyperpars["alpha1_max"]-
                           self.hyperpars["alpha1_min"])

        alpha2_norm = 1.0/(self.hyperpars["alpha2_max"]-
                           self.hyperpars["alpha2_min"])

        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-
                              self.hyperpars["amplitude_min"])

        x_break_norm = 1.0/(self.hyperpars["x_break_max"]-
                            self.hyperpars["x_break_min"])

        self.prior_norm = np.log(alpha1_norm*alpha2_norm*
                                 amplitude_norm*x_break_norm)

        self.bpl = BrokenPowerLaw(self.hyperpars)


    def zero_prior(self, alpha1, amplitude, alpha2, x_break):
        assert self.bpl.logprior(alpha1, alpha2, x_break, amplitude) == logmin

    def nonzero_prior(self, alpha1, amplitude, alpha2, x_break):
        assert self.bpl.logprior(alpha1, alpha2, x_break, amplitude) == \
               self.prior_norm


    def test_prior(self):

        alpha1 = [1.0, 10.0]
        alpha2 = [1.0, 10.0]
        amplitude = [2.0, -20.0]
        x_break = [np.log(50.0), np.log(1000.0)]

        for i, a1 in enumerate(alpha1):
            for j, amp in enumerate(amplitude):
                for k, a2 in enumerate(alpha2):
                    for l, br in enumerate(x_break):
                        if i == 1 or j == 1 or k == 1 or l == 1:
                            yield self.zero_prior, a1, amp, a2, br
                        else:
                            yield self.nonzero_prior, a1, amp, a2, br


class TestLorentzianPrior(object):

    def setUp(self):

        self.hyperpars = {"gamma_min":-1.0, "gamma_max":5.0,
                     "amplitude_min":-10.0, "amplitude_max":10.0,
                     "x0_min":0.0, "x0_max":100.0}

        gamma_norm = 1.0/(self.hyperpars["gamma_max"]-
                          self.hyperpars["gamma_min"])

        amplitude_norm = 1.0/(self.hyperpars["amplitude_max"]-
                              self.hyperpars["amplitude_min"])

        x0_norm = 1.0/(self.hyperpars["x0_max"]-self.hyperpars["x0_min"])
        self.prior_norm = np.log(gamma_norm*amplitude_norm*x0_norm)
        self.lorentzian = Lorentzian(self.hyperpars)

    def zero_prior(self, gamma, amplitude, x0):
        assert self.lorentzian.logprior(x0, gamma, amplitude) == logmin

    def nonzero_prior(self, gamma, amplitude, x0):
        assert self.lorentzian.logprior(x0, gamma, amplitude) == self.prior_norm

    def test_prior(self):

        gamma = [2.0, -10.0]
        amplitude = [5.0, -20.0]
        x0 = [10.0, -5.0]

        for i,g in enumerate(gamma):
            for j,a in enumerate(amplitude):
                for k, x in enumerate(x0):
                    pars = [g, a, x]
                    if i == 1 or j == 1 or k == 1:
                        yield self.zero_prior, g, a, x
                    else:
                        yield self.nonzero_prior, g, a, x
