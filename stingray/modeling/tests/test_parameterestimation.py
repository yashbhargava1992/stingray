import numpy as np
import scipy.stats

from astropy.tests.helper import pytest
from astropy.modeling import models
from astropy.modeling.fitting import _fitter_to_model_params

from stingray import Powerspectrum

from stingray.modeling import ParameterEstimation, PSDParEst, OptimizationResults
from stingray.modeling import PSDPosterior, set_logprior

from statsmodels.tools.numdiff import approx_hess

class TestParameterEstimation(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise*2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        cls.priors = {"amplitude":p_amplitude}
        cls.lpost = PSDPosterior(cls.ps, cls.model)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

    def test_par_est_initializes(self):
        pe = ParameterEstimation()


    def test_parest_stores_max_post_correctly(self):
        """
        Make sure the keyword for Maximum A Posteriori fits is stored correctly
        as a default.
        """
        pe = ParameterEstimation()

        assert pe.max_post is True, "max_post should be set to True as a default."

    def test_fit_fails_when_object_is_not_posterior_or_likelihood(self):
        x = np.ones(10)
        y = np.ones(10)
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit(x, y)

    def test_fit_fails_without_lpost_or_t0(self):
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit()

    def test_fit_fails_without_t0(self):
        pe = ParameterEstimation()
        with pytest.raises(TypeError):
            res = pe.fit(np.ones(10))

    def test_fit_fails_with_incorrect_number_of_parameters(self):
        pe = ParameterEstimation()
        t0 = [1,2]
        with pytest.raises(ValueError):
            res = pe.fit(self.lpost, t0)

    def test_fit_method_works_with_correct_parameter(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)

    def test_fit_method_returns_optimization_results_object(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)
        assert isinstance(res, OptimizationResults), "res must be of type OptimizationResults"

    def test_compute_lrt_fails_when_garbage_goes_in(self):
        pe = ParameterEstimation()
        t0 = [2.0]

        with pytest.raises(TypeError):
            pe.compute_lrt(self.lpost, t0, None, t0)

        with pytest.raises(ValueError):
            pe.compute_lrt(self.lpost, t0[:-1], self.lpost, t0)

    def test_compute_lrt_sets_max_post_to_false(self):
        t0 = [2.0]
        pe = ParameterEstimation(max_post=True)

        assert pe.max_post is True
        delta_deviance = pe.compute_lrt(self.lpost, t0, self.lpost, t0)

        assert pe.max_post is False

    def test_compute_lrt_computes_deviance_correctly(self):

        t0 = [2.0]
        pe = ParameterEstimation()

        delta_deviance = pe.compute_lrt(self.lpost, t0, self.lpost, t0)

        assert delta_deviance < 1e-7


    def test_sampler_runs(self):

        pe = ParameterEstimation()
        sample_res = pe.sample(self.lpost, [2.0], nwalkers=200, niter=100,
                               burnin=100, print_results=True, plot=False)


        assert sample_res.acceptance > 0.25





class TestOptimizationResults(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise*2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0

        cls.model = models.Const1D()

        p_amplitude = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        cls.priors = {"amplitude":p_amplitude}
        cls.lpost = PSDPosterior(cls.ps, cls.model)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "BFGS"
        cls.max_post = True
        cls.t0 = [2.0]
        cls.neg = True
        cls.opt = scipy.optimize.minimize(cls.lpost, cls.t0,
                                          method=cls.fitmethod,
                                          args=cls.neg, tol=1.e-10)

    def test_object_initializes(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

    def test_object_has_right_attributes(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        assert hasattr(res, "p_opt")
        assert hasattr(res, "result")
        assert hasattr(res, "deviance")
        assert hasattr(res, "aic")
        assert hasattr(res, "bic")
        assert hasattr(res, "model")
        assert isinstance(res.model, models.Const1D)

    def test_p_opt_is_correct(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert res.p_opt == self.opt.x, "res.p_opt must be the same as opt.x!"
        assert np.isclose(res.p_opt[0], 2.0, atol=0.1, rtol=0.1)

    def test_model_is_same_as_in_lpost(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert res.model == self.lpost.model

    def test_result_is_same_as_in_opt(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        assert res.result == self.opt.fun

    def test_compute_model_works_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)
        mean_model = np.ones_like(self.lpost.x)*self.opt.x[0]
        assert np.all(res.mfit == mean_model), "res.model should be exactly " \
                                               "the model for the data."

    def test_compute_criteria_computes_criteria_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        test_aic = 1694708.7566869266
        test_bic = 1694720.5721974846
        test_deviance = 3389411.675487054

        assert np.isclose(res.aic, test_aic, atol=0.1, rtol=0.1)
        assert np.isclose(res.bic, test_bic, atol=0.1, rtol=0.1)
        assert np.isclose(res.deviance, test_deviance, atol=0.1, rtol=0.1)

    def test_merit_calculated_correctly(self):
        res = OptimizationResults(self.lpost, self.opt, neg=self.neg)

        test_merit = 999563.81710186403
        assert np.isclose(res.merit, test_merit, atol=0.1, rtol=0.1)

    def test_res_is_of_correct_type(self):
        pe = ParameterEstimation()
        t0 = [2.0]
        res = pe.fit(self.lpost, t0)

        assert isinstance(res, OptimizationResults)


class OptimizationResultsSubclassDummy(OptimizationResults):

    def __init__(self, lpost, res, neg):
        self.neg = neg
        self.result = res.fun
        self.p_opt = res.x
        self.model = lpost.model


class TestOptimizationResultInternalFunctions(object):

    @classmethod
    def setup_class(cls):
        m = 1
        nfreq = 1000000
        freq = np.linspace(1, 1000, nfreq)

        np.random.seed(100) # set the seed for the random number generator
        noise = np.random.exponential(size=nfreq)

        cls.model = models.PowerLaw1D() + models.Const1D()
        cls.model.x_0_0.fixed = True

        cls.alpha_0 = 2.0
        cls.amplitude_0 = 100.0
        cls.amplitude_1 = 2.0

        cls.model.alpha_0 = cls.alpha_0
        cls.model.amplitude_0 = cls.amplitude_0
        cls.model.amplitude_1 = cls.amplitude_1

        p = cls.model(freq)
        power = noise*p

        ps = Powerspectrum()
        ps.freq = freq
        ps.power = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        cls.ps = ps
        cls.a_mean, cls.a_var = 2.0, 1.0
        cls.a2_mean, cls.a2_var = 100.0, 10.0

        p_amplitude_1 = lambda amplitude: \
            scipy.stats.norm(loc=cls.a_mean, scale=cls.a_var).pdf(amplitude)

        p_alpha_0 = lambda alpha: \
            scipy.stats.uniform(0.0, 5.0).pdf(alpha)

        p_amplitude_0 = lambda amplitude: \
            scipy.stats.norm(loc=cls.a2_mean, scale=cls.a2_var).pdf(amplitude)

        cls.priors = {"amplitude_1": p_amplitude_1,
                      "amplitude_0": p_amplitude_0,
                      "alpha_0": p_alpha_0}

        cls.lpost = PSDPosterior(cls.ps, cls.model)
        cls.lpost.logprior = set_logprior(cls.lpost, cls.priors)

        cls.fitmethod = "BFGS"
        cls.max_post = True
        cls.t0 = [cls.amplitude_0, cls.alpha_0, cls.amplitude_1]
        cls.neg = True
        cls.opt = scipy.optimize.minimize(cls.lpost, cls.t0,
                                          method=cls.fitmethod,
                                          args=cls.neg, tol=1.e-5)


        cls.optres = OptimizationResultsSubclassDummy(cls.lpost, cls.opt,
                                                      neg=True)


    def test_compute_model(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_model(self.lpost)

        assert hasattr(optres, "mfit"), "OptimizationResult object should have mfit " \
                            "attribute at this point!"

        _fitter_to_model_params(self.model, self.opt.x)
        mfit_test = self.model(self.lpost.x)

        assert np.all(optres.mfit == mfit_test)

    def test_compute_statistics_computes_mfit(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        assert hasattr(optres, "mfit") is False
        optres._compute_statistics(self.lpost)

        assert hasattr(optres, "mfit")

    def test_compute_statistics_computes_all_statistics(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_statistics(self.lpost)

        assert hasattr(optres, "merit")
        assert hasattr(optres, "dof")
        assert hasattr(optres, "sexp")
        assert hasattr(optres, "ssd")
        assert hasattr(optres, "sobs")

    def test_compute_statistics_returns_correct_values(self):
        test_merit = 1000377.6603212412
        test_dof = 999997.0
        test_sexp = 6000000.0
        test_ssd =  3464.1016151377544
        test_sobs = 819.79384402855862

        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_statistics(self.lpost)

        assert np.isclose(test_merit, optres.merit, atol=0.01, rtol=0.01)
        assert test_dof == optres.dof
        assert test_sexp == optres.sexp
        assert test_ssd == optres.ssd
        assert np.isclose(test_sobs, optres.sobs, atol=0.01, rtol=0.01)


    def test_compute_criteria_returns_correct_attributes(self):

        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_criteria(self.lpost)

        assert hasattr(optres, "aic")
        assert hasattr(optres, "bic")
        assert hasattr(optres, "deviance")

    def test_compute_criteria_returns_correct_values(self):
        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_criteria(self.lpost)

        test_aic = 1708892.3207432728
        test_bic = 1708927.7672749467
        test_deviance = 3417761.061446513

        assert np.isclose(test_aic, optres.aic)
        assert np.isclose(test_bic, optres.bic)
        assert np.isclose(test_deviance, optres.deviance)

    def test_compute_covariance_with_hess_inverse(self):

        optres = OptimizationResultsSubclassDummy(self.lpost, self.opt,
                                                  neg=True)

        optres._compute_covariance(self.lpost, self.opt)

        assert np.all(optres.cov == np.asarray(self.opt.hess_inv))
        assert np.all(optres.err == np.sqrt(np.diag(self.opt.hess_inv)))

    def test_compute_covariance_without_hess_inverse(self):

        fitmethod = "powell"
        opt = scipy.optimize.minimize(self.lpost, self.t0,
                                      method=fitmethod,
                                      args=self.neg, tol=1.e-10)

        optres = OptimizationResultsSubclassDummy(self.lpost, opt,
                                                  neg=True)

        optres._compute_covariance(self.lpost, opt)

        phess = approx_hess(opt.x, self.lpost)
        hess_inv = np.linalg.inv(phess)

        assert np.all(optres.cov == hess_inv)
        assert np.all(optres.err == np.sqrt(np.diag(np.abs(hess_inv))))

