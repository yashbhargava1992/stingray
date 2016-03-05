import numpy as np
import scipy.stats

from nose.tools import eq_, raises

from stingray import Powerspectrum
from stingray import Posterior, PSDPosterior
from stingray import Const

np.random.seed(20150907)



class PosteriorClassDummy(Posterior):
    """
    This is a test class that tests the basic functionality of the
    Posterior superclass.
    """
    def __init__(self, x, y, model):
        Posterior.__init__(self, x, y, model)


    def  loglikelihood(self, t0, neg=False):
        loglike = 1.0
        return loglike

    def logprior(self, t0):
        lp = 2.0
        return lp


class TestPosterior(object):

    def setUp(self):
        self.x = np.arange(100)
        self.y = np.ones(self.x.shape[0])
        self.model = Const(hyperpars={"a_mean":2.0, "a_var":1.0})
        self.p = PosteriorClassDummy(self.x,self.y,self.model)


    def test_inputs(self):
        assert np.allclose(self.p.x, self.x)
        assert np.allclose(self.p.y, self.y)
        assert isinstance(self.p.model, Const)

    def test_call_method_positive(self):
        t0 = [1,2,3]
        post = self.p(t0, neg=False)
        assert post == 3.0

    def test_call_method_negative(self):
        t0 = [1,2,3]
        post = self.p(t0, neg=True)
        assert post == -3.0




class TestPSDPosterior(object):

    def setUp(self):
        m = 1
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = np.random.exponential(size=nfreq)
        power = noise*2.0

        ps = Powerspectrum()
        ps.freq = freq
        ps.ps = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        self.ps = ps
        self.a_mean, self.a_var = 2.0, 1.0

        self.model = Const(hyperpars={"a_mean":self.a_mean, "a_var":self.a_var})


    @raises(AssertionError)
    def test_logprior_fails_without_prior(self):
        model = Const()
        lpost = PSDPosterior(self.ps, model)
        lpost.logprior([1])

    def test_making_posterior(self):
        lpost = PSDPosterior(self.ps, self.model)
        #print(lpost.x)
        #print(self.ps.freq)
        assert lpost.x.all() == self.ps.freq.all()
        assert lpost.y.all() == self.ps.ps.all()

    @raises(AssertionError)
    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps, self.model)
        lpost([2,3])


    @raises(AssertionError)
    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps, self.model)
        lpost([2,3])


    @raises(AssertionError)
    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps, self.model)
        lpost([2,3])


    @raises(AssertionError)
    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps, self.model)
        lpost([2,3])


    @raises(AssertionError)
    def test_correct_number_of_parameters(self):
        lpost = PSDPosterior(self.ps, self.model)
        lpost([2,3])


    def test_logprior(self):
        t0 = [2.0]

        lpost = PSDPosterior(self.ps, self.model)
        lp_test = lpost.logprior(t0)
        lp = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        assert lp == lp_test

    def test_loglikelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        loglike = -np.sum(self.ps.ps[1:]/m + np.log(m))

        lpost = PSDPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike, loglike_test)

    @raises(AssertionError)
    def test_loglikelihood_correctly_leaves_out_zeroth_freq(self):
        t0 = [2.0]
        m = self.model(self.ps.freq, t0)
        loglike = -np.sum(self.ps.ps/m + np.log(m))

        lpost = PSDPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=False)

        assert np.isclose(loglike_test, loglike, atol=1.e-10, rtol=1.e-10)



    def test_negative_loglikelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        loglike = np.sum(self.ps.ps[1:]/m + np.log(m))

        lpost = PSDPosterior(self.ps, self.model)
        loglike_test = lpost.loglikelihood(t0, neg=True)

        assert np.isclose(loglike, loglike_test)


    def test_posterior(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        lpost = PSDPosterior(self.ps, self.model)
        post_test = lpost(t0, neg=False)

        loglike = -np.sum(self.ps.ps[1:]/m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = loglike + logprior

        assert np.isclose(post_test, post, atol=1.e-10)

    def test_negative_posterior(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        lpost = PSDPosterior(self.ps, self.model)
        post_test = lpost(t0, neg=True)

        loglike = -np.sum(self.ps.ps[1:]/m + np.log(m))
        logprior = np.log(scipy.stats.norm(2.0, 1.0).pdf(t0))
        post = -loglike - logprior

        assert np.isclose(post_test, post, atol=1.e-10)

class TestPerPosteriorAveragedPeriodogram(object):

    def setUp(self):
        m = 10
        nfreq = 1000000
        freq = np.arange(nfreq)
        noise = scipy.stats.chi2(2.*m).rvs(size=nfreq)/np.float(m)
        power = noise

        ps = Powerspectrum()
        ps.freq = freq
        ps.ps = power
        ps.m = m
        ps.df = freq[1]-freq[0]
        ps.norm = "leahy"

        self.ps = ps
        self.a_mean, self.a_var = 2.0, 1.0

        self.model = Const(hyperpars={"a_mean":self.a_mean, "a_var":self.a_var})

    def test_likelihood(self):
        t0 = [2.0]
        m = self.model(self.ps.freq[1:], t0)
        ## TODO: Finish this test!

## TODO: Write tests for Lightcurve posterior
## TODO: Write tests for Gaussian posterior
