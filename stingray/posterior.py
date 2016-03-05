__all__ = ["Posterior", "PSDPosterior",
           "LightcurvePosterior", "GaussianPosterior"]

import numpy as np
from scipy.special import gamma as scipy_gamma

## TODO: Find out whether there is a gamma function in numpy!

from stingray import Powerspectrum, AveragedPowerspectrum
from stingray.parametricmodels import logmin

class Posterior(object):

    def __init__(self,x, y, model):
        """



        """
        self.x = x
        self.y = y

        ### model is a parametric model
        self.model = model

    def logprior(self, t0):
        """
        The logarithm of the prior distribution for the
        model defined in self.model.

        Parameters:
        ------------
        t0: {list | numpy.ndarray}
            The list with parameters for the model

        Returns:
        --------
        logp: float
            The logarithm of the prior distribution for the model and
            parameters given.
        """
        assert hasattr(self.model, "logprior")
        assert np.size(t0) == self.model.npar, "Input parameters must " \
                                               "match model parameters!"

        return self.model.logprior(*t0)


    ### use standard definition of the likelihood as the product of all
    def loglikelihood(self, t0, neg=False):
        print("If you're calling this method, something is wrong!")
        return 0.0

    def logposterior(self, t0, neg=False):
        lpost = self.loglikelihood(t0) + self.logprior(t0)
        if neg == True:
            return -lpost
        else:
            return lpost

    def __call__(self, t0, neg=False):
        return self.logposterior(t0, neg=neg)


class PSDPosterior(Posterior):

    def __init__(self, ps, model):
        """
        Posterior distribution for power spectra.
        Uses an exponential distribution for the errors in the likelihood,
        or a $\chi^2$ distribution with $2M$ degrees of freedom, where $M$ is
        the number of frequency bins or power spectra averaged in each bin.


        Parameters
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.


        Attributes
        ----------
        ps: {Powerspectrum | AveragedPowerspectrum} instance
            the Powerspectrum object containing the data

        m: int, optional, default is 1
            The number of averaged periodograms or frequency bins in ps.
            Useful for binned/averaged periodograms, since the value of
            m will change the likelihood function!

        x: numpy.ndarray
            The independent variable (list of frequencies) stored in ps.freq

        y: numpy.ndarray
            The dependent variable (list of powers) stored in ps.ps

        model: instance of any subclass of parameterclass.ParametricModel
               The model for the power spectrum. Note that in order to define
               the posterior properly, the ParametricModel subclass must be
               instantiated with the hyperpars parameter set, or there won't
               be a prior to be calculated! If all this object is used
               for a maximum likelihood-style analysis, no prior is required.

        """
        self.m = ps.m
        Posterior.__init__(self, ps.freq, ps.ps, model)

    def loglikelihood(self, t0, neg=False):
        """
        The log-likelihood for the model defined in self.model
        and the parameters in t0. Uses an exponential model for
        the errors.

        Parameters:
        ------------
        t0: {list | numpy.ndarray}
            The list with parameters for the model

        Returns:
        --------
        logl: float
            The logarithm of the likelihood function for the model and
            parameters given.

        """
        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x[1:], *t0)

        if self.m == 1:
            res = -np.sum(np.log(funcval)) - np.sum(self.y[1:]/funcval)
        else:
            res = -2.0*self.m*(np.sum(np.log(funcval)) +
                               np.sum(self.y[1:]/funcval) +
                               np.sum((2.0/float(2.*self.m) - 1.0)*
                                      np.log(self.y[1:])))

        if np.isnan(res):
            res = logmin
        elif res == np.inf or np.isfinite(res) == False:
            res = logmin

        if neg:
            return -res
        else:
            return res


class LightcurvePosterior(Posterior):

    def __init__(self, lc, model):
        """
        Posterior for Poisson lightcurve data. Primary intended use is for
        modelling X-ray light curves, but alternative uses are conceivable.

        Parameters
        ----------
        lc: lightcurve.Lightcurve object
            Object containing the light curve to be modelled

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.
        Attributes
        ----------
        lc: lightcurve.Lightcurve instance
            the Lightcurve object containing the data

        x: numpy.ndarray
            The independent variable (list of frequencies) stored in ps.freq

        y: numpy.ndarray
            The dependent variable (list of powers) stored in ps.ps

        model: instance of any subclass of parameterclass.ParametricModel
               The model for the power spectrum. Note that in order to define
               the posterior properly, the ParametricModel subclass must be
               instantiated with the hyperpars parameter set, or there won't
               be a prior to be calculated! If all this object is used
               for a maximum likelihood-style analysis, no prior is required.

        """
        self.lc = lc
        Posterior.__init__(self, lc.time, lc.counts, model)



    def loglikelihood(self, t0, neg=False):

        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x[1:], *t0)

        res = -funcval + self.y*np.log(funcval) - scipy_gamma(self.y + 1.)

        if np.isnan(res):
            res = logmin
        elif res == np.inf or np.isfinite(res) == False:
            res = logmin

        if neg:
            return -res
        else:
            return res


class GaussianPosterior(Posterior):

    def __init__(self, x, y, model):
        """
        A general class for two-dimensional data following a Gaussian
        sampling distribution.

        Parameters
        ----------
        x: numpy.ndarray
            independent variable

        y: numpy.ndarray
            dependent variable

        model: instance of any subclass of parameterclass.ParametricModel
            The model for the power spectrum. Note that in order to define
            the posterior properly, the ParametricModel subclass must be
            instantiated with the hyperpars parameter set, or there won't
            be a prior to be calculated! If all this object is used
            for a maximum likelihood-style analysis, no prior is required.

        """
        Posterior.__init__(self, x, y, model)


    def loglikelihood(self, t0, neg=False):

        assert np.size(t0) == self.model.npar, "Input parameters must" \
                                               " match model parameters!"

        funcval = self.model(self.x[1:], *t0)

        #res = -funcval + self.y*np.log(funcval) - scipy_gamma(self.y + 1.)
        ## TODO: Add Gaussian likelihood
        res = 1.0


        if np.isnan(res):
            res = logmin
        elif res == np.inf or np.isfinite(res) == False:
            res = logmin

        if neg:
            return -res
        else:
            return res
