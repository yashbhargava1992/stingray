import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import functools
import tensorflow_probability.substrates.jax as tfp

from jax import jit, random

from tinygp import GaussianProcess, kernels
from stingray import Lightcurve

from jaxns import ExactNestedSampler
from jaxns import TerminationCondition

# from jaxns import analytic_log_evidence
from jaxns import Prior, Model

jax.config.update("jax_enable_x64", True)

tfpd = tfp.distributions
tfpb = tfp.bijectors

__all__ = ["GP", "GPResult"]


def get_kernel(kernel_type, kernel_params):
    """
    Function for producing the kernel for the Gaussian Process.
    Returns the selected Tinygp kernel

    Parameters
    ----------
    kernel_type: string
        The type of kernel to be used for the Gaussian Process
        To be selected from the kernels already implemented

    kernel_params: dict
        Dictionary containing the parameters for the kernel
        Should contain the parameters for the selected kernel

    """
    if kernel_type == "QPO_plus_RN":
        kernel = kernels.quasisep.Exp(
            scale=1 / kernel_params["crn"], sigma=(kernel_params["arn"]) ** 0.5
        ) + kernels.quasisep.Celerite(
            a=kernel_params["aqpo"],
            b=0.0,
            c=kernel_params["cqpo"],
            d=2 * jnp.pi * kernel_params["freq"],
        )
        return kernel
    elif kernel_type == "RN":
        kernel = kernels.quasisep.Exp(
            scale=1 / kernel_params["crn"], sigma=(kernel_params["arn"]) ** 0.5
        )
        return kernel


def get_mean(mean_type, mean_params):
    """
    Function for producing the mean for the Gaussian Process.

    Parameters
    ----------
    mean_type: string
        The type of mean to be used for the Gaussian Process
        To be selected from the mean functions already implemented

    mean_params: dict
        Dictionary containing the parameters for the mean
        Should contain the parameters for the selected mean

    """
    if mean_type == "gaussian":
        mean = functools.partial(_gaussian, mean_params=mean_params)
    elif mean_type == "exponential":
        mean = functools.partial(_exponential, mean_params=mean_params)
    elif mean_type == "constant":
        mean = functools.partial(_constant, mean_params=mean_params)
    return mean


def _gaussian(t, mean_params):
    return mean_params["A"] * jnp.exp(
        -((t - mean_params["t0"]) ** 2) / (2 * (mean_params["sig"] ** 2))
    )


def _exponential(t, mean_params):
    return mean_params["A"] * jnp.exp(-jnp.abs((t - mean_params["t0"])) / mean_params["sig"])


def _constant(t, mean_params):
    return mean_params["A"] * jnp.ones_like(t)


class GP:
    """
    Makes a GP object which takes in a Stingray.Lightcurve and fits a Gaussian
    Process on the lightcurve data, for the given kernel.

    Parameters
    ----------
    lc: Stingray.Lightcurve object
        The lightcurve on which the gaussian process, is to be fitted

    Model_type: string tuple
        Has two strings with the first being the name of the kernel type
        and the secound being the mean type

    Model_parameter: dict, default = None
        Dictionary conatining the parameters for the mean and kernel
        The keys should be accourding to the selected kernel and mean
        coressponding to the Model_type
        By default, it takes a value None, and the kernel and mean are
        then bulit using the pre-set parameters.

    Other Parameters
    ----------------
    kernel: class: `TinyGp.kernel` object
        The tinygp kernel for the GP

    mean: class: `TinyGp.mean` object
        The tinygp mean for the GP

    maingp: class: `TinyGp.GaussianProcess` object
        The tinygp gaussian process made on the lightcurve

    """

    def __init__(self, Lc: Lightcurve, Model_type: tuple, Model_params: dict = None) -> None:
        self.lc = Lc
        self.Model_type = Model_type
        self.Model_param = Model_params
        self.kernel = get_kernel(self.Model_type[0], self.Model_param)
        self.mean = get_mean(self.Model_type[1], self.Model_param)
        self.maingp = GaussianProcess(
            self.kernel, Lc.time, mean=self.mean, diag=Model_params["diag"]
        )

    def get_logprob(self):
        """
        Returns the logprobability of the lightcurves counts for the
        given kernel for the Gaussian Process
        """
        cond = self.maingp.condition(self.lc.counts)
        return cond.log_probability

    def get_model(self):
        """
        Returns the model of the Gaussian Process
        """
        return (self.Model_type, self.Model_param)

    def plot_kernel(self):
        """
        Plots the kernel of the Gaussian Process
        """
        X = self.lc.time
        Y = self.kernel(X, np.array([0.0]))
        plt.plot(X, Y)
        plt.xlabel("distance")
        plt.ylabel("Value")
        plt.title("Kernel Function")

    def plot_originalgp(self, sample_no=1, seed=0):
        """
        Plots samples obtained from the gaussian process for the kernel

        Parameters
        ----------
        sample_no: int , default = 1
            Number of GP samples to be taken

        """
        X_test = self.lc.time
        _, ax = plt.subplots(1, 1, figsize=(10, 3))
        y_samp = self.maingp.sample(jax.random.PRNGKey(seed), shape=(sample_no,))
        ax.plot(X_test, y_samp[0], "C0", lw=0.5, alpha=0.5, label="samples")
        ax.plot(X_test, y_samp[1:].T, "C0", lw=0.5, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("counts")
        ax.legend(loc="best")

    def plot_gp(self, sample_no=1, seed=0):
        """
        Plots gaussian process, conditioned on the lightcurve
        Also, plots the lightcurve along with it

        Parameters
        ----------
        sample_no: int , default = 1
            Number of GP samples to be taken

        """
        X_test = self.lc.time

        _, ax = plt.subplots(1, 1, figsize=(10, 3))
        _, cond_gp = self.maingp.condition(self.lc.counts, X_test)
        mu = cond_gp.mean
        # std = np.sqrt(cond_gp.variance)

        ax.plot(self.lc.time, self.lc.counts, lw=2, color="blue", label="Lightcurve")
        ax.plot(X_test, mu, "C1", label="Gaussian Process")
        y_samp = cond_gp.sample(jax.random.PRNGKey(seed), shape=(sample_no,))
        ax.plot(X_test, y_samp[0], "C0", lw=0.5, alpha=0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("counts")
        ax.legend(loc="best")


def get_prior(kernel_type, mean_type, **kwargs):
    """
    A prior generator function based on given values

    Parameters
    ----------
    kwargs:
        All possible keyword arguments to construct the prior.

    Returns
    -------
    The Prior function.
    The arguments of the prior function are in the order of
    Kernel arguments (RN arguments, QPO arguments),
    Mean arguments
    Non Windowed arguments

    """
    kwargs["T"] = kwargs["Times"][-1] - kwargs["Times"][0]  # Total time
    kwargs["f"] = 1 / (kwargs["Times"][1] - kwargs["Times"][0])  # Sampling frequency
    kwargs["min"] = jnp.min(kwargs["counts"])
    kwargs["max"] = jnp.max(kwargs["counts"])
    kwargs["span"] = kwargs["max"] - kwargs["min"]

    def RNprior_model():
        arn = yield Prior(tfpd.Uniform(0.1 * kwargs["span"], 2 * kwargs["span"]), name="arn")
        crn = yield Prior(tfpd.Uniform(jnp.log(1 / kwargs["T"]), jnp.log(kwargs["f"])), name="crn")

        A = yield Prior(tfpd.Uniform(0.1 * kwargs["span"], 2 * kwargs["span"]), name="A")
        t0 = yield Prior(
            tfpd.Uniform(
                kwargs["Times"][0] - 0.1 * kwargs["T"], kwargs["Times"][-1] + 0.1 * kwargs["T"]
            ),
            name="t0",
        )
        sig = yield Prior(tfpd.Uniform(0.5 * 1 / kwargs["f"], 2 * kwargs["T"]), name="sig")
        return arn, crn, A, t0, sig

    if (kernel_type == "RN") & ((mean_type == "gaussian") | (mean_type == "exponential")):
        return RNprior_model

    def QPOprior_model():
        arn = yield Prior(tfpd.Uniform(0.1 * kwargs["span"], 2 * kwargs["span"]), name="arn")
        crn = yield Prior(tfpd.Uniform(jnp.log(1 / kwargs["T"]), jnp.log(kwargs["f"])), name="crn")
        aqpo = yield Prior(tfpd.Uniform(0.1 * kwargs["span"], 2 * kwargs["span"]), name="aqpo")
        cqpo = yield Prior(tfpd.Uniform(1 / 10 / kwargs["T"], jnp.log(kwargs["f"])), name="cqpo")
        freq = yield Prior(tfpd.Uniform(2 / kwargs["T"], kwargs["f"] / 2), name="freq")

        A = yield Prior(tfpd.Uniform(0.1 * kwargs["span"], 2 * kwargs["span"]), name="A")
        t0 = yield Prior(
            tfpd.Uniform(
                kwargs["Times"][0] - 0.1 * kwargs["T"], kwargs["Times"][-1] + 0.1 * kwargs["T"]
            ),
            name="t0",
        )
        sig = yield Prior(tfpd.Uniform(0.5 * 1 / kwargs["f"], 2 * kwargs["T"]), name="sig")

        return arn, crn, aqpo, cqpo, freq, A, t0, sig

    if (kernel_type == "QPO_plus_RN") & ((mean_type == "gaussian") | (mean_type == "exponential")):
        return QPOprior_model


def get_likelihood(kernel_type, mean_type, **kwargs):
    """
    A likelihood generator function based on given values
    """

    @jit
    def RNlog_likelihood(arn, crn, A, t0, sig):
        rnlikelihood_params = {
            "arn": arn,
            "crn": crn,
            "aqpo": 0.0,
            "cqpo": 0.0,
            "freq": 0.0,
        }

        mean_params = {
            "A": A,
            "t0": t0,
            "sig": sig,
        }

        kernel = get_kernel(kernel_type="RN", kernel_params=rnlikelihood_params)

        mean = get_mean(mean_type=mean_type, mean_params=mean_params)

        gp = GaussianProcess(kernel, kwargs["Times"], mean=mean)
        return gp.log_probability(kwargs["counts"])

    if (kernel_type == "RN") & ((mean_type == "gaussian") | (mean_type == "exponential")):
        return RNlog_likelihood

    @jit
    def QPOlog_likelihood(arn, crn, aqpo, cqpo, freq, A, t0, sig):
        qpolikelihood_params = {
            "arn": arn,
            "crn": crn,
            "aqpo": aqpo,
            "cqpo": cqpo,
            "freq": freq,
        }

        mean_params = {
            "A": A,
            "t0": t0,
            "sig": sig,
        }

        kernel = get_kernel(kernel_type="RN", kernel_params=qpolikelihood_params)
        mean = get_mean(mean_type=mean_type, mean_params=mean_params)

        gp = GaussianProcess(kernel, kwargs["Times"], mean=mean)
        return gp.log_probability(kwargs["counts"])

    if (kernel_type == "QPO_plus_RN") & ((mean_type == "gaussian") | (mean_type == "exponential")):
        return QPOlog_likelihood


class GPResult:
    """
    Makes a GP regressor for a given GP class and a prior over it.
    Provides the sampled hyperparameters and tabulates their charachtersistics
    Using jaxns for nested sampling and evidence analysis

    Parameters
    ----------
    GP: class: GP
        The initial GP class, on which we will apply our regressor.

    prior_type: string tuple
        Has two strings with the first being the name of the kernel type
        and the secound being the mean type for the prior

    prior_parameters: dict, default = None
        Dictionary containing the parameters for the mean and kernel priors
        The keys should be accourding to the selected kernel and mean
        prior coressponding to the prior_type
        By default, it takes a value None, and the kernel and mean priors are
        then bulit using the pre-set parameters.

    Other Parameters
    ----------------
    lc: Stingray.Lightcurve object
        The lightcurve on which the gaussian process regression, is to be done

    """

    def __init__(self, GP: GP, prior_type: tuple, prior_parameters=None) -> None:
        self.gpclass = GP
        self.prior_type = prior_type
        self.prior_parameters = prior_parameters
        self.lc = GP.lc

    def run_sampling(self):
        """
        Runs a sampling process for the hyperparameters for the GP model.
        Based on No U turn Sampling from the numpyro module
        """

        dict = {"Times": self.lc.time, "counts": self.lc.counts}
        self.prior_model = get_prior(self.prior_type[0], self.prior_type[1], **dict)
        self.likelihood = get_likelihood(self.prior_type[0], self.prior_type[1], **dict)

        NSmodel = Model(prior_model=self.prior_model, log_likelihood=self.likelihood)

        NSmodel.sanity_check(random.PRNGKey(10), S=100)

        self.Exact_ns = ExactNestedSampler(NSmodel, num_live_points=500, max_samples=1e4)
        Termination_reason, State = self.Exact_ns(
            random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4)
        )
        self.Results = self.Exact_ns.to_results(State, Termination_reason)

    def print_summary(self):
        """
        Prints a summary table for the model parameters
        """
        self.Exact_ns.summary(self.Results)

    def plot_diagnostics(self):
        """
        Plots the diagnostic plots for the sampling process
        """
        self.Exact_ns.plot_diagnostics(self.Results)

    def corner_plot(self):
        """
        Plots the corner plot for the sampled hyperparameters
        """
        self.Exact_ns.plot_corner(self.Results)

    def get_parameters(self):
        """
        Returns the optimal parameters for the model based on the NUTS sampling
        """

        pass

    def plot_posterior(self, X_test):
        """
        Plots posterior gaussian process, conditioned on the lightcurve
        Also, plots the lightcurve along with it

        Parameters
        ----------
        X_test: jnp.array
            Array over which the Gaussian process values are to be obtained
            Can be made default with lc.times as default

        """

        pass
