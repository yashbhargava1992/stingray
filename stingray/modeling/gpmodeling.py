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

__all__ = ["GP"]


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
    elif mean_type == "skew_gaussian":
        mean = functools.partial(_skew_gaussian, mean_params=mean_params)
    elif mean_type == "skew_exponential":
        mean = functools.partial(_skew_exponential, mean_params=mean_params)
    elif mean_type == "fred":
        mean = functools.partial(_fred, mean_params=mean_params)
    return mean


def _gaussian(t, mean_params):
    """A gaussian flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.
    A:  jnp.int
        Amplitude of the flare.
    t0:
        The location of the maximum.
    sig1:
        The width parameter for the gaussian.

    Returns
    -------
    The y values for the gaussian flare.
    """
    A = jnp.atleast_1d(mean_params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(mean_params["t0"])[:, jnp.newaxis]
    sig = jnp.atleast_1d(mean_params["sig"])[:, jnp.newaxis]

    return jnp.sum(A * jnp.exp(-((t - t0) ** 2) / (2 * (sig**2))), axis=0)


def _exponential(t, mean_params):
    """An exponential flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.
    A:  jnp.int
        Amplitude of the flare.
    t0:
        The location of the maximum.
    sig1:
        The width parameter for the exponential.

    Returns
    -------
    The y values for exponential flare.
    """
    A = jnp.atleast_1d(mean_params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(mean_params["t0"])[:, jnp.newaxis]
    sig = jnp.atleast_1d(mean_params["sig"])[:, jnp.newaxis]

    return jnp.sum(A * jnp.exp(-jnp.abs(t - t0) / (2 * (sig**2))), axis=0)


def _constant(t, mean_params):
    """A constant mean shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.
    A:  jnp.int
        Constant amplitude of the flare.

    Returns
    -------
    The constant value.
    """
    return mean_params["A"] * jnp.ones_like(t)


def _skew_gaussian(t, mean_params):
    """A skew gaussian flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.
    A:  jnp.int
        Amplitude of the flare.
    t0:
        The location of the maximum.
    sig1:
        The width parameter for the rising edge.
    sig2:
        The width parameter for the falling edge.

    Returns
    -------
    The y values for skew gaussian flare.
    """
    return mean_params["A"] * jnp.where(
        t > mean_params["t0"],
        jnp.exp(-((t - mean_params["t0"]) ** 2) / (2 * (mean_params["sig2"] ** 2))),
        jnp.exp(-((t - mean_params["t0"]) ** 2) / (2 * (mean_params["sig1"] ** 2))),
    )


def _skew_exponential(t, mean_params):
    """A skew exponential flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.
    A:  jnp.int
        Amplitude of the flare.
    t0:
        The location of the maximum.
    sig1:
        The width parameter for the rising edge.
    sig2:
        The width parameter for the falling edge.

    Returns
    -------
    The y values for exponential flare.
    """
    return mean_params["A"] * jnp.where(
        t > mean_params["t0"],
        jnp.exp(-(t - mean_params["t0"]) / mean_params["sig2"]),
        jnp.exp((t - mean_params["t0"]) / mean_params["sig1"]),
    )


def _fred(t, mean_params):
    """A fast rise exponential decay (FRED) flare shape.

    Parameters
    ----------
    t:  jnp.ndarray
        The time coordinates.
    A:  jnp.int
        Amplitude of the flare.
    t0:
        The location of the maximum.
    phi:
        Symmetry parameter of the flare.
    delta:
        Offset parameter of the flare.

    Returns
    -------
    The y values for exponential flare.
    """
    return (
        mean_params["A"]
        * jnp.exp(
            -mean_params["phi"]
            * (
                (t + mean_params["delta"]) / mean_params["t0"]
                + mean_params["t0"] / (t + mean_params["delta"])
            )
        )
        * jnp.exp(2 * mean_params["phi"])
    )


def get_kernel_params(kernel_type):
    if kernel_type == "RN":
        return ["arn", "crn"]
    elif kernel_type == "QPO_plus_RN":
        return ["arn", "crn", "aqpo", "cqpo", "freq"]


def get_mean_params(mean_type):
    if (mean_type == "gaussian") or (mean_type == "exponential"):
        return ["A", "t0", "sig"]
    elif mean_type == "constant":
        return ["A"]
    elif (mean_type == "skew_gaussian") or (mean_type == "skew_exponential"):
        return ["A", "t0", "sig1", "sig2"]
    elif mean_type == "fred":
        return ["A", "t0", "delta", "phi"]


def get_gp_params(kernel_type, mean_type):
    kernel_params = get_kernel_params(kernel_type)
    mean_params = get_mean_params(mean_type)
    kernel_params.extend(mean_params)
    return kernel_params


def get_prior(params_list, prior_dict):
    """
    A prior generator function based on given values

    Parameters
    ----------
    params_list:
        A list in order of the parameters to be used.

    prior_dict:
        A dictionary of the priors of parameters to be used.

    Returns
    -------
    The Prior function.
    The arguments of the prior function are in the order of
    Kernel arguments (RN arguments, QPO arguments),
    Mean arguments
    Non Windowed arguments

    """

    def prior_model():
        prior_list = []
        for i in params_list:
            if isinstance(prior_dict[i], tfpd.Distribution):
                parameter = yield Prior(prior_dict[i], name=i)
            else:
                parameter = yield prior_dict[i]
            prior_list.append(parameter)
        return tuple(prior_list)

    return prior_model


def get_likelihood(params_list, kernel_type, mean_type, **kwargs):
    """
    A likelihood generator function based on given values

    Parameters
    ----------
    params_list:
        A list in order of the parameters to be used.

    prior_dict:
        A dictionary of the priors of parameters to be used.

    kernel_type:
        The type of kernel to be used in the model.

    mean_type:
        The type of mean to be used in the model.

    """

    @jit
    def likelihood_model(*args):
        dict = {}
        for i, params in enumerate(params_list):
            dict[params] = args[i]
        kernel = get_kernel(kernel_type=kernel_type, kernel_params=dict)
        mean = get_mean(mean_type=mean_type, mean_params=dict)
        gp = GaussianProcess(kernel, kwargs["Times"], mean_value=mean(kwargs["Times"]))
        return gp.log_probability(kwargs["counts"])

    return likelihood_model


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

    def __init__(self, Lc: Lightcurve) -> None:
        self.lc = Lc
        self.time = Lc.time
        self.counts = Lc.counts

    def fit(self, kernel=None, mean=None, **kwargs):
        self.kernel = kernel
        self.mean = mean
        self.maingp = GaussianProcess(
            self.kernel, self.time, mean_value=self.mean(self.time), diag=kwargs["diag"]
        )

    def get_logprob(self):
        """
        Returns the logprobability of the lightcurves counts for the
        given kernel for the Gaussian Process
        """
        cond = self.maingp.condition(self.lc.counts)
        return cond.log_probability

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

    def sample(self, prior_model=None, likelihood_model=None, **kwargs):
        """
        Makes a Jaxns nested sampler over the Gaussian Process, given the
        prior and likelihood model

        Parameters
        ----------
        prior_model: jaxns.prior.PriorModelType object
            A prior generator object

        likelihood_model: jaxns.types.LikelihoodType object
            A likelihood fucntion which takes in the arguments of the prior
            model and returns the loglikelihood of the model

        Returns
        ----------
        Results: jaxns.results.NestedSamplerResults object
            The results of the nested sampling process

        """

        self.prior_model = prior_model
        self.likelihood_model = likelihood_model

        NSmodel = Model(prior_model=self.prior_model, log_likelihood=self.likelihood_model)
        NSmodel.sanity_check(random.PRNGKey(10), S=100)

        self.Exact_ns = ExactNestedSampler(NSmodel, num_live_points=500, max_samples=1e4)
        Termination_reason, State = self.Exact_ns(
            random.PRNGKey(42), term_cond=TerminationCondition(live_evidence_frac=1e-4)
        )
        self.Results = self.Exact_ns.to_results(State, Termination_reason)
        print("Simulation Complete")

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

    def plot_cornerplot(self):
        """
        Plots the corner plot for the sampled hyperparameters
        """
        self.Exact_ns.plot_cornerplot(self.Results)

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
