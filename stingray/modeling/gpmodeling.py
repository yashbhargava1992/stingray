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
from jaxns.utils import resample

jax.config.update("jax_enable_x64", True)

tfpd = tfp.distributions
tfpb = tfp.bijectors

__all__ = ["GPResult"]


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
    A = jnp.atleast_1d(mean_params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(mean_params["t0"])[:, jnp.newaxis]
    sig1 = jnp.atleast_1d(mean_params["sig1"])[:, jnp.newaxis]
    sig2 = jnp.atleast_1d(mean_params["sig2"])[:, jnp.newaxis]

    return jnp.sum(
        A
        * jnp.where(
            t > t0,
            jnp.exp(-((t - t0) ** 2) / (2 * (sig2**2))),
            jnp.exp(-((t - t0) ** 2) / (2 * (sig1**2))),
        ),
        axis=0,
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
    A = jnp.atleast_1d(mean_params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(mean_params["t0"])[:, jnp.newaxis]
    sig1 = jnp.atleast_1d(mean_params["sig1"])[:, jnp.newaxis]
    sig2 = jnp.atleast_1d(mean_params["sig2"])[:, jnp.newaxis]

    return jnp.sum(
        A
        * jnp.where(
            t > t0,
            jnp.exp(-(t - t0) / (2 * (sig2**2))),
            jnp.exp((t - t0) / (2 * (sig1**2))),
        ),
        axis=0,
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
    A = jnp.atleast_1d(mean_params["A"])[:, jnp.newaxis]
    t0 = jnp.atleast_1d(mean_params["t0"])[:, jnp.newaxis]
    phi = jnp.atleast_1d(mean_params["phi"])[:, jnp.newaxis]
    delta = jnp.atleast_1d(mean_params["delta"])[:, jnp.newaxis]

    return jnp.sum(
        A * jnp.exp(-phi * ((t + delta) / t0 + t0 / (t + delta))) * jnp.exp(2 * phi), axis=0
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


class GPResult:
    """
    Makes a GPResult object which takes in a Stingray.Lightcurve and samples parameters of a model
    (Gaussian Process) based on the given prior and log_likelihood function.

    Parameters
    ----------
    lc: Stingray.Lightcurve object
        The lightcurve on which the bayesian inference is to be done

    Other Parameters
    ----------------
    time : class: np.array
        The array containing the times of the lightcurve

    counts : class: np.array
        The array containing the photon counts of the lightcurve

    """

    def __init__(self, Lc: Lightcurve) -> None:
        self.lc = Lc
        self.time = Lc.time
        self.counts = Lc.counts
        self.Result = None

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

    def get_evidence(self):
        """
        Returns the log evidence of the model
        """
        return self.Results.log_Z_mean

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

    def get_parameters_names(self):
        """
        Returns the names of the parameters
        """
        return sorted(self.Results.samples.keys())

    def get_max_posterior_parameters(self):
        """
        Returns the optimal parameters for the model based on the NUTS sampling
        """
        max_post_idx = jnp.argmax(self.Results.log_posterior_density)
        map_points = jax.tree_map(lambda x: x[max_post_idx], self.Results.samples)

        return map_points

    def get_max_likelihood_parameters(self):
        """
        Retruns the maximum likelihood parameters
        """
        max_like_idx = jnp.argmax(self.Results.log_L_samples)
        max_like_points = jax.tree_map(lambda x: x[max_like_idx], self.Results.samples)

        return max_like_points

    def posterior_plot(self, name: str, n=0):
        """
        Plots the posterior histogram for the given parameter
        """
        nsamples = self.Results.total_num_samples
        samples = self.Results.samples[name].reshape((nsamples, -1))[:, n]
        plt.hist(
            samples, bins="auto", density=True, alpha=1.0, label=name, fc="None", edgecolor="black"
        )
        mean1 = jnp.mean(self.Results.samples[name])
        std1 = jnp.std(self.Results.samples[name])
        plt.axvline(mean1, color="red", linestyle="dashed", label="mean")
        plt.axvline(mean1 + std1, color="green", linestyle="dotted")
        plt.axvline(mean1 - std1, linestyle="dotted", color="green")
        plt.legend()
        plt.plot()

        pass

    def weighted_posterior_plot(self, name: str, n=0, rkey=random.PRNGKey(1234)):
        """
        Returns the weighted posterior histogram for the given parameter
        """
        nsamples = self.Results.total_num_samples
        log_p = self.Results.log_dp_mean
        samples = self.Results.samples[name].reshape((nsamples, -1))[:, n]

        weights = jnp.where(jnp.isfinite(samples), jnp.exp(log_p), 0.0)
        log_weights = jnp.where(jnp.isfinite(samples), log_p, -jnp.inf)
        samples_resampled = resample(
            rkey, samples, log_weights, S=max(10, int(self.Results.ESS)), replace=True
        )

        nbins = max(10, int(jnp.sqrt(self.Results.ESS)) + 1)
        binsx = jnp.linspace(*jnp.percentile(samples_resampled, jnp.asarray([0, 100])), 2 * nbins)

        plt.hist(
            np.asarray(samples_resampled),
            bins=binsx,
            density=True,
            alpha=1.0,
            label=name,
            fc="None",
            edgecolor="black",
        )
        sample_mean = jnp.average(samples, weights=weights)
        sample_std = jnp.sqrt(jnp.average((samples - sample_mean) ** 2, weights=weights))
        plt.axvline(sample_mean, color="red", linestyle="dashed", label="mean")
        plt.axvline(sample_mean + sample_std, color="green", linestyle="dotted")
        plt.axvline(sample_mean - sample_std, linestyle="dotted", color="green")
        plt.legend()
        plt.plot()

    def corner_plot(self, param1: str, param2: str, n1=0, n2=0, rkey=random.PRNGKey(1234)):
        """
        Plots the corner plot for the given parameters
        """
        nsamples = self.Results.total_num_samples
        log_p = self.Results.log_dp_mean
        samples1 = self.Results.samples[param1].reshape((nsamples, -1))[:, n1]
        samples2 = self.Results.samples[param2].reshape((nsamples, -1))[:, n2]

        log_weights = jnp.where(jnp.isfinite(samples2), log_p, -jnp.inf)
        nbins = max(10, int(jnp.sqrt(self.Results.ESS)) + 1)

        samples_resampled = resample(
            rkey,
            jnp.stack([samples1, samples2], axis=-1),
            log_weights,
            S=max(10, int(self.Results.ESS)),
            replace=True,
        )
        plt.hist2d(
            samples_resampled[:, 1],
            samples_resampled[:, 0],
            bins=(nbins, nbins),
            density=True,
            cmap="GnBu",
        )
        plt.plot()

        pass
