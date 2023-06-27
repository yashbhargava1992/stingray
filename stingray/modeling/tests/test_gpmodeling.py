import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import matplotlib.pyplot as plt

from tinygp import GaussianProcess, kernels
from stingray.modeling.gpmodeling import get_kernel, get_mean, GP


class Testget_kernel(object):
    def setup_class(self):
        self.x = np.linspace(0, 1, 5)
        self.kernel_params = {"arn": 1.0, "aqpo": 1.0, "crn": 1.0, "cqpo": 1.0, "freq": 1.0}

    def test_get_kernel_qpo_plus_rn(self):
        kernel_qpo_plus_rn = kernels.quasisep.Exp(
            scale=1 / 1, sigma=(1) ** 0.5
        ) + kernels.quasisep.Celerite(
            a=1,
            b=0.0,
            c=1,
            d=2 * jnp.pi * 1,
        )
        kernel_qpo_plus_rn_test = get_kernel("QPO_plus_RN", self.kernel_params)
        assert (
            kernel_qpo_plus_rn(self.x, jnp.array([0.0]))
            == kernel_qpo_plus_rn_test(self.x, jnp.array([0.0]))
        ).all()

    def test_get_kernel_rn(self):
        kernel_rn = kernels.quasisep.Exp(scale=1 / 1, sigma=(1) ** 0.5)
        kernel_rn_test = get_kernel("RN", self.kernel_params)
        assert (
            kernel_rn(self.x, jnp.array([0.0])) == kernel_rn_test(self.x, jnp.array([0.0]))
        ).all()


class Testget_mean(object):
    def setup_class(self):
        self.t = np.linspace(0, 5, 10)
        self.mean_params = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "sig": jnp.array([0.2, 0.1]),
        }
        self.skew_mean_params = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "sig1": jnp.array([0.2, 0.1]),
            "sig2": jnp.array([0.3, 0.4]),
        }
        self.fred_mean_params = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "phi": jnp.array([4.0, 5.0]),
            "delta": jnp.array([0.3, 0.4]),
        }

    def test_get_mean_gaussian(self):
        result_gaussian = 3 * jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.2**2))) + 4 * jnp.exp(
            -((self.t - 0.7) ** 2) / (2 * (0.1**2))
        )
        assert (get_mean("gaussian", self.mean_params)(self.t) == result_gaussian).all()

    def test_get_mean_exponential(self):
        result_exponential = 3 * jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.2**2))) + 4 * jnp.exp(
            -jnp.abs(self.t - 0.7) / (2 * (0.1**2))
        )
        assert (get_mean("exponential", self.mean_params)(self.t) == result_exponential).all()

    def test_get_mean_constant(self):
        result_constant = 3 * jnp.ones_like(self.t)
        const_param_dict = {"A": jnp.array([3.0])}
        assert (get_mean("constant", const_param_dict)(self.t) == result_constant).all()

    def test_get_mean_skew_gaussian(self):
        result_skew_gaussian = 3.0 * jnp.where(
            self.t > 0.2,
            jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.3**2))),
            jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.2**2))),
        ) + 4.0 * jnp.where(
            self.t > 0.7,
            jnp.exp(-((self.t - 0.7) ** 2) / (2 * (0.4**2))),
            jnp.exp(-((self.t - 0.7) ** 2) / (2 * (0.1**2))),
        )
        assert (
            get_mean("skew_gaussian", self.skew_mean_params)(self.t) == result_skew_gaussian
        ).all()

    def test_get_mean_skew_exponential(self):
        result_skew_exponential = 3.0 * jnp.where(
            self.t > 0.2,
            jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.3**2))),
            jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.2**2))),
        ) + 4.0 * jnp.where(
            self.t > 0.7,
            jnp.exp(-jnp.abs(self.t - 0.7) / (2 * (0.4**2))),
            jnp.exp(-jnp.abs(self.t - 0.7) / (2 * (0.1**2))),
        )
        assert (
            get_mean("skew_exponential", self.skew_mean_params)(self.t) == result_skew_exponential
        ).all()

    def test_get_mean_fred(self):
        result_fred = 3.0 * jnp.exp(-4.0 * ((self.t + 0.3) / 0.2 + 0.2 / (self.t + 0.3))) * jnp.exp(
            2 * 4.0
        ) + 4.0 * jnp.exp(-5.0 * ((self.t + 0.4) / 0.7 + 0.7 / (self.t + 0.4))) * jnp.exp(2 * 5.0)
        assert (get_mean("fred", self.fred_mean_params)(self.t) == result_fred).all()
