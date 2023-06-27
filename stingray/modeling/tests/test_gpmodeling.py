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
        self.mean_params_gaussian = {
            "A": jnp.array([3.0, 4.0]),
            "t0": jnp.array([0.2, 0.7]),
            "sig": jnp.array([0.2, 0.1]),
        }

    def test_get_mean_gaussian(self):
        result_gaussian = 3 * jnp.exp(-((self.t - 0.2) ** 2) / (2 * (0.2**2))) + 4 * jnp.exp(
            -((self.t - 0.7) ** 2) / (2 * (0.1**2))
        )
        assert (get_mean("gaussian", self.mean_params_gaussian)(self.t) == result_gaussian).all()

    def test_get_mean_exponential(self):
        result_exponential = 3 * jnp.exp(-jnp.abs(self.t - 0.2) / (2 * (0.2**2))) + 4 * jnp.exp(
            -jnp.abs(self.t - 0.7) / (2 * (0.1**2))
        )
        assert (
            get_mean("exponential", self.mean_params_gaussian)(self.t) == result_exponential
        ).all()
