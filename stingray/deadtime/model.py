from stingray.utils import njit, prange
from math import gamma
import numpy as np
import matplotlib.pyplot as plt
from astropy import log


def r_in(td, r_0):
    """Calculate incident countrate given dead time and detected countrate."""
    tau = 1 / r_0
    return 1. / (tau - td)


def r_det(td, r_i):
    """Calculate detected countrate given dead time and incident countrate."""
    tau = 1 / r_i
    return 1. / (tau + td)


@njit()
def factorial(n):
    return gamma(n + 1)


@njit()
def Gn(x, n):
    s = 0
    for l in range(0, n):
        s += (n - l) / factorial(l) * x**l
    return np.exp(-x) * s


@njit()
def heaviside(x):
    """Heaviside function.

    Examples
    --------
    >>> heaviside(2)
    1
    >>> heaviside(-1)
    0
    """
    if x >= 0:
        return 1
    else:
        return 0


@njit()
def h(k, n, td, tb, tau):
    # Typo in Zhang+95 corrected. k * tb, not k * td
    if k * tb < n * td:
        return 0
    return (k - n*(td + tau) / tb +
            tau / tb * Gn((k * tb - n * td)/tau, n))


INFINITE = 100
@njit()
def A0(r0, td, tb, tau):
    s = 0
    for n in range(1, INFINITE):
        s += h(1, n, td, tb, tau)

    return r0 * tb * (1 + 2 * s)


@njit()
def A(k, r0, td, tb, tau):
    if k == 0:
        return A0(r0, td, tb, tau)
    # Equation 39
    s = 0
    for n in range(1, INFINITE):
        s += h(k + 1, n, td, tb, tau) - 2 * h(k, n, td, tb, tau) + h(k - 1, n, td, tb, tau)

    return r0 * tb * s


def check_A(rate, td, tb, max_k=100, save_to=None):
    """Ak ->r0**2tb**2 for k->infty"""
    tau = 1 / rate
    r0 = r_det(td, rate)

    value = r0 ** 2 * tb**2
    fig = plt.figure()
    for k in range(max_k):
        plt.scatter(k, A(k, r0, td, tb, tau), color='k')
    plt.axhline(value, ls='--', color='k')
    plt.xlabel('$k$')
    plt.ylabel('$A_k$')
    if save_to is not None:
        plt.savefig(save_to)
        plt.close(fig)

@njit()
def B(k, r0, td, tb, tau):
    if k == 0:
        return 2 * (A(0, r0, td, tb, tau) - r0**2 * tb**2) / (r0*tb)

    return 4 * (A(k, r0, td, tb, tau) - r0**2 * tb**2) / (r0*tb)

@njit()
def safe_B(k, r0, td, tb, tau, limit_k=60):
    if k > limit_k:
        return 0
    return B(k, r0, td, tb, tau)


def check_B(rate, td, tb, max_k=100, save_to=None):
    """Ak ->r0**2tb**2 for k->infty."""
    tau = 1 / rate
    r0 = r_det(td, rate)

    fig = plt.figure()
    for k in range(max_k):
        plt.scatter(k, B(k, r0, td, tb, tau), color='k')
    plt.axhline(0, ls='--', color='k')
    plt.xlabel('$k$')
    plt.ylabel('$B_k$')
    if save_to is not None:
        plt.savefig(save_to)
        plt.close(fig)

@njit(parallel=True)
def _inner_loop_pds_zhang(N, tau, r0, td, tb, limit_k=60):
    P = np.zeros(N // 2)
    for j in prange(N//2):
        eq8_sum = 0
        for k in range(1, N):
            eq8_sum += (N - k) / N * safe_B(
                k, r0, td, tb, tau, limit_k=limit_k) * np.cos(2 * np.pi * j * k / N)

        P[j] = safe_B(0, r0, td, tb, tau) + eq8_sum

    return P


def pds_model_zhang(N, rate, td, tb, limit_k=60):
    tau = 1 / rate
    r0 = r_det(td, rate)
    # Nph = N / tau
    log.info("Calculating PDS model (update)")
    P = _inner_loop_pds_zhang(N, tau, r0, td, tb, limit_k=limit_k)

    maxf = 0.5 / tb
    df = maxf / len(P)
    freqs = np.arange(0, maxf, df)

    return freqs, P
