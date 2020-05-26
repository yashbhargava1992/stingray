import warnings
from collections.abc import Iterable

import numpy as np
from scipy import stats
from stingray.utils import simon
from stingray.utils import vectorize, float64, float32, int32, int64


__all__ = ['p_multitrial_from_single_trial',
           'p_single_trial_from_p_multitrial',
           'fold_profile_probability',
           'fold_detection_level',
           'pds_probability',
           'pds_detection_level',
           'z2_n_detection_level',
           'z2_n_probability',
           'classical_pvalue']

@vectorize([float64(float32, int32),
            float64(float32, int64),
            float64(float64, int32),
            float64(float64, int64)], nopython=True)
def _logp_multitrial_from_single_logp(logp1, n):
    """Calculate a multi-trial p-value from the log of a single-trial one.

    This allows to work around Numba's limitation on longdoubles, a way to
    vectorize the computation when we need longdouble precision.

    Parameters
    ----------
    logp1 : float
        The logarithm (base 10) of the significance at which we reject the null
        hypothesis on each single trial.
    n : int
        The number of trials

    Returns
    -------
    logpn : float
        The log10 of the significance at which we reject the null hypothesis
        after multiple trials
    """
    # If the the probability is very small (p1 * n) < 1e-6, use Bonferroni
    # approximation.
    logn = np.log10(n)
    if logp1 + logn < -6:
        return logp1 + logn

    return np.log10(1 - (1 - 10.**logp1) ** n)


def p_multitrial_from_single_trial(p1, n):
    r"""Calculate a multi-trial p-value from a single-trial one.

    Calling *p* the probability of a single success, the Binomial
    distributions says that the probability *at least* one outcome
    in n trials is

    .. math::

        P(k\geq 1) = \sum_{k\geq 1} \binom{n}{k} p^k (1-p)^{(n-k)}

    or more simply, using P(k ≥ 0) = 1

    .. math::

        P(k\geq 1) = 1 - \binom{n}{0} (1-p)^n = 1 - (1-p)^n


    Parameters
    ----------
    p1 : float
        The significance at which we reject the null hypothesis on
        each single trial.
    n : int
        The number of trials

    Returns
    -------
    pn : float
        The significance at which we reject the null hypothesis
        after multiple trials
    """
    logpn = _logp_multitrial_from_single_logp(
        np.log10(p1).astype(np.double), n)

    return 10.**np.longdouble(logpn)


@vectorize([float64(float32, int32),
            float64(float32, int64),
            float64(float64, int32),
            float64(float64, int64)], nopython=True)
def _logp_single_trial_from_logp_multitrial(logpn, n):
    """Calculate a multi-trial p-value from the log of a single-trial one.

    This allows to work around Numba's limitation on longdoubles, a way to
    vectorize the computation when we need longdouble precision.

    Parameters
    ----------
    logpn : float
        The logarithm (base 10) of the significance at which we want to reject
        the null hypothesis after multiple trials
    n : int
        The number of trials

    Returns
    -------
    logp1 : float
        The log10 of the significance at which we reject the null hypothesis on
        each single trial.
    """
    logn = np.log10(n)
    # If the the probability is very small, use Bonferroni approximation.
    if logpn < -6:
        return logpn - logn

    # Numerical errors arise when pn is very close to 1. (logpn ~ 0)
    if 1 - 10.**logpn < np.finfo(np.double).resolution * 1000:
        return np.nan

    p1 = 1 - np.power(1 - 10.**logpn, 1 / n)
    return np.log10(p1)


def p_single_trial_from_p_multitrial(pn, n):
    r"""Calculate the single-trial p-value from a total p-value

    Let us say that we want to reject a null hypothesis at the
    ``pn`` level, after executing ``n`` different measurements.
    This might be the case because, e.g., we
    want to have a 1% probability of detecting a signal in an
    entire power spectrum, and we need to correct the detection
    level accordingly.

    The typical procedure is dividing the initial probability
    (often called _epsilon_) by the number of trials. This is
    called the Bonferroni correction and it is often a good
    approximation, when ``pn`` is low: ``p1 = pn / n``.

    However, if ``pn`` is close to 1, this approximation gives
    incorrect results.

    Here we calculate this probability by inverting the Binomial
    problem. Given that (see ``p_multitrial_from_single_trial``)
    the probability of getting more than one hit in n trials,
    given the single-trial probability *p*, is

    .. math ::

        P (k \geq 1) =  1 - (1 - p)^n,

    we get the single trial probability from the multi-trial one
    from

    .. math ::

        p = 1 - (1 - P)^{(1/n)}

    This is also known as Šidák correction.

    Parameters
    ----------
    pn : float
        The significance at which we want to reject the null
        hypothesis after multiple trials
    n : int
        The number of trials

    Returns
    -------
    p1 : float
        The significance at which we reject the null hypothesis on
        each single trial.
    """

    logp = _logp_single_trial_from_logp_multitrial(
        np.log10(pn).astype(np.float64), n)

    if np.any(np.isnan(logp)):
        if np.any(1 - pn < np.finfo(np.double).resolution * 1000):
            warnings.warn("Multi-trial probability is very close to 1.")
            warnings.warn("The problem is ill-conditioned. Returning NaN")

    return 10**logp


def fold_profile_probability(stat, nbin, ntrial=1):
    """Calculate the probability of a certain folded profile, due to noise.

    Parameters
    ----------
    stat : float
        The epoch folding statistics
    nbin : int
        The number of bins in the profile

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile

    Returns
    -------
    p : float
        The probability that the profile has been produced by noise
    """
    p1 = stats.chi2.sf(stat, (nbin - 1))
    return p_multitrial_from_single_trial(p1, ntrial)


def fold_detection_level(nbin, epsilon=0.01, ntrial=1):
    """Return the detection level for a folded profile.

    See Leahy et al. (1983).

    Parameters
    ----------
    nbin : int
        The number of bins in the profile
    epsilon : float, default 0.01
        The fractional probability that the signal has been produced
        by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile

    Returns
    -------
    detlev : float
        The epoch folding statistics corresponding to a probability
        epsilon * 100 % that the signal has been produced by noise
    """
    epsilon = p_single_trial_from_p_multitrial(epsilon, ntrial)
    return stats.chi2.isf(epsilon.astype(np.double), nbin - 1)


def z2_n_probability(z2, n=2, ntrial=1, n_summed_spectra=1):
    """Calculate the probability of a certain folded profile, due to noise.

    Parameters
    ----------
    z2 : float
        A Z^2_n statistics value
    n : int, default 2
        The ``n`` in $Z^2_n$ (number of harmonics, including the fundamental)

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    n_summed_spectra : int
        Number of Z_2^n periodograms that were averaged to obtain z2

    Returns
    -------
    p : float
        The probability that the Z^2_n value has been produced by noise
    """
    epsilon_1 = stats.chi2.sf(z2 * n_summed_spectra,
                              2 * n * n_summed_spectra)
    epsilon = p_multitrial_from_single_trial(epsilon_1, ntrial)
    return epsilon


def z2_n_detection_level(n=2, epsilon=0.01, ntrial=1, n_summed_spectra=1):
    """Return the detection level for the Z^2_n statistics.

    See Buccheri et al. (1983), Bendat and Piersol (1971).

    Parameters
    ----------
    n : int, default 2
        The ``n`` in $Z^2_n$ (number of harmonics, including the fundamental)
    epsilon : float, default 0.01
        The fractional probability that the signal has been produced by noise

    Other Parameters
    ----------------
    ntrial : int
        The number of trials executed to find this profile
    n_summed_spectra : int
        Number of Z_2^n periodograms that are being averaged

    Returns
    -------
    detlev : float
        The epoch folding statistics corresponding to a probability
        epsilon * 100 % that the signal has been produced by noise
    """

    epsilon = p_single_trial_from_p_multitrial(epsilon, ntrial)
    retlev = stats.chi2.isf(epsilon.astype(np.double),
                            2 * n_summed_spectra * n) / (n_summed_spectra)

    return retlev


def pds_probability(level, ntrial=1, n_summed_spectra=1, n_rebin=1):
    r"""Give the probability of a given power level in PDS.

    Return the probability of a certain power level in a Power Density
    Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)

    Parameters
    ----------
    level : float or array of floats
        The power level for which we are calculating the probability

    Other Parameters
    ----------------
    ntrial : int
        The number of *independent* trials (the independent bins of the PDS)
    n_summed_spectra : int
        The number of power density spectra that have been averaged to obtain
        this power level
    n_rebin : int
        The number of power density bins that have been averaged to obtain
        this power level

    Returns
    -------
    epsilon : float
        The probability value(s)
    """

    epsilon_1 = stats.chi2.sf(level * n_summed_spectra * n_rebin,
                              2 * n_summed_spectra * n_rebin)

    epsilon = p_multitrial_from_single_trial(epsilon_1, ntrial)
    return epsilon


def pds_detection_level(epsilon=0.01, ntrial=1, n_summed_spectra=1, n_rebin=1):
    r"""Detection level for a PDS.

    Return the detection level (with probability 1 - epsilon) for a Power
    Density Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)

    Parameters
    ----------
    epsilon : float
        The single-trial probability value(s)

    Other Parameters
    ----------------
    ntrial : int
        The number of *independent* trials (the independent bins of the PDS)
    n_summed_spectra : int
        The number of power density spectra that have been averaged to obtain
        this power level
    n_rebin : int
        The number of power density bins that have been averaged to obtain
        this power level

    Examples
    --------
    >>> np.isclose(pds_detection_level(0.1), 4.6, atol=0.1)
    True
    >>> np.allclose(pds_detection_level(0.1, n_rebin=[1]), [4.6], atol=0.1)
    True
    """
    epsilon = p_single_trial_from_p_multitrial(epsilon, ntrial)
    epsilon = epsilon.astype(np.double)
    if isinstance(n_rebin, Iterable):
        retlev = [stats.chi2.isf(epsilon, 2 * n_summed_spectra * r) /
                  (n_summed_spectra * r) for r in n_rebin]
        retlev = np.array(retlev)
    else:
        r = n_rebin
        retlev = stats.chi2.isf(epsilon, 2 * n_summed_spectra * r) \
            / (n_summed_spectra * r)
    return retlev


def classical_pvalue(power, nspec):
    """
    Note:
    This is stingray's original implementation of the probability
    distribution for the power spectrum. It is superseded by the
    implementation in pds_probability for practical purposes, but
    remains here for backwards compatibility and for its educational
    value as a clear, explicit implementation of the correct
    probability distribution.

    Compute the probability of detecting the current power under
    the assumption that there is no periodic oscillation in the data.

    This computes the single-trial p-value that the power was
    observed under the null hypothesis that there is no signal in
    the data.

    Important: the underlying assumptions that make this calculation valid
    are:

    1. the powers in the power spectrum follow a chi-square distribution
    2. the power spectrum is normalized according to [Leahy 1983]_, such
       that the powers have a mean of 2 and a variance of 4
    3. there is only white noise in the light curve. That is, there is no
       aperiodic variability that would change the overall shape of the power
       spectrum.

    Also note that the p-value is for a *single trial*, i.e. the power
    currently being tested. If more than one power or more than one power
    spectrum are being tested, the resulting p-value must be corrected for the
    number of trials (Bonferroni correction).

    Mathematical formulation in [Groth 1975]_.
    Original implementation in IDL by Anna L. Watts.

    Parameters
    ----------
    power :  float
        The squared Fourier amplitude of a spectrum to be evaluated

    nspec : int
        The number of spectra or frequency bins averaged in ``power``.
        This matters because averaging spectra or frequency bins increases
        the signal-to-noise ratio, i.e. makes the statistical distributions
        of the noise narrower, such that a smaller power might be very
        significant in averaged spectra even though it would not be in a single
        power spectrum.

    Returns
    -------
    pval : float
        The classical p-value of the observed power being consistent with
        the null hypothesis of white noise

    References
    ----------

    * .. [Leahy 1983] https://ui.adsabs.harvard.edu/#abs/1983ApJ...266..160L/abstract
    * .. [Groth 1975] https://ui.adsabs.harvard.edu/#abs/1975ApJS...29..285G/abstract

    """

    warnings.warn("This function was substituted by pds_probability.",
                  DeprecationWarning)

    if not np.isfinite(power):
        raise ValueError("power must be a finite floating point number!")

    if power < 0:
        raise ValueError("power must be a positive real number!")

    if not np.isfinite(nspec):
        raise ValueError("nspec must be a finite integer number")

    if nspec < 1:
        raise ValueError("nspec must be larger or equal to 1")

    if not np.isclose(nspec % 1, 0):
        raise ValueError("nspec must be an integer number!")

    # If the power is really big, it's safe to say it's significant,
    # and the p-value will be nearly zero
    if (power * nspec) > 30000:
        simon("Probability of no signal too miniscule to calculate.")
        return 0.0

    else:
        pval = _pavnosigfun(power, nspec)
        return pval


def _pavnosigfun(power, nspec):
    """
    Helper function doing the actual calculation of the p-value.

    Parameters
    ----------
    power : float
        The measured candidate power

    nspec : int
        The number of power spectral bins that were averaged in `power`
        (note: can be either through averaging spectra or neighbouring bins)
    """
    sum = 0.0
    m = nspec - 1

    pn = power * nspec

    while m >= 0:

        s = 0.0
        for i in range(int(m) - 1):
            s += np.log(float(m - i))

        logterm = m * np.log(pn / 2) - pn / 2 - s
        term = np.exp(logterm)
        ratio = sum / term

        if ratio > 1.0e15:
            return sum

        sum += term
        m -= 1

    return sum
