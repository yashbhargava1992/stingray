import warnings
from collections.abc import Iterable

import numpy as np
from scipy import stats


__all__ = ['p_multitrial_from_single_trial',
           'p_single_trial_from_p_multitrial',
           'fold_profile_probability',
           'fold_detection_level',
           'z2_n_detection_level',
           'z2_n_probability']


def p_multitrial_from_single_trial(p1, n):
    """Calculate a multi-trial p-value from a single-trial one.

    Calling _p_ the probability of a single success, the Binomial
    distributions says that the probability _at least_ one outcome
    in n trials is
                         n
    P (k ≥ 1) =   Σ    (   ) p^k (1 - p)^(n - k)
                k ≥ 1    k

    or more simply, using P(k ≥ 0) = 1
                          n
    P (k ≥ 1) =   1 -   (   ) (1 - p)^n = 1 - (1 - p)^n
                          0

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
    return 1 - (1 - np.longdouble(p1))**np.longdouble(n)


def p_single_trial_from_p_multitrial(pn, n):
    """Calculate the single-trial p-value from a total p-value

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
    given the single-trial probability _p_, is

    P (k ≥ 1) =  1 - (1 - p)^n,

    we get the single trial probability from the multi-trial one
    from

    p = 1 - (1 - P)^(1/n)

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

    Parameters
    ----------
    pn : float or array of float (same size as p1)
        The probability of at least one success in ``n`` trials, each with
        probability ``p1``
    n : int
        The number of trials

    Returns
    -------
    p1 : float or array of floats, same size as ``pn``
        the probability of success in each trial.
    """
    if isinstance(n, Iterable):
        return np.array([p_single_trial_from_p_multitrial(pn, ni)
                         for ni in n])

    # Numerical errors arise when pn is very close to 1.
    if 1 - pn < np.finfo(np.longdouble).resolution * 1000:
        warnings.warn("Multi-trial probability is very close to 1.")
        warnings.warn("The problem is ill-conditioned. Returning NaN")
        return np.nan

    p1 = 1 - np.power(1 - np.longdouble(pn), 1/np.longdouble(n))
    return p1


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
