__all__ = ["classical_pvalue"]

import numpy as np

def classical_pvalue(power, nspec):
    """
    Compute the probability of detecting the current power under
    the assumption that there is no periodic oscillation in the data.

    This computes the single-trial p-value that the power was
    observed under the null hypothesis that there is no signal in
    the data.

    Important: the underlying assumptions that make this calculation valid
    are:
    (1) the powers in the power spectrum follow a chi-square distribution
    (2) the power spectrum is normalized according to Leahy (1984), such
    that the powers have a mean of 2 and a variance of 4
    (3) there is only white noise in the light curve. That is, there is no
    aperiodic variability that would change the overall shape of the power
    spectrum.

    Also note that the p-value is for a *single trial*, i.e. the power currently
    being tested. If more than one power or more than one power spectrum are
    being tested, the resulting p-value must be corrected for the number
    of trials (Bonferroni correction).

    Mathematical formulation in Groth, 1975.
    Original implementation in IDL by Anna L. Watts.

    Parameters
    ----------
    power :  float
        The squared Fourier amplitude of a spectrum to be evaluated

    nspec : int
        The number of spectra or frequency bins averaged in `power`.
        This matters because averaging spectra or frequency bins increases
        the signal-to-noise ratio, i.e. makes the statistical distributions
        of the noise narrower, such that a smaller power might be very
        significant in averaged spectra even though it would not be in a single
        power spectrum.

    """

    assert np.isfinite(power), "power must be a finite floating point number!"
    assert power > 0.0, "power must be a positive real number!"
    assert np.isfinite(nspec), "nspec must be a finite integer number"
    assert nspec >= 1, "nspec must be larger or equal to 1"
    assert np.isclose(nspec%1, 0), "nspec must be an integer number!"

    ## If the power is really big, it's safe to say it's significant,
    ## and the p-value will be nearly zero
    if power*nspec > 30000:
        print("Probability of no signal too miniscule to calculate.")
        return 0.0

    else:
        pval = _pavnosigfun(power, nspec)
        return pval



def _pavnosigfun(power, nspec):
    """
    Helper function doing the actual calculation of the p-value.
    """
    sum = 0.0
    m = nspec-1

    pn = power*nspec

    while m >= 0:

        s = 0.0
        for i in xrange(int(m)-1):
            s += np.log(float(m-i))

        logterm = m*np.log(pn/2.0) - pn/2.0 - s
        term = np.exp(logterm)
        ratio = sum/term

        if ratio > 1.0e15:
            return sum

        sum += term
        m -= 1

    return sum
