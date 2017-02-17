
import numpy as np
from astropy.modeling import models

from stingray.modeling import PSDParEst
from stingray import Powerspectrum

__all__ = ["fit_lorentzians"]

def fit_lorentzians(ps, nlor, starting_pars, fit_whitenoise=True,
                    max_post=False, priors=None,
                    fitmethod="L-BFGS-B"):
    """
    Fit a number of Lorentzians to a power spectrum, possibly including white
    noise. Each Lorentzian has three parameters (amplitude, centroid position,
    full-width at half maximum), plus one extra parameter if the white noise
    level should be fit as well. Priors for each parameter can be included in
    case `max_post = True`, in which case the function will attempt a
    Maximum-A-Posteriori fit. Priors must be specified as a dictionary with one
    entry for each parameter.
    The parameter names are `(amplitude_i, x_0_i, fwhm_i)` for each `i` out of
    a total of `N` Lorentzians. The white noise level has a parameter
    `amplitude_(N+1)`. For example, a model with two Lorentzians and a
    white noise level would have parameters:
    [amplitude_0, x_0_0, fwhm_0, amplitude_1, x_0_1, fwhm_1, amplitude_2].

    Parameters
    ----------
    ps : Powerspectrum
        A Powerspectrum object with the data to be fit

    nlor : int
        The number of Lorentzians to fit

    starting_pars : iterable
        The list of starting guesses for the optimizer. See explanation above
        for ordering of parameters in this list.

    fit_whitenoise : bool, optional, default True
        If True, the code will attempt to fit a white noise level along with
        the Lorentzians. Be sure to include a starting parameter for the
        optimizer in `starting_pars`!

    max_post : bool, optional, default False
        If True, perform a Maximum-A-Posteriori fit of the data rather than a
        Maximum Likelihood fit. Note that this requires priors to be specified,
        otherwise this will cause an exception!

    priors : {dict | None}, optional, default None
        Dictionary with priors for the MAP fit. This should be of the form
        {"parameter name": probability distribution, ...}

    fitmethod : string, optional, default "L-BFGS-B"
        Specifies an optimization algorithm to use. Supply any valid option for
        `scipy.optimize.minimize`.

    Returns
    -------
    parest : PSDParEst object
        A PSDParEst object for further analysis

    res : OptimizationResults object
        The OptimizationResults object storing useful results and quantities
        relating to the fit

    Example
    -------

    We start by making an example power spectrum with three Lorentzians
    >>> np.random.seed(400)
    >>> nlor = 3

    >>> x_0_0 = 0.5
    >>> x_0_1 = 2.0
    >>> x_0_2 = 7.5

    >>> amplitude_0 = 150.0
    >>> amplitude_1 = 50.0
    >>> amplitude_2 = 15.0

    >>> fwhm_0 = 0.1
    >>> fwhm_1 = 1.0
    >>> fwhm_2 = 0.5

    We will also include a white noise level:
    >>> whitenoise = 2.0

    >>> model = models.Lorentz1D(amplitude_0, x_0_0, fwhm_0) + \\
    ...         models.Lorentz1D(amplitude_1, x_0_1, fwhm_1) + \\
    ...         models.Lorentz1D(amplitude_2, x_0_2, fwhm_2) + \\
    ...         models.Const1D(whitenoise)

    >>> freq = np.linspace(0.01, 10.0, 10.0/0.01)
    >>> p = model(freq)
    >>> noise = np.random.exponential(size=len(freq))

    >>> power = p*noise
    >>> ps = Powerspectrum()
    >>> ps.freq = freq
    >>> ps.power = powers
    >>> ps.df = ps.freq[1] - ps.freq[0]
    >>> ps.m = 1

    Now we have to guess starting parameters. For each Lorentzian, we have
    amplitude, centroid position and fwhm, and this pattern repeats for each
    Lorentzian in the fit. The white noise level is the last parameter.
    >>> t0 = [150, 0.4, 0.2, 50, 2.3, 0.6, 20, 8.0, 0.4, 2.1]

    We're ready for doing the fit:
    >>> parest, res = fit_lorentzians(ps_new, nlor, t0)

    `res` contains a whole array of useful information about the fit, for
    example the parameters at the optimum:
    >>> res.p_opt

    """

    model = models.Lorentz1D()

    if nlor > 1:
        for i in range( nlor -1):
            model += models.Lorentz1D()

    if fit_whitenoise:
        model += models.Const1D()

    parest = PSDParEst(ps, fitmethod=fitmethod, max_post=max_post)
    res = parest.fit(model, starting_pars, neg=True, priors=priors)

    return parest, res
