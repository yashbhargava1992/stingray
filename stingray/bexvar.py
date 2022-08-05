import warnings
import numpy as np
import scipy.stats
import scipy.optimize

# check whether ultranest is installed for sampling
try:
    from ultranest import ReactiveNestedSampler
    can_sample = True
except ImportError:
    can_sample = False


__all__ = ['bexvar', 'bexvar_from_table']


def _lscg_gen(src_counts, bkg_counts, bkg_area, rate_conversion, density_gp):
    """
    Generates a grid of log(Source count rates), `log_src_crs_grid` applicable
    to this particular light curve, with appropriately designated limits, for
    a faster and more accurate run of `_estimate_source_cr_marginalised()`
    and `_calculate_bexvar()`.

    Parameters
    ----------
    src_counts : Iterable, `:class:numpy.array` or `:class:List` of floats
        A list or array of counts observed from source region in each bin.

    bkg_counts : iterable, `:class:numpy.array` or `:class:List` of floats
        A list or array of counts observed from background region in each bin.

    bkg_area : iterable, `:class:numpy.array` or `:class:List` of floats
        A list or array of background area from where the ``bkg_counts`` were
        obtained for each bin.

    rate_conversion : iterable, `:class:numpy.array` or `:class:List` of floats
        A list or array of rate conversion constants. Obtained by multipllying
        fractional exposer with time interval for each bin.

    density_gp : int
        A number specifying density of grid points in the output ``log_src_crs_grid``.

    Returns
    -------
    log_src_crs_grid : iterable, `:class:numpy.array` of floats
        An array (grid) of log(Source count rates).
    """

    # lowest count rate
    a = scipy.special.gammaincinv(src_counts + 1, 0.001) / rate_conversion
    # highest background count rate
    b = scipy.special.gammaincinv(bkg_counts + 1, 0.999) / (rate_conversion * bkg_area)

    mindiff = min(a - b)
    if mindiff > 0: # minimum background-subtracted rate is positive
        m0 = np.log10(mindiff)
    else: # minimum background-subtracted rate is negative (more background than source)
        m0 = -1
    # highest count rate (including background)
    c = scipy.special.gammaincinv(src_counts + 1, 0.999) / rate_conversion
    m1 = np.log10(c.max())

    # add a bit of padding to the bottom and top
    lo = m0 - 0.05 * (m1 - m0)
    hi = m1 + 0.05 * (m1 - m0)
    span = hi - lo
    if lo < -1:
        log_src_crs_grid = np.linspace(-1.0, hi, int(np.ceil(density_gp * (hi + 1.0))))
    else:
        log_src_crs_grid = np.linspace(lo, hi, int(np.ceil(density_gp * 1.05 * span)))

    return log_src_crs_grid


def _estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, \
        rate_conversion):
    """
    Compute the PDF at positions in log(source count rate)s grid ``log_src_crs_grid``
    for observing ``src_counts`` counts in the source region of size src_area,
    and ``bkg_counts`` counts in the background region of size ``bkg_area``.

    Parameters
    ----------
    log_src_crs_grid : iterable, `:class:numpy.array` of floats
        An array (grid) of log(Source count rates).
    src_counts : float
        Source region counts in one of the bin.
    bkg_counts : float
        Background region counts in the bin.
    bkg_area : float
        Background region area in the bin.
    rate_conversion : float
        Rate conversion constant for the bin. Used to convert counts to count rate.

    Returns
    -------
    weights: float
    """
    # background counts give background count rates deterministically
    N = 1000
    u = np.linspace(0, 1, N)[1:-1]
    bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) / bkg_area

    def prob(log_src_cr):
        src_cr = 10**log_src_cr * rate_conversion
        like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
        return like

    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])

    if weights.sum() <= 0:
        print("Sum is" , weights.sum(), \
             "\n Maximum log(src_count_rate) ",np.log10(src_counts.max() / rate_conversion), \
             "\n range of log_src_count_rate_grid ",  log_src_crs_grid[0], log_src_crs_grid[-1])
        warnings.warn("Weight problem! sum is <= 0")

    weights /= weights.sum()

    return weights

def _calculate_bexvar(log_src_crs_grid, pdfs):
    """
    Assumes that the source count rate is log-normal distributed.
    returns posterior samples of the mean and standard deviation of that distribution.

    Parameters
    ----------
    log_src_crs_grid : Iterable, `:class:numpy.array` of flots
        An array (grid) of log(Source count rates).
    pdfs : Iterable, `:class:List` of floats
        An array of PDFs for each object defined over the
        log(source count rate) grid ``log_src_crs_grid``.

    Returns
    -------
    log_sigma : iterable, `:class:numpy.array` of flots.
        An array of posterior samples of log(standard deviation) or that of
        log(Bayesian excess varience).
    """

    if not can_sample:
        raise ImportError("ultranest not installed! Can't sample!")

    def transform(cube):
        params = cube.copy()
        params[0] = cube[0] * (log_src_crs_grid[-1] - log_src_crs_grid[0]) + log_src_crs_grid[0]
        params[1] = 10**(cube[1]*4 - 2)
        return params

    def loglike(params):
        log_mean = params[0]
        log_sigma = params[1]
        # compute for each grid log-countrate its probability, according to log_mean, log_sigma
        variance_pdf = scipy.stats.norm.pdf(log_src_crs_grid, log_mean, log_sigma)
        # multiply that probability with the precomputed probabilities (pdfs)
        likes = np.log((variance_pdf.reshape((1, -1)) * pdfs).mean(axis=1) + 1e-100)
        like = likes.sum()
        if not np.isfinite(like):
            like = -1e300
        return like

    sampler = ReactiveNestedSampler(['logmean', 'logsigma'], loglike,
    transform=transform, vectorized=False)
    samples = sampler.run(viz_callback=False)['samples']
    sampler.print_results()
    log_mean, log_sigma = samples.transpose()

    return log_sigma


def bexvar(time, time_del, src_counts, bg_counts=None, bg_ratio=None, frac_exp=None):
    """
    Given a light curve data, Computes a Bayesian excess variance of count rate,
    by estimating mean and variance of the log of the count rate.

    Parameters
    ----------
    time : Iterable, `:class:numpy.array` or `:class:List` of floats, optional, default ``None``
        A list or array of time stamps for a light curve.

    time_del : iterable, `:class:numpy.array` or `:class:List` of floats
        A list or array of time intervals for each bin of light curve.

    src_counts : iterable, `:class:numpy.array` or `:class:List` of floats
        A list or array of counts observed from source region in each bin.

     bg_counts : iterable, `:class:numpy.array` or `:class:List` of floats, optional, default ``None``
        A list or array of counts observed from background region in each bin. If ``None``
        we assume it as a numpy array of zeros, of length equal to length of ``src_counts``.

    bg_ratio : iterable, `:class:numpy.array` or `:class:List` of floats, optional, default ``None``
        A list or array of source region area to background region area ratio in each bin.
        If ``None`` we assume it as a numpy array of ones, of length equal the to length of
        ``src_counts``.

    frac_exp : iterable, `:class:numpy.array` or `:class:List` of floats, optional, default ``None``
        A list or array of fractional exposers in each bin. If ``None`` we assume it as
        a numpy array of ones, of length equal to length of ``src_counts``.

    Returns
    -------
    posterior_log_sigma_src_cr : iterable, `:class:List` of floats
        Contains an array of posterior samples of log(Sigma source count rate)
        (i.e. log(Bayesian excess varience) of source count rates).
    """

    if bg_counts is None:
        bg_counts = np.zeros(src_counts.shape[0])
    if bg_ratio is None:
        bg_ratio = np.ones(src_counts.shape[0])
    if frac_exp is None:
        frac_exp = np.ones(src_counts.shape[0])

    # makes sure that data with frac_exp <= 0.1 gets discarded
    time = time[frac_exp > 0.1]
    time_del = time_del[frac_exp > 0.1]
    src_counts = src_counts[frac_exp > 0.1]
    bg_counts = bg_counts[frac_exp > 0.1]
    bg_ratio = bg_ratio[frac_exp > 0.1]
    frac_exp = frac_exp[frac_exp > 0.1]

    bg_area = 1. / bg_ratio
    rate_conversion = frac_exp * time_del

    log_src_crs_grid = _lscg_gen(src_counts, bg_counts, bg_area, rate_conversion, 100)

    src_posteriors = []

    print("preparing time bin posteriors...")
    for xi, ci, bci, bgareai, rate_conversion in \
            zip(time, src_counts, bg_counts, bg_area, rate_conversion):

        pdf = _estimate_source_cr_marginalised(log_src_crs_grid, ci, bci, bgareai, rate_conversion)
        src_posteriors.append(pdf)

    src_posteriors = np.array(src_posteriors)

    print("running bexvar...")
    posterior_log_sigma_src_cr = _calculate_bexvar(log_src_crs_grid, src_posteriors)
    print("running bexvar... done")

    return posterior_log_sigma_src_cr
