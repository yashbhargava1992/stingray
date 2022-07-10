import warnings
import numpy as np
import scipy.stats, scipy.optimize
from ultranest import ReactiveNestedSampler

__all__ = ['Bexvar']

def _lscg_gen(src_counts, bkg_counts, bkg_area, rate_conversion, density_gp):
    """ 
    Generates a log_src_crs_grid applicable to this particular light curve, 
    with appropriately designated limits, for a faster and more accurate 
    run of estimate_source_cr_marginalised and bexvar 
    """
    # lowest count rate
    a = scipy.special.gammaincinv(src_counts + 1, 0.001) / rate_conversion
    # highest background count rate
    b = scipy.special.gammaincinv(bkg_counts + 1, 0.999) / (rate_conversion * bkg_area)
    mindiff = min(a - b)
    if mindiff > 0: # background-subtracted rate is positive
        m0 = np.log10(mindiff)
    else: # more background than source -> subtraction negative somewhere
        m0 = -1
    # highest count rate (including background)
    c = scipy.special.gammaincinv(src_counts + 1, 0.999) / rate_conversion
    m1 = np.log10(c.max())
    # print(src_counts, bkg_counts, a, b, m0, m1)

    # add a bit of padding to the bottom and top
    lo = m0 - 0.05 * (m1 - m0)
    hi = m1 + 0.05 * (m1 - m0)
    span = hi - lo
    if lo < -1:
        log_src_crs_grid = np.linspace(-1.0, hi, int(np.ceil(density_gp * (hi + 1.0))))
    else:
        log_src_crs_grid = np.linspace(lo, hi, int(np.ceil(density_gp * 1.05 * span)))

    return log_src_crs_grid

def _estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, rate_conversion):
    """ 
    Compute the PDF at positions in log(source count rate)s grid log_src_crs_grid 
    for observing src_counts counts in the source region of size src_area,
    and bkg_counts counts in the background region of size bkg_area.
    
    """
    # background counts give background cr deterministically
    N = 1000
    u = np.linspace(0, 1, N)[1:-1]
    bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) / bkg_area
    
    def prob(log_src_cr):
        src_cr = 10**log_src_cr * rate_conversion
        like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
        return like
    
    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])
    if not weights.sum() > 0:
        print("WARNING: Weight problem! sum is", weights.sum(), np.log10(src_counts.max() / rate_conversion), log_src_crs_grid[0], log_src_crs_grid[-1])
    weights /= weights.sum()
    
    return weights

def _bexvar(log_src_crs_grid, pdfs):
    """ 
    Assumes that the source count rate is log-normal distributed.
    returns posterior samples of the mean and std of that distribution.
    
    pdfs: PDFs for each object 
        defined over the log-source count rate grid log_src_crs_grid.
    
    returns (log_mean, log_std), each an array of posterior samples.
    """
    
    def transform(cube):
        params = cube.copy()
        params[0] = cube[0] * (log_src_crs_grid[-1] - log_src_crs_grid[0]) + log_src_crs_grid[0]
        params[1] = 10**(cube[1]*4 - 2)
        return params
    
    def loglike(params):
        log_mean  = params[0]
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
    
    return log_mean, log_sigma


def Bexvar(LightCurve):
    """
    Given an eROSITA SRCTOOL light curve,
    Computes a Bayesian excess variance, by estimating mean and variance of
    the log of the count rate. If the light curve contains counts for multiple 
    bands then it iterates over each band to produce Bayesian excess variance
    corresponding to data in each band.
    
    Parameters
    ----------
    LightCurve : `~astropy.table.Table` object 
        Contains light curve for which Bayesian excess variance is to be calculated.

    Returns
     posterior_logcr_sigma : list of ndarrays  
        Contains an array of posterior samples of log_srs_countrate_sigma corresponding 
        to each band in the light curve.
    """

    nbands = LightCurve['COUNTS'].shape[1]

    # Initializing an empty array which will contain the posterior samples of log_srs_countrate_sigma
    posterior_logcr_sigma = [] 
    
    for band in range(nbands):
        print("band %d" % band)
        lc = LightCurve[LightCurve['FRACEXP'][:,band] > 0.1]
        x = lc['TIME'] - lc['TIME'][0]
        bc = lc['BACK_COUNTS'][:,band]
        c = lc['COUNTS'][:,band]
        bgarea = 1. / lc['BACKRATIO']
        fe = lc['FRACEXP'][:,band]
        rate_conversion = fe * lc['TIMEDEL']

        log_src_crs_grid = _lscg_gen(c, bc, bgarea, rate_conversion, 100)
        
        src_posteriors = []

        print("preparing time bin posteriors...")
        for xi, ci, bci, bgareai, rate_conversioni in zip(x, c, bc, bgarea, rate_conversion):
            pdf = _estimate_source_cr_marginalised(log_src_crs_grid, ci, bci, bgareai, rate_conversioni)
            src_posteriors.append(pdf)

        src_posteriors = np.array(src_posteriors)
        
        
        print("running bexvar...")
        logcr_mean, logcr_sigma = _bexvar(log_src_crs_grid, src_posteriors)
        print("running bexvar... done")
        
        posterior_logcr_sigma.append(logcr_sigma)
    
    return posterior_logcr_sigma
