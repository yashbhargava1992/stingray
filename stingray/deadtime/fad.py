import warnings
import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from astropy import log
from astropy.table import Table

from ..gti import cross_two_gtis, bin_intervals_from_gtis


def _get_fourier_intv(lc, start_ind, end_ind):
    time = lc.time[start_ind:end_ind]
    counts = lc.counts[start_ind:end_ind]

    fourier = scipy.fftpack.fft(counts)

    freq = scipy.fftpack.fftfreq(len(time), lc.dt)
    good = freq > 0

    return freq[good], fourier[good], fourier[good] * np.sqrt(2 / (np.sum(counts)))


def calculate_FAD_correction(lc1, lc2, segment_size, gti=None,
                             plot=False, smoothing_alg='gauss',
                             smoothing_length=None, verbose=False, tolerance=0.05,
                             strict=False, all_leahy=False):
    """Calculate Frequency Amplitude Difference-corrected (cross) power spectra.

    Reference: Bachetti \& Huppenkothen, 2018, ApJ, 853L, 21

    Parameters
    ----------
    lc1: class:`stingray.ligthtcurve.Lightcurve`
    lc1: class:`stingray.ligthtcurve.Lightcurve`
    segment_size: float
        Length of the segments to be averaged

    Other parameters
    ----------------
    plot : bool
        Plot diagnostics
    smoothing_alg : {'gauss', 'spline'}
        Smoothing algorithm
    smoothing_length : int
        Number of bins to smooth in gaussian window smoothing
    verbose: bool
        Print out information on the outcome of the algorithm (recommended)
    tolerance : float
        Accepted relative error on the FAD-corrected Fourier amplitude, to be
        used as success diagnostics.
        Should be
        ```
        stdtheor = 2 / np.sqrt(n)
        std = (average_corrected_fourier_diff / n).std()
        np.abs((std - stdtheor) / stdtheor) < tolerance
    strict : bool
        Fail if condition on tolerance is not met.

    Returns
    -------
    results : class:`astropy.Table` object
        This table contains the results of the FAD correction, in its columns:
        pds1: the corrected PDS of ``lc1``
        pds2: the corrected PDS of ``lc2``
        cs: the corrected cospectrum
        ptot: the corrected PDS of lc1 + lc2

    """
    if smoothing_length is None:
        smoothing_length = segment_size * 3
    if gti is None:
        gti = cross_two_gtis(lc1.gti, lc2.gti)
    lc1.gti = gti
    lc2.gti = gti
    lc1.apply_gtis()
    lc2.apply_gtis()
    summed_lc = lc1 + lc2
    start_inds, end_inds = \
        bin_intervals_from_gtis(gti, segment_size, lc1.time,
                                dt=lc1.dt)
    freq = 0
    pds1 = 0
    pds2 = 0
    ptot = 0
    cs = 0
    n = 0
    average_diff = average_diff_uncorr = 0

    if plot:
        plt.figure()

    for start_ind, end_ind in zip(start_inds, end_inds):
        freq, f1, f1_leahy = _get_fourier_intv(lc1, start_ind, end_ind)
        freq, f2, f2_leahy = _get_fourier_intv(lc2, start_ind, end_ind)
        freq, ftot, ftot_leahy = \
            _get_fourier_intv(summed_lc, start_ind, end_ind)

        fourier_diff = f1_leahy - f2_leahy

        if smoothing_alg == 'gauss':
            smooth_real = gaussian_filter1d(fourier_diff.real ** 2,
                                            smoothing_length)
        else:
            raise ValueError("Unknown smoothing algorithm: {}".format(
                smoothing_alg))

        if plot:
            plt.scatter(freq, fourier_diff, s=1)

        if all_leahy:
            f1 = f1_leahy
            f2 = f2_leahy
            ftot = ftot_leahy
        p1 = (f1 * f1.conj()).real
        p1 = p1 / smooth_real * 2
        p2 = (f2 * f2.conj()).real
        p2 = p2 / smooth_real * 2
        pt = (ftot * ftot.conj()).real
        pt = pt / smooth_real * 2

        c = (f2 * f1.conj()).real
        c = c / smooth_real * 2

        if n == 0 and plot:
            plt.plot(freq, smooth_real, zorder=10, lw=3)
            plt.plot(freq, f1_leahy, zorder=5, lw=1)
            plt.plot(freq, f2_leahy, zorder=5, lw=1)

        ptot += pt
        pds1 += p1
        pds2 += p2
        cs += c
        average_diff += fourier_diff / smooth_real ** 0.5 * np.sqrt(2)
        average_diff_uncorr += fourier_diff
        n += 1

    std = (average_diff / n).std()
    stdtheor = 2 / np.sqrt(n)
    stduncorr = (average_diff_uncorr / n).std()
    is_compliant = np.abs((std - stdtheor) / stdtheor) < tolerance
    verbose_string = \
        '''
    -------- FAD correction ----------
    I smoothed over {smoothing_length} power spectral bins
    {n} intervals averaged.
    The uncorrected standard deviation of the Fourier
    differences is {stduncorr} (dead-time affected!)
    The final standard deviation of the FAD-corrected
    Fourier differences is {std}. For the results to be
    acceptable, this should be close to {stdtheor}
    to within {tolerance} %.
    In this case, the results ARE {compl}complying.
    {additional}
    ----------------------------------
    '''.format(smoothing_length=smoothing_length,
               n=n,
               stduncorr=stduncorr,
               std=std,
               stdtheor=stdtheor,
               tolerance=tolerance * 100,
               compl='NOT ' if not is_compliant else '',
               additional='Maybe something is not right.' if not is_compliant else '')

    print(verbose_string)
    if verbose and is_compliant:
        log.info(verbose_string)
    elif not is_compliant:
        warnings.warn(verbose_string)

    if strict:
        assert is_compliant

    results = Table()

    results['freq'] = freq
    results['pds1'] = pds1 / n
    results['pds2'] = pds2 / n
    results['cs'] = cs / n
    results['ptot'] = ptot / n
    results['fad'] = average_diff / n
    results.meta['fad_delta'] = (std - stdtheor) / stdtheor
    results.meta['is_compliant'] = is_compliant
    results.meta['n'] = n
    results.meta['smoothing_length'] = smoothing_length

    return results
