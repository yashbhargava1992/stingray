from collections.abc import Iterable

import numpy as np
import scipy
from scipy import special
from astropy import log
from astropy.table import Table

try:
    from tqdm import tqdm as show_progress
except ImportError:
    def show_progress(a):
        return a

from ..utils import njit, prange

from ..gti import create_gti_mask


@njit()
def pds_from_fft(spectr, nph):
    return (spectr * spectr.conj()).real * 2 / nph


def probability_of_power(level, nbins, n_summed_spectra=1, n_rebin=1):
    r"""Give the probability of a given power level in PDS.

    Return the probability of a certain power level in a Power Density
    Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)
    """
    try:
        from scipy import stats
    except Exception:  # pragma: no cover
        raise Exception('You need Scipy to use this function')

    epsilon = nbins * stats.chi2.sf(level * n_summed_spectra * n_rebin,
                                    2 * n_summed_spectra * n_rebin)
    return epsilon


def detection_level(nbins, epsilon=0.01, n_summed_spectra=1, n_rebin=1):
    r"""Detection level for a PDS.

    Return the detection level (with probability 1 - epsilon) for a Power
    Density Spectrum of nbins bins, normalized a la Leahy (1983), based on
    the 2-dof :math:`{\chi}^2` statistics, corrected for rebinning (n_rebin)
    and multiple PDS averaging (n_summed_spectra)
    Examples
    --------
    >>> np.isclose(detection_level(1, 0.1), 4.6, atol=0.1)
    True
    >>> np.allclose(detection_level(1, 0.1, n_rebin=[1]), [4.6], atol=0.1)
    True
    """
    try:
        from scipy import stats
    except Exception:  # pragma: no cover
        raise Exception('You need Scipy to use this function')

    if not isinstance(n_rebin, Iterable):
        r = n_rebin
        retlev = stats.chi2.isf(epsilon / nbins, 2 * n_summed_spectra * r) \
            / (n_summed_spectra * r)
    else:
        retlev = [stats.chi2.isf(epsilon / nbins, 2 * n_summed_spectra * r) /
                  (n_summed_spectra * r) for r in n_rebin]
        retlev = np.array(retlev)
    return retlev


def _create_responses(range_z):
    log.info("Creating responses")
    responses = []
    for j, z in enumerate(show_progress(range_z)):
        # fdot = z / T**2
        if( np.abs( z ) < 0.01 ):
             responses.append(0)
             continue

        m = np.max([np.abs(np.int( 2 * z)), 40]) #np.abs( np.rint( ( 2 * z ) ) )#maximum_m_idx
        sign = z / np.abs(z)
        absz = np.abs(z)
        factor = sign * 1 / scipy.sqrt( 2 * absz)

        q_ks = np.arange(-m / 2, m / 2+ 1)

        exponentials = scipy.exp(1j * np.pi * q_ks**2 / z)

        Yks = sign * scipy.sqrt( 2 / absz ) * q_ks
        Zks = sign * scipy.sqrt( 2 / absz ) * ( q_ks + z )
        # print(Yks, Zks)
        [SZs, CZs] = special.fresnel(Zks)
        [SYs, CYs] = special.fresnel(Yks)
        weights = SZs - SYs + 1j * (CYs - CZs)
        responses.append(weights * exponentials * factor)
    return responses


@njit()
def convolve(a, b):
    return np.convolve(a, b)


def _calculate_all_convolutions(A, responses, n_photons, freq_intv_to_search,
                               detlev, debug=False, interbin=False):
    log.info("Convolving FFT with responses...")
    candidate_powers = [0.]
    candidate_rs = [1]
    candidate_js = [2]
    r_freqs = np.arange(A.size)
    len_responses = len(responses)
    if debug:
        fobj = open('accelsearch_dump.dat', 'w')
    for j in show_progress(prange(len_responses)):
        response = responses[j]
        if np.asarray(response).size == 1:
            accel = A
        else:
            accel = convolve(A, response)
            new_size = accel.size
            diff = new_size - A.size
            accel = accel[diff // 2: diff // 2 + A.size]

        rf = r_freqs[freq_intv_to_search]
        accel = accel[freq_intv_to_search]
        if interbin:
            rf, accel = \
                interbin_fft(rf, accel)

        powers_to_search = pds_from_fft(accel, n_photons)

        if debug:
            print(*powers_to_search, file=fobj)

        candidate = powers_to_search > detlev
        rs = rf[candidate]
        cand_powers= powers_to_search[candidate]
        for i in range(len(rs)):
            r = rs[i]
            cand_power = cand_powers[i]
            candidate_powers.append(cand_power)
            candidate_rs.append(r)
            candidate_js.append(j)
    if debug:
        fobj.close()
    return candidate_rs[1:], candidate_js[1:], candidate_powers[1:]


def accelsearch(times, signal, delta_z=1, fmin=1, fmax=1e32,
                GTI=None, zmax=100, candidate_file=None, ref_time=0,
                debug=False, interbin=False):
    """Find pulsars with accelerated search.

    The theory behind these methods is described in Ransom+02, AJ 124, 1788.

    Parameters
    ----------
    times : array of floats
        An evenly spaced list of times
    signal : array of floats
        The light curve, in counts; same length as ``times``
    Other parameters
    ----------------
    delta_z : float
        The spacing in ``z`` space
    fmin : float, default 1.
        Minimum frequency to search
    fmax : float, default 1e32
        Maximum frequency to search
    GTI : ``[[gti00, gti01], [gti10, gti11], ...]``, default None
        Good Time Intervals. If None, it assumes the full range
        ``[[time[0] - dt / 2 -- time[-1] + dt / 2]]``
    zmax : int, default 100
        Maximum frequency derivative to search (pos and neg), in bins.
        It corresponds to ``fdot_max = zmax / T**2``, where ``T`` is the
        length of the observation.
    candidate_file : str, default None
        Save the final candidate table in this file.
    ref_time : float, default 0
        Reference time for the times
    """
    times = np.asarray(times)
    signal = np.asarray(signal)
    dt = times[1] - times[0]
    if GTI is not None:
        GTI = np.asarray(GTI)
        # Fill in the data with a constant outside GTI
        gti_mask = create_gti_mask(times, GTI)
        bti_mask = ~gti_mask
        signal[bti_mask] = np.median(signal[gti_mask])
    else:
        GTI = np.array(
            [[times[0] - dt /2, times[-1] + dt / 2]])

    n_photons = np.sum(signal)
    spectr = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(spectr), dt)
    T = times[-1] - times[0] + dt

    freq_intv_to_search = (freq >= fmin) & (freq < fmax)
    log.info("Starting search over full plane...")
    start_z = -zmax
    end_z = zmax
    range_z = np.arange(start_z,end_z, delta_z)
    log.info("min and max possible r_dot: {}--{}".format(delta_z/T**2,
                                                         np.max(range_z)/T**2))
    freqs_to_search = freq[freq_intv_to_search]

    candidate_table = Table(
        names=['time', 'power', 'prob', 'frequency', 'fdot', 'fddot'],
        dtype=[float] * 6)

    detlev = detection_level(freqs_to_search.size, epsilon=0.015)

    RESPONSES = _create_responses(range_z)

    candidate_rs, candidate_js, candidate_powers = \
        _calculate_all_convolutions(spectr, RESPONSES, n_photons,
                                    freq_intv_to_search, detlev,
                                    debug=debug, interbin=interbin)

    for r, j, cand_power in zip(candidate_rs, candidate_js, candidate_powers):
        z = range_z[j]
        cand_freq = r / T
        fdot = z / T**2
        prob = probability_of_power(cand_power, freqs_to_search.size)
        candidate_table.add_row(
            [ref_time + GTI[0, 0], cand_power, prob, cand_freq, fdot, 0])

    if candidate_file is not None:
        candidate_table.write(candidate_file + '.csv', overwrite=True)

    return candidate_table


def interbin_fft(freq, fft):
    """Interbinning, a la van der Klis 1989.

    Allows to recover some sensitivity when the pulsation frequency
    is close to bin edge.

    Examples
    --------
    >>> import numpy as np
    >>> freq = [0, 0.5, 1, -1, -0.5]
    >>> fft = np.array([1, 0, 1, 1, 0], dtype=float)
    >>> f, F = interbin_fft(freq, fft)
    >>> np.allclose(f, [0, 0.25, 0.5, 0.75, 1, -1, -0.75, -0.5, -0.25])
    True
    >>> pi_4 = np.pi / 4
    >>> np.allclose(F, [1, -pi_4, 0, pi_4, 1, 1, -pi_4, 0, pi_4])
    True
    """
    import numpy as np

    freq = np.asarray(freq)
    fft = np.asarray(fft)

    neglast = freq[-1] < 0
    if neglast:
        order = np.argsort(freq)
        freq = freq[order]
        fft = fft[order]

    N = freq.size

    new_N = 2 * N - 1

    new_freqs = np.linspace(freq[0], freq[-1], new_N)

    new_fft = np.zeros(new_N, dtype=type(fft[0]))
    new_fft[::2] = fft
    new_fft[1::2] = (fft[1:] - fft[:-1]) * np.pi / 4

    if neglast:
        fneg = new_freqs < 0
        fpos = ~fneg
        new_freqs = np.concatenate((new_freqs[fpos], new_freqs[fneg]))
        new_fft = np.concatenate((new_fft[fpos], new_fft[fneg]))

    return new_freqs, new_fft
