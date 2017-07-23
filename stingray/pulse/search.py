from __future__ import division, print_function
import numpy as np
from .pulsar import stat, fold_events, z_n
from ..utils import jit, HAS_NUMBA
from ..utils import contiguous_regions


@jit(nopython=True)
def _pulse_phase_fast(time, f, buffer_array):
    for i in range(len(time)):
        buffer_array[i] = time[i] * f
        buffer_array[i] -= np.floor(buffer_array[i])
    return buffer_array


def _folding_search(stat_func, times, frequencies, segment_size=5000):
    stats = np.zeros_like(frequencies)
    times = (times - times[0]).astype(np.float64)
    length = times[-1]
    if length < segment_size:
        segment_size = length
    start_times = np.arange(times[0], times[-1], segment_size)
    count = 0
    for s in start_times:
        ts = times[(times >=s) & (times < s + segment_size)]
        buffer = np.zeros_like(ts)
        if len(ts) < 1 or ts[-1] - ts[0] < 0.2 * segment_size:
            continue
        for i, f in enumerate(frequencies):
            phases = _pulse_phase_fast(ts, f, buffer)
            stats[i] += stat_func(phases)
        count += 1
    return frequencies, stats / count


@jit(nopython=True)
def _bincount_fast(phase):
    return np.bincount(phase)


@jit(nopython=True)
def _profile_fast(phase, nbin=128):
    phase_bin = np.zeros(len(phase) + 2, dtype=np.int64)
    # This is done to force bincount from 0 to nbin -1
    phase_bin[-1] = nbin - 1
    phase_bin[-2] = 0
    for i in range(len(phase)):
        phase_bin[i] = np.int64(np.floor(phase[i] * nbin))
    bc = _bincount_fast(phase_bin)
    bc[0] -= 1
    bc[-1] -= 1
    return bc


def epoch_folding_search(times, frequencies, nbin=128, segment_size=5000,
                         expocorr=False):
    """Performs epoch folding at trial frequencies in photon data.

    If no exposure correction is needed and numba is installed, it uses a fast
    algorithm to perform the folding. Otherwise, it runs a *much* slower
    algorithm, which however yields a more precise result.
    The search can be done in segments and the results averaged. Use
    segment_size to control this

    Parameters
    ----------
    times : array-like
        the event arrival times
    frequencies : array-like
        the trial values for the frequencies

    Other Parameters
    ----------------
    nbin : int
        the number of bins of the folded profiles
    segment_size : float
        the length of the segments to be averaged in the periodogram
    expocorr : bool
        correct for the exposure (Use it if the period is comparable to the
        length of the good time intervals.)
    """
    if expocorr or not HAS_NUMBA:
        return \
            _folding_search(lambda x: stat(fold_events(np.sort(x), 1,
                                                       nbin=nbin,
                                                       expocorr=expocorr)[1]),
                               times, frequencies, segment_size=segment_size)

    return _folding_search(lambda x: stat(_profile_fast(x, nbin=nbin)),
                           times, frequencies, segment_size=segment_size)


def z_n_search(times, frequencies, nharm=4, nbin=128, segment_size=5000,
               expocorr=False):
    """Calculates the Z^2_n statistics at trial frequencies in photon data.

    The "real" Z^2_n statistics is very slow. Therefore, in this function data
    are folded first, and then the statistics is calculated using the value of
    the profile as an additional normalization term.
    The two methods are mostly equivalent. However, the number of bins has to
    be chosen wisely: if the number of bins is too small, the search for high
    harmonics is ineffective.
    If no exposure correction is needed and numba is installed, it uses a fast
    algorithm to perform the folding. Otherwise, it runs a *much* slower
    algorithm, which however yields a more precise result.
    The search can be done in segments and the results averaged. Use
    segment_size to control this

    Parameters
    ----------
    times : array-like
        the event arrival times
    frequencies : array-like
        the trial values for the frequencies

    Other Parameters
    ----------------
    nbin : int
        the number of bins of the folded profiles
    segment_size : float
        the length of the segments to be averaged in the periodogram
    expocorr : bool
        correct for the exposure (Use it if the period is comparable to the
        length of the good time intervals.)
    """
    phase = np.arange(0, 1, 1 / nbin)
    if expocorr or not HAS_NUMBA:
        return \
            _folding_search(
                lambda x: z_n(phase, n=nharm,
                              norm=fold_events(np.sort(x), 1, nbin=nbin,
                                               expocorr=expocorr)[1]),
                               times, frequencies, segment_size=segment_size)

    return _folding_search(lambda x: z_n(phase, n=nharm,
                                         norm=_profile_fast(x, nbin=nbin)),
                           times, frequencies, segment_size=segment_size)


def search_best_peaks(x, stat, threshold):
    """Search peaks above threshold in an epoch folding periodogram.

    If more values of stat are above threshold and are contiguous, only the
    largest one is returned (see Examples).

    Parameters
    ----------
    x : array-like
        The x axis of the periodogram (frequencies, periods, ...)
    stat : array-like
        The y axis. It must have the same shape as x
    threshold : float
        The threshold value over which we look for peaks in the stat array

    Examples
    --------
    >>> # Test multiple peaks
    >>> x = np.arange(10)
    >>> stat = [0, 0, 0.5, 0, 0, 1, 1, 2, 1, 0]
    >>> best = search_best_peaks(x, stat, 0.5)
    >>> len(best)
    2
    >>> best[0]
    2.0
    >>> best[1]
    7.0
    >>> # Test no peak above threshold
    >>> x = np.arange(10)
    >>> stat = [0, 0, 0.4, 0, 0, 0, 0, 0, 0, 0]
    >>> search_best_peaks(x, stat, 0.5)
    []

    Returns
    -------
    best_x : array-like
        the array containing the peaks above threshold. If no peaks are above
        threshold, an empty list is returned.
    """
    stat = np.asarray(stat)
    x = np.asarray(x)
    peaks = stat >= threshold
    regions = contiguous_regions(peaks)
    if len(regions) == 0:
        return []
    best_x = np.zeros(len(regions))
    for i, r in enumerate(regions):
        best_x[i] = x[r[0]:r[1]][np.argmax(stat[r[0]:r[1]])]

    return best_x
