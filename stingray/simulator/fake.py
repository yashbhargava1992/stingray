
"""Functions to simulate event lists data."""

import numpy as np
import numpy.random as ra
from stingray.simulator.utils import _assign_value_if_none
import logging
import warnings

def fake_events_from_lc(
        times, lc, use_spline=False, bin_time=None):

    """Create events from a light curve.

    Parameters
    ----------
    times : array-like
        the center time of each light curve bin
    lc : array-like
        light curve, in units of counts/bin

    Returns
    -------
    event_list : array-like
        Simulated arrival times
    """
    try:
        import scipy.interpolate as sci
    except:
        if use_spline:
            warnings.warn("Scipy not available. ",
                          "use_spline option cannot be used.")
            use_spline = False

    # Cast as numpy arrays in case inputs are in list format
    times = np.asarray(times)
    lc = np.asarray(lc)

    bin_time = _assign_value_if_none(bin_time, times[1] - times[0])
    n_bin = len(lc)

    bin_start = 0

    maxlc = np.max(lc)
    intlc = maxlc * n_bin

    n_events_predict = int(intlc + 10 * np.sqrt(intlc))

    # Max number of events per chunk must be < 100000
    events_per_bin_predict = n_events_predict / n_bin
    if use_spline:
        max_bin = np.max([4, 1000000 / events_per_bin_predict])
        logging.debug("Using splines")
    else:
        max_bin = np.max([4, 5000000 / events_per_bin_predict])

    ev_list = np.zeros(n_events_predict)

    nev = 0

    while bin_start < n_bin:
        t0 = times[bin_start]
        bin_stop = min([bin_start + max_bin, n_bin + 1])
        lc_filt = lc[bin_start:bin_stop]
        t_filt = times[bin_start:bin_stop]
        logging.debug("{} {}".format(t_filt[0] - bin_time / 2,
                                     t_filt[-1] + bin_time / 2))

        length = t_filt[-1] - t_filt[0]
        n_bin_filt = len(lc_filt)
        n_to_simulate = n_bin_filt * max(lc_filt)
        safety_factor = 10
        if n_to_simulate > 10000:
            safety_factor = 4.

        n_to_simulate += safety_factor * np.sqrt(n_to_simulate)
        n_to_simulate = int(np.ceil(n_to_simulate))

        n_predict = ra.poisson(np.sum(lc_filt))

        random_ts = ra.uniform(t_filt[0] - bin_time / 2,
                               t_filt[-1] + bin_time / 2, n_to_simulate)

        logging.debug(random_ts[random_ts < 0])

        random_amps = ra.uniform(0, max(lc_filt), n_to_simulate)
        if use_spline:
            # print("Creating spline representation")
            lc_spl = sci.splrep(t_filt, lc_filt, s=np.longdouble(0), k=1)

            pts = sci.splev(random_ts, lc_spl)
        else:
            rough_bins = np.rint((random_ts - t0) / bin_time)
            rough_bins = rough_bins.astype(int)

            pts = lc_filt[rough_bins]
            #pts = [lc_filt[bin] for bin in rough_bins]

        good = random_amps < pts
        len1 = len(random_ts)
        random_ts = random_ts[good]
        len2 = len(random_ts)
        logging.debug("Max LC, nbin: {0} {1}".format(max(lc_filt), n_bin_filt))
        logging.debug("{0} Events generated".format(len1))
        logging.debug("{0} Events predicted".format(n_predict))
        logging.debug("{0} Events rejected".format(len1 - len2))
        random_ts = random_ts[:n_predict]
        random_ts.sort()
        new_nev = len(random_ts)
        ev_list[nev:nev + new_nev] = random_ts[:]
        nev += new_nev
        logging.debug(
            "{0} good events created ({1} ev/s)".format(new_nev,
                                                        new_nev / length))
        bin_start += max_bin

    # Discard all zero entries at the end!
    ev_list = ev_list[:nev]
    ev_list.sort()
    return ev_list
