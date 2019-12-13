import warnings
import copy

import numpy as np
import numpy.random as ra
from astropy import log
from astropy.logger import AstropyUserWarning

from ..utils import njit
from ..events import EventList


def _paralyzable_dead_time(event_list, dead_time):
    mask = np.ones(len(event_list), dtype=bool)
    dead_time_end = event_list + dead_time
    bad = dead_time_end[:-1] > event_list[1:]
    # Easy: paralyzable case. Here, events coming during dead time produce
    # more dead time. So...
    mask[1:][bad] = False

    return event_list[mask], mask


@njit()
def _nonpar_core(event_list, dead_time_end, mask):
    for i in range(1, len(event_list)):
        if (event_list[i] < dead_time_end[i - 1]):
            dead_time_end[i] = dead_time_end[i - 1]
            mask[i] = False
    return mask


def _non_paralyzable_dead_time(event_list, dead_time):
    event_list_dbl = (event_list - event_list[0]).astype(np.double)
    dead_time_end = event_list_dbl + np.double(dead_time)
    mask = np.ones(event_list_dbl.size, dtype=bool)
    mask = _nonpar_core(event_list_dbl, dead_time_end, mask)
    return event_list[mask], mask


class DeadtimeFilterOutput(object):
    uf_events = None
    is_event = None
    deadtime = None
    mask = None
    bkg = None


def filter_for_deadtime(event_list, deadtime, bkg_ev_list=None,
                        dt_sigma=None, paralyzable=False,
                        additional_data=None, return_all=False):
    """Filter an event list for a given dead time.

    Parameters
    ----------
    ev_list : array-like or class:`stingray.events.EventList`
        The event list
    deadtime: float
        The (mean, if not constant) value of deadtime

    Other Parameters
    ----------------
    bkg_ev_list : array-like
        A background event list that affects dead time
    dt_sigma : float
        The standard deviation of a non-constant dead time around deadtime.
    return_all : bool
        If True, return the mask that filters the input event list to obtain
        the output event list.

    Returns
    -------
    new_ev_list : class:`stingray.events.EventList` object
        The filtered event list
    additional_output : dict
        Object with all the following attributes. Only returned if
        `return_all` is True
    uf_events : array-like
        Unfiltered event list (events + background)
    is_event : array-like
        Boolean values; True if event, False if background
    deadtime : array-like
        Dead time values
    bkg : array-like
        The filtered background event list
    mask : array-like, optional
        The mask that filters the input event list and produces the output
        event list.

    """
    additional_output = DeadtimeFilterOutput()
    if not isinstance(event_list, EventList):
        event_list_obj = EventList(event_list)
    else:
        event_list_obj = event_list

    ev_list = event_list_obj.time

    if deadtime <= 0.:
        return copy.deepcopy(event_list)

    # Create the total lightcurve, and a "kind" array that keeps track
    # of the events classified as "signal" (True) and "background" (False)
    if bkg_ev_list is not None:
        tot_ev_list = np.append(ev_list, bkg_ev_list)
        ev_kind = np.append(np.ones(len(ev_list), dtype=bool),
                            np.zeros(len(bkg_ev_list), dtype=bool))
        order = np.argsort(tot_ev_list)
        tot_ev_list = tot_ev_list[order]
        ev_kind = ev_kind[order]
        del order
    else:
        tot_ev_list = ev_list
        ev_kind = np.ones(len(ev_list), dtype=bool)

    nevents = len(tot_ev_list)
    all_ev_kind = ev_kind.copy()

    if dt_sigma is not None:
        deadtime_values = ra.normal(deadtime, dt_sigma, nevents)
    else:
        deadtime_values = np.zeros(nevents) + deadtime

    initial_len = len(tot_ev_list)

    if paralyzable:
        tot_ev_list, saved_mask = \
            _paralyzable_dead_time(tot_ev_list, deadtime_values)

    else:
        tot_ev_list, saved_mask = \
            _non_paralyzable_dead_time(tot_ev_list, deadtime_values)

    ev_kind = ev_kind[saved_mask]
    deadtime_values = deadtime_values[saved_mask]
    final_len = len(tot_ev_list)
    log.info(
        'filter_for_deadtime: '
        '{0}/{1} events rejected'.format(initial_len - final_len,
                                         initial_len))
    retval = EventList(time=tot_ev_list[ev_kind], mjdref=event_list_obj.mjdref)

    if hasattr(event_list_obj, 'pi') and event_list_obj.pi is not None:
        warnings.warn(
            "PI information is lost during dead time filtering",
            AstropyUserWarning)

    if not isinstance(event_list, EventList):
        retval = retval.time

    if return_all:
        additional_output.uf_events = tot_ev_list
        additional_output.is_event = ev_kind
        additional_output.deadtime = deadtime_values
        additional_output.mask = saved_mask[all_ev_kind]
        additional_output.bkg = tot_ev_list[np.logical_not(ev_kind)]
        retval = [retval, additional_output]

    return retval
