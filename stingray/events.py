"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np
from lightcurve import Lightcurve
from stingray.simulator.events import events_to_lc, lc_to_events, assign_energies


class EventList(object):
    def __init__(self, times, energies=None, mjdref=0, dt=0, notes="", gti=None, pi=None,
                 pha=None):
        """
        Make an event list object from an array of time stamps

        Parameters
        ----------
        times: iterable
            A list or array of time stamps

        Other Parameters
        ----------------
        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        energies: iterable
            A list of array of photon energy values

        mjdref : float
            The MJD used as a reference for the time array.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        pha : float, numpy.ndarray
            Photon energies

        Attributes
        ----------
        time: numpy.ndarray
            The array of event arrival times, in seconds from the reference
            MJD (self.mjdref)

        energies: numpy.ndarray
            The array of photon energy values

        ncounts: int
            The number of data points in the event list

        dt: float
            The time resolution of the events. Only relevant when using events
            to produce light curves with similar bin time.

        mjdref : float
            The MJD used as a reference for the time array.

        gtis: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            Good Time Intervals

        pi : integer, numpy.ndarray
            PI channels

        pha : float, numpy.ndarray
            Photon energies

        """
        self.time = np.array(times, dtype=np.longdouble)
        self.energies = np.array(energies)
        self.ncounts = len(times)
        self.notes = notes
        self.dt = dt
        self.mjdref = mjdref
        self.gti = gti
        self.pi = pi
        self.pha = pha

    @staticmethod
    def from_fits(fname, **kwargs):
        from .utils import _assign_value_if_none
        from .io import load_events_and_gtis

        ret = load_events_and_gtis(fname, **kwargs)

        times = ret.ev_list
        gti = ret.gtis

        pi = _assign_value_if_none(ret.additional_data, "PI", None)
        pha = _assign_value_if_none(ret.additional_data, "PHA", None)
        dt = ret.dt
        mjdref = ret.mjdref

        return EventList(times, gti=gti, pi=pi, pha=pha, dt=dt, mjdref=mjdref)

    def from_lc(lc):
        """
        Make an event list with light curve as input.

        Parameters
        ----------
        lc: `Lightcurve` object
        """ 

        self = EventList(events_to_lc(lc.counts))

    def to_lc(bin_time, start_time=None, stop_time=None, center_time=True):
        """
        Convert event list to a light curve object.

        Parameters
        ----------
        bin_time : float
            Binning time of the light curve

        Other Parameters
        ----------------
        start_time : float
            Initial time of the light curve
        stop_time : float
            Stop time of the light curve
        center_time: bool
            If False, time is the start of the bin. Otherwise, the center

        Returns
        -------
        lc: `Lightcurve` object
        """

        times, counts = events_to_lc(self.time, bin_time, start_time,
            stop_time, center_time)

        return Lightcurve(times, counts)

    def energies(spectrum):
        """
        Assign energies to event list.

        Parameters
        ----------
        spectrum: 2-d array or list
            Energies versus corresponding fluxes. The 2-d array or list must
            have energies across the first dimension and fluxes across the
            second one.
        """

        self.energies = assign_energies(self.ncounts, spectrum)
