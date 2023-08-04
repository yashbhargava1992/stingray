"""
Definition of :class:`EventList`.

:class:`EventList` is used to handle photon arrival times.
"""

import copy
import logging
import pickle
import warnings
from collections.abc import Iterable

import numpy as np
import numpy.random as ra
from astropy.table import Table

from .base import StingrayObject, StingrayTimeseries
from .filters import get_deadtime_mask
from .gti import append_gtis, check_separate, cross_gtis, generate_indices_of_boundaries
from .io import load_events_and_gtis
from .lightcurve import Lightcurve
from .utils import assign_value_if_none, simon, interpret_times

__all__ = ["EventList"]


class EventList(StingrayTimeseries):
    """
    Basic class for event list data. Event lists generally correspond to individual events (e.g. photons)
    recorded by the detector, and their associated properties. For X-ray data where this type commonly occurs,
    events are time stamps of when a photon arrived in the detector, and (optionally) the photon energy associated
    with the event.

    Parameters
    ----------
    time: iterable
        A list or array of time stamps

    Other Parameters
    ----------------
    dt: float
        The time resolution of the events. Only relevant when using events
        to produce light curves with similar bin time.

    energy: iterable
        A list of array of photon energy values in keV

    mjdref : float
        The MJD used as a reference for the time array.

    ncounts: int
        Number of desired data points in event list.

    gtis: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals

    pi : integer, numpy.ndarray
        PI channels

    notes : str
        Any useful annotations

    high_precision : bool
        Change the precision of self.time to float128. Useful while dealing with fast pulsars.

    mission : str
        Mission that recorded the data (e.g. NICER)

    instr : str
        Instrument onboard the mission

    header : str
        The full header of the original FITS file, if relevant

    detector_id : iterable
        The detector that recorded each photon (if the instrument has more than
        one, e.g. XMM/EPIC-pn)

    timeref : str
        The time reference, as recorded in the FITS file (e.g. SOLARSYSTEM)

    timesys : str
        The time system, as recorded in the FITS file (e.g. TDB)

    ephem : str
        The JPL ephemeris used to barycenter the data, if any (e.g. DE430)

    **other_kw :
        Used internally. Any other keyword arguments will be ignored

    Attributes
    ----------
    time: numpy.ndarray
        The array of event arrival times, in seconds from the reference
        MJD defined in ``mjdref``

    energy: numpy.ndarray
        The array of photon energy values

    ncounts: int
        The number of data points in the event list

    dt: float
        The time resolution of the events. Only relevant when using events
        to produce light curves with similar bin time.

    mjdref : float
        The MJD used as a reference for the time array.

    gtis: ``[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]``
        Good Time Intervals

    pi : integer, numpy.ndarray
        PI channels

    high_precision : bool
        Change the precision of self.time to float128. Useful while dealing with fast pulsars.

    mission : str
        Mission that recorded the data (e.g. NICER)

    instr : str
        Instrument onboard the mission

    detector_id : iterable
        The detector that recoded each photon, if relevant (e.g. XMM, Chandra)

    header : str
        The full header of the original FITS file, if relevant

    """

    main_array_attr = "time"

    def __init__(
        self,
        time=None,
        energy=None,
        ncounts=None,
        mjdref=0,
        dt=0,
        notes="",
        gti=None,
        pi=None,
        high_precision=False,
        mission=None,
        instr=None,
        header=None,
        detector_id=None,
        ephem=None,
        timeref=None,
        timesys=None,
        **other_kw,
    ):
        StingrayObject.__init__(self)

        self.energy = None if energy is None else np.asarray(energy)
        self.notes = notes
        self.dt = dt
        self.mjdref = mjdref
        self.gti = np.asarray(gti) if gti is not None else None
        self.pi = None if pi is None else np.asarray(pi)
        self.ncounts = ncounts
        self.mission = mission
        self.instr = instr
        self.detector_id = detector_id
        self.header = header
        self.ephem = ephem
        self.timeref = timeref
        self.timesys = timesys

        if other_kw != {}:
            warnings.warn(f"Unrecognized keywords: {list(other_kw.keys())}")

        if time is not None:
            time, mjdref = interpret_times(time, mjdref)
            if not high_precision:
                self.time = np.asarray(time)
            else:
                self.time = np.asarray(time, dtype=np.longdouble)
            self.ncounts = self.time.size
        else:
            self.time = None

        if (self.time is not None) and (self.energy is not None):
            if self.time.size != self.energy.size:
                raise ValueError("Lengths of time and energy must be equal.")

    def to_lc(self, dt, tstart=None, tseg=None):
        """
        Convert event list to a :class:`stingray.Lightcurve` object.

        Parameters
        ----------
        dt: float
            Binning time of the light curve

        Other Parameters
        ----------------
        tstart : float
            Start time of the light curve

        tseg: float
            Total duration of light curve

        Returns
        -------
        lc: :class:`stingray.Lightcurve` object
        """
        if tstart is None and self.gti is not None:
            tstart = self.gti[0][0]
            tseg = self.gti[-1][1] - tstart

        return Lightcurve.make_lightcurve(
            self.time, dt, tstart=tstart, gti=self.gti, tseg=tseg, mjdref=self.mjdref
        )

    def to_lc_iter(self, dt, segment_size=None):
        """Convert event list to a generator of Lightcurves.

        Parameters
        ----------
        dt: float
            Binning time of the light curves

        Other parameters
        ----------------
        segment_size : float, default None
            Optional segment size. If None, use the GTI boundaries

        Returns
        -------
        lc_gen: `generator`
            Generates one :class:`stingray.Lightcurve` object for each GTI or segment
        """

        segment_iter = generate_indices_of_boundaries(
            self.time, self.gti, segment_size=segment_size, dt=0
        )

        for st, end, idx_st, idx_end in segment_iter:
            tseg = end - st

            lc = Lightcurve.make_lightcurve(
                self.time[idx_st : idx_end + 1],
                dt,
                tstart=st,
                gti=np.asarray([[st, end]]),
                tseg=tseg,
                mjdref=self.mjdref,
                use_hist=True,
            )
            yield lc

    def to_lc_list(self, dt, segment_size=None):
        """Convert event list to a list of Lightcurves.

        Parameters
        ----------
        dt: float
            Binning time of the light curves

        Other parameters
        ----------------
        segment_size : float, default None
            Optional segment size. If None, use the GTI boundaries

        Returns
        -------
        lc_list: `List`
            List containig one :class:`stingray.Lightcurve` object for each GTI or segment
        """
        return list(self.to_lc_iter(dt, segment_size))

    @staticmethod
    def from_lc(lc):
        """
        Create an :class:`EventList` from a :class:`stingray.Lightcurve` object. Note that all
        events in a given time bin will have the same time stamp.

        Parameters
        ----------
        lc: :class:`stingray.Lightcurve` object
            Light curve to use for creation of the event list.

        Returns
        -------
        ev: :class:`EventList` object
            The resulting list of photon arrival times generated from the light curve.
        """

        # Multiply times by number of counts
        times = [[i] * int(j) for i, j in zip(lc.time, lc.counts)]
        # Concatenate all lists
        times = [i for j in times for i in j]

        return EventList(time=times, gti=lc.gti)

    def simulate_times(self, lc, use_spline=False, bin_time=None):
        """Simulate times from an input light curve.

        Randomly simulate photon arrival times to an :class:`EventList` from a
        :class:`stingray.Lightcurve` object, using the inverse CDF method.

        ..note::
            Preferably use model light curves containing **no Poisson noise**,
            as this method will intrinsically add Poisson noise to them.

        Parameters
        ----------
        lc: :class:`stingray.Lightcurve` object

        Other Parameters
        ----------------
        use_spline : bool
            Approximate the light curve with a spline to avoid binning effects

        bin_time : float default None
            Ignored and deprecated, maintained for backwards compatibility.

        Returns
        -------
        times : array-like
            Simulated photon arrival times
        """
        # Need import here, or there will be a circular import
        from .simulator.base import simulate_times

        if bin_time is not None:
            warnings.warn("Bin time will be ignored in simulate_times", DeprecationWarning)

        self.time = simulate_times(lc, use_spline=use_spline)
        self.gti = lc.gti
        self.ncounts = len(self.time)

    def simulate_energies(self, spectrum, use_spline=False):
        """
        Assign (simulate) energies to event list from a spectrum.

        Parameters
        ----------
        spectrum: 2-d array or list [energies, spectrum]
            Energies versus corresponding fluxes. The 2-d array or list must
            have energies across the first dimension and fluxes across the
            second one. If the dimension of the energies is the same as
            spectrum, they are interpreted as bin centers.
            If it is longer by one, they are interpreted as proper bin edges
            (similarly to the bins of `np.histogram`).
            Note that for non-uniformly binned spectra, it is advisable to pass
            the exact edges.
        """
        from .simulator.base import simulate_with_inverse_cdf

        if self.ncounts is None:
            simon("Either set time values or explicity provide counts.")
            return

        if isinstance(spectrum, list) or isinstance(spectrum, np.ndarray):
            energy = np.asarray(spectrum)[0]
            fluxes = np.asarray(spectrum)[1]

            if not isinstance(energy, np.ndarray):
                raise IndexError("Spectrum must be a 2-d array or list")

        else:
            raise TypeError("Spectrum must be a 2-d array or list")

        if energy.size == fluxes.size:
            de = energy[1] - energy[0]
            energy = np.concatenate([energy - de / 2, [energy[-1] + de / 2]])

        self.energy = simulate_with_inverse_cdf(
            fluxes, self.ncounts, edges=energy, sorted=False, interp_kind="linear"
        )

    def sort(self, inplace=False):
        """Sort the event list in time.

        Other parameters
        ----------------
        inplace : bool, default False
            Sort in place. If False, return a new event list.

        Returns
        -------
        eventlist : `EventList`
            The sorted event list. If ``inplace=True``, it will be a shallow copy
            of ``self``.

        Examples
        --------
        >>> events = EventList(time=[0, 2, 1], energy=[0.3, 2, 0.5], pi=[3, 20, 5])
        >>> e1 = events.sort()
        >>> np.allclose(e1.time, [0, 1, 2])
        True
        >>> np.allclose(e1.energy, [0.3, 0.5, 2])
        True
        >>> np.allclose(e1.pi, [3, 5, 20])
        True

        But the original event list has not been altered (``inplace=False`` by
        default):
        >>> np.allclose(events.time, [0, 2, 1])
        True

        Let's do it in place instead
        >>> e2 = events.sort(inplace=True)
        >>> np.allclose(e2.time, [0, 1, 2])
        True

        In this case, the original event list has been altered.
        >>> np.allclose(events.time, [0, 1, 2])
        True

        """
        order = np.argsort(self.time)
        return self.apply_mask(order, inplace=inplace)

    def join(self, other):
        """
        Join two :class:`EventList` objects into one.

        If both are empty, an empty :class:`EventList` is returned.

        GTIs are crossed if the event lists are over a common time interval,
        and appended otherwise.

        Standard attributes such as ``pi`` and ``energy`` remain ``None`` if they are ``None``
        in both. Otherwise, 0 is used as a default value for the :class:`EventList` where they
        were None. Arbitrary attributes (e.g., Stokes parameters in polarimetric data) are
        created and concatenated using the same convention.

        Parameters
        ----------
        other : :class:`EventList` object or class:`list` of :class:`EventList` objects
            The other :class:`EventList` object which is supposed to be joined with.
            If other is a list, it is assumed to be a list of :class:`EventList` objects
            and they are all joined, one by one.

        Returns
        -------
        `ev_new` : :class:`EventList` object
            The resulting :class:`EventList` object.
        """

        ev_new = EventList()

        if isinstance(other, EventList):
            others = [other]
        else:
            others = other

        # First of all, check if there are empty event lists
        for obj in others:
            if getattr(obj, "time", None) is None or np.size(obj.time) == 0:
                warnings.warning("One of the event lists you are concatenating is empty.")
                others.remove(obj)

        if self.time is None:
            self.time = []

        for other in others:
            # Tolerance for MJDREF:1 microsecond
            if not np.isclose(self.mjdref, other.mjdref, atol=1e-6 / 86400):
                other = other.change_mjdref(self.mjdref)

        all_objs = [self] + others
        dts = list(set([getattr(obj, "dt", None) for obj in all_objs]))
        if len(dts) != 1:
            warnings.warning("The time resolution is different. Using the rougher by default")

            ev_new.dt = np.max(dts)

        ev_new.time = np.concatenate([getattr(obj, "time", np.array([])) for obj in all_objs])
        order = np.argsort(ev_new.time)
        ev_new.time = ev_new.time[order]

        def _get_set_from_many_lists(lists):
            all_vals = []
            for l in lists:
                all_vals += l
            return set(all_vals)

        for attr in _get_set_from_many_lists([obj.array_attrs() for obj in all_objs]):
            # if it's here, it means that it's an array attr in at least one object.
            # So, everywhere it's None, it needs to be set to 0s of the same length as time
            new_attr_values = []
            for obj in all_objs:
                other_attr = getattr(other, attr, None)
                if getattr(obj, attr, None) is None:
                    new_attr_values.append(np.zeros_like(obj.time))
                else:
                    new_attr_values.append(getattr(obj, attr))

            new_attr = np.concatenate(new_attr_values)[order]
            setattr(ev_new, attr, new_attr)

        if np.all([obj.gti is None for obj in all_objs]):
            ev_new.gti = None
        else:
            all_gti_lists = []

            for obj in all_objs:
                if obj.gti is None and len(obj.time) > 0:
                    obj.gti = assign_value_if_none(
                        obj.gti,
                        np.asarray([[obj.time[0] - obj.dt / 2, obj.time[-1] + obj.dt / 2]]),
                    )
                if obj.gti is not None:
                    all_gti_lists.append(obj.gti)

            new_gtis = all_gti_lists[0]
            for gti in all_gti_lists[1:]:
                if check_separate(new_gtis, gti):
                    new_gtis = append_gtis(new_gtis, gti)
                    logging.info(
                        "GTIs in these two event lists do not overlap at all."
                        "Merging instead of returning an overlap."
                    )
                else:
                    new_gtis = cross_gtis([new_gtis, gti])

        all_meta_attrs = _get_set_from_many_lists([obj.meta_attrs() for obj in all_objs])
        # The attributes being treated separately are removed from the standard treatment
        # When energy, pi etc. are None, they might appear in the meta_attrs, so we
        # also add them to the list of attributes to be removed if present.
        to_remove = ["gti", "header", "ncounts", "dt"] + ev_new.array_attrs()

        for attrs in to_remove:
            if attrs in all_meta_attrs:
                all_meta_attrs.remove(attrs)

        logging.info("The header attribute will be removed from the output event list.")

        def _safe_concatenate(a, b):
            if isinstance(a, str) and isinstance(b, str):
                return a + "," + b
            else:
                if isinstance(a, tuple):
                    return a + (b,)
                return (a, b)

        for attr in all_meta_attrs:
            self_attr = getattr(self, attr, None)
            new_val = self_attr
            for other in others:
                other_attr = getattr(other, attr, None)
                if self_attr != other_attr:
                    warnings.warn(
                        "Attribute " + attr + " is different in the event lists being merged."
                    )
                    new_val = _safe_concatenate(new_val, other_attr)
            setattr(ev_new, attr, new_val)

        ev_new.mjdref = self.mjdref

        return ev_new

    @classmethod
    def read(cls, filename, fmt=None, **kwargs):
        r"""Read a :class:`EventList` object from file.

        Currently supported formats are

        * pickle (not recommended for long-term storage)
        * hea or ogip : FITS Event files from (well, some) HEASARC-supported missions.
        * any other formats compatible with the writers in
          :class:`astropy.table.Table` (ascii.ecsv, hdf5, etc.)

        Files that need the :class:`astropy.table.Table` interface MUST contain
        at least a ``time`` column. Other recognized columns are ``energy`` and
        ``pi``.
        The default ascii format is enhanced CSV (ECSV). Data formats
        supporting the serialization of metadata (such as ECSV and HDF5) can
        contain all eventlist attributes such as ``mission``, ``gti``, etc with
        no significant loss of information. Other file formats might lose part
        of the metadata, so must be used with care.

        Parameters
        ----------
        filename: str
            Path and file name for the file to be read.

        fmt: str
            Available options are 'pickle', 'hea', and any `Table`-supported
            format such as 'hdf5', 'ascii.ecsv', etc.

        Other parameters
        ----------------
        kwargs : dict
            Any further keyword arguments to be passed to `load_events_and_gtis`
            for reading in event lists in OGIP/HEASOFT format

        Returns
        -------
        ev: :class:`EventList` object
            The :class:`EventList` object reconstructed from file
        """

        if fmt is not None and fmt.lower() in ("hea", "ogip"):
            evtdata = load_events_and_gtis(filename, **kwargs)

            evt = EventList(
                time=evtdata.ev_list,
                gti=evtdata.gti_list,
                pi=evtdata.pi_list,
                energy=evtdata.energy_list,
                mjdref=evtdata.mjdref,
                instr=evtdata.instr,
                mission=evtdata.mission,
                header=evtdata.header,
                detector_id=evtdata.detector_id,
                ephem=evtdata.ephem,
                timeref=evtdata.timeref,
                timesys=evtdata.timesys,
            )
            if "additional_columns" in kwargs:
                for key in evtdata.additional_data:
                    if not hasattr(evt, key.lower()):
                        setattr(evt, key.lower(), evtdata.additional_data[key])
            return evt

        return super().read(filename=filename, fmt=fmt)

    def filter_energy_range(self, energy_range, inplace=False, use_pi=False):
        """Filter the event list from a given energy range.

        Parameters
        ----------
        energy_range: [float, float]
            Energy range in keV, or in PI channel (if ``use_pi`` is True)

        Other Parameters
        ----------------
        inplace : bool, default False
            Do the change in place (modify current event list). Otherwise, copy
            to a new event list.
        use_pi : bool, default False
            Use PI channel instead of energy in keV

        Examples
        --------
        >>> events = EventList(time=[0, 1, 2], energy=[0.3, 0.5, 2], pi=[3, 5, 20])
        >>> e1 = events.filter_energy_range([0, 1])
        >>> np.allclose(e1.time, [0, 1])
        True
        >>> np.allclose(events.time, [0, 1, 2])
        True
        >>> e2 = events.filter_energy_range([0, 10], use_pi=True, inplace=True)
        >>> np.allclose(e2.time, [0, 1])
        True
        >>> np.allclose(events.time, [0, 1])
        True

        """
        if use_pi:
            energies = self.pi
        else:
            energies = self.energy
        mask = (energies >= energy_range[0]) & (energies < energy_range[1])

        return self.apply_mask(mask, inplace=inplace)

    def apply_mask(self, mask, inplace=False):
        """Apply a mask to all array attributes of the event list

        Parameters
        ----------
        mask : array of ``bool``
            The mask. Has to be of the same length as ``self.time``

        Other parameters
        ----------------
        inplace : bool
            If True, overwrite the current event list. Otherwise, return a new one.

        Examples
        --------
        >>> evt = EventList(time=[0, 1, 2], mission="nustar")
        >>> evt.bubuattr = [222, 111, 333]
        >>> newev0 = evt.apply_mask([True, True, False], inplace=False);
        >>> newev1 = evt.apply_mask([True, True, False], inplace=True);
        >>> newev0.mission == "nustar"
        True
        >>> np.allclose(newev0.time, [0, 1])
        True
        >>> np.allclose(newev0.bubuattr, [222, 111])
        True
        >>> np.allclose(newev1.time, [0, 1])
        True
        >>> evt is newev1
        True
        """
        array_attrs = self.array_attrs()

        if inplace:
            new_ev = self
        else:
            new_ev = EventList()
            for attr in self.meta_attrs():
                setattr(new_ev, attr, copy.deepcopy(getattr(self, attr)))

        for attr in array_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(new_ev, attr, copy.deepcopy(np.asarray(getattr(self, attr))[mask]))
        return new_ev

    def apply_deadtime(self, deadtime, inplace=False, **kwargs):
        """Apply deadtime filter to this event list.

        Additional arguments in ``kwargs`` are passed to `get_deadtime_mask`

        Parameters
        ----------
        deadtime : float
            Value of dead time to apply to data
        inplace : bool, default False
            If True, apply the deadtime to the current event list. Otherwise,
            return a new event list.

        Returns
        -------
        new_event_list : `EventList` object
            Filtered event list. if `inplace` is True, this is the input object
            filtered for deadtime, otherwise this is a new object.
        additional_output : object
            Only returned if `return_all` is True. See `get_deadtime_mask` for
            more details.

        Examples
        --------
        >>> events = np.array([1, 1.05, 1.07, 1.08, 1.1, 2, 2.2, 3, 3.1, 3.2])
        >>> events = EventList(events)
        >>> events.pi=np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        >>> events.energy=np.array([1, 2, 2, 2, 2, 1, 1, 1, 2, 1])
        >>> events.mjdref = 10
        >>> filt_events, retval = events.apply_deadtime(0.11, inplace=False,
        ...                                             verbose=False,
        ...                                             return_all=True)
        >>> filt_events is events
        False
        >>> expected = np.array([1, 2, 2.2, 3, 3.2])
        >>> np.allclose(filt_events.time, expected)
        True
        >>> np.allclose(filt_events.pi, 1)
        True
        >>> np.allclose(filt_events.energy, 1)
        True
        >>> np.allclose(events.pi, 1)
        False
        >>> filt_events = events.apply_deadtime(0.11, inplace=True,
        ...                                     verbose=False)
        >>> filt_events is events
        True
        """
        local_retall = kwargs.pop("return_all", False)

        mask, retall = get_deadtime_mask(self.time, deadtime, return_all=True, **kwargs)

        new_ev = self.apply_mask(mask, inplace=inplace)

        if local_retall:
            new_ev = [new_ev, retall]

        return new_ev
