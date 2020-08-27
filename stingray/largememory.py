import os
import warnings

import numpy as np

import stingray

from .events import EventList
from .lightcurve import Lightcurve
from .utils import genDataPath, randomNameGenerate

HAS_ZARR = False
try:
    import zarr

    HAS_ZARR = True
    from numcodecs import Blosc
except ImportError:
    warnings.warn(
        "Large Datasets may not be processed efficiently due to "
        "computational constraints")

__all__ = ['createChunkedSpectra', 'saveData', 'retreiveData']


def _saveChunkLC(lc, fname):
    """
    Save Lightcurve temporarily on disk.

    Parameters
    ----------
    lc: :class:`stingray.Lightcurve` object
        Lightcurve to be saved

    fname: string
        High Level diretory name where Lightcurve is to be saved.
    """
    # Creating a Nested Store and multiple groups for temporary saving
    store = zarr.NestedDirectoryStore(fname)
    lc_data_group = zarr.group(store=store, overwrite=True)
    main_data_group = lc_data_group.create_group('main_data', overwrite=True)
    meta_data_group = lc_data_group.create_group('meta_data', overwrite=True)

    compressor = Blosc(cname='lz4', clevel=1, shuffle=-1)  # Tested

    # REVIEW: Max chunk size can be 8388608 or 2**23. This efficiently balances time, memory. Memory consumption restricted to 9.1 GB

    main_data_group.create_dataset(name='times',
                                   data=lc.time,
                                   compressor=compressor,
                                   overwrite=True,
                                   chunks=(8388608, ))

    main_data_group.create_dataset(name='counts',
                                   data=lc.counts,
                                   compressor=compressor,
                                   overwrite=True,
                                   chunks=(8388608, ))

    # REVIEW: Count_err calculation takes a lot of memory
    main_data_group.create_dataset(name='count_err',
                                   data=lc.counts_err,
                                   compressor=compressor,
                                   overwrite=True,
                                   chunks=(8388608, ))

    # FIXME: GTI's are not consistently saved
    main_data_group.create_dataset(name='gti', data=lc.gti, overwrite=True)

    meta_data_group.create_dataset(name='dt',
                                   data=lc.dt,
                                   compressor=compressor,
                                   overwrite=True)

    meta_data_group.create_dataset(name='err_dist',
                                   data=lc.err_dist,
                                   compressor=compressor,
                                   overwrite=True)

    meta_data_group.create_dataset(name='mjdref',
                                   data=lc.mjdref,
                                   compressor=compressor,
                                   overwrite=True)


def _saveChunkEV(ev, fname):
    """
    Save EventList temporarily on disk.

    Parameters
    ----------
    ev: :class:`stingray.events.EventList` object
        EventList to be saved

    fname: string
        High Level diretory name where EventList is to be saved.
    """
    # Creating a Nested Store and multiple groups for temporary saving
    store = zarr.NestedDirectoryStore(fname)
    ev_data_group = zarr.group(store=store, overwrite=True)
    main_data_group = ev_data_group.create_group('main_data', overwrite=True)
    meta_data_group = ev_data_group.create_group('meta_data', overwrite=True)

    compressor = Blosc(cname='lz4', clevel=1, shuffle=-1)  # Tested

    # REVIEW: Max chunk size can be 8388608 or 2**23. This efficiently balances time, memory. Memory consumption restricted to 9.1 GB
    main_data_group.create_dataset(name='times',
                                   data=ev.time,
                                   compressor=compressor,
                                   overwrite=True,
                                   chunks=(8388608, ))

    meta_data_group.create_dataset(name='dt',
                                   data=ev.dt,
                                   compressor=compressor,
                                   overwrite=True)

    meta_data_group.create_dataset(name='ncounts',
                                   data=ev.ncounts,
                                   compressor=compressor,
                                   overwrite=True)

    meta_data_group.create_dataset(name='mjdref',
                                   data=ev.mjdref,
                                   compressor=compressor,
                                   overwrite=True)

    if ev.energy:
        main_data_group.create_dataset(name='energy',
                                       data=ev.energy,
                                       compressor=compressor,
                                       overwrite=True,
                                       chunks=(8388608, ))

    # FIXME: GTI's are not consistently saved
    if ev.gti:
        main_data_group.create_dataset(name='gti', data=ev.gti, overwrite=True)

    if ev.pi:
        main_data_group.create_dataset(name='pi_channel',
                                       data=ev.pi,
                                       compressor=compressor,
                                       overwrite=True)


def saveData(data_obj, f_name=randomNameGenerate()):
    """
    Saves Lightcurve/EventList or any such data in chunks to disk.

    Parameters
    ----------
    data_obj : :class:`stingray.Lightcurve` or :class:`stingray.events.EventList` object
        Data to be stored on the disk.
    f_name : string, optional
        Name of top level directory where data is to be stored, by default randomNameGenerate()

    Returns
    -------
    string
        Name of top level directory where data is to be stored

    Raises
    ------
    ValueError
        If data is not a Lightcurve or EventList
    """
    if isinstance(data_obj, Lightcurve):
        _saveChunkLC(data_obj, f_name)

    elif isinstance(data_obj, EventList):
        _saveChunkEV(data_obj, f_name)

    return f_name


def _retrieveDataLC(data_path, chunk_size=0, offset=0, raw=False):
    """
    Retrieve data from stored Lightcurve on disk.

    Parameters
    ----------
    data_path : list
        Path to datastore.
    chunk_size : int
        Size of data to be retrieved.
    offset : int, optional
        Offset or start element to read the array from, by default 0
    raw : bool, optional
        Only to be used for if raw memory mapped zarr arrays are to be obtained, by default False

    Returns
    -------
    :class:`stingray.Lightcurve` object
        Lightcurve retrieved from store.

    Raises
    ------
    ValueError
        If offset provided is larger than size of array.
    """
    times = zarr.open_array(store=data_path[0], mode='r', path='times')
    counts = zarr.open_array(store=data_path[0], mode='r', path='counts')
    count_err = zarr.open_array(store=data_path[0], mode='r', path='count_err')
    gti = zarr.open_array(store=data_path[0], mode='r', path='gti')

    dt = zarr.open_array(store=data_path[1], mode='r', path='dt')
    mjdref = zarr.open_array(store=data_path[1], mode='r', path='mjdref')
    err_dist = zarr.open_array(store=data_path[1], mode='r', path='err_dist')

    if raw:
        return times, counts, count_err, gti, dt, err_dist, mjdref
    else:
        if chunk_size > times.size or chunk_size == 0:
            chunk_size = times.size
            warnings.warn(
                f"The chunk size is set to the size of the whole array {chunk_size}"
            )

        if offset > times.size:
            raise ValueError(f"No element read. Offset cannot be larger than size of array {times.size}")

        return Lightcurve(
            time=times.get_basic_selection(slice(offset, chunk_size)),
            counts=counts.get_basic_selection(slice(offset, chunk_size)),
            err=count_err.get_basic_selection(slice(offset, chunk_size)),
            gti=gti,
            dt=dt,
            err_dist=err_dist,
            mjdref=mjdref,
            skip_checks=True)


def _retrieveDataEV(data_path, chunk_size=0, offset=0, raw=False):
    """
    Retrieve data from stored Lightcurve on disk.

    Parameters
    ----------
    data_path : list
        Path to datastore.
    chunk_size : int
        Size of data to be retrieved.
    offset : int, optional
        Offset or start element to read the array from, by default 0
    raw : bool, optional
        Only to be used for if raw memory mapped zarr arrays are to be obtained, by default False

    Returns
    -------
    :class:`stingray.events.EventList` object
        EventList retrieved from store.

    Raises
    ------
    ValueError
        If offset provided is larger than size of array.
    """
    times = zarr.open_array(store=data_path[0], mode='r', path='times')

    dt = zarr.open_array(store=data_path[1], mode='r', path='dt')
    ncounts = zarr.open_array(store=data_path[1], mode='r', path='ncounts')
    mjdref = zarr.open_array(store=data_path[1], mode='r', path='mjdref')

    try:
        energy = zarr.open_array(store=data_path[0], mode='r', path='energy')
    except ValueError:
        energy = None

    try:
        gti = zarr.open_array(store=data_path[0], mode='r', path='gti')
    except ValueError:
        gti = None

    try:
        pi_channel = zarr.open_array(store=data_path[0],
                                     mode='r',
                                     path='pi_channel')
    except ValueError:
        pi_channel = None

    if raw:
        return times, energy, ncounts, mjdref, dt, gti, pi_channel
    else:
        if chunk_size > times.size or chunk_size > energy.size or chunk_size == 0:
            chunk_size = times.size if times.size is not None else energy.size
            warnings.warn(
                f"The chunk size is set to the size of the whole array {chunk_size}"
            )

        if offset > times.size or offset > energy.size:
            raise ValueError(
                "No element read. Offset cannot be larger than size of array")

        return EventList(
            time=times.get_basic_selection(slice(offset, chunk_size)),
            energy=energy.get_basic_selection(slice(offset, chunk_size)),
            ncounts=ncounts.get_basic_selection(slice(offset, chunk_size)),
            mjdref=mjdref,
            dt=dt,
            gti=gti,
            pi=pi_channel)


def retreiveData(data_type, f_name, path=os.getcwd(), chunk_data=False, chunk_size=None, offset=None, raw=False):
    """
    Retrieves Lightcurve/EventList or any such data from disk.

    Parameters
    ----------
    data_type : string
        Type of data to retrieve i.e. Lightcurve, Eventlist data to retrieve.
    f_name1 : string
        Top level directory name for datastore
    f_name2 : string, optional
        Top level directory name for datastore, by default None
    path : string, optional
        path to retrieve data from, by default os.getcwd()
    chunk_data : bool, optional
        If only a chunk of data is to be retrieved, by default False
    chunk_size : int, optional
        Number of values to be retrieved, by default None
    offset : int, optional
        Start offset from where values are to be retrieved, by default None
    raw : bool, optional
        Only to be used for if raw memory mapped zarr arrays are to be obtained, by default False

    Raises
    ------
    ValueError
        If datatype is not Lightcurve or EventList
    """
    data_path = genDataPath(f_name, path)

    if data_type == 'Lightcurve':
        if chunk_data is True and chunk_size is not None:
            _retrieveDataLC(data_path, chunk_size, offset)
        else:
            _retrieveDataLC(data_path, raw)

    elif data_type == 'EventList':
        if chunk_data is True and chunk_size is not None:
            _retrieveDataEV(data_path, chunk_size, offset)
        else:
            _retrieveDataEV(data_path, raw)

    else:
        raise TypeError((f"Invalid input data type: {data_type}"))


def _combineSpectra(final_spectra):
    """
    Create a final spectra that is the mean of all spectra.
    Parameters
    ----------
    final_spectra: :class:`stingray.AveragedCrossspectrum/AveragedPowerspectrum' object
        Summed spectra of all spectra
    Returns
    -------
    object
        Final resulting spectra.
    """
    final_spectra.freq /= final_spectra.m
    final_spectra.power /= final_spectra.m
    final_spectra.unnorm_power /= final_spectra.m
    # REVIEW: final_spectra.power_err /= final_spectra.m

    if isinstance(final_spectra, stingray.AveragedCrossspectrum) and not \
            isinstance(final_spectra, stingray.AveragedPowerspectrum):
        final_spectra.pds1.power /= final_spectra.m
        final_spectra.pds2.power /= final_spectra.m

    return final_spectra


def _addSpectra(final_spectra, curr_spec, flag):
    """
    Add various Spectra(AveragedCrossspectrum/AveragedPowerspectrum) for combination.

    Parameters
    ----------
    final_spectra: object
        Final Combined AveragedCrossspectrum or AveragedPowerspectrum
    curr_spec: object
        AveragedCrossspectrum/AveragedPowerspectrum to be combined
    flag: bool
        Indicator variable

    Returns
    -------
    object
        Combined AveragedCrossspectrum/AveragedPowerspectrum
    """
    if flag:
        final_spectra = curr_spec
        final_spectra.freq = final_spectra.freq.astype('float128')
        final_spectra.power = final_spectra.power.astype('complex256')
        final_spectra.unnorm_power = final_spectra.unnorm_power.astype(
            'complex256')

        return final_spectra

    np.multiply(np.add(final_spectra.freq, curr_spec.freq),
                curr_spec.m,
                out=final_spectra.freq)
    np.multiply(np.add(final_spectra.power, curr_spec.power),
                curr_spec.m,
                out=final_spectra.power)
    np.multiply(np.add(final_spectra.unnorm_power, curr_spec.unnorm_power),
                curr_spec.m,
                out=final_spectra.unnorm_power)
    np.sqrt(np.add(np.square(final_spectra.power_err),
                   np.square(curr_spec.power_err)),
            out=final_spectra.power_err)

    final_spectra.m += curr_spec.m
    final_spectra.df = (final_spectra.df + curr_spec.df) / 2
    final_spectra.gti = np.concatenate((final_spectra.gti, curr_spec.gti))

    if isinstance(final_spectra, stingray.AveragedPowerspectrum):
        final_spectra.nphots += curr_spec.nphots

    elif isinstance(final_spectra, stingray.AveragedCrossspectrum):
        np.multiply(np.add(final_spectra.pds1.power, curr_spec.pds1.power),
                    curr_spec.m,
                    out=final_spectra.pds1.power)
        np.multiply(np.add(final_spectra.pds2.power, curr_spec.pds2.power),
                    curr_spec.m,
                    out=final_spectra.pds1.power)
        final_spectra.nphots1 += curr_spec.nphots1
        final_spectra.nphots2 += curr_spec.nphots2

    return final_spectra


def _chunkLCSpec(data_path, spec_type, segment_size, norm, gti, power_type,
                 silent):
    """
    Create a chunked spectra from Lightcurve stored on disk.

    Parameters
    ----------
    data_path : string
        Path to stored Lightcurve or EventList chunks on disk.
    spec_type : string
        Type of spectra to create AveragedCrossspectrum or AveragedPowerspectrum.
    segment_size: float
        The size of each segment to average in the AveragedCrossspectrum/AveragedPowerspectrum.
    norm : {``frac``, ``abs``, ``leahy``, ``none``}
        The normalization of the (real part of the) cross spectrum.
    gti : 2-d float array
        `[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!
    power_type : string
        Parameter to choose among complete, real part and magnitude of
         the cross spectrum. None for AveragedPowerspectrum
    silent : bool
        Do not show a progress bar when generating an averaged cross spectrum.
        Useful for the batch execution of many spectra
    dt1: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList

    Returns
    -------
    object
        Summed computed spectra.

    Raises
    ------
    ValueError
        If spectra is not AveragedCrossspectrum or AveragedPowerspectrum
    """
    times, counts, count_err, gti, dt, err_dist, mjdref = _retrieveDataLC(data_path[0:2], raw=True)

    if spec_type == 'AveragedPowerspectrum':
        fin_spec = stingray.AveragedPowerspectrum()

    elif spec_type == 'AveragedCrossspectrum':
        fin_spec = stingray.AveragedCrossspectrum()
        times_other, counts_other, count_err_other, gti_other, dt_other, err_dist_other, mjdref_other = _retrieveDataLC(
            data_path[2:4], raw=True)

    else:
        raise ValueError

    flag = True
    for i in range(times.chunks[0], times.size, times.chunks[0]):
        lc1 = Lightcurve(
            time=times.get_basic_selection(slice(i - times.chunks[0], i)),
            counts=counts.get_basic_selection(slice(i - times.chunks[0], i)),
            err=count_err.get_basic_selection(slice(i - times.chunks[0], i)),
            err_dist=str(err_dist[...]),
            mjdref=mjdref[...],
            dt=dt[...],
            skip_checks=True)

        if isinstance(fin_spec, stingray.AveragedPowerspectrum):
            if segment_size < lc1.time.size / 8192:
                warnings.warn(
                    f"It is advisable to have the segment size greater than or equal to {lc1.time.size / 8192}. Very small segment sizes may greatly increase computation times."
                )

            avg_spec = stingray.AveragedPowerspectrum(data=lc1, segment_size=lc1.time.size / segment_size, norm=norm, gti=gti, silent=silent, large_data=False)

        elif isinstance(fin_spec, stingray.AveragedCrossspectrum):
            lc2 = Lightcurve(time=times_other.get_basic_selection(slice(i - times.chunks[0], i)), counts=counts_other.get_basic_selection(slice(i - times.chunks[0], i)), err=count_err_other.get_basic_selection(slice(i - times.chunks[0], i)), err_dist=str(err_dist_other[...]), mjdref=mjdref_other[...], dt=dt_other[...], skip_checks=True)

            if segment_size < lc1.time.size / 4096:
                warnings.warn(
                    f"It is advisable to have the segment size greater than or equal to {lc1.time.size / 4096}. Very small segment sizes may greatly increase computation times."
                )

            avg_spec = stingray.AveragedCrossspectrum(data1=lc1, data2=lc2, segment_size=lc1.time.size / segment_size, norm=norm, gti=gti, power_type=power_type, silent=silent, large_data=False)

        if flag:
            prev_freq = avg_spec.freq
            fin_spec = _addSpectra(fin_spec, avg_spec, flag)
        else:
            if np.array_equal(prev_freq, avg_spec.freq):
                fin_spec = _addSpectra(fin_spec, avg_spec, flag)
                prev_freq = avg_spec.freq
            else:
                raise ValueError((
                    f"Spectra have unequal frequencies {avg_spec.freq.shape}{prev_freq.shape}"
                ))

        flag = False

    return fin_spec


def _chunkEVSpec(data_path, spec_type, segment_size, norm, gti, power_type,
                 silent, dt1):
    """
    Create a chunked spectra from EventList stored on disk.

    Parameters
    ----------
    data_path : string
        Path to stored Lightcurve or EventList chunks on disk.
    spec_type : string
        Type of spectra to create AveragedCrossspectrum or AveragedPowerspectrum.
    segment_size: float
        The size of each segment to average in the AveragedCrossspectrum/AveragedPowerspectrum.
    norm : {``frac``, ``abs``, ``leahy``, ``none``}
        The normalization of the (real part of the) cross spectrum.
    gti : 2-d float array
        `[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!
    power_type : string
        Parameter to choose among complete, real part and magnitude of
         the cross spectrum. None for AveragedPowerspectrum
    silent : bool
        Do not show a progress bar when generating an averaged cross spectrum.
        Useful for the batch execution of many spectra
    dt1: float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList

    Returns
    -------
    object
        Summed computed spectra.

    Raises
    ------
    ValueError
        If spectra is not AveragedCrossspectrum or AveragedPowerspectrum
    """
    times, energy, ncounts, mjdref, dt, gti, pi_channel = _retrieveDataEV(
        data_path[0:2], raw=True)

    if spec_type == 'AveragedPowerspectrum':
        fin_spec = stingray.AveragedPowerspectrum()

    elif spec_type == 'AveragedCrossspectrum':
        fin_spec = stingray.AveragedCrossspectrum()
        times_other, energy_other, ncounts_other, mjdref_other, dt_other, gti_other, pi_channel_other = _retrieveDataEV(
            data_path[2:4], raw=True)

    else:
        raise ValueError

    flag = True
    for i in range(times.chunks[0], times.size, times.chunks[0]):
        ev1 = EventList(
            time=times.get_basic_selection(slice(i - times.chunks[0], i)),
            energy=energy.get_basic_selection(slice(i - times.chunks[0], i))
            if energy is not None else None,
            ncounts=ncounts[...],
            mjdref=mjdref[...],
            dt=dt[...],
            gti=gti[...] if gti is not None else None,
            pi=pi_channel[...] if pi_channel is not None else None)

        if spec_type == 'AveragedPowerspectrum':
            if segment_size < ev1.time.size / 8192:
                warnings.warn(
                    f"It is advisable to have the segment size greater than or equal to {ev1.time.size / 8192}. Very small segment sizes may greatly increase computation times."
                )

            avg_spec = stingray.AveragedPowerspectrum(data=ev1, segment_size=ev1.time.size / segment_size, norm=norm, gti=gti, silent=silent, dt=dt1, large_data=False)

        elif spec_type == 'AveragedCrossspectrum':
            ev2 = EventList(time=times_other.get_basic_selection(slice(i - times.chunks[0], i)), energy=energy_other.get_basic_selection(slice(i - times.chunks[0], i) if energy_other is not None else None, ncounts=ncounts_other[...], mjdref=mjdref_other[...], dt=dt_other[...], gti=gti_other[...] if gti_other is not None else None, pi=pi_channel_other[...] if pi_channel_other is not None else None))

            if segment_size < ev1.time.size / 4096:
                warnings.warn(
                    f"It is advisable to have the segment size greater than or equal to {ev1.time.size / 4096}. Very small segment sizes may greatly increase computation times."
                )

            avg_spec = stingray.AveragedCrossspectrum(data1=ev1, data2=ev2, segment_size=ev1.time.size / segment_size, norm=norm, gti=gti, power_type=power_type, silent=silent, dt=dt1, large_data=False)

        if flag:
            prev_freq = avg_spec.freq
            fin_spec = _addSpectra(fin_spec, avg_spec, flag)
        else:
            if np.array_equal(prev_freq, avg_spec.freq):
                fin_spec = _addSpectra(fin_spec, avg_spec, flag)
                prev_freq = avg_spec.freq
            else:
                raise ValueError((
                    f"Spectra have unequal frequencies {avg_spec.freq.shape}{prev_freq.shape}"
                ))

        flag = False

    return fin_spec


def createChunkedSpectra(data_type, spec_type, data_path, segment_size, norm,
                         gti, power_type, silent, dt=None):
    """
    Create a chunked spectra from zarr files stored on disk.

    Parameters
    ----------
    data_type : string
        Data in Lightcurve or EventList
    spec_type : string
        Type of spectra to create AveragedCrossspectrum or AveragedPowerspectrum.
    data_path : list
        Path to datastore.
    segment_size: float
        The size of each segment to average in the AveragedCrossspectrum/AveragedPowerspectrum.
    norm : {``frac``, ``abs``, ``leahy``, ``none``}
        The normalization of the (real part of the) cross spectrum.
    gti : 2-d float array
        `[[gti0_0, gti0_1], [gti1_0, gti1_1], ...]`` -- Good Time intervals.
        This choice overrides the GTIs in the single light curves. Use with
        care!
    power_type : string
        Parameter to choose among complete, real part and magnitude of
         the cross spectrum. None for AveragedPowerspectrum
    silent : bool
        Do not show a progress bar when generating an averaged cross spectrum.
        Useful for the batch execution of many spectra
    dt : float
        The time resolution of the light curve. Only needed when constructing
        light curves in the case where data1 or data2 are of :class:EventList

    Returns
    -------
    object
        Final computed spectra.
    """
    if data_type == 'Lightcurve':
        fin_spec = _chunkLCSpec(data_path=data_path, spec_type=spec_type, segment_size=segment_size, norm=norm, gti=gti, power_type=power_type, silent=silent)

    elif data_type == 'EventList':
        fin_spec = _chunkEVSpec(data_path=data_path, spec_type=spec_type, segment_size=segment_size, norm=norm, gti=gti, power_type=power_type, silent=silent, dt1=dt)

    return _combineSpectra(fin_spec)
