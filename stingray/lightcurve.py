"""
Definition of :class:`Lightcurve`.

:class:`Lightcurve` is used to create light curves out of photon counting data
or to save existing light curves in a class that's easy to use.
"""

__all__ = ["Lightcurve", "moving_bins"]

import numpy as np

class Lightcurve(object):
    def __init__(self, time, counts = None, timestep=None, tseg=None, tstart = None):
        """
        Make a light curve object, either from an array of time stamps and an array of counts,
        or from a list of photon arrival times.

        Parameters
        ----------
        time: iterable
            Either a list or an array of photon arrival times (if counts is None) or a list of
            time stamps for a light curve (if counts is not None)

        counts: iterable, optional, default None
            if `time` contains a list of time stamps and not a list of photn arrival times, this
            list or array should contain the corresponding counts in each bin (note:
            **not** the count rate, i.e. counts/second, but the counts/timestep).

        timestep: float, optional, default None
            If `time` is a list of photon arrival times, `timestep` needs to be set to the time
            resolution of the light curve. If `counts` is set (the input is a light curve and not
            a set of photon arrival times), this keyword will be ignored.

        tseg: float, optional, default None
            If not None, and the input is a list of photon arrival times, this will be the total duration
            of the light curve, either starting from the first photon (if tstart == None) or from tstart.
            Otherwise the duration will be the difference between the arrival times of the last and the first
            photon.

        tstart: float, optional, default None
            if not None, and the input is a list of photon arrival times, this will be the start time
            of the light curve. Otherwise the start time of the light curve coincides with the arrival
            time of the first photon.


        Attributes
        ----------

        """

        if counts is None:
            ### TOA has a list of photon times of arrival
            self.ncounts = np.asarray(time).shape[0]
            self.tstart = tstart
            self.makeLightcurve(time, timestep, tseg=tseg)
            
        else:
            self.time = np.asarray(time)
            self.counts = np.asarray(counts)
            self.res = time[1] - time[0]
            self.countrate = self.counts/self.res
            self.tseg = self.time[-1] - self.time[0] + self.res

    def makeLightcurve(self, time, timestep):
        """
        Make a light curve out of photon arrival times.

        Parameters
        ----------
        time: iterable
            list of photon arrival times

        timestep: float
            time resolution of the light curve (the bin width)


        Attributes
        ----------
        self.time: numpy.ndarray
            list with mid-bin time stamps

        self.counts: numpy.ndarray
            The number of photons per time bin

        self.countrate: numpy.ndarray
            the same as self.counts expressed in counts/second
        """

        ## tstart is an optional parameter to set a starting time for the light curve
        ## in case this does not coincide with the first photon
        if self.tstart is None:
            ## if tstart is not set, assume light curve starts with first photon
            tstart = time[0]
        else:
            tstart = self.tstart
        ### number of bins in light curve

        ## compute the number of bins in the light curve
        ## for cases where tseg/timestep are not integer, computer one
        ## last time bin more that we have to subtract in the end
        if not self.tseg:
            tseg = time[-1] - time[0]

        timebin = np.ceil(tseg/timestep)
        frac = (tseg/timestep) - int(timebin - 1)

        tend = tstart + timebin*timestep

        counts, histbins = np.histogram(time, bins=timebin, range = [tstart, tend])
        self.res = histbins[1] - histbins[0]

        self.time = np.array([histbins[0] + 0.5*self.res + n*self.res for n in range(int(timebin))])
        self.countrate = self.counts/self.res

        #print("len timebins: " + str(len(timebins)))
        if frac > 0.0:
            self.counts = np.asarray(counts[:-1])
            self.time = np.array(self.time[:-1])

        else:
            self.counts = np.asarray(counts)
            self.time = self.time

    def rebinLightcurve(self, newres, method='sum'):
        ### calculate number of bins in new light curve
        nbins = np.floor(self.tseg/newres)+1
        self.binres = self.tseg/nbins
        print "New time resolution is: " + str(self.binres)

        #print("I am here")
        bintime, bincounts, _ = self._rebin_new(self.time, self.counts, newres, method)
        return Lightcurve(bintime, bincounts)

    def _rebin_new(self, time, counts, dtnew, method='sum'):

        try:
            step_size = float(dtnew)/float(self.res)
        except AttributeError:
            step_size = float(dtnew)/float(self.df)

        output = []
        for i in np.arange(0, len(counts), step_size):
            total = 0
            #print "Bin is " + str(i)

            prev_frac = int(i+1) - i
            prev_bin = int(i)
            #print "Fractional part of bin %d is %f"  %(prev_bin, prev_frac)
            total += prev_frac * counts[prev_bin]

            if i + step_size < len(time):
                # Fractional part of next bin:
                next_frac = i+step_size - int(i+step_size)
                next_bin = int(i+step_size)
                #print "Fractional part of bin %d is %f"  %(next_bin, next_frac)
                total += next_frac * counts[next_bin]

            #print "Fully included bins: %d to %d" % (int(i+1), int(i+step_size)-1)
            total += sum(counts[int(i+1):int(i+step_size)])
            output.append(total)

        tnew = np.arange(len(output))*dtnew + time[0]
        if method in ['mean', 'avg', 'average', 'arithmetic mean']:
            cbinnew = output
            cbin = np.array(cbinnew)/float(step_size)
        elif method not in ['sum']:
            raise Exception("Method for summing or averaging not recognized. Please enter either 'sum' or 'mean'.")
        else:
            cbin = output

        return tnew, cbin, dtnew


def moving_bins(time, timestep=1.0, duration=10.0, startdiff=1.0, tstart=None):
    """
    Chop up light curve in pieces and save each piece in
    a separate :class:`Lightcurve` object.
    """
    lcs = []
    if tstart is None:
        tstart = time[0]


    tend = tstart + duration

    while tend <= time[-1] :
        stind = time.searchsorted(tstart)
        eind = time.searchsorted(tend)
        tnew = time[stind:eind]
        lcs.append(Lightcurve(tnew, timestep=timestep, tseg=duration, tstart=tstart))
        tstart += startdiff
        tend += startdiff

    return lcs
