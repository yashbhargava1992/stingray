from __future__ import division, print_function, absolute_import
import numpy as np

"""
Implementation of time and energy averaged responses from 2-d 
transfer functions.
"""

class TransferFunction(object):

    def __init__(self, data, dt=1, de=1, tstart=0, estart=0):
        """
        Create or retrieve a transfer function, and form
        time and energy averaged responses.

        Parameters
        ----------
        data : numpy 2-d array / 2-d list
            inner/first dimension (or column counts) represents time
            and outer/second dimension (or rows counts) represents energy.

            As an example, if you have you 2-d model define by 'arr', then
            arr[1][5] defines a time of 5(s) and energy of 1(keV) [assuming
            'dt' and 'de' are 1 and 'tstart' and 'estart' are 0.]

        dt : float, default 1
            time interval

        de : float, default 1
            energy interval

        tstart : float, default 0
            initial time value across time axis

        estart : float, default 0
            initial energy value across energy axis

        Attributes
        ----------
        time : numpy.ndarray
            energy-averaged response of 2-d transfer function

        energy : numpy.ndarray
            time-averaged response of 2-d transfer function
        """
        
        self.data = np.asarray(data)
        self.dt = dt
        self.de = de
        self.tstart = tstart
        self.estart = estart
        self.time = None
        self.energy = None

        if len(data[0]) < 2:
            raise ValueError('Number of columns should be greater than 1.')

        if len(data[:]) < 2:
            raise ValueError('Number of rows should be greater than 1.')

    def time_response(self):
        """
        Form a time-averaged response of 2-d transfer function. 

        Returns
        -------
        energy : numpy.ndarray
            time-averaged response of 2-d transfer function
        """
        
        self.time = np.mean(self.data, axis=0)

    def energy_response(self):
        """
        Form an energy-averaged response of 2-d transfer function. 

        Returns
        -------
        time : numpy.ndarray
            energy-averaged response of 2-d transfer function
        """
        
        self.energy = np.mean(self.data, axis=1)

    def plot(self, response='2d'):
        """
        Plot 'time', 'energy' or 2-d response using matplotlib.

        In case of 1-d response, 'time' and 'energy' would appear
        along x-axis and corresponding flux across y-axis. In case
        of 2-d response, a spectrograph would be formed with 'time'
        along x-axis and 'energy' along y-axis.

        Parameters
        ----------
        response : str
            type of response. accepts 'time', 'energy', '2d'

        marker : str, default '-'
            Line style and color of the plot. Line styles and colors are
            combined in a single format string, as in ``'bo'`` for blue
            circles. See `matplotlib.pyplot.plot` for more options.

        filename : str
            the name of file to save plot to. If a default of 'None' is
            picked, plot is not saved.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plot()")

        if response not in ['time', 'energy', '2d']:
            raise ValueError("Response value is not recognized. Available"
                            "response types are ")

        fig = plt.figure()
        
    @staticmethod
    def read(format_='pickle'):
        """
        Reads transfer function from a 'fits', 'pickle', 'hdf5' or 
        'ascii' file.

        Parameter
        ---------
        format_ : str
            the format of the file to be retrieved - accepts 'pickle'
            , 'hdf5', 'ascii' and 'fits'

        Returns
        -------
        data : class instance
            `TransferFunction` object
        """
        
        pass

    def write(self, format_='pickle'):
        """
        Writes a transfer function to 'pickle', 'hdf5', 'ascii' or
        'fits' file. 

        Parameters
        ----------
        format_ : str
            the format of the file to be saved - accepts 'pickle'
            , 'hdf5', 'ascii' and 'fits'
        """

        pass
"""
Implementation of artificial methods to create energy-averaged 
responses for quick testing.
"""

def simple_ir(dt, start=0, width=1000, intensity=1):
    """
    Construct a simple impulse response using start time, 
    width and scaling intensity.
    To create a delta impulse response, set width to 1.

    Parameters
    ----------
    dt : float
        Time resolution

    start : int
        start time of impulse response
    
    width : int
        width of impulse response
    
    intensity : float
        scaling parameter to set the intensity of delayed emission
        corresponding to direct emission.

    Returns
    -------
    h : numpy.ndarray
        Constructed impulse response
    """

    # Fill in 0 entries until the start time
    h_zeros = np.zeros(start/dt)

    # Define constant impulse response
    h_ones = np.ones(width/dt) * intensity

    return np.append(h_zeros, h_ones)

def relativistic_ir(dt, t1=3, t2=4, t3=10, p1=1, p2=1.4, rise=0.6, decay=0.1):
    """
    Construct a realistic impulse response considering the relativistic
    effects.

    Parameters
    ----------
    dt : float
        Time resolution

    t1 : int
        primary peak time
    
    t2 : int
        secondary peak time
    
    t3 : int
        end time
    
    p1 : float
        value of primary peak
    
    p2 : float
        value of secondary peak
    
    rise : float
        slope of rising exponential from primary peak to secondary peak
    
    decay : float
        slope of decaying exponential from secondary peak to end time

    Returns
    -------
    h: numpy.ndarray
        Constructed impulse response
    """

    assert t2>t1, 'Secondary peak must be after primary peak.'
    assert t3>t2, 'End time must be after secondary peak.'
    assert p2>p1, 'Secondary peak must be greater than primary peak.'

    # Append zeros before start time
    h_primary = np.append(np.zeros(int(t1/dt)), p1)

    # Create a rising exponential of user-provided slope
    x = np.linspace(t1/dt, t2/dt, (t2-t1)/dt)
    h_rise = np.exp(rise*x)
    
    # Evaluate a factor for scaling exponential
    factor = np.max(h_rise)/(p2-p1)
    h_secondary = (h_rise/factor) + p1

    # Create a decaying exponential until the end time
    x = np.linspace(t2/dt, t3/dt, (t3-t2)/dt)
    h_decay = (np.exp((-decay)*(x-4/dt))) 

    # Add the three responses
    h = np.append(h_primary, h_secondary)
    h = np.append(h, h_decay)

    return h
