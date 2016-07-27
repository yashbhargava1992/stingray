from __future__ import division, print_function, absolute_import
import numpy as np

"""
Implementation of time and energy averaged responses from 2-d 
transfer functions.
"""

class TransferFunction(object):

    def __init__(self, data, dt=1, de=1):
        """
        Create or retrieve a transfer function, and form
        time and energy averaged responses.

        Parameters
        ----------
        data: numpy 2-d array
            Has time values across the first dimension, and energy 
            values across the second one.

        dt: float
            time interval

        de: float
            energy interval

        Attributes
        ----------
        time: numpy.ndarray
            energy-averaged response of 2-d transfer function

        energy: numpy.ndarray
            time-averaged response of 2-d transfer function
        """
        
        pass

    def average_time(self):
        """
        Form a time-averaged response of 2-d transfer function. 

        Returns
        -------
        energy: numpy.ndarray
            time-averaged response of 2-d transfer function
        """
        
        pass

    def average_energy(self):
        """
        Form an energy-averaged response of 2-d transfer function. 

        Returns
        -------
        time: numpy.ndarray
            energy-averaged response of 2-d transfer function
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
    dt: float
        Time resolution

    start: int
        start time of impulse response
    
    width: int
        width of impulse response
    
    intensity: float
        scaling parameter to set the intensity of delayed emission
        corresponding to direct emission.

    Returns
    -------
    h: numpy.ndarray
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
    dt: float
        Time resolution

    t1: int
        primary peak time
    
    t2: int
        secondary peak time
    
    t3: int
        end time
    
    p1: float
        value of primary peak
    
    p2: float
        value of secondary peak
    
    rise: float
        slope of rising exponential from primary peak to secondary peak
    
    decay: float
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
