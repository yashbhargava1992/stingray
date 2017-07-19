from astropy.modeling import models, fitting

def sinc(x):
    
    """
    Calculate a sinc function.
    
    sinc(x)=sin(x)/x
    
    Parameters
    ----------
    x : array-like
        
    Returns
    -------
    values : array-like 

    """
    x_is_zero = x == 0
    values = np.sin(x) / x
    values[x_is_zero] = 1.
    return values

def sinc_square_model(x, amplitude=1., mean=0., width=1.):
    
    """
    Calculate a sinc-squared function.
    
    (sin(x)/x)**2
    
    Parameters
    ----------
    x: array-like
    
    Other Parameters
    ----------
    amplitude : float
        the value for x=mean   
    mean : float
        mean of the sinc function    
    width : float
        width of the sinc function
    
    Returns
    -------
    sqvalues : array-like
         Return square of sinc function


    """
    sqvalues = amplitude * sinc((x-mean)/width) ** 2
    return sqvalues

def sinc_square_deriv(x, amplitude=1., mean=0., width=1.):
    
    """
    Calculate partial derivatives of sinc-squared.
    
    Parameters
    ----------
    x: array-like

    Other Parameters
    ----------
    amplitude : float
        the value for x=mean
    mean : float
        mean of the sinc function   
    width : float
        width of the sinc function
    
    Returns
    -------
    d_amplitude : array-like
         partial derivative of sinc-squared function
         with respect to the amplitude
    d_mean : array-like
         partial derivative of sinc-squared function
         with respect to the mean
    d_width : array-like
         partial derivative of sinc-squared function
         with respect to the width

    """
    
    x_is_zero = x == mean
    
    d_x = 2 * amplitude * sinc((x-mean)/width) * (x * np.cos((x-mean)/width) - np.sin((x-mean)/width)) / ((x-mean)/width)**2
    d_amplitude = sinc((x-mean)/width)**2
    d_mean = d_x*(-1/width)
    d_width = d_x*(-(x-mean)/(width)**2)
    
    d_x[x_is_zero] = 0
    
    return [d_amplitude, d_mean, d_width]

SincModel = models.custom_model(sinc_square_model, fit_deriv=sinc_square_deriv)

def fitSinc(x, y, amp=1.5, mean=0., width=1.):
    """
    Fit a sinc function to x,y values.
    
    Parameters
    ----------
    x : array-like
    y : array-like
    
    Other Parameters
    ----------------
    amp : float
        The initial value for the amplitude
    mean : float
        The initial value for the mean of the sinc
    width : float
        The initial value for the width of the sinc
     
    Returns
    -------
    sincfit : function
        The best-fit function, accepting x as input 
        and returning the best-fit model as output
    """
    sinc_in = SincModel(amplitude=amp, mean=mean,width=width)
    fit_sinc = fitting.LevMarLSQFitter()
    sincfit = fit_sinc(sinc_in, x, y)
    return sincfit
