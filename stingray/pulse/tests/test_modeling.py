import numpy as np
from stingray.pulse.modeling import fitSinc

np.random.seed(0)


def test_sinc_function():
    x = np.linspace(-5., 5., 200)
    y = 2 * (np.sin(x)/x)**2
    y += np.random.normal(0., 0.1, x.shape)

    s = fitSinc(x,y)
    
    assert np.abs(s.mean) < 0.1 
    assert np.abs(s.amplitude - 2) < 0.1 
    assert np.abs(s.width - 1) < 0.1
    
    
def test_gaussian_function():
    x = np.linspace(-5., 5., 200)
    y = 2 * np.exp(-0.5 * (x - 1.3)**2 / 0.7**2)
    y += np.random.normal(0., 0.1, x.shape)

    gs = fit_gaussian(x,y) 
    
    assert np.abs(gs.mean - 1.3) < 0.1
    assert np.abs(gs.amplitude - 2) < 0.1
    assert np.abs(gs.stddev - 0.7) < 0.1
