import numpy as np
from stingray.pulse.modeling import fit_sinc, fit_gaussian

np.random.seed(0)


def test_sinc_function():
    x = np.linspace(-5., 5., 200)
    y = 2 * (np.sin(x)/x)**2
    y += np.random.normal(0., 0.1, x.shape)

    s = fit_sinc(x, y)

    assert np.abs(s.mean) < 0.1
    assert np.abs(s.amplitude - 2) < 0.1
    assert np.abs(s.width - 1) < 0.1


def test_sinc_fixed():
    x = np.linspace(-5., 5., 200)
    y = 2 * (np.sin(x)/x)**2
    y += np.random.normal(0., 0.1, x.shape)

    sf = fit_sinc(x, y, mean=1., fixed={"mean": True, "amplitude": False})
    assert sf.mean.fixed
    assert not sf.amplitude.fixed


def test_gaussian_function():
    x = np.linspace(-5., 5., 200)
    y = 2 * np.exp(-0.5 * (x - 1.3)**2 / 0.7**2)
    y += np.random.normal(0., 0.1, x.shape)

    gs = fit_gaussian(x, y)

    assert np.abs(gs.mean - 1.3) < 0.1
    assert np.abs(gs.amplitude - 2) < 0.1
    assert np.abs(gs.stddev - 0.7) < 0.1


def test_gaussian_bounds():
    x = np.linspace(-5., 5., 200)
    y = 2 * np.exp(-0.5 * (x - 1.3)**2 / 0.7**2)
    y += np.random.normal(0., 0.1, x.shape)

    gs = fit_gaussian(x, y,
                      bounds={"mean": [1., 1.6], "amplitude":[1.7, 2.3]})


def test_gaussian_fixed():
    x = np.linspace(-5., 5., 200)
    y = 2 * np.exp(-0.5 * (x - 1.3)**2 / 0.7**2)
    y += np.random.normal(0., 0.1, x.shape)

    gs = fit_gaussian(x, y, mean=1.3, fixed={"mean": True, "amplitude": False})
    assert gs.mean.fixed
    assert not gs.amplitude.fixed


def test_gaussian_tied():
    x = np.linspace(-5., 5., 200)
    y = 2 * np.exp(-0.5 * (x - 1.3)**2 / 0.7**2)
    y += np.random.normal(0., 0.1, x.shape)

    def tiedgaussian(model):
        mean = model.amplitude / 2
        return mean

    gs = fit_gaussian(x, y, tied={"mean": tiedgaussian})

    assert np.abs(gs.mean/gs.amplitude - 0.5) < 0.1
