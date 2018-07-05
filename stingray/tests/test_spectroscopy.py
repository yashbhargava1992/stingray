import numpy as np
import pytest
import stingray.spectroscopy as spec
from stingray.filters import Window1D
from stingray import Crossspectrum
from scipy.fftpack import fft
from astropy.modeling import models
from astropy.table import Table, Column

from os import remove


def test_load_lc_fits():
    dt = 0.1
    n_seg = 5
    n_seconds = 100
    output_file = 'out.fits'

    ref_lc = np.arange(0, n_seconds, dt)
    ci_lc = np.array(np.array_split(np.arange(0, n_seconds*n_seg, dt), n_seg))

    n_bins = len(ref_lc)

    lightcurves = Table()
    lightcurves.add_column(Column(name='REF', data=ref_lc.T))
    lightcurves.add_column(Column(name='CI', data=ci_lc.T))
    lightcurves.meta['N_BINS'] = n_bins
    lightcurves.meta['DT'] = dt
    lightcurves.meta['N_SEG'] = n_seg
    lightcurves.meta['NSECONDS'] = n_seconds
    lightcurves.write(output_file, format='fits', overwrite=True)

    ref, ci, meta = spec.load_lc_fits(output_file, counts_type=True)
    remove(output_file)

    assert np.all(ref_lc == ref)
    assert np.all(ci_lc == ci)
    assert meta['N_BINS'] == n_bins
    assert meta['DT'] == dt
    assert meta['N_SEG'] == n_seg
    assert meta['NSECONDS'] == n_seconds


def test_psi_distance():
    a = np.arange(-10, 11)
    dist = spec.psi_distance(0, a)
    assert np.all(np.abs(a) == dist)


def test_get_new_df():
    np.random.seed(150)

    amplitude_0 = 200.0
    amplitude_1 = 100.0
    amplitude_2 = 50.0

    x_0_0 = 0.5
    x_0_1 = 2.0
    x_0_2 = 7.5

    fwhm_0 = 0.1
    fwhm_1 = 1.0
    fwhm_2 = 0.5

    whitenoise = 100.0

    model = models.Lorentz1D(amplitude_0, x_0_0, fwhm_0) + \
        models.Lorentz1D(amplitude_1, x_0_1, fwhm_1) + \
        models.Lorentz1D(amplitude_2, x_0_2, fwhm_2) + \
        models.Const1D(whitenoise)

    freq = np.linspace(0.01, 10.0, 10.0 / 0.01)
    p = model(freq)
    noise = np.random.exponential(size=len(freq))

    power = p * noise
    cs = Crossspectrum()
    cs.freq = freq
    cs.power = power
    cs.df = cs.freq[1] - cs.freq[0]
    cs.n = len(freq)
    cs.m = 1

    assert np.isclose(cs.df, spec.get_new_df(cs, cs.n), rtol=0.001)


def test_waveform():
    mu = 1
    avg_sigma_1 = 1
    avg_sigma_2 = 2
    cap_phi_1 = np.pi/2
    cap_phi_2 = np.pi/2
    x = np.linspace(0, 2 * 2 * np.pi, 100)

    y1 = spec.waveform(x, mu, avg_sigma_1, avg_sigma_2, cap_phi_1, cap_phi_2)
    y2 = mu * (1 + np.sqrt(2) * (avg_sigma_1 * np.cos(x - cap_phi_1) +
                                 avg_sigma_2 * np.cos(2 * x - cap_phi_2)))

    assert np.all(y1 == y2)


def test_x_2_function():
    a = [-2, -1, 0, 1, 2]
    y1 = spec.x_2_function(0, a)
    y2 = np.sum(np.array(a) ** 2)
    assert np.all(y1 == y2)


def test_compute_rms():
    np.random.seed(150)

    amplitude_0 = 200.0
    amplitude_1 = 100.0
    amplitude_2 = 50.0

    x_0_0 = 0.5
    x_0_1 = 2.0
    x_0_2 = 7.5

    fwhm_0 = 0.1
    fwhm_1 = 1.0
    fwhm_2 = 0.5

    whitenoise = 100.0

    model = models.Lorentz1D(amplitude_0, x_0_0, fwhm_0) + \
        models.Lorentz1D(amplitude_1, x_0_1, fwhm_1) + \
        models.Lorentz1D(amplitude_2, x_0_2, fwhm_2) + \
        models.Const1D(whitenoise)

    freq = np.linspace(-10.0, 10.0, 10.0 / 0.01)
    p = model(freq)
    noise = np.random.exponential(size=len(freq))

    power = p * noise
    cs = Crossspectrum()
    cs.freq = freq
    cs.power = power
    cs.df = cs.freq[1] - cs.freq[0]
    cs.n = len(freq)
    cs.m = 1

    rms = np.sqrt(np.sum(model(cs.freq) * cs.df)).mean()

    assert rms == spec.compute_rms(cs, model, criteria="all")

    rms_pos = np.sqrt(np.sum(model(cs.freq[cs.freq > 0]) * cs.df)).mean()

    assert rms_pos == spec.compute_rms(cs, model, criteria="posfreq")

    optimal_filter = Window1D(model)
    optimal_filter_freq = optimal_filter(cs.freq)
    filtered_cs_power = optimal_filter_freq * np.abs(model(cs.freq))

    rms = np.sqrt(np.sum(filtered_cs_power * cs.df)).mean()
    assert rms == spec.compute_rms(cs, model, criteria="window")

    with pytest.raises(ValueError):
        spec.compute_rms(cs, model, criteria="filter")


def test_ccf():
    x = np.arange(100)
    power = fft(x)
    n_bins = 10
    ps_rms = 4
    ccf1 = x * (2 / n_bins / ps_rms)
    ccf2 = spec.ccf(power, n_bins, ps_rms)
    print(ccf1-ccf2)
    assert np.all(np.isclose(ccf1, ccf2, atol=0.000001, rtol=0.000001))
