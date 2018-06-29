import astropy
from astropy.table import Table
from stingray import Lightcurve, Powerspectrum, AveragedPowerspectrum, DynamicalPowerspectrum, Crossspectrum, AveragedCrossspectrum
from astropy.modeling import models, fitting
from stingray.modeling.scripts import fit_powerspectrum, fit_lorentzians, fit_crossspectrum
from stingray.utils import standard_error
from stingray.filters import Optimal1D, Window1D

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.fftpack import ifft


def load_lc_fits(file, rate_type=True):
    lc_fits = Table.read(file)
    meta = lc_fits.meta
    dt = meta['DT']

    if rate_type:
        ref = np.asarray(lc_fits['REF'].T * dt, dtype=np.float64)
        ci = np.asarray(lc_fits['CI'].T * dt, dtype=np.float64)
    else:
        ref = np.asarray(lc_fits['REF'].T, dtype=np.float64)
        ci = np.asarray(lc_fits['CI'].T, dtype=np.float64)

    return ref, ci, meta


def get_new_df(spectrum):
    """
    Return the new df used to rebin the spectrum.

    Parameters
    ----------
    spectrum : Powerspectrum or Crossspectrum class instance

    Returns
    -------
    new_df : float

    """
    lin_psd, f_bin_edges, something = binned_statistic(spectrum.freq, spectrum.power[0:int(spectrum.n/2+1)],
                                statistic='mean', bins=1600)
    new_df = np.median(np.diff(f_bin_edges))
    return new_df


def ccf(cs_power, ps_rms, n_bins):
    """
    Return the normalised cross-correlation function of the cross spectrum. It
    is normalised using the average RMS of power spectrum.

    Parameters
    ----------
    cs_power : 1-d float array
        Power of cross spectrum.

    ps_rms : 1-d float array
        RMS of power spectrum.

    n_bins : int
        Number of buns.

    Returns
    -------
    ccf_real_norm : 1-d float array
        Real part of normalised ccf.
    """
    ccf = ifft(cs_power)  # inverse fast fourier transformation
    ccf_real = ccf.real  # real part of ccf
    ccf_real_norm = ccf_real * (2 / n_bins / ps_rms)  # normalisation
    return ccf_real_norm


def ccf_error(ref_counts, ci_counts_0, cs_res_model, rebin_log_factor, meta,
              ps_rms, filter_type="optimal"):
    n_seg = meta['N_SEG']
    n_seconds = meta['NSECONDS']
    dt = meta['DT']
    n_bins = meta['N_BINS']

    seg_ref_counts = np.array(np.split(ref_counts, n_seg))
    seg_ci_counts = np.array(np.split(ci_counts_0, n_seg))
    seg_css = np.array([])
    seg_ccfs = np.array([])
    seg_times = np.arange(0, n_seconds, dt)  # light curve time bins

    print(seg_ref_counts.shape, seg_ci_counts.shape)

    for i in range(n_seg):  # for each segment
        # Creating cross spectrum
        seg_ci_lc = Lightcurve(seg_times, seg_ci_counts[i],
                               dt=dt)  # CoI light curve
        seg_ref_lc = Lightcurve(seg_times, seg_ref_counts[i],
                                dt=dt)  # reference band light curve
        seg_cs = Crossspectrum(lc2=seg_ci_lc, lc1=seg_ref_lc, norm='leahy',
                               power_type="absolute")  # cross spectrum
        seg_cs = seg_cs.rebin_log(rebin_log_factor)  # cross spectrum rebinning

        # applying filter
        if filter_type == "optimal":
            cs_filter = Optimal1D(cs_res_model)
            filter_freq = cs_filter(seg_cs.freq)
            filtered_seg_cs_power = filter_freq * np.abs(seg_cs.power)
        else:
            cs_filter = Window1D(cs_res_model)
            filter_freq = cs_filter(seg_cs.freq)
            filtered_seg_cs_power = filter_freq * np.abs(seg_cs.power)

        # calculating normalized ccf
        seg_ccf = ifft(filtered_seg_cs_power)  # inverse FFT
        seg_ccf_real = seg_ccf.real  # real part of ccf
        seg_ccf_real_norm = seg_ccf_real * (
                    2 / n_bins / ps_rms)  # normalisation

        if i == 0:
            seg_css = np.hstack((seg_css, np.array(filtered_seg_cs_power)))
        else:
            seg_css = np.vstack((seg_css, np.array(filtered_seg_cs_power)))
        if i == 0:
            seg_ccfs = np.hstack((seg_ccfs, np.array(seg_ccf_real_norm)))
        else:
            seg_ccfs = np.vstack((seg_ccfs, np.array(seg_ccf_real_norm)))

    # ccf after taking avg
    avg_seg_css = np.average(seg_css, axis=0)  # average of cross spectrum
    avg_seg_ccf = ccf(avg_seg_css, ps_rms, n_bins)

    error = standard_error(seg_ccfs, avg_seg_ccf)
    return error
