import numpy as np
import pytest
from stingray.lightcurve import Lightcurve
from stingray.deadtime.fad import calculate_FAD_correction
from stingray.deadtime.filters import filter_for_deadtime
import matplotlib.pyplot as plt

# np.random.seed(2134791)

def generate_events(length, ncounts):
    ev = np.random.uniform(0, length, ncounts)
    ev.sort()
    return ev


def generate_deadtime_lc(ev, dt, tstart=0, tseg=None, deadtime=2.5e-3):
    ev = filter_for_deadtime(ev, deadtime)
    return Lightcurve.make_lightcurve(ev, dt=dt, tstart=tstart, tseg=tseg,
                                      gti=np.array([[tstart, tseg]]))


@pytest.mark.parametrize('ctrate', [0.5, 5, 50, 200])
def test_fad_power_spectrum_compliant(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.
    ncounts = np.int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)
    ncounts1 = np.sum(lc1.counts)
    ncounts2 = np.sum(lc2.counts)

    results = \
        calculate_FAD_correction(lc1, lc2, segment_size, plot=True,
                          smoothing_alg='gauss',
                          strict=True, verbose=True,
                          tolerance=0.05)

    pds1_f = results['pds1']
    pds2_f = results['pds2']
    cs_f = results['cs']
    ptot_f = results['ptot']

    n = length / segment_size
    ncounts_per_intv1 = ncounts1 * segment_size / length
    ncounts_per_intv2 = ncounts2 * segment_size / length
    ncounts_per_intvtot = (ncounts1 + ncounts2) * segment_size / length
    ncounts_per_intv_geomav = np.sqrt(ncounts1 * ncounts2) * segment_size / length

    pds_std_theor = 2 / np.sqrt(n)
    cs_std_theor = np.sqrt(2 / n)

    assert np.isclose(pds1_f.std() * 2 / ncounts_per_intv1, pds_std_theor, rtol=0.1)
    assert np.isclose(pds2_f.std() * 2 / ncounts_per_intv2, pds_std_theor, rtol=0.1)
    assert np.isclose(cs_f.std() * 2 / ncounts_per_intv_geomav, cs_std_theor, rtol=0.1)
    assert np.isclose(ptot_f.std() * 2 / ncounts_per_intvtot, pds_std_theor, rtol=0.1)


@pytest.mark.parametrize('ctrate', [50])
def test_fad_power_spectrum_non_compliant(ctrate):
    dt = 0.1
    deadtime = 2.5e-3
    length = 25600
    segment_size = 256.
    ncounts = np.int(ctrate * length)
    ev1 = generate_events(length, ncounts)
    ev2 = generate_events(length, ncounts)

    lc1 = generate_deadtime_lc(ev1, dt, tstart=0, tseg=length, deadtime=deadtime)
    lc2 = generate_deadtime_lc(ev2, dt, tstart=0, tseg=length, deadtime=deadtime)

    with pytest.warns(UserWarning) as record:
        results = \
            calculate_FAD_correction(lc1, lc2, segment_size, plot=True,
                              smoothing_alg='gauss',
                              strict=False, verbose=False,
                              tolerance=0.0001)
    assert np.any(["results ARE NOT complying"
                   in r.message.args[0] for r in record])

    is_compliant = results.meta['is_compliant']

    assert not is_compliant
