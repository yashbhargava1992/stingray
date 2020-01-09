import pytest
import numpy as np

from stingray.stats import *


@pytest.mark.parametrize('ntrial', [1, 10, 100, 1000, 10000, 100000])
def test_p_single_from_multi(ntrial):
    epsilon_1 = 0.00000001
    epsilon_n = p_multitrial_from_single_trial(epsilon_1, ntrial)
    epsilon_1_corr = \
        p_single_trial_from_p_multitrial(epsilon_n, ntrial)

    assert np.isclose(epsilon_1_corr, epsilon_1, rtol=1e-2)


def test_p_single_from_multi_fails():
    epsilon_n = 1
    with pytest.warns(UserWarning) as record:
        p1 = p_single_trial_from_p_multitrial(epsilon_n, 1000)

    assert np.any(["Multi-trial probability is very close to 1"
                   in r.message.args[0] for r in record])
    assert np.isnan(p1)


def test_fold_detection_level():
    """Test pulse phase calculation, frequency only."""
    np.testing.assert_almost_equal(fold_detection_level(16, 0.01),
                                   30.577914166892498)
    epsilon_corr = p_single_trial_from_p_multitrial(0.01, 2)
    np.testing.assert_almost_equal(
        fold_detection_level(16, 0.01, ntrial=2),
        fold_detection_level(16, epsilon_corr))


def test_zn_detection_level():
    np.testing.assert_almost_equal(z2_n_detection_level(2),
                                   13.276704135987625)
    epsilon_corr = p_single_trial_from_p_multitrial(0.01, 2)
    np.testing.assert_almost_equal(z2_n_detection_level(4, 0.01, ntrial=2),
                                   z2_n_detection_level(4, epsilon_corr))


@pytest.mark.parametrize('ntrial', [1, 10, 100, 1000, 100000])
def test_fold_probability(ntrial):
    detlev = fold_detection_level(16, 0.1, ntrial=ntrial)
    np.testing.assert_almost_equal(fold_profile_probability(detlev, 16,
                                                            ntrial=ntrial),
                                   0.1)


@pytest.mark.parametrize('ntrial', [1, 10, 100, 1000, 100000])
def test_zn_probability(ntrial):
    detlev = z2_n_detection_level(2, 0.1, ntrial=ntrial)
    np.testing.assert_almost_equal(z2_n_probability(detlev, 2, ntrial=ntrial),
                                   0.1)
