import numpy as np
from nose.tools import raises

from stingray import classical_pvalue

np.random.seed(20150907)

class TestClassicalSignificances(object):

    def test_function_runs(self):
        power = 2.0
        nspec = 1.0
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_power_is_not_infinite(self):
        power = np.inf
        nspec = 1
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_power_is_not_infinite2(self):
        power = -np.inf
        nspec = 1
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_power_is_non_nan(self):
        power = np.nan
        nspec = 1
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_power_is_positive(self):
        power = -2.0
        nspec = 1.0
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_nspec_is_not_infinite(self):
        power = 2.0
        nspec = np.inf
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_nspec_is_not_infinite2(self):
        power = 2.0
        nspec = -np.inf
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_nspec_is_not_nan(self):
        power = 2.0
        nspec = np.nan
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_nspec_is_positive(self):
        power = 2.0
        nspec = -1.0
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_nspec_is_nonzero(self):
        power = 2.0
        nspec = 0.0
        classical_pvalue(power, nspec)

    @raises(AssertionError)
    def test_nspec_is_an_integer_number(self):
        power = 2.0
        nspec = 2.5
        classical_pvalue(power, nspec)

    def test_nspec_float_type_okay(self):
        power = 2.0
        nspec = 2.0
        classical_pvalue(power, nspec)

    def test_pvalue_decreases_with_increasing_power(self):
        power1 = 2.0
        power2 = 20.0
        nspec = 1.0
        pval1 = classical_pvalue(power1, nspec)
        pval2 = classical_pvalue(power2, nspec)

        assert pval1-pval2 > 0.0

    def test_pvalue_must_decrease_with_increasing_nspec(self):

        power = 3.0
        nspec1 = 1.0
        nspec2 = 10.0

        pval1 = classical_pvalue(power, nspec1)
        pval2 = classical_pvalue(power, nspec2)

        assert pval1-pval2 > 0.0

    def test_very_large_powers_produce_zero_prob(self):
        power = 31000.0
        nspec = 1
        pval = classical_pvalue(power, nspec)
        assert np.isclose(pval, 0.0)
        