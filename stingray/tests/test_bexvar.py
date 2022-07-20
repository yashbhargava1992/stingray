import warnings
import pytest
import os


_HAS_ULTRANEST = True

try:
    import ultranest
except ImportError:
    _HAS_ULTRANEST = False

@pytest.mark.skipif("not _HAS_ULTRANEST")
def test_bexvar():
    # An empty test function at the moment.
    pass
