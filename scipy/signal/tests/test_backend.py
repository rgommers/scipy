import numpy as np
import scipy.signal
from scipy.signal import set_backend
from scipy.signal.tests import mock_backend

from numpy.testing import assert_equal
import pytest


fnames = ['upfirdn']


funcs = [getattr(scipy.signal, fname) for fname in fnames]
mocks = [getattr(mock_backend, fname) for fname in fnames]


@pytest.mark.parametrize("func, mock", zip(funcs, mocks))
def test_backend_call(func, mock):
    """
    Checks fake backend dispatch.
    """
    x = np.array([[0, 0], [1, 1]])

    with set_backend(mock_backend, only=True):
        mock.number_calls = 0
        y = func(x)
        assert_equal(y, mock.return_value)
        assert_equal(mock.number_calls, 1)
