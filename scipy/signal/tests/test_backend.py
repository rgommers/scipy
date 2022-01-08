import numpy as np
import scipy.signal
from scipy.signal import set_backend
from scipy.signal.tests import mock_backend

from numpy.testing import assert_equal
import pytest


fnames = [
    'upfirdn', 'sepfir2d',
    # signaltools
    'correlate', 'correlation_lags', 'correlate2d',
    'convolve', 'convolve2d', 'fftconvolve', 'oaconvolve',
    'order_filter', 'medfilt', 'medfilt2d', 'wiener', 'lfilter',
    'lfiltic', 'sosfilt', 'deconvolve', 'hilbert', 'hilbert2',
    'cmplx_sort', 'unique_roots', 'invres', 'invresz', 'residue',
    'residuez', 'resample', 'resample_poly', 'detrend',
    'lfilter_zi', 'sosfilt_zi', 'sosfiltfilt', 'choose_conv_method',
    'filtfilt', 'decimate', 'vectorstrength',
    # waveforms
    'sawtooth', 'square', 'gausspulse', 'chirp', 'sweep_poly',
    'unit_impulse',
    # spectrum analysis
    'periodogram', 'welch', 'lombscargle', 'csd', 'coherence',
    'spectrogram', 'stft', 'istft', 'check_COLA', 'check_NOLA',
    # bsplines
    'spline_filter', 'bspline', 'gauss_spline', 'cubic', 'quadratic',
    'cspline1d', 'qspline1d', 'cspline1d_eval', 'qspline1d_eval',
    # peak finding
    'argrelmin', 'argrelmax', 'argrelextrema', 'peak_prominences',
    'peak_widths', 'find_peaks', 'find_peaks_cwt'
]


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
