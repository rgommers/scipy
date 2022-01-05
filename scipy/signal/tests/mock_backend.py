import numpy as np


class _MockFunction:
    def __init__(self, return_value=None):
        self.number_calls = 0
        self.return_value = return_value
        self.last_args = ([], {})

    def __call__(self, *args, **kwargs):
        self.number_calls += 1
        self.last_args = (args, kwargs)
        return self.return_value


method_names = [
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
    'unit_impulse'
]

for name in method_names:
    globals()[name] = _MockFunction(np.array([[0, 0], [1, 1]]))


__ua_domain__ = "numpy.scipy.signal"


def __ua_function__(method, args, kwargs):
    fn = globals().get(method.__name__)
    return (fn(*args, **kwargs) if fn is not None else NotImplemented)
