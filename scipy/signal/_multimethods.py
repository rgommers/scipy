import functools
import numpy as np
from scipy._lib.uarray import Dispatchable, all_of_type, create_multimethod
from scipy.signal import _api
from scipy.signal._backend import scalar_or_array

__all__ = [
    'upfirdn', 'sepfir2d', 'correlate', 'correlation_lags', 'correlate2d',
    'convolve', 'convolve2d', 'fftconvolve', 'oaconvolve',
    'order_filter', 'medfilt', 'medfilt2d', 'wiener', 'lfilter',
    'lfiltic', 'sosfilt', 'deconvolve', 'hilbert', 'hilbert2',
    'cmplx_sort', 'unique_roots', 'invres', 'invresz', 'residue',
    'residuez', 'resample', 'resample_poly', 'detrend',
    'lfilter_zi', 'sosfilt_zi', 'sosfiltfilt', 'choose_conv_method',
    'filtfilt', 'decimate', 'vectorstrength'
]


_create_signal = functools.partial(create_multimethod,
                                   domain="numpy.scipy.signal")

_mark_scalar_or_array = functools.partial(
                            Dispatchable,
                            dispatch_type=scalar_or_array,
                            coercible=True
                        )


def _get_docs(func):
    func.__doc__ = getattr(_api, func.__name__).__doc__
    return func


def _h_x_replacer(args, kwargs, dispatchables):
    def self_method(h, x, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_h_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def upfirdn(h, x, up=1, down=1, axis=-1, mode='constant', cval=0):
    return h, x


def _input_hrow_hcol_replacer(args, kwargs, dispatchables):
    def self_method(input, hrow, hcol, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_input_hrow_hcol_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sepfir2d(input, hrow, hcol):
    return input, hrow, hcol


########################## signaltools functions ###############################

def _in1_in2_replacer(args, kwargs, dispatchables):
    def self_method(in1, in2, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def correlate(in1, in2, mode='full', method='auto'):
    return in1, in2


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def fftconvolve(in1, in2, mode="full", axes=None):
    return in1, in2


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def oaconvolve(in1, in2, mode="full", axes=None):
    return in1, in2


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def choose_conv_method(in1, in2, mode='full', measure=False):
    return in1, in2


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def convolve(in1, in2, mode='full', method='auto'):
    return in1, in2


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    return in1, in2


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def correlate2d(in1, in2, mode='full', boundary='fill', fillvalue=0):
    return in1, in2


def _in1len_in2len_replacer(args, kwargs, dispatchables):
    def self_method(in1_len, in2_len, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_in1len_in2len_replacer)
@all_of_type(np.ndarray)
@_get_docs
def correlation_lags(in1_len, in2_len, mode='full'):
    return in1_len, in2_len


def _a_domain_rank_replacer(args, kwargs, dispatchables):
    def self_method(a, domain, rank, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_a_domain_rank_replacer)
@all_of_type(np.ndarray)
@_get_docs
def order_filter(a, domain, rank):
    return a, domain, rank


def _volume_replacer(args, kwargs, dispatchables):
    def self_method(volume, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_volume_replacer)
@all_of_type(np.ndarray)
@_get_docs
def medfilt(volume, kernel_size=None):
    return (volume, )


def _im_replacer(args, kwargs, dispatchables):
    def self_method(im, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_im_replacer)
@all_of_type(np.ndarray)
@_get_docs
def wiener(im, mysize=None, noise=None):
    return (im, )


def _input_replacer(args, kwargs, dispatchables):
    def self_method(input, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_input_replacer)
@all_of_type(np.ndarray)
@_get_docs
def medfilt2d(input, kernel_size=3):
    return (input, )


def _b_a_x_replacer(args, kwargs, dispatchables):
    def self_method(b, a, x, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_b_a_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def lfilter(b, a, x, axis=-1, zi=None):
    return b, a, x


@_create_signal(_b_a_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad',
             irlen=None):
    return b, a, x


def _b_a_y_replacer(args, kwargs, dispatchables):
    def self_method(b, a, y, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_b_a_y_replacer)
@all_of_type(np.ndarray)
@_get_docs
def lfiltic(b, a, y, x=None):
    return b, a, y

def _signal_divisor_replacer(args, kwargs, dispatchables):
    def self_method(signal, divisor, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_signal_divisor_replacer)
@all_of_type(np.ndarray)
@_get_docs
def deconvolve(signal, divisor):
    return signal, divisor


def _x_replacer(args, kwargs, dispatchables):
    def self_method(x, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def hilbert(x, N=None, axis=-1):
    return (x, )


@_create_signal(_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def hilbert2(x, N=None):
    return (x, )


def _p_replacer(args, kwargs, dispatchables):
    def self_method(p, *args, **kwargs):
        print('pppp', p)
        print('pppaaa', np.asarray(p))
        print('dddp', dispatchables, dispatchables[0])
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_p_replacer)
@all_of_type(np.ndarray)
@_get_docs
def cmplx_sort(p):
    return (p, )


@_create_signal(_p_replacer)
@all_of_type(np.ndarray)
@_get_docs
def unique_roots(p, tol=1e-3, rtype='min'):
    return (p, )


def _r_p_k_replacer(args, kwargs, dispatchables):
    def self_method(r, p, k, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_r_p_k_replacer)
@all_of_type(np.ndarray)
@_get_docs
def invres(r, p, k, tol=1e-3, rtype='avg'):
    return r, p, k


@_create_signal(_r_p_k_replacer)
@all_of_type(np.ndarray)
@_get_docs
def invresz(r, p, k, tol=1e-3, rtype='avg'):
    return r, p, k


def _b_a_replacer(args, kwargs, dispatchables):
    def self_method(b, a, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_b_a_replacer)
@all_of_type(np.ndarray)
@_get_docs
def residue(b, a, tol=1e-3, rtype='avg'):
    return b, a


@_create_signal(_b_a_replacer)
@all_of_type(np.ndarray)
@_get_docs
def residuez(b, a, tol=1e-3, rtype='avg'):
    return b, a


@_create_signal(_b_a_replacer)
@all_of_type(np.ndarray)
@_get_docs
def lfilter_zi(b, a):
    return b, a


def _x_num_replacer(args, kwargs, dispatchables):
    def self_method(x, num, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_num_replacer)
@all_of_type(np.ndarray)
@_get_docs
def resample(x, num, t=None, axis=0, window=None, domain='time'):
    return x, Dispatchable(num, int)


def _x_up_down_replacer(args, kwargs, dispatchables):
    def self_method(x, up, down, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_up_down_replacer)
@all_of_type(np.ndarray)
@_get_docs
def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0),
                  padtype='constant', cval=None):
    return x, up, down


def _events_period_replacer(args, kwargs, dispatchables):
    def self_method(events, period, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_events_period_replacer)
@all_of_type(np.ndarray)
@_get_docs
def vectorstrength(events, period):
    return events, period


def _data_replacer(args, kwargs, dispatchables):
    def self_method(data, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_data_replacer)
@all_of_type(np.ndarray)
@_get_docs
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    return (data, )


def _sos_replacer(args, kwargs, dispatchables):
    def self_method(sos, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_sos_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sosfilt_zi(sos):
    return (sos, )


def _sos_x_replacer(args, kwargs, dispatchables):
    def self_method(sos, x, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)

@_create_signal(_sos_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sosfilt(sos, x, axis=-1, zi=None):
    return sos, x

@_create_signal(_sos_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sosfiltfilt(sos, x, axis=-1, padtype='odd', padlen=None):
    return sos, x


def _x_q_replacer(args, kwargs, dispatchables):
    def self_method(x, q, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_q_replacer)
@all_of_type(np.ndarray)
@_get_docs
def decimate(x, q, n=None, ftype='iir', axis=-1, zero_phase=True):
    return x, Dispatchable(q, int)
