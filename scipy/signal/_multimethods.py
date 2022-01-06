import functools
import numpy as np
from scipy._lib.uarray import Dispatchable, all_of_type, create_multimethod
from scipy.signal import _api
from scipy.signal._backend import scalar_tuple_array

__all__ = [
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
]


_create_signal = functools.partial(
                    create_multimethod,
                    domain="numpy.scipy.signal"
                 )


_mark_scalar_tuple_array = functools.partial(
                            Dispatchable,
                            dispatch_type=scalar_tuple_array,
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
    return _mark_scalar_tuple_array(in1), _mark_scalar_tuple_array(in2)


def _in1_in2_axes_replacer(args, kwargs, dispatchables):
    def self_method(in1, in2, mode="full", axes=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                mode, dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_in1_in2_axes_replacer)
@all_of_type(np.ndarray)
@_get_docs
def fftconvolve(in1, in2, mode="full", axes=None):
    return (_mark_scalar_tuple_array(in1), _mark_scalar_tuple_array(in2),
           _mark_scalar_tuple_array(axes))


@_create_signal(_in1_in2_axes_replacer)
@all_of_type(np.ndarray)
@_get_docs
def oaconvolve(in1, in2, mode="full", axes=None):
    return (_mark_scalar_tuple_array(in1), _mark_scalar_tuple_array(in2),
           _mark_scalar_tuple_array(axes))


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def choose_conv_method(in1, in2, mode='full', measure=False):
    return _mark_scalar_tuple_array(in1), _mark_scalar_tuple_array(in2)


@_create_signal(_in1_in2_replacer)
@all_of_type(np.ndarray)
@_get_docs
def convolve(in1, in2, mode='full', method='auto'):
    return _mark_scalar_tuple_array(in1), _mark_scalar_tuple_array(in2)


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
    return _mark_scalar_tuple_array(in1_len), _mark_scalar_tuple_array(in2_len)


def _a_domain_rank_replacer(args, kwargs, dispatchables):
    def self_method(a, domain, rank, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_a_domain_rank_replacer)
@all_of_type(np.ndarray)
@_get_docs
def order_filter(a, domain, rank):
    return a, _mark_scalar_tuple_array(domain), Dispatchable(rank, int)


def _volume_kernelsize_replacer(args, kwargs, dispatchables):
    def self_method(volume, kernel_size=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_volume_kernelsize_replacer)
@all_of_type(np.ndarray)
@_get_docs
def medfilt(volume, kernel_size=None):
    return volume, _mark_scalar_tuple_array(kernel_size)


def _im_mysize_replacer(args, kwargs, dispatchables):
    def self_method(im, mysize=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_im_mysize_replacer)
@all_of_type(np.ndarray)
@_get_docs
def wiener(im, mysize=None, noise=None):
    return im, _mark_scalar_tuple_array(mysize)


def _input_kernelsize_replacer(args, kwargs, dispatchables):
    def self_method(input, kernel_size=3, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_input_kernelsize_replacer)
@all_of_type(np.ndarray)
@_get_docs
def medfilt2d(input, kernel_size=3):
    return input, _mark_scalar_tuple_array(kernel_size)


def _b_a_x_axis_zi_replacer(args, kwargs, dispatchables):
    def self_method(b, a, x, axis=-1, zi=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1], dispatchables[2],
                dispatchables[3], dispatchables[4]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_b_a_x_axis_zi_replacer)
@all_of_type(np.ndarray)
@_get_docs
def lfilter(b, a, x, axis=-1, zi=None):
    return b, a, x, Dispatchable(axis, int), _mark_scalar_tuple_array(zi)


def _b_a_x_replacer(args, kwargs, dispatchables):
    def self_method(b, a, x, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_b_a_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None, method='pad',
             irlen=None):
    return b, a, x


def _b_a_y_x_replacer(args, kwargs, dispatchables):
    def self_method(b, a, y, x=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2], dispatchables[3]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_b_a_y_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def lfiltic(b, a, y, x=None):
    return b, a, y, x


def _signal_divisor_replacer(args, kwargs, dispatchables):
    def self_method(signal, divisor, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_signal_divisor_replacer)
@all_of_type(np.ndarray)
@_get_docs
def deconvolve(signal, divisor):
    return _mark_scalar_tuple_array(signal), _mark_scalar_tuple_array(divisor)


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
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_p_replacer)
@all_of_type(np.ndarray)
@_get_docs
def cmplx_sort(p):
    return (_mark_scalar_tuple_array(p), )


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


def _x_num_t_window_replacer(args, kwargs, dispatchables):
    def self_method(x, num, t=None, axis=0, window=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1], dispatchables[2],
                axis, dispatchables[3]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_num_t_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def resample(x, num, t=None, axis=0, window=None, domain='time'):
    return x, Dispatchable(num, int), t, _mark_scalar_tuple_array(window)


def _x_up_down_window_replacer(args, kwargs, dispatchables):
    def self_method(x, up, down, axis=0, window=('kaiser', 5.0),
                    *args, **kwargs):
        return (dispatchables[0], dispatchables[1], dispatchables[2],
                axis, dispatchables[3]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_up_down_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0),
                  padtype='constant', cval=None):
    return (x, Dispatchable(up, int), Dispatchable(down, int),
            _mark_scalar_tuple_array(window))


def _events_period_replacer(args, kwargs, dispatchables):
    def self_method(events, period, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_events_period_replacer)
@all_of_type(np.ndarray)
@_get_docs
def vectorstrength(events, period):
    return events, _mark_scalar_tuple_array(period)


def _data_axis_type_bp_replacer(args, kwargs, dispatchables):
    def self_method(data, axis=-1, type='linear', bp=0, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2], dispatchables[3]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_data_axis_type_bp_replacer)
@all_of_type(np.ndarray)
@_get_docs
def detrend(data, axis=-1, type='linear', bp=0, overwrite_data=False):
    return (data, Dispatchable(axis, int), Dispatchable(type, str),
            _mark_scalar_tuple_array(bp))


def _sos_replacer(args, kwargs, dispatchables):
    def self_method(sos, *args, **kwargs):
        return (dispatchables[0], ) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_sos_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sosfilt_zi(sos):
    return (sos, )


def _sos_x_zi_replacer(args, kwargs, dispatchables):
    def self_method(sos, x, axis=-1, zi=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                axis, dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_sos_x_zi_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sosfilt(sos, x, axis=-1, zi=None):
    return sos, x, _mark_scalar_tuple_array(zi)


def _sos_x_replacer(args, kwargs, dispatchables):
    def self_method(sos, x, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


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


############################### waveforms ######################################


def _t_width_replacer(args, kwargs, dispatchables):
    def self_method(t, width=1, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_t_width_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sawtooth(t, width=1):
    return _mark_scalar_tuple_array(t), _mark_scalar_tuple_array(width)


def _t_duty_replacer(args, kwargs, dispatchables):
    def self_method(t, duty=0.5, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_t_duty_replacer)
@all_of_type(np.ndarray)
@_get_docs
def square(t, duty=0.5):
    return _mark_scalar_tuple_array(t), _mark_scalar_tuple_array(duty)


def _t_replacer(args, kwargs, dispatchables):
    def self_method(t, *args, **kwargs):
        return (dispatchables[0],) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_t_replacer)
@all_of_type(np.ndarray)
@_get_docs
def gausspulse(t, fc=1000, bw=0.5, bwr=-6, tpr=-60, retquad=False,
               retenv=False):
    return (_mark_scalar_tuple_array(t), )


def _t_f0_t1_f1_replacer(args, kwargs, dispatchables):
    def self_method(t, f0, t1, f1, *args, **kwargs):
        return (dispatchables[0], dispatchables[1], dispatchables[2],
                dispatchables[3]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_t_f0_t1_f1_replacer)
@all_of_type(np.ndarray)
@_get_docs
def chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    return (_mark_scalar_tuple_array(t), Dispatchable(f0, float),
            Dispatchable(t1, float), Dispatchable(f1, float))


def _t_poly_replacer(args, kwargs, dispatchables):
    def self_method(t, poly, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_t_poly_replacer)
@all_of_type(np.ndarray)
@_get_docs
def sweep_poly(t, poly, phi=0):
    return t, poly


def _shape_idx_replacer(args, kwargs, dispatchables):
    def self_method(shape, idx=None, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_shape_idx_replacer)
@all_of_type(np.ndarray)
@_get_docs
def unit_impulse(shape, idx=None, dtype=float):
    return _mark_scalar_tuple_array(shape), _mark_scalar_tuple_array(idx)


############################ spectrum analysis #################################

def _x_y_freqs_replacer(args, kwargs, dispatchables):
    def self_method(x, y, freqs, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_y_freqs_replacer)
@all_of_type(np.ndarray)
@_get_docs
def lombscargle(x, y, freqs, precenter=False, normalize=False):
    return x, y, freqs


def _x_fs_window_replacer(args, kwargs, dispatchables):
    def self_method(x, fs=1.0, window='', *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant',
                return_onesided=True, scaling='density', axis=-1):
    return x, Dispatchable(fs, float), _mark_scalar_tuple_array(window)


@_create_signal(_x_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, average='mean'):
    return x, Dispatchable(fs, float), _mark_scalar_tuple_array(window)


@_create_signal(_x_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def spectrogram(x, fs=1.0, window=('tukey', .25), nperseg=None, noverlap=None,
                nfft=None, detrend='constant', return_onesided=True,
                scaling='density', axis=-1, mode='psd'):
    return x, Dispatchable(fs, float), _mark_scalar_tuple_array(window)


@_create_signal(_x_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
         detrend=False, return_onesided=True, boundary='zeros', padded=True,
         axis=-1):
    return x, Dispatchable(fs, float), _mark_scalar_tuple_array(window)


def _x_y_fs_window_replacer(args, kwargs, dispatchables):
    def self_method(x, y, fs=1.0, window='', *args, **kwargs):
        return (dispatchables[0], dispatchables[1], dispatchables[2],
                dispatchables[3]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_x_y_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean'):
    return x, y, Dispatchable(fs, float), _mark_scalar_tuple_array(window)


@_create_signal(_x_y_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1):
    return x, y, Dispatchable(fs, float), _mark_scalar_tuple_array(window)


def _window_nperseg_noverlap_replacer(args, kwargs, dispatchables):
    def self_method(window, nperseg, noverlap, *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_window_nperseg_noverlap_replacer)
@all_of_type(np.ndarray)
@_get_docs
def check_COLA(window, nperseg, noverlap, tol=1e-10):
    return (_mark_scalar_tuple_array(window), Dispatchable(nperseg, int),
            Dispatchable(noverlap, int))


@_create_signal(_window_nperseg_noverlap_replacer)
@all_of_type(np.ndarray)
@_get_docs
def check_NOLA(window, nperseg, noverlap, tol=1e-10):
    return (_mark_scalar_tuple_array(window), Dispatchable(nperseg, int),
            Dispatchable(noverlap, int))


def _Zxx_fs_window_replacer(args, kwargs, dispatchables):
    def self_method(Zxx, fs=1.0, window='', *args, **kwargs):
        return (dispatchables[0], dispatchables[1],
                dispatchables[2]) + args, kwargs

    return self_method(*args, **kwargs)


@_create_signal(_Zxx_fs_window_replacer)
@all_of_type(np.ndarray)
@_get_docs
def istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2):
    return Zxx, Dispatchable(fs, float), _mark_scalar_tuple_array(window)
