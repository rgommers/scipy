import functools
import numpy as np
from scipy._lib.uarray import all_of_type, create_multimethod
from scipy.signal import _api


__all__ = ['upfirdn']


_create_signal = functools.partial(create_multimethod,
                                   domain="numpy.scipy.signal")


def _h_x_replacer(args, kwargs, dispatchables):
    def self_method(h, x, *args, **kwargs):
        return (dispatchables[0], dispatchables[1]) + args, kwargs

    return self_method(*args, **kwargs)


def _get_docs(func):
    func.__doc__ = getattr(_api, func.__name__).__doc__
    return func


@_create_signal(_h_x_replacer)
@all_of_type(np.ndarray)
@_get_docs
def upfirdn(h, x, up=1, down=1, axis=-1, mode='constant', cval=0):
    return h, x
