import scipy._lib.uarray as ua
from scipy.signal import _api
import numpy as np

__all__ = [
    'register_backend', 'set_backend',
    'set_global_backend', 'skip_backend'
]


class scalar_or_array:
    """
    Special case argument that can be either a scalar or array
    for __ua_convert__.
    """
    pass


class tuple_str_array:
    """
    Special case argument that can be either a string, tuple or array
    for __ua_convert__.
    """


class _ScipySignalBackend:
    __ua_domain__ = "numpy.scipy.signal"


    @staticmethod
    def __ua_function__(method, args, kwargs):
        fn = getattr(_api, method.__name__, None)

        if fn is None:
            return NotImplemented

        return fn(*args, **kwargs)


    @ua.wrap_single_convertor
    def __ua_convert__(value, dispatch_type, coerce):
        if value is None:
            return None

        if dispatch_type is np.ndarray:
            if not coerce and not isinstance(value, np.ndarray):
                return NotImplemented

            return np.asarray(value)

        elif dispatch_type is np.dtype:
            return np.dtype(value)

        elif dispatch_type is scalar_or_array:
            if np.isscalar(value):
                return value
            elif not coerce and not isinstance(value, np.ndarray):
                return NotImplemented

            return np.asarray(value)

        elif dispatch_type is tuple_str_array:
            if np.isscalar(value) or isinstance(value, (str, tuple)):
                return value
            elif not coerce and not isinstance(value, np.ndarray):
                return NotImplemented

            return np.asarray(value)

        return value


_named_backends = {
    'scipy': _ScipySignalBackend,
}


def _backend_from_arg(backend):
    if isinstance(backend, str):
        try:
            backend = _named_backends[backend]
        except KeyError as e:
            raise ValueError('Unknown backend {}'.format(backend)) from e

    if backend.__ua_domain__ != 'numpy.scipy.signal':
        raise ValueError('Backend does not implement "numpy.scipy.signal"')

    return backend


def set_global_backend(backend, coerce=False, only=False, try_last=False):
    backend = _backend_from_arg(backend)
    ua.set_global_backend(backend, coerce=coerce, only=only, try_last=try_last)


def register_backend(backend):
    backend = _backend_from_arg(backend)
    ua.register_backend(backend)


def set_backend(backend, coerce=True, only=False):
    backend = _backend_from_arg(backend)
    return ua.set_backend(backend, coerce=coerce, only=only)


def skip_backend(backend):
    backend = _backend_from_arg(backend)
    return ua.skip_backend(backend)


set_global_backend('scipy', coerce=True, try_last=True)