from __future__ import division
import numpy as np
from ._step_generators import _generate_step
from scipy import misc
from scipy.ndimage.filters import convolve1d
from ._derivative_numdiff import extrapolate


def _increments(n, h):
    ei = np.zeros(np.shape(h), float)
    for k in range(n):
        ei[k] = h[k]
        yield ei
        ei[k] = 0.0


def jacobian(f, x, **options):
    """
    Jacobian of a function

    Parameters
    ----------
    f : function
    x : array
        parameters at which the jacobian is to be evaluated
    options : dict
        options for specifying the method, order of jacobian,
        order of error and other parameters for step generation.

    Returns
    -------
    jacobian : array
        jacobian

    Note
    ----
    1. This implementation is adapted from numdifftools :
    https://github.com/pbrod/numdifftools/tree/master/numdifftools

    2.  If fun returns a 1d array, jacobian returns a 2d array.
    If a 2d array is returned by fun (e.g., with a value for
    each observation), it returns a 3d array.

    Examples
    --------
    >>> jacobian(lambda x: [[x[0]*x[1]**2], [x[0]*x[1]]], [[1,2],[3,4]])
    [array([[ 4.,  4.],[ 2.,  1.]]), array([[ 16.,  24.],[  4.,   3.]])]

     References
    ----------
    1. https://github.com/pbrod/numdifftools/tree/master/numdifftools

    2. https://en.wikipedia.org/wiki/Finite_difference

    3. D Levy, Numerical Differentiation, Section 5
    """
    x = np.asarray(np.atleast_1d(x))
    x = np.transpose(x)
    method = options.pop('method', 'central')
    n = 1
    order = options.pop('order', 2)
    step = options.pop('step', None)
    if step not in ['max_step', 'min_step', None]:
        raise ValueError('step can only take values'
                         ' as `max_step` or `min_step`')
    step_ratio = options.pop('step_ratio', None)
    if n == 0:
        jacobian = f(x)
    else:
        if step_ratio is None:
            if n == 1:
                step_ratio = 2.0
            else:
                step_ratio = 1.6
        if step is None:
            step = 'max_step'
        options.update(x=x, n=n, order=order,
                       method=method, step=step, step_ratio=step_ratio)
        step_gen = _generate_step(**options)
        steps = [stepi for stepi in step_gen]
        fact = 1.0
        step_ratio_inv = 1.0 / step_ratio
        ni = len(x)
        if method is 'central':
            fxi = np.asarray(f(x))
            results = [[(np.asarray((f(x + hi)) - np.asarray(f(x - hi))) / 2.0)
                        for hi in _increments(ni, h)] for h in steps]
            fd_step = 2
            if n % 2 == 1:
                offset = 1
            else:
                offset = 2
        if method is 'forward':
            fxi = np.asarray(f(x))
            results = [[(np.asarray(f(x + hi)) - fxi) for hi in
                        _increments(ni, h)] for h in steps]
            fd_step = 1
            offset = 1
        if method is 'backward':
            fxi = np.asarray(f(x))
            results = [[(fxi - np.asarray(f(x - hi))) for hi in
                        _increments(ni, h)] for h in steps]
            fd_step = 1
            offset = 1
        results = np.asarray(results)
        if np.size(fxi) > 1:
            one = np.ones_like(fxi)
            steps = [np.array([one * h[i] for i in range(ni)]) for h in steps]
        original_shape = list(np.shape(np.atleast_1d(results[0].squeeze())))
        axes = [0, 1, 2][:len(original_shape)]
        axes[:2] = axes[1::-1]
        original_shape[:2] = original_shape[1::-1]
        fun = np.vstack([np.atleast_1d(r.squeeze()).transpose(axes).ravel()
                        for r in results])
        h = np.vstack([np.atleast_1d(r.squeeze()).transpose(axes).ravel()
                       for r in steps])
        if len(original_shape) == 1:
            original_shape = (1, ) + tuple(original_shape)
        else:
            original_shape = tuple(original_shape)
        if fun.size != h.size:
                raise ValueError('fun did not return data of correct '
                                 'size (it must be vectorized)')
        richardson_step = 1
        if method is 'central':
            richardson_step = 2
        richardson_order = max(
                (order // richardson_step) * richardson_step, richardson_step)
        richarson_terms = 2
        num_terms = (n+richardson_order-1) // richardson_step
        term = (n-1) // richardson_step
        c = fact / misc.factorial(
                np.arange(offset, fd_step * num_terms + offset, fd_step))
        [i, j] = np.ogrid[0:num_terms, 0:num_terms]
        fd = np.atleast_2d(c[j] * step_ratio_inv**(i * (fd_step * j + offset)))
        fd = np.linalg.pinv(fd)
        if n % 2 == 0 and method is 'backward':
            fdi = -fd[term]
        else:
            fdi = fd[term]
        if h.shape[0] < n + order - 1:
            raise ValueError('num_steps must be larger than n + order - 1')
        fdiff = convolve1d(fun, fdi[::-1], axis=0, origin=(fdi.size - 1) // 2)
        jacobian = fdiff / (h**n)
        num_steps = max(h.shape[0] + 1 - fdi.size, 1)
        jacobian = extrapolate(order, richarson_terms, richardson_step,
                               step_ratio, jacobian[:num_steps],
                               h[:num_steps], original_shape)
    if jacobian.ndim == 3:
        jacobian = np.transpose(jacobian)
    jacobian = [np.transpose(jac) for jac in jacobian]
    return jacobian


def gradient(f, x, **options):
    return jacobian(f, x, **options)
