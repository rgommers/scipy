from __future__ import division
import numpy as np
from ._step_generators import _generate_step
from scipy import misc
from scipy.ndimage.filters import convolve1d
from scipy import linalg
import warnings

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny


def choose_derivative_with_least_error(der, errors):
    try:
        median = np.nanmedian(errors, axis=0)
        p75 = np.nanpercentile(errors, 75, axis=0)
        p25 = np.nanpercentile(errors, 25, axis=0)
        iqr = np.abs(p75-p25)
        a_median = np.abs(median)
        outliers = (((abs(errors) < (a_median / 10)) +
                    (abs(errors) > (a_median * 10))) * (a_median > 1e-8) +
                    ((errors < p25-1.5*iqr) + (p75+1.5*iqr < der)))
        errors = errors + outliers * np.abs(errors - median)
    except ValueError:
        warnings.warn('the results cannot be trusted if a slice'
                      'contains only NaNs and Infs.')
        errors = 0 * errors

    try:
        result = np.nanargmin(errors, axis=0)
        result = np.asarray(result, dtype=float)
        min_errors = np.nanmin(errors, axis=0)
        for i, min_error in enumerate(min_errors):
            idx = np.flatnonzero(errors[:, i] == min_error)
            result[i] = (der[idx[idx.size // 2]][i])
        return result
    except ValueError:
        warnings.warn('the results cannot be trusted if a slice'
                      'contains only NaNs and Infs.')
        result = np.zeros(der.shape[1], dtype=float)
        for i in range(der.shape[1]):
            result[i] = der[0][i]
        return result


def extrapolate(order, num_terms, step, step_ratio, results, steps, shape):
    res_shape = results.shape[0]
    if res_shape is None:
        res_shape = num_terms + 1
    num_terms = min(num_terms, res_shape - 1)
    if num_terms > 0:
        i, j = np.ogrid[0:num_terms + 1, 0:num_terms]
        r_mat = np.ones((num_terms + 1, num_terms + 1))
        r_mat[:, 1:] = (1.0 / step_ratio)**(i * (step * j + order))
        r_mat = linalg.pinv(r_mat)[0]
    else:
        r_mat = np.ones((1, ))
    new_sequence = convolve1d(results,
                              r_mat[::-1], axis=0, origin=(num_terms) // 2)
    new_sequence = new_sequence[:res_shape + 1 - r_mat.size]
    steps = steps[:res_shape + 1 - r_mat.size]
    if len(new_sequence) > 2:
        steps = steps[2:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conv1 = new_sequence[0:-2]
            conv2 = new_sequence[1:-1]
            conv3 = new_sequence[2:]
            err1 = conv2 - conv1
            err2 = conv3 - conv2
            tol1 = np.maximum(np.abs(conv1), np.abs(conv2)) * EPS
            tol2 = np.maximum(np.abs(conv3), np.abs(conv2)) * EPS
            err1[np.abs(err1) < TINY] = TINY
            err2[np.abs(err2) < TINY] = TINY
            ss = 1.0 / err2 - 1.0 / err1 + TINY
            small = abs(ss * conv2) <= 1.0e-3
            converged = (np.abs(err1) <= tol1) & (np.abs(err2) <= tol2) | small
            new_seq = np.where(converged, conv3 * 1.0, conv2 + 1.0 / ss)
            abserr = np.where(converged, tol2 * 10, np.abs(new_seq - conv3))
            abserr = np.abs(err1) + np.abs(err2) + abserr
            result = choose_derivative_with_least_error(new_seq, abserr)
            result = result.reshape(shape)
    else:
        abserr = np.abs(new_sequence) * 0
        result = choose_derivative_with_least_error(new_sequence, abserr)
        result = result.reshape(shape)
    return result


def derivative(f, x, **options):
    """
    Derivative of a function

    Parameters
    ----------
    f : function
        ``f(x)`` returning one value.
        ``f(x)`` should be univariate.
    x : array
        parameters at which the derivative is evaluated
    options : dict
        options for specifying the method, order of derivative,
        order of error and other parameters for step generation.

    Returns
    -------
    derivative : array
        derivative

    Note
    ----
    This implementation is adapted from numdifftools :
    https://github.com/pbrod/numdifftools/tree/master/numdifftools

    Examples
    --------
    >>> derivative((lambda x: x**2), [1,2])
    [2,4]

     References
    ----------
    1. https://github.com/pbrod/numdifftools/tree/master/numdifftools

    2. https://en.wikipedia.org/wiki/Finite_difference

    3. D Levy, Numerical Differentiation, Section 5
    """

    x = np.asarray(x)
    method = options.pop('method', 'central')
    n = options.pop('n', 1)
    order = options.pop('order', 2)
    step = options.pop('step', None)
    if step not in ['max_step', 'min_step', None]:
        raise ValueError('step can only take values'
                         ' as `max_step` or `min_step`')
    step_ratio = options.pop('step_ratio', None)
    if n == 0:
        derivative = f(x)
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
        if n % 2 == 0 and method is 'central':
            fxi = f(x)
            results = [((f(x + h) + f(x - h)) / 2.0 - fxi) for h in steps]
            fd_step = 2
            offset = 2
        if n % 2 == 1 and method is 'central':
            fxi = 0.0
            results = [((f(x + h) - f(x - h)) / 2.0) for h in steps]
            fd_step = 2
            offset = 1
        if method is 'forward':
            fxi = f(x)
            results = [(f(x + h) - fxi) for h in steps]
            fd_step = 1
            offset = 1
        if method is 'backward':
            fxi = f(x)
            results = [(fxi - f(x - h)) for h in steps]
            fd_step = 1
            offset = 1
        fun = np.vstack(list(np.ravel(r)) for r in results)
        h = np.vstack(list(
                np.ravel(np.ones(np.shape(
                                         results[0]))*step)) for step in steps)
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
        derivative = fdiff / (h**n)
        num_steps = max(h.shape[0] + 1 - fdi.size, 1)
        derivative = extrapolate(order, richarson_terms, richardson_step,
                                 step_ratio, derivative[:num_steps],
                                 h[:num_steps], np.shape(results[0]))
    return derivative
