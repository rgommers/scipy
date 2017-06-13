from __future__ import division, absolute_import
import numpy as np
from numpy.testing import (assert_, run_module_suite, assert_allclose)
from scipy.diff._derivative import derivative, gradient, jacobian


class Test(object):
    def test_derivative(self):
        funcs = [
                (lambda x: x**2, lambda x: 2*x),
                (lambda x: x**3 + x**2, lambda x: 3*x**2 + 2*x),
                (lambda x: np.exp(x), lambda x: np.exp(x)),
                (lambda x: np.cos(x), lambda x: -np.sin(x)),
                (lambda x: np.cosh(x), lambda x: np.sinh(x)),
        ]
        for func in funcs:
            for method in ['central', 'forward']:
                np.random.seed(0)
                rand = 2*10*np.random.rand(1, 100)-10
                df = derivative(func[0], rand, method=method)
                f = func[1](rand)
                assert_(np.allclose(df, f, atol=1e-12))

    def test_gradient(self):
        funcs = [
                (lambda x, y: np.sin(x) + np.exp(y),
                 lambda x, y: [np.cos(x), np.exp(y)]),
                (lambda x, y: x**3 + y**2, lambda x, y: [3*x**2, 2*y]),
        ]
        for func in funcs:
            np.random.seed(0)
            rand = 2*10*np.random.rand(2, 100)-10
            df = gradient(func[0], np.transpose(rand))
            f = np.reshape(np.transpose(func[1](*rand)), df.shape)
            assert_(np.allclose(df, f, atol=1e-12))

    def test_jacobian(self):
        funcs = [
            (lambda x, y: [x**3 + y**2, x*y**3],
             lambda x, y: [[3*x**2, y**3], [2*y, 3*x*y**2]]),
            (lambda x, y: [np.sin(x) + np.exp(y), 2*x*y],
             lambda x, y: [[np.cos(x), 2*y], [np.exp(y), 2*x]]),
        ]
        for func in funcs:
            np.random.seed(0)
            rand = 2*10*np.random.rand(2, 100)-10
            df = jacobian(func[0], np.transpose(rand))
            f = np.reshape(np.transpose(func[1](*rand)), df.shape)
            assert_allclose(df, f, atol=1e-7)


if __name__ == '__main__':
    run_module_suite()
