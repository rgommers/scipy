from __future__ import division, absolute_import
import numpy as np
from numpy.testing import (assert_, run_module_suite)
from scipy.diff.statmodels._derivative import derivative, gradient, jacobian


def fun(x):
    return x**2


def df_fun(x):
    return 2*x


def fun2(x, y):
    return x**3 + y**2


def df_fun2(x, y):
    return [3*x**2, 2*y]


def fun3(x, y):
    return [x**3 + y**2, x*y**3]


def df_fun3(x, y):
    return [[3*x**2, y**3], [2*y, 3*x*y**2]]


class Test(object):
    def test_derivative(self):
        for method in ['central', 'forward']:
            np.random.seed(0)
            rand = 2*10000*np.random.rand(1, 100000)-10000
            df = derivative(fun, rand, method=method)
            f = df_fun(rand)
            assert_(np.allclose(df, f))

    def test_gradient(self):
        np.random.seed(0)
        rand = 2*1000*np.random.rand(2, 1000)-1000
        df = gradient(fun2, np.transpose(rand))
        f = np.reshape(np.transpose(df_fun2(*rand)), df.shape)
        assert_(np.allclose(df, f, rtol=1e-3))

    def test_jacobian(self):
        np.random.seed(0)
        rand = 2*1000*np.random.rand(2, 1000)-1000
        df = jacobian(fun3, np.transpose(rand))
        f = np.transpose(df_fun3(*rand))
        assert_(np.allclose(df, f, rtol=1e-2))


if __name__ == '__main__':
    run_module_suite()
