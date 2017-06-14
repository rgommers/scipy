from __future__ import division, absolute_import
import numpy as np
from numpy.testing import (run_module_suite, assert_allclose)
from scipy.diff._jacobian_numdiff import jacobian


class Test(object):
    def test_scalar_univariate_functions(self):
        funcs = [
                (lambda x: x**2, lambda x: 2*x),
                (lambda x: x**3 + x**2, lambda x: 3*x**2 + 2*x),
                (lambda x: np.exp(x), lambda x: np.exp(x)),
                (lambda x: np.cos(x), lambda x: -np.sin(x)),
                (lambda x: np.cosh(x), lambda x: np.sinh(x)),
        ]
        for func in funcs:
            methods = ['central', 'forward', 'backward']
            for method in methods:
                np.random.seed(0)
                rand = 2*10*np.random.rand()-10
                df = jacobian(func[0], rand, method=method)
                f = func[1](rand)
                assert_allclose(df, f)

    def test_scalar_functions(self):
        funcs = [
                (lambda x: x[0]*x[1]**2, lambda x: [[x[1]**2, 2*x[0]*x[1]]]),
                (lambda x: x[0]**3 + x[1]**2, lambda x: [[3*x[0]**2, 2*x[1]]]),
                (lambda x: np.exp(x[0]) + np.exp(x[1]),
                 lambda x: [[np.exp(x[0]), np.exp(x[1])]]),
                (lambda x: np.cos(x[0])+np.sin(x[1]),
                 lambda x: [[-np.sin(x[0]), np.cos(x[1])]]),
        ]
        for func in funcs:
            methods = ['central', 'forward', 'backward']
            for method in methods:
                np.random.seed(0)
                rand = 2*10*np.random.rand(10, 2)-10
                df = jacobian(func[0], rand, method=method)
                f = func[1](np.transpose(rand))
                df = np.asarray(df)
                f = np.reshape(np.transpose(np.asarray(f)), df.shape)
                assert_allclose(df, f, atol=1e-10)

    def test_vector_functions(self):
        funcs = [
                (lambda x: [[x[0]*x[1]**2], [x[0]*x[1]]],
                 lambda x: [[x[1]**2, x[1]], [2*x[0]*x[1], x[0]]]),
                (lambda x: [[x[0]**3 + x[1]**2], [x[0]**2 + x[1]**3]],
                 lambda x: [[3*x[0]**2, 2*x[0]], [2*x[1], 3*x[1]**2]]),
                (lambda x: [[np.exp(x[0]) + np.exp(x[1])],
                 [np.exp(x[0]+x[1])]],
                 lambda x: [[np.exp(x[0]), np.exp(x[0]+x[1])],
                 [np.exp(x[1]), np.exp(x[0]+x[1])]]),
                (lambda x: [[np.cos(x[0])+np.sin(x[1])],
                 [np.cosh(x[0])+np.sinh(x[1])]],
                 lambda x: [[-np.sin(x[0]), np.sinh(x[0])],
                 [np.cos(x[1]), np.cosh(x[1])]]),
        ]
        for func in funcs:
            np.random.seed(0)
            rand = 2*10*np.random.rand(10, 2)-10
            df = jacobian(func[0], rand)
            f = func[1](np.transpose(rand))
            df = np.asarray(df)
            f = np.reshape(np.transpose(np.asarray(f)), df.shape)
            assert_allclose(df, f)


if __name__ == '__main__':
    run_module_suite()
