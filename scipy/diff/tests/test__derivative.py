from __future__ import division, absolute_import
import numpy as np
from numpy.testing import (assert_, run_module_suite, assert_allclose)
from scipy.diff._derivative import jacobian as jacobian1
from scipy.diff._derivative import gradient as gradient1
from scipy.diff._derivative import derivative as derivative1
from scipy.diff._derivative_numdiff import derivative as derivative2
from scipy.diff._jacobian_numdiff import jacobian as jacobian2
from scipy.diff._jacobian_numdiff import gradient as gradient2


class Test(object):
    def __init__(self, derivative=None, jacobian=None,
                 gradient=None, atol=None):
        self.derivative = derivative
        self.jacobian = jacobian
        self.gradient = gradient
        self.atol = atol

    def der(self):
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
                df = self.derivative(func[0], rand, method=method)
                f = func[1](rand)
                assert_(np.allclose(df, f, atol=1e-12))

    def grad(self):
        funcs = [
                (lambda x: x[0]*x[1]**2, lambda x: [[x[1]**2, 2*x[0]*x[1]]]),
                (lambda x: x[0]**3 + x[1]**2, lambda x: [[3*x[0]**2, 2*x[1]]]),
                (lambda x: np.exp(x[0]) + np.exp(x[1]),
                 lambda x: [[np.exp(x[0]), np.exp(x[1])]]),
                (lambda x: np.cos(x[0])+np.sin(x[1]),
                 lambda x: [[-np.sin(x[0]), np.cos(x[1])]]),
        ]
        for func in funcs:
            np.random.seed(0)
            rand = 2*10*np.random.rand(10, 2)-10
            df = self.gradient(func[0], rand)
            f = func[1](np.transpose(rand))
            df = np.asarray(df)
            f = np.reshape(np.transpose(np.asarray(f)), df.shape)
            assert_allclose(df, f, atol=self.atol)

    def jac(self):
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
            df = self.jacobian(func[0], rand)
            f = func[1](np.transpose(rand))
            df = np.asarray(df)
            f = np.reshape(np.transpose(np.asarray(f)), df.shape)
            assert_allclose(df, f, atol=1e-7)


class Test_numdiff(Test):
    def __init__(self):
        super(Test_numdiff, self).__init__(derivative=derivative2,
                                           jacobian=jacobian2,
                                           gradient=gradient2, atol=1e-10)

    def test_der(self):
        self.der()

    def test_grad(self):
        self.grad()

    def test_jac(self):
        self.jac()

    def test_for_higher_orders(self):
            true_vals = (0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0)
            methods = ['central', 'forward', 'backward']
            for method in methods:
                n_max = dict(central=6).get(method, 5)
                for n in range(0, n_max + 1):
                    true_val = true_vals[n]
                    for order in range(2, 7, 2):
                        d3cos = derivative2(np.cos, np.pi/2.0, n=n,
                                            order=order, method=method)
                        assert_(np.allclose(d3cos,
                                            np.asarray(true_val), atol=1e-4))


class Test_statsmodels(Test):
    def __init__(self):
        super(Test_statsmodels, self).__init__(derivative=derivative1,
                                               jacobian=jacobian1,
                                               gradient=gradient1, atol=1e-6)

    def test_derivative(self):
        self.der()

    def test_gradient(self):
        self.grad()

    def test_jacobian(self):
        self.jac()


if __name__ == '__main__':
    run_module_suite()
