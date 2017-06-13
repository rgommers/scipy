from __future__ import division, absolute_import
import numpy as np
from numpy.testing import (assert_, run_module_suite)
from scipy.diff._derivative_numdiff import derivative


class Test(object):
    def test_funcs(self):
        funcs = [
                (lambda x: x**2, lambda x: 2*x),
                (lambda x: x**3 + x**2, lambda x: 3*x**2 + 2*x),
                (lambda x: np.exp(x), lambda x: np.exp(x)),
                (lambda x: np.cos(x), lambda x: -np.sin(x)),
                (lambda x: np.cosh(x), lambda x: np.sinh(x)),
        ]
        for func in funcs:
            np.random.seed(0)
            rand = 2*100*np.random.rand(1, 100)-100
            df = derivative(func[0], rand)
            f = func[1](rand)
            assert_(np.allclose(df, f))

    def test_for_higher_orders(self):
            true_vals = (0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0)
            methods = ['central', 'forward', 'backward']
            for method in methods:
                n_max = dict(central=6).get(method, 5)
                for n in range(0, n_max + 1):
                    true_val = true_vals[n]
                    for order in range(2, 7, 2):
                        d3cos = derivative(np.cos, np.pi/2.0, n=n, order=order,
                                           method=method)
                        assert_(np.allclose(d3cos,
                                            np.asarray(true_val), atol=1e-4))


if __name__ == '__main__':
    run_module_suite()
