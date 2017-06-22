import numpy as np
from numpy.testing import (assert_, run_module_suite)
from scipy.diff._step_generators import _generate_step

EPS = np.finfo(float).eps


class Test(object):
    def test_step_ratio(self):
        for n in range(1, 3):
            if n == 1:
                est = 2.0
            else:
                est = 1.6

            step_gen = _generate_step(n=n, step='max_step')
            step_gen = [step for step in step_gen]
            step_ratio = step_gen[0] / step_gen[1]
            assert_(np.allclose(step_ratio, est))

            step_gen = _generate_step(n=n, step='min_step')
            step_gen = [step for step in step_gen]
            step_ratio = step_gen[1] / step_gen[0]
            assert_(np.allclose(step_ratio, est))

    def test_base_step(self):
        for step in ['max_step', 'min_step']:
            if step == 'max_step':
                base_step = 2.0
            else:
                base_step = EPS**(1. / 2.5)

            step_gen = _generate_step(step=step)
            step_gen = [stepi for stepi in step_gen]
            assert_(np.allclose(step_gen[0], base_step))

            step_gen = _generate_step(step=step, base_step=10.0)
            step_gen = [stepi for stepi in step_gen]
            assert_(np.allclose(step_gen[0], 10.0))


if __name__ == '__main__':
    run_module_suite()
