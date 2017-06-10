import numpy as np
from numpy.testing import (assert_, run_module_suite)
from scipy.diff.numdifftools._step_generators import _generate_step


def steps(base_step, step_ratio, num_steps, sign):
    for i in range(num_steps):
            steps = base_step * step_ratio ** (sign*i)
            if (np.abs(steps) > 0).all():
                yield steps


class Test(object):
    def test_min_step(self):
        for base_step in range(1, 10000):
            np.random.seed(0)
            step_ratio = 20*np.random.randint(2, 20)
            base_step = float(base_step)
            step_gen = _generate_step(
                                    base_step=base_step, step_ratio=step_ratio,
                                    num_steps=100, step='min_step')
            steps_gen = steps(
                            base_step=base_step, step_ratio=step_ratio,
                            num_steps=100, sign=1)
            step_gen = [step for step in step_gen]
            steps_gen = [step for step in steps_gen]
            assert_(np.allclose(steps_gen, steps_gen))

    def test_max_step(self):
        for base_step in range(1, 10000):
            np.random.seed(0)
            step_ratio = 20*np.random.randint(2, 20)
            base_step = float(base_step)
            step_gen = _generate_step(
                                    base_step=base_step, step_ratio=step_ratio,
                                    num_steps=100, step='max_step')
            steps_gen = steps(
                            base_step=base_step, step_ratio=step_ratio,
                            num_steps=100, sign=-1)
            step_gen = [step for step in step_gen]
            steps_gen = [step for step in steps_gen]
            assert_(np.allclose(steps_gen, steps_gen))


if __name__ == '__main__':
    run_module_suite()
