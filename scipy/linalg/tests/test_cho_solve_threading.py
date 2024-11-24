from concurrent.futures import ThreadPoolExecutor
import multiprocessing.pool

import numpy as np

from scipy.linalg import cho_solve


def test_cho_solve_gh_21479():
    from threadpoolctl import threadpool_limits
    threadpool_limits(10, user_api="blas")

    rng = np.random.default_rng(0)
    L = rng.normal(size=(50, 50))
    W_sr = rng.normal(size=50)

    def func(i):
        cho_solve((L, True), np.diag(W_sr))

    with multiprocessing.pool.Pool(2):
        tpe = ThreadPoolExecutor(max_workers=2)
        tpe.map(func, range(10))
