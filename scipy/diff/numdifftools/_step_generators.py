from __future__ import division
import numpy as np
EPS = np.finfo(float).eps


def _generate_step(
                x=None, n=1, order=2, method='central', base_step=None,
                scale=500, num_steps=15, step_nom=None, step_ratio=None,
                offset=0, step=None):
    '''
    Generates steps in an adaptive manner

    Parameters
    ----------
    x : array, optional
        parameters at which steps are to be generated.
        Default is [1].

    n : int, optional
        order of the differentiation.
        Default is 1.

    order : int, optional
            defines the order of the error term
            in the Taylor approximation used.
            For 'central' method, it must be an even number.
            Default is 2.

    method : {'central', 'forward', 'backward'}, optional
             defines the method to be used.
             Default is 'central'

    scale : real scalar, optional
            scale used in base step.
            Default is 500

    base_step : float, array-like, optional
                Defines the start step.

    step_ratio : real scalar, optional, default 2
                 Ratio between sequential steps generated.
                 Note: Ratio > 1
                 If None then step_ratio is 2 for n=1
                 otherwise step_ratio is 1.6.

    num_steps : scalar integer, optional, default max(15, min_num_steps)
                defines number of steps generated. It should be larger than
                min_num_steps = (n + order - 1) / fact where fact is 1, 2 or 4
                depending on differentiation method used.

    step_nom :  array-like, default maximum(log(1+|x|), 1)
                Nominal step is the same shape as x.

    offset : real scalar, optional, default 0
             offset to the base step.

    step : {'max_step', 'min_step'}, optional, defult 'max_step'
            Defines the nature of the steps to be generated,
            increasing or decreasing.

    Returns
    -------
    steps : array
            array of generated sequence.

    Example
    -------
    >>> step_gen = _generate_step(base_step=0.25, step_ratio=2,
                                    num_steps=4, step='min_step')
    >>> [s for s in step_gen()]
    [0.25, 0.5, 1.0, 2.0]
    '''

    if x is None:
        x = np.asarray(1)
    x = np.asarray(x)
    if step_nom is None:
        step_nom = np.log1p(np.abs(x)).clip(min=1.0)
    else:
        step_nom = np.asarray(np.ones(x.shape)*step_nom)
    if step_ratio is None:
        if n == 1:
            step_ratio = 2.0
        else:
            step_ratio = 1.6
    min_num_steps = int(n+order-1)
    if method is 'central':
        min_num_steps = min_num_steps//2
    min_num_steps = max(min_num_steps, 1)
    num_steps = max(num_steps, min_num_steps)
    if step in ['max_step', None]:
        if base_step is None:
            base_step = 2.0
        base_step = base_step*step_nom
        for i in range(num_steps):
            steps = base_step * step_ratio ** (-i + offset)
            if (np.abs(steps) > 0).all():
                yield steps
    if step is 'min_step':
        if base_step is None:
            base_step = EPS ** (1. / scale)
        base_step = base_step*step_nom
        for i in range(num_steps):
            steps = base_step * step_ratio ** (i + offset)
            if (np.abs(steps) > 0).all():
                yield steps
