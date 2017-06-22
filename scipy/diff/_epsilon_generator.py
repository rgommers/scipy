import numpy as np

EPS = np.MachAr().eps


def _epsilon(x, s=3, eps=None):
    """
    Generates steps for computing derivatives.

    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated.
    s : int, optional
        scale for generation of steps.
    eps : array, optional
        value of the steps

    Returns
    -------
    epsilon : array
        steps
    """

    if eps is not None:
        if np.isscalar(eps):
            epsilon = np.empty(x.size)
            epsilon.fill(eps)
        else:
            epsilon = np.asarray(eps)
            if x.shape != epsilon.shape:
                raise ValueError(
                    'epsilon should either be a scalar'
                    'or must have same shape as x')
    else:
        epsilon = EPS**(1./s)*np.maximum(np.abs(x), 0.1)

    return epsilon
