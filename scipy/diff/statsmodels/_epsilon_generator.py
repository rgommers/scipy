import numpy as np

EPS = np.MachAr().eps


def _epsilon(x, s, eps):

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
