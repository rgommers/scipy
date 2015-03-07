"""
==========================================
Miscellaneous routines (:mod:`scipy.misc`)
==========================================

.. currentmodule:: scipy.misc

Various utilities that don't have another home.

Note that the Python Imaging Library (PIL) is not a dependency
of SciPy and therefore the `pilutil` module is not available on
systems that don't have PIL installed.

.. autosummary::
   :toctree: generated/

   bytescale - Byte scales an array (image)
   central_diff_weights - Weights for an n-point central m-th derivative
   comb - Combinations of N things taken k at a time, "N choose k" (imported from scipy.special)
   derivative - Find the n-th derivative of a function at a point
   factorial  - The factorial function, n! = special.gamma(n+1) (imported from scipy.special)
   factorial2 - Double factorial, (n!)! (imported from scipy.special)
   factorialk - (...((n!)!)!...)! where there are k '!' (imported from scipy.special)
   fromimage - Return a copy of a PIL image as a numpy array
   imfilter - Simple filtering of an image
   imread - Read an image file from a filename
   imresize - Resize an image
   imrotate - Rotate an image counter-clockwise
   imsave - Save an array to an image file
   imshow - Simple showing of an image through an external viewer
   info - Get help information for a function, class, or module
   lena - Get classic image processing example image Lena
   logsumexp - Compute the log of the sum of exponentials of input elements
   pade - Pade approximation to function as the ratio of two polynomials
   toimage - Takes a numpy array and returns a PIL image
   who - Print the Numpy arrays in the given dictionary

"""

from __future__ import division, print_function, absolute_import

__all__ = ['who', 'source', 'info', 'doccer']

import sys

import numpy as np

from . import doccer
from .common import *

who = np.deprecate(np.who, message="misc.who is deprecated in Scipy 0.16.0, "
                                "please use it as numpy.who")
source = np.deprecate(np.source,
                      message="misc.source is deprecated in Scipy0.16.0, "
                              "please use it as numpy.source")

from scipy.special import comb, factorial, factorial2, factorialk
comb = np.deprecate(comb, message="misc.comb is deprecated in Scipy 0.16.0, "
                                  "please use it as scipy.special.comb")
factorial = np.deprecate(factorial,
                         message="misc.factorial is deprecated in Scipy 0.16.0, "
                                 "please use it as scipy.special.factorial")
factorial2 = np.deprecate(factorial2,
                          message="misc.factorial2 is deprecated in Scipy "
                                  "0.16.0, please use it as "
                                  "scipy.special.factorial2")
factorialk = np.deprecate(factorialk,
                          message="misc.factorialk is deprecated in Scipy "
                                  "0.16.0, please use it as "
                                  "scipy.special.factorialk")

from numpy import info as _info
def info(object=None, maxwidth=76, output=sys.stdout, toplevel='scipy'):
    return _info(object, maxwidth, output, toplevel)
info.__doc__ = _info.__doc__
del sys
del np

try:
    from .pilutil import *
    from . import pilutil
    __all__ += pilutil.__all__
    del pilutil
except ImportError:
    pass

from . import common
__all__ += common.__all__
del common

from numpy.testing import Tester
test = Tester().test
