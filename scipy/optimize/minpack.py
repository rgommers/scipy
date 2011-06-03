from _minpack_py import *

import warnings
msg = """
optimize.minpack has been renamed to optimize._minpack_py. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

