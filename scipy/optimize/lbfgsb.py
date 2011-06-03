from _lbfgsb_py import *

import warnings
msg = """
optimize.lbfgsb has been renamed to optimize._lbfgsb_py. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

