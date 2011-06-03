from _slsqp_py import *

import warnings
msg = """
optimize.slsqp has been renamed to optimize._slsqp_py. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

