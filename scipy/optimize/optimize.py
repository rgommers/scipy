from _optimize import *

import warnings
msg = """
optimize.optimize has been renamed to optimize._optimize. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

