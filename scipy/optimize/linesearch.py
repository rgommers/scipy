from _linesearch import *

import warnings
msg = """
optimize.linesearch has been renamed to optimize._linesearch. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

