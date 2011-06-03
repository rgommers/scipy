from _zeros_py import *

import warnings
msg = """
optimize.zeros has been renamed to optimize._zeros_py. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

