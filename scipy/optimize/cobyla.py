from _cobyla_py import *

import warnings
msg = """
optimize.cobyla has been renamed to optimize._cobyla_py. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

