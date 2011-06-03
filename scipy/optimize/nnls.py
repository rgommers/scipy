from _nnls_py import *

import warnings
msg = """
optimize.nnls has been renamed to optimize._nnls_py. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

