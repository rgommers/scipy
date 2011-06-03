from _tnc import *

import warnings
msg = """
optimize.tnc has been renamed to optimize._tnc. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

