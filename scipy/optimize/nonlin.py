from _nonlin import *

import warnings
msg = """
optimize.nonlin has been renamed to optimize._nonlin. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)


