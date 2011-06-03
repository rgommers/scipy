from _anneal import *

import warnings
msg = """
optimize.anneal has been renamed to optimize._anneal. It is not a
public module, please import from the optimize namespace instead.
"""
warnings.warn(msg, DeprecationWarning)

