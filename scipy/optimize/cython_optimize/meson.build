py3.install_sources(
    ['__init__.py',
     'c_zeros.pxd', 
     '_zeros.pxd'],
    pure: false,
    subdir: 'scipy/optimize'
)

# FIXME: Fix the following
# FAILED: scipy/optimize/cython_optimize._zeros.cpython-39-x86_64-linux-gnu.so.p/scipy/optimize/_zeros.pyx.c 
# cython --fast-fail -3 -o scipy/optimize/cython_optimize._zeros.cpython-39-x86_64-linux-gnu.so.p/scipy/optimize/_zeros.pyx.c scipy/optimize/_zeros.pyx

# Error compiling Cython file:
# ------------------------------------------------------------
# ...
# """Cython wrappers for root finders in c_zeros"""

# from . cimport c_zeros
# ^
# ------------------------------------------------------------

# _zeros_pyx = custom_target('_zeros_pyx',
#   output : '_zeros.pyx',
#   input : '_zeros.pyx.in',
#   command : [py3, tempita, '@INPUT@', '-o', '@OUTDIR@'],
# )

# _zeros = py3.extension_module('_zeros',
#   [_zeros_pyx],
#   include_directories: ['.', inc_np],
#   dependencies: [py3_dep],
#   install: true,
#   subdir: 'scipy/optimize')