# Need to copy it all to build dir, because of cimport statement
# TODO: dependency on cython/blas_lapack.pxd
_givens_elimination_c = custom_target('_givens_elimination_c',
  output: 'givens_elimination.c',
  input: [
    'givens_elimination.pyx',
    _cython_tree,
    # _cython_blas_pxd, _cython_lapack_pxd
  ],
  command: [cython, '-3', '--fast-fail',
            '@INPUT0@', '-o', '@OUTPUT@', '-I', '@BUILD_ROOT@'],
)

_lsq = py3.extension_module('givens_elimination',
  _givens_elimination_c,
  include_directories: [inc_np],
  dependencies: [py3_dep],
  install: true,
  subdir: 'scipy/optimize/_lsq'
)

py3.install_sources(
    ['__init__.py', 
     'least_squares.py', 
     'trf_linear.py', 
     'common.py', 
     'bvls.py', 
     'lsq_linear.py', 
     'dogbox.py', 
     'trf.py'],
    pure: false,
    subdir: 'scipy/optimize/_lsq'
)
