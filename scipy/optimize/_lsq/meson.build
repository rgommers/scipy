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


_lsq = py3.extension_module('_lsq',
  ['givens_elimination.pyx'],
  include_directories: [inc_np],
  dependencies: [py3_dep],
  install: true,
  subdir: 'scipy/optimize/_lsq')
