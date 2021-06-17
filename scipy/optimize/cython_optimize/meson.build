py3.install_sources(
    ['c_zeros.pxd', 
     '_zeros.pxd'],
    pure: false,
    subdir: 'scipy/optimize'
)

run_command(py3, [cythonize, '.'], check : true)


_zeros = py3.extension_module('_zeros',
  ['_zeros.pyx'],
  include_directories: [inc_np],
  dependencies: [py3_dep],
  install: true,
  subdir: 'scipy/optimize')