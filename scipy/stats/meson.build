fortran_ignore_warnings = [
  '-Wno-tabs', '-Wno-conversion',
  '-Wno-argument-mismatch', '-Wno-unused-dummy-argument',
  '-Wno-maybe-uninitialized', '-Wno-unused-label',
  '-Wno-unused-variable']

subdir('tests')

statlib_lib = static_library('statlib_lib',
    ['statlib/ansari.f',
     'statlib/spearman.f',
     'statlib/swilk.f'])

statlib_module = custom_target('statlib_module',
  output : ['statlib-f2pywrappers.f', 'statlibmodule.c'],
  input : 'statlib.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

statlib = py3.extension_module('statlib',
  [statlib_module],
  c_args: numpy_nodepr_api,
  fortran_args: fortran_ignore_warnings,
  link_with : statlib_lib,
  dependencies : [py3_dep],
  include_directories : [inc_np, inc_f2py],
  install : true,
  subdir : 'scipy/stats')

# TODO: Add c_args: numpy_nodepr_api 
_stats = py3.extension_module('_stats',
  ['_stats.pyx'],
  dependencies : [py3_dep],
  include_directories : [inc_np],
  install : true,
  subdir : 'scipy/stats')

mvn_module = custom_target('mvn_module',
  output : ['mvn-f2pywrappers.f', 'mvnmodule.c'],
  input : 'mvn.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

mvn = py3.extension_module('mvn',
  [mvn_module, 'mvndst.f'],
  c_args: numpy_nodepr_api,
  fortran_args: fortran_ignore_warnings,
  dependencies : [py3_dep],
  include_directories : [inc_np, inc_f2py],
  install : true,
  subdir : 'scipy/stats')

_sobol = py3.extension_module('_sobol',
  ['_sobol.pyx'],
  c_args: numpy_nodepr_api,
  dependencies : [py3_dep],
  include_directories : [inc_np],
  install : true,
  subdir : 'scipy/stats')

py3.install_sources(
  ['_sobol_direction_numbers.npz'],
  pure : false,
  subdir : 'scipy/stats'
)

run_command(cython, '--cplus', '_qmc_cy.pyx')
_qmc_cy = py3.extension_module('_qmc_cy',
  ['_qmc_cy.cpp'],
  cpp_args: ['-pthread'],
  dependencies : [py3_dep],
  include_directories : [inc_np],
  install : true,
  subdir : 'scipy/stats')

py3.install_sources(
  ['biasedurn.pxd'],
  pure : false,
  subdir : 'scipy/stats'
)

npyrandom_src = run_command(py3, '-c', 
  'import numpy, os; print(os.path.dirname(numpy.random.__file__) + "/lib/libnpyrandom.a")')
# npyrandom_lib = static_library('npyrandom_lib',
#  [npyrandom_src])

npymath_src = run_command(py3, '-c', 
  'import numpy, os; print(os.path.dirname(numpy.random.__file__) + "/lib/libnpymath.a")')
# npymath_lib = static_library('npymath_lib',
#  [npymath_src])

incdir_biasedurn = incdir_numpy / '..' / '..' / 'random' / 'lib'
inc_baisedurn = include_directories(incdir_biasedurn)
biasedurn = py3.extension_module('biasedurn',
  ['biasedurn.cxx', 'biasedurn/impls.cpp',
   'biasedurn/fnchyppr.cpp', 'biasedurn/wnchyppr.cpp',
   'biasedurn/stoc1.cpp', 'biasedurn/stoc3.cpp'],
    include_directories: [inc_np, inc_baisedurn],
#    link_with: [npyrandom, npymath],
#    define_macros=[('R_BUILD', None)],
#    language='c++',
    dependencies: ['biasedurn/stocR.h', npyrandom_src, npymath_src],
)