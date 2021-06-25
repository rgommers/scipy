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

pthread = dependency('threads')
run_command(cython, '--cplus', '_qmc_cy.pyx')
_qmc_cy = py3.extension_module('_qmc_cy',
  ['_qmc_cy.cpp'],
  dependencies : [py3_dep, pthread],
  include_directories : [inc_np],
  install : true,
  subdir : 'scipy/stats')

py3.install_sources(
  ['biasedurn.pxd'],
  pure : false,
  subdir : 'scipy/stats'
)

npymath_path = incdir_numpy / '..' / 'lib'
npymath_lib = meson.get_compiler('c').find_library('npymath', dirs: npymath_path)
npyrandom_path = incdir_numpy / '..' / '..' / 'random' / 'lib'
npyrandom_lib = meson.get_compiler('c').find_library('npyrandom', dirs: npyrandom_path)
run_command(py3, '_generate_pyx.py')
run_command(cython, '--cplus', 'biasedurn.pyx', '-o', 'biasedurn.cpp')
biasedurn = py3.extension_module('biasedurn',
  ['biasedurn/stocR.h', 
   'biasedurn.cpp', 'biasedurn/impls.cpp',
   'biasedurn/fnchyppr.cpp', 'biasedurn/wnchyppr.cpp',
   'biasedurn/stoc1.cpp', 'biasedurn/stoc3.cpp'],
   c_args: numpy_nodepr_api,
   cpp_args: ['-DR_BUILD'],
   include_directories: [inc_np],
   dependencies: [py3_dep, npyrandom_lib, npymath_lib]
)

# subdir('_boost')

py3.install_sources(
  ['_qmc_cy.pyi', 
   '_sobol.pyi'],
  pure : false,
  subdir : 'scipy/stats'
)

py3.install_sources(['qmc.py',
  '_distn_infrastructure.py',
  '_hypotests.py',
  '__init__.py',
  '_mannwhitneyu.py',
  '_distr_params.py',
  'mstats_basic.py',
  'mstats_extras.py',
  '_wilcoxon_data.py',
  '_tukeylambda_stats.py',
  '_binomtest.py',
  'stats.py',
  '_page_trend_test.py',
  'mstats.py',
  '_discrete_distns.py',
  '_result_classes.py',
  '_entropy.py',
  'morestats.py',
  '_constants.py',
  '_stats_mstats_common.py',
  '_common.py',
  '_multivariate.py',
  '_generate_pyx.py',
  '_continuous_distns.py',
  '_binned_statistic.py',
  'setup.py',
  '_rvs_sampling.py',
  '_crosstab.py',
  '_qmc.py',
  'distributions.py',
  'kde.py',
  '_relative_risk.py',
  'contingency.py',
  '_ksstats.py',
  '_bootstrap.py',],
  pure : false,
  subdir : 'scipy/stats'
)