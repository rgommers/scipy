fortran_ignore_warnings = [
  '-Wno-tabs', '-Wno-conversion',
  '-Wno-argument-mismatch', '-Wno-unused-dummy-argument',
  '-Wno-maybe-uninitialized', '-Wno-unused-label',
  '-Wno-unused-variable']

statlib_lib = static_library('statlib_lib',
  ['statlib/ansari.f',
   'statlib/spearman.f',
   'statlib/swilk.f'
  ],
  fortran_args: '-Wno-unused-variable',
)

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

# TODO: relies on cython_special
_stats_c = custom_target('_stats_c',
  output: '_stats.c',
  input: [
    '_stats.pyx',
    _cython_tree,
    # _cython_special_pxd
  ],
  command: [cython, '-3', '--fast-fail',
            '@INPUT0@', '-o', '@OUTPUT@', '-I', '@BUILD_ROOT@'],
)


_stats = py3.extension_module('_stats',
  _stats_c,
  dependencies : [py3_dep],
  include_directories : [inc_np],
  install : true,
  subdir : 'scipy/stats'
)

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
_qmc_cpp = custom_target('_qmc_cpp',
  output : '_qmc_cy.cpp',
  input : '_qmc_cy.pyx',
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
)
_qmc_cy = py3.extension_module('_qmc_cy',
  _qmc_cpp,
  cpp_args: '-Wno-cpp',  # TODO: replace with numpy_nodepr_api
  dependencies : [py3_dep, pthread],
  include_directories : [inc_np],
  install : true,
  subdir : 'scipy/stats')

_stats_pxd = custom_target('_stats_pxd',
  output : [
    '__init__.py',
    '_stats.pxd',
    'biasedurn.pxd',
  ],
  input : [
    '__init__.py',
    '_stats.pxd',
    'biasedurn.pxd',
  ],
  command : ['cp', '@INPUT@', '-t', '@OUTDIR@'],
)

_stats_gen_pyx = custom_target('_stats_gen_pyx',
  output : [
    'biasedurn.pyx',
    'beta_ufunc.pyx',
    'binom_ufunc.pyx',
    'func_defs.pxd',
    'hypergeom_ufunc.pyx',
    'nbinom_ufunc.pyx',
    'templated_pyufunc.pxd',
  ],
  input : '_generate_pyx.py',
  command : [py3, '@INPUT@', '-o', '@OUTDIR@']
)
_biasedurn_cpp = custom_target('_biasedurn_cpp',
  output : 'biasedurn.cpp',
  input : [
    _stats_gen_pyx[0],  # biasedurn.pyx
    _stats_pxd,
    _cython_tree,
  ],
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT0@', '-o', '@OUTPUT@']
)
biasedurn = py3.extension_module('biasedurn',
  [_biasedurn_cpp,
   'biasedurn/fnchyppr.cpp',
   'biasedurn/impls.cpp',
   'biasedurn/stoc1.cpp',
   'biasedurn/stoc3.cpp',
   'biasedurn/stocR.h', 
   'biasedurn/wnchyppr.cpp',
  ],
  cpp_args: ['-DR_BUILD', numpy_nodepr_api],
  include_directories: [inc_np],
  dependencies: [py3_dep, npyrandom_lib, npymath_lib]
)

py3.install_sources([
  'qmc.py',
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
  '_bootstrap.py'],
  pure : false,
  subdir : 'scipy/stats'
)

py3.install_sources([
  'biasedurn.pxd',
  '_stats.pxd'],
  pure : false,
  subdir : 'scipy/stats'
)

py3.install_sources([
  '_qmc_cy.pyi',
  '_sobol.pyi'],
  pure : false,
  subdir : 'scipy/stats'
)

subdir('_boost')

subdir('tests')
