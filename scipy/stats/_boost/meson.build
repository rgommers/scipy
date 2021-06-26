cpp_args = ['-DBOOST_MATH_DOMAIN_ERROR_POLICY=ignore_error']

if meson.get_compiler('cpp', native: true).sizeof('void*') > 4
    cpp_args += ['-DBOOST_MATH_PROMOTE_DOUBLE_POLICY=false',
                 numpy_nodepr_api]
endif

include = include_directories('include')
inc_boost = include_directories('..' / '..' / '_lib' / 'boost')

# hypergeom_ufunc
_hypergeom_cy = custom_target('_hypergeom_cy',
  output : 'hypergeom_ufunc.cpp',
  input : _stats_gen_pyx[4],  # hypergeom_ufunc.pyx
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
)
hypergeom_ufunc = py3.extension_module('hypergeom_ufunc',
    _hypergeom_cy,
    include_directories: [include, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

# nbinom_ufunc
_nbinom_cy = custom_target('_nbinom_cy',
  output : 'nbinom_ufunc.cpp',
  input : _stats_gen_pyx[5],  # nbinom_ufunc.pyx
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
)
nbinom_ufunc = py3.extension_module('nbinom_ufunc',
    _nbinom_cy,
    include_directories: [include, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

# beta_ufunc
_beta_cy = custom_target('_beta_cy',
  output : 'beta_ufunc.cpp',
  input : _stats_gen_pyx[1],  # beta_ufunc.pyx
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
)
beta_ufunc = py3.extension_module('beta_ufunc',
    _beta_cy,
    include_directories: [include, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

# binom_ufunc
_binom_cy = custom_target('_binom_cy',
  output : 'binom_ufunc.cpp',
  input : _stats_gen_pyx[2],  # binom_ufunc.pyx
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
)
binom_ufunc = py3.extension_module('binom_ufunc',
    _binom_cy,
    include_directories: [include, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

py3.install_sources(['__init__.py'],
  pure : false,
  subdir : 'scipy/stats/_boost'
)
