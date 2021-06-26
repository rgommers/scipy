cpp_args = ['-DBOOST_MATH_DOMAIN_ERROR_POLICY=ignore_error']

if meson.get_compiler('cpp', native: true).sizeof('void*') > 4
    cpp_args += ['-DBOOST_MATH_PROMOTE_DOUBLE_POLICY=false']
endif

include = include_directories('include')
src = include_directories('src')
inc_boost = include_directories('..' / '..' / '_lib' / 'boost')

run_command(cython, '--cplus', 'src/hypergeom_ufunc.pyx')
hypergeom_ufunc = py3.extension_module('hypergeom_ufunc',
    ['src/hypergeom_ufunc.cpp'],
    c_args: numpy_nodepr_api,
    include_directories: [include, src, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

run_command(cython, '--cplus', 'src/nbinom_ufunc.pyx')
nbinom_ufunc = py3.extension_module('nbinom_ufunc',
    ['src/nbinom_ufunc.cpp'],
    c_args: numpy_nodepr_api,
    include_directories: [include, src, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

run_command(cython, '--cplus', 'src/beta_ufunc.pyx')
beta_ufunc = py3.extension_module('beta_ufunc',
    ['src/beta_ufunc.cpp'],
    c_args: numpy_nodepr_api,
    include_directories: [include, src, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)

run_command(cython, '--cplus', 'src/binom_ufunc.pyx')
binom_ufunc = py3.extension_module('binom_ufunc',
    ['src/binom_ufunc.cpp'],
    c_args: numpy_nodepr_api,
    include_directories: [include, src, inc_np, inc_boost],
    cpp_args: cpp_args,
    dependencies: [py3_dep]
)