project(
  'SciPy',
  'c', 'cpp', 'cython', 'fortran',
  version: '1.7.0.dev0+ababababab', # FIXME: git hash
  license: 'BSD-3',
  meson_version: '>= 0.54',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
    'cpp_std=c++14',  # TODO: use c++11 if 14 is not available
  ],
)

cython = find_program('cython')
generate_f2pymod = files('tools/generate_f2pymod.py')
tempita = files('scipy/_build_utils/tempita.py')

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation()
py3_dep = py3.dependency()

subdir('scipy')
