project(
  'SciPy',
  'c', 'cpp', 'cython', 'fortran',
  version: '1.7.0.dev0+ababababab', # FIXME: git hash
  license: 'BSD-3',
  meson_version: '>= 0.54',
  default_options: [
    'buildtype=debugoptimized',
    'c_std=c99',
  ],
)

cython = find_program('cython')

# https://mesonbuild.com/Python-module.html
py_mod = import('python')
py3 = py_mod.find_installation()
py3_dep = py3.dependency()

subdir('scipy')
