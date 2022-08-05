# Vendored from numpy, see `numpy/core/setup.py` and `numpy/core/src/npymath`
npymath_sources = [
  '_signbit.c',
  'halffloat.c',
  'ieee754.c',
  'npy_math.c',
  'npy_math_complex.c',
]

# TODO: should this use `/GL-` on Windows? See lib_opts_if_msvc in
#       numpy.distutils
npymath_lib = static_library('npymath', npymath_sources)
