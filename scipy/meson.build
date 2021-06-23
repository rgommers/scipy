# NumPy include directory - needed in all submodules
incdir_numpy = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# pybind11 include directory - needed in several submodules
incdir_pybind11 = run_command(py3,
  ['-c', 'import pybind11; print(pybind11.get_include())'],
  check : true
).stdout().strip()

inc_pybind11 = include_directories(incdir_pybind11)

# Pythran include directory
incdir_pythran = run_command(py3,
  ['-c', 'import os; os.chdir(".."); import pythran; print(os.path.dirname(pythran.__file__));'],
  check : true
).stdout().strip()

inc_pythran = include_directories(incdir_pythran)

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
#   py3.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'


# TODO: 64-bit BLAS and LAPACK
blas = dependency('openblas')
lapack = dependency('openblas')

# TODO: deal with case where we do need the g77 ABI wrappers (see
#       _build_utils/_fortran.py)
g77_abi_wrappers = files('_build_utils' / 'src' / 'wrap_dummy_g77_abi.f')


python_sources = [
  '__config__.py',  # TODO: generate dynamically
  '__init__.py',
  '_distributor_init.py',
  'conftest.py',
  'version.py',  # TODO: generate dynamically
]

py3.install_sources(
    python_sources,
    pure: false,
    subdir: 'scipy'
)


# Needed to trick Cython, it won't do a relative import outside a package
_dummy_init = custom_target('_dummy_init',
  output : '__init__.py',
  input : '__init__.py',
  command : ['cp', '@INPUT@', '@OUTDIR@'],
)
_linalg_pxd = custom_target('_linalg_pxd',
  output : 'linalg.pxd',
  input : 'linalg.pxd',
  command : ['cp', '@INPUT@', '@OUTDIR@'],
)

# Note that subdirs with the most heavy builds should come first (that
# parallelizes better)
subdir('sparse')
subdir('stats')
subdir('fft')
subdir('_lib')
subdir('cluster')
subdir('constants')
subdir('fftpack')
subdir('integrate')
subdir('linalg')
subdir('ndimage')
subdir('odr')
subdir('optimize')
