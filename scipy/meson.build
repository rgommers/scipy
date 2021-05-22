# NumPy include directory - needed in all submodules
incdir_numpy = run_command(py3,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

# pybind11 include directory - needed in several submodules
incdir_pybind11 = run_command(py3,
  ['-c', 'import pybind11; print(pybind11.get_include())'],
  check : true
).stdout().strip()

inc_pybind11 = include_directories(incdir_pybind11)

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
#   py3.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION'

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

subdir('_lib')
subdir('cluster')
subdir('constants')
subdir('fft')
subdir('fftpack')
subdir('ndimage')
subdir('odr')
