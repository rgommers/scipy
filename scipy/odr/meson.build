odrpack = static_library('odrpack',
  [
    'odrpack/d_lpk.f',
    'odrpack/d_mprec.f',
    'odrpack/d_odr.f',
    'odrpack/dlunoc.f'
  ],
  fortran_args: [_fflag_Wno_conversion, _fflag_ilp64],
)

py3.extension_module('__odrpack',
  '__odrpack.c',
  link_with: odrpack,
  link_args: version_link_args,
  c_args: numpy_nodepr_api,
  dependencies: [blas_ilp64, np_dep],
  install: true,
  link_language: 'fortran',
  subdir: 'scipy/odr'
)


python_sources = [
  '__init__.py',
  '_add_newdocs.py',
  '_models.py',
  '_odrpack.py',
  'models.py',
  'odrpack.py'
]

py3.install_sources(
  python_sources,
  subdir: 'scipy/odr'
)

subdir('tests')
