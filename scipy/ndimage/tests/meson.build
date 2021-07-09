python_sources = [
  '__init__.py',
  'test_datatypes.py',
  'test_fourier.py',
  'test_measurements.py',
  'test_splines.py',
  'dots.png',
  'test_c_api.py',
  'test_filters.py',
  'test_interpolation.py',
  'test_morphology.py',
]

py3.install_sources(
    python_sources,
    pure: false,
    subdir: 'scipy/ndimage/tests'
)

py3.install_sources(
  ['data/label_inputs.txt',
   'data/label_results.txt',
   'data/label_strels.txt',
  ],
  pure: false,
  subdir: 'scipy/ndimage/tests/data'
)
