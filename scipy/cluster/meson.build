# NOTE: depends on scipy.linalg.cython_blas/lapack!

#slib = py3.extension_module('_hierarchy',
#  '_hierarchy.pyx',
#  dependencies : py3_dep)
#
#slib = py3.extension_module('_optimal_leaf_ordering',
#  '_optimal_leaf_ordering.pyx',
#  dependencies : py3_dep)
#
#slib = py3.extension_module('_vq',
#  '_vq.pyx',
#  dependencies : py3_dep)


python_sources = [
  '__init__.py',
  'hierarchy.py',
  'vq.py',
]


py3.install_sources(
    python_sources,
    pure: false,             # Will be installed next to binaries
    subdir: 'scipy/cluster'  # Folder relative to site-packages to install to
)

subdir('tests')
