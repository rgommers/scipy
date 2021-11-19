py3.install_sources([
    '__init__.py',
    '_svds.py',
    '_svds_doc.py'
  ],
  pure: false,
  subdir: 'scipy/sparse/linalg/_eigen'
)

subdir('arpack')
subdir('lobpcg')
# Install the tests directory once the failing eigen tests are fixed.
# See details at: https://github.com/scipy/scipy/pull/14868#issuecomment-964444731
# Discussion at: https://github.com/rgommers/scipy/pull/93
# subdir('tests')
