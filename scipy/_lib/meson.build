py3.extension_module('_ccallback_c',
  '_ccallback_c.pyx',
  include_directories : 'src',
  dependencies : py3_dep,
  install : true,
  subdir : 'scipy/_lib')

py3.extension_module('_test_ccallback',
  'src/_test_ccallback.c',
  include_directories : 'src',
  dependencies : py3_dep,
  install : true,
  subdir : 'scipy/_lib')

py3.extension_module('_fpumode',
  '_fpumode.c',
  include_directories : 'src',
  dependencies : py3_dep,
  install : true,
  subdir : 'scipy/_lib')

py3.extension_module('_test_deprecation_call',
  '_test_deprecation_call.pyx',
  include_directories : 'src',
  dependencies : py3_dep,
  install : true,
  subdir : 'scipy/_lib')

py3.extension_module('_test_deprecation_def',
  '_test_deprecation_def.pyx',
  include_directories : 'src',
  dependencies : py3_dep,
  install : true,
  subdir : 'scipy/_lib')

# FIXME: add messagestream extension. We may not need that config file,
#        can we pass around a configuration object instead?

python_sources = [
  '__init__.py',
  '_bunch.py',
  '_ccallback.py',
  '_disjoint_set.py',
  '_gcutils.py',
  '_pep440.py',
  '_testutils.py',
  '_threadsafety.py',
  '_tmpdirs.py',
  '_util.py',
  'decorator.py',
  'deprecation.py',
  'doccer.py',
  'uarray.py',
]

py3.install_sources(
    python_sources,
    pure: false,          # Will be installed next to binaries
    subdir: 'scipy/_lib'  # Folder relative to site-packages to install to
)

subdir('_uarray')
subdir('tests')
