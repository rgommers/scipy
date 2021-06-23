fortran_ignore_warnings = [
  '-Wno-tabs', '-Wno-conversion',
  '-Wno-argument-mismatch', '-Wno-unused-dummy-argument',
  '-Wno-maybe-uninitialized', '-Wno-unused-label',
  '-Wno-unused-variable']

subdir('tests')

statlib_lib = static_library('statlib_lib',
    ['statlib/ansari.f',
     'statlib/spearman.f',
     'statlib/swilk.f'])

statlib_module = custom_target('statlib_module',
  output : ['statlib-f2pywrappers.f', 'statlibmodule.c'],
  input : 'statlib.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

statlib = py3.extension_module('statlib',
  [statlib_module],
  c_args: numpy_nodepr_api,
  fortran_args: fortran_ignore_warnings,
  link_with : statlib_lib,
  dependencies : [py3_dep],
  include_directories : [inc_np, inc_f2py],
  install : true,
  subdir : 'scipy/stats')