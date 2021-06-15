include_dirs = [incdir_numpy, '../_lib/src']
fortran_ignore_warnings = [
  '-Wno-tabs', '-Wno-conversion',
  '-Wno-argument-mismatch', '-Wno-unused-dummy-argument',
  '-Wno-maybe-uninitialized', '-Wno-unused-label',
  '-Wno-unused-variable']
c_ignore_warnings = ['-Wno-unused-variable', 
                     '-Wno-unused-const-variable']

minpack_lib = static_library('minpack', ['minpack/hybrj.f', 
     'minpack/lmstr1.f', 
     'minpack/lmdif.f', 
     'minpack/fdjac1.f', 
     'minpack/fdjac2.f', 
     'minpack/enorm.f', 
     'minpack/lmder1.f', 
     'minpack/r1mpyq.f', 
     'minpack/r1updt.f', 
     'minpack/lmdif1.f', 
     'minpack/rwupdt.f', 
     'minpack/qform.f', 
     'minpack/qrfac.f', 
     'minpack/hybrd1.f', 
     'minpack/lmder.f', 
     'minpack/chkder.f', 
     'minpack/qrsolv.f', 
     'minpack/lmstr.f', 
     'minpack/hybrj1.f', 
     'minpack/dpmpar.f', 
     'minpack/hybrd.f', 
     'minpack/dogleg.f', 
     'minpack/lmpar.f'])

_minpack = py3.extension_module('_minpack',
  ['minpack.h', '__minpack.h', '_minpackmodule.c'],
  link_with : minpack_lib,
  c_args : [numpy_nodepr_api] + c_ignore_warnings,
  dependencies : [py3_dep],
  include_directories : include_dirs,
  install : true,
  subdir : 'scipy/optimize')

cpp_args = []
link_args = []
if host_machine.system() == 'darwin'
  if compiler.has_argument('-mmacosx-version-min=10.9')
      cpp_args = ['-mmacosx-version-min=10.9'] + c_ignore_warnings
      link_args = ['-mmacosx-version-min=10.9']
  endif
endif

rectangular_lasp = static_library('rectangular_lasp',
     ['rectangular_lsap/rectangular_lsap.h',
      'rectangular_lsap/rectangular_lsap.cpp'],
      cpp_args: cpp_args + c_ignore_warnings,
      link_args: link_args)

_lasp = py3.extension_module('_lasp',
  ['_lsap_module.c'],
  link_with : rectangular_lasp,
  c_args : [numpy_nodepr_api] + c_ignore_warnings,
  dependencies : [py3_dep],
  include_directories : include_dirs,
  install : true,
  subdir : 'scipy/optimize')

rootfind_lib = static_library('rootfind', 
     ['Zeros/zeros.h',
     'Zeros/brentq.c', 
     'Zeros/ridder.c', 
     'Zeros/bisect.c', 
     'Zeros/brenth.c'])

_minpack = py3.extension_module('_zeros',
  ['zeros.c',],
  link_with : rootfind_lib,
  c_args : [numpy_nodepr_api] + c_ignore_warnings,
  dependencies : [py3_dep],
  include_directories : include_dirs,
  install : true,
  subdir : 'scipy/optimize')

lbfgsb_module = custom_target('lbfgsb_module',
  output : ['_lbfgsb-f2pywrappers.f', '_lbfgsbmodule.c'],
  input : 'lbfgsb_src/lbfgsb.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

_lbfgsb = py3.extension_module('_lbfgsb', 
  ['lbfgsb_src/lbfgsb.f', 'lbfgsb_src/linpack.f', 
  'lbfgsb_src/timer.f', lbfgsb_module, fortranobject_c],
  c_args : [numpy_nodepr_api] + c_ignore_warnings,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

moduleTNC = py3.extension_module('moduleTNC',
  ['tnc/tnc.h', 'tnc/moduleTNC.c', 'tnc/tnc.c'],
  c_args: [numpy_nodepr_api] + c_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

cobyla_module = custom_target('cobyla_module',
  output : ['_cobylamodule.c'],
  input : 'cobyla/cobyla.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

_cobyla = py3.extension_module('_cobyla',
  [cobyla_module, fortranobject_c, 'cobyla/cobyla2.f', 'cobyla/trstlp.f'],
  c_args: [numpy_nodepr_api] + c_ignore_warnings,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

minpack2_module = custom_target('minpack2_module',
  output : ['minpack2module.c'],
  input : 'minpack2/minpack2.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

minpack2 = py3.extension_module('minpack2',
  [minpack2_module, fortranobject_c, 'minpack2/dcsrch.f', 'minpack2/dcstep.f'],
  c_args: [numpy_nodepr_api] + c_ignore_warnings,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')
  
slsqp_module = custom_target('slsqp_module',
  output : ['_slsqpmodule.c'],
  input : 'slsqp/slsqp.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

_slsqp = py3.extension_module('_slsqp',
  [slsqp_module, fortranobject_c, 'slsqp/slsqp_optmz.f'],
  c_args: [numpy_nodepr_api] + c_ignore_warnings,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')