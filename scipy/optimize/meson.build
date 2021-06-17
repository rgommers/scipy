include_dirs = [inc_np, '../_lib/src']
fortran_ignore_warnings = [
  '-Wno-tabs', '-Wno-conversion',
  '-Wno-argument-mismatch', '-Wno-unused-dummy-argument',
  '-Wno-maybe-uninitialized', '-Wno-unused-label',
  '-Wno-unused-variable']

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
  c_args : numpy_nodepr_api,
  dependencies : [py3_dep],
  include_directories : include_dirs,
  install : true,
  subdir : 'scipy/optimize')

cpp_args = []
link_args = []
if host_machine.system() == 'darwin'
  if compiler.has_argument('-mmacosx-version-min=10.9')
      cpp_args = ['-mmacosx-version-min=10.9']
      link_args = ['-mmacosx-version-min=10.9']
  endif
endif

rectangular_lasp = static_library('rectangular_lasp',
     ['rectangular_lsap/rectangular_lsap.h',
      'rectangular_lsap/rectangular_lsap.cpp'],
      cpp_args: cpp_args,
      link_args: link_args)

_lasp = py3.extension_module('_lasp',
  ['_lsap_module.c'],
  link_with : rectangular_lasp,
  c_args : numpy_nodepr_api,
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

_zeros = py3.extension_module('_zeros',
  ['zeros.c',],
  link_with : rootfind_lib,
  c_args : numpy_nodepr_api,
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
  c_args : numpy_nodepr_api,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

moduleTNC = py3.extension_module('moduleTNC',
  ['tnc/tnc.h', 'tnc/moduleTNC.c', 'tnc/tnc.c'],
  c_args: numpy_nodepr_api,
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
  c_args: numpy_nodepr_api,
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
  c_args: numpy_nodepr_api,
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
  c_args: numpy_nodepr_api,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

__nnls_module = custom_target('__nnls_module',
  output : ['__nnlsmodule.c'],
  input : '__nnls/nnls.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

py3.install_sources(
    ['__nnls.pyi'],
    pure: false,
    subdir: 'scipy/optimize'
)

__nnls = py3.extension_module('__nnls',
  ['__nnls/nnls.f', __nnls_module, fortranobject_c],
  c_args: numpy_nodepr_api,
  fortran_args : fortran_ignore_warnings,
  include_directories : [inc_np, inc_f2py],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

res = run_command(py3, '-c', 'import os; print(os.environ.get("SCIPY_USE_PYTHRAN", 1))')
if res.returncode() == 0
  _group_columns = custom_target('_group_columns',
    output: ['_group_columns.cpp'],
    input : '_group_columns.py',
    command : [pythran, '-E', '@INPUT@', '-o', '@OUTDIR@/_group_columns.cpp'],
  )
  _group_columns = py3.extension_module('_group_columns',
  [_group_columns],
  cpp_args: ['-Wno-unused-function', '-Wno-int-in-bool-context'],
  include_directories: [inc_pythran],
  install: true,
  subdir: 'scipy/optimize')
else
  _group_columns = py3.extension_module('_group_columns',
  ['_group_columns.pyx'],
  include_directories : [inc_np],
  dependencies : [py3_dep],
  install: true,
  subdir: 'scipy/optimize')
endif

_bglu_dense = py3.extension_module('_bglu_dense',
  ['_bglu_dense.pyx'],
  include_directories: [inc_np],
  dependencies: [py3_dep],
  install: true,
  subdir: 'scipy/optimize')

subdir('_lsq')
subdir('_trlib')
subdir('_trustregion_constr')
subdir('cython_optimize')

subdir('_shgo_lib')

subdir('_highs')
subdir('tests')

py3.install_sources(
    ['lbfgsb_src/README'],
    pure: false,
    subdir: 'scipy/optimize'
)

py3.install_sources(
  ['_shgo.py', 
   'tnc.py', 
   '_linprog.py', 
   '_trustregion_ncg.py', 
   'zeros.py', 
   '_differentiable_functions.py', 
   '_spectral.py', 
   '__init__.py', 
   '_root.py', 
   '_linprog_doc.py', 
   'cobyla.py', 
   '_qap.py', 
   '_linprog_util.py', 
   'minpack.py', 
   '_dual_annealing.py', 
   '_hessian_update_strategy.py', 
   '_nnls.py', 
   '_trustregion_dogleg.py', 
   '_root_scalar.py', 
   '_linprog_rs.py', 
   '_trustregion_exact.py', 
   'linesearch.py', 
   'slsqp.py', 
   '_trustregion_krylov.py', 
   'lbfgsb.py', 
   '_linprog_simplex.py', 
   '_differentialevolution.py', 
   '_constraints.py', 
   '_linprog_ip.py', 
   '_trustregion.py', 
   '_lsap.py', 
   '_minimize.py', 
   'optimize.py', 
   '_remove_redundancy.py', 
   '_linprog_highs.py', 
   '_numdiff.py', 
   '_tstutils.py', 
   '_basinhopping.py', 
   'nonlin.py'],
   pure: false,
   subdir: 'scipy/optimize'
)
