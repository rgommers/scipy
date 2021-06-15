include_dirs = [incdir_numpy, '../_lib/src']

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
  c_args : [numpy_nodepr_api],
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
  c_args : numpy_nodepr_api,
  dependencies : [py3_dep],
  include_directories : include_dirs,
  install : true,
  subdir : 'scipy/optimize')
