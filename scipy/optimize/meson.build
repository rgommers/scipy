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

compiler = meson.get_compiler('cpp')
compiler_id = compiler.get_id()
std_flag = ''
if compiler_id == 'msvc'
  if compiler.has_argument('/std=c++14')
    std_flag = '/std=c++14'
  endif
elif compiler_id == 'intelw'
  if compiler.has_argument('/Qstd=c++14')
    std_flag = '/Qstd=c++14'
  elif compiler.has_argument('/Qstd=c++11')
    std_flag = '/Qstd=c++11'
  endif
else
  if compiler.has_argument('-std=c++14')
    std_flag = '-std=c++14'
  elif compiler.has_argument('-std=c++11')
    std_flag = '-std=c++11'
  endif
endif

if std_flag != ''
  cpp_args = [std_flag]
else
  cpp_args = []
endif
link_args = []
if host_machine.system() == 'darwin'
  if compiler.has_argument('-mmacosx-version-min=10.9')
      cpp_args = cpp_args + ['-mmacosx-version-min=10.9']
      link_args = ['-mmacosx-version-min=10.9']
  endif
endif

rectangular_lasp = static_library('rectangular_lasp',
     ['rectangular_lsap/rectangular_lsap.h',
      'rectangular_lsap/rectangular_lsap.cpp'],
      cpp_args: cpp_args,
      link_args: link_args)

compiler = meson.get_compiler('c')
compiler_id = compiler.get_id()
if compiler_id == 'msvc'
  std_flag = ''
elif compiler_id == 'intelw'
  if compiler.has_argument('/Qstd=c99')
    std_flag = '/Qstd=c99'
  else
    std_flag = ''
  endif
else
  if compiler.has_argument('-std=c99')
    std_flag = '-std=c99'
  else
    std_flag = ''
  endif
endif

_lasp = py3.extension_module('_lasp',
  ['_lsap_module.c'],
  link_with : rectangular_lasp,
  c_args : [numpy_nodepr_api] + [std_flag],
  dependencies : [py3_dep],
  include_directories : include_dirs,
  install : true,
  subdir : 'scipy/optimize')