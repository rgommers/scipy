qhull_headers = ['qhull_src/src/mem_r.h',
'qhull_src/src/poly_r.h',
'qhull_src/src/io_r.h',
'qhull_src/src/qhull_ra.h',
'qhull_src/src/merge_r.h',
'qhull_src/src/user_r.h',
'qhull_src/src/qset_r.h',
'qhull_src/src/random_r.h',
'qhull_src/src/libqhull_r.h',
'qhull_src/src/geom_r.h',
'qhull_src/src/stat_r.h']

qhull_src = ['qhull_src/src/poly_r.c',
'qhull_src/src/stat_r.c',
'qhull_src/src/merge_r.c',
'qhull_src/src/userprintf_r.c',
'qhull_src/src/userprintf_rbox_r.c',
'qhull_src/src/usermem_r.c',
'qhull_src/src/geom_r.c',
'qhull_src/src/libqhull_r.c',
'qhull_src/src/user_r.c',
'qhull_src/src/random_r.c',
'qhull_src/src/geom2_r.c',
'qhull_src/src/mem_r.c',
'qhull_src/src/global_r.c',
'qhull_src/src/poly2_r.c',
'qhull_src/src/io_r.c',
'qhull_src/src/rboxlib_r.c',
'qhull_src/src/qset_r.c']

# TODO: Add c_args: numpy_nodepr_api, once NumPy API usage is corrected.
qhull = py3.extension_module('qhull',
  qhull_headers + ['qhull.pyx', 'qhull_misc.h', 'qhull_misc.c'] + qhull_src,
  include_directories: [incdir_numpy, '../_lib', '../_build_utils/src'],
  dependencies: [py3_dep],
  install : true,
  subdir : 'scipy/spatial')


# subdir('tests')
# subdir('transform')