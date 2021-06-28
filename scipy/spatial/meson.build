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
  ['qhull.pyx', 'qhull_misc.h', 'qhull_misc.c'] + qhull_src,
  include_directories: [incdir_numpy, '../_lib', '../_build_utils/src', 'qhull_src/src'],
  dependencies: [py3_dep],
  install : true,
  subdir : 'scipy/spatial')

ckdtree_src = ['ckdtree/src/query.cxx',
  'ckdtree/src/build.cxx',
  'ckdtree/src/query_pairs.cxx',
  'ckdtree/src/count_neighbors.cxx',
  'ckdtree/src/query_ball_point.cxx',
  'ckdtree/src/query_ball_tree.cxx',
  'ckdtree/src/sparse_distances.cxx']

ckdtree_cpp = custom_target('ckdtree_cpp',
  output : 'ckdtree.cpp',
  input : 'ckdtree.pyx',
  command : [cython, '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
)

ckdtree = py3.extension_module('ckdtree',
  ckdtree_src + [ckdtree_cpp],
  cpp_args: '-Wno-cpp', # TODO: replace with numpy_nodepr_api
  include_directories: [incdir_numpy, '../_lib', '../_build_utils/src', 'ckdtree/src'],
  dependencies: [py3_dep],
  install : true,
  subdir : 'scipy/spatial')

# TODO: Add c_args: numpy_nodepr_api,
# once NumPy API usage is corrected (can be done in main SciPy Repo)
_distance_wrap = py3.extension_module('_distance_wrap',
  'src/distance_wrap.c',
  include_directories: [incdir_numpy, '../_lib'],
  dependencies: [py3_dep, npymath_lib],
  install : true,
  subdir : 'scipy/spatial')

# subdir('tests')
# subdir('transform')