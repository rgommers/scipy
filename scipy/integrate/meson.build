fortran_ignore_warnings = [
  '-Wno-tabs', '-Wno-conversion',
  '-Wno-argument-mismatch', '-Wno-unused-dummy-argument',
  '-Wno-maybe-uninitialized', '-Wno-unused-label',
  '-Wno-unused-variable']

mach_src = ['mach/d1mach.f', 'mach/xerror.f']
quadpack_src = ['quadpack/dqagie.f',
    'quadpack/dqk15i.f',
    'quadpack/dqawo.f',
    'quadpack/dqelg.f',
    'quadpack/dqage.f',
    'quadpack/dqng.f',
    'quadpack/dqwgts.f',
    'quadpack/dqwgtc.f',
    'quadpack/dqag.f',
    'quadpack/dqcheb.f',
    'quadpack/dqc25s.f',
    'quadpack/dqagpe.f',
    'quadpack/dqk51.f',
    'quadpack/dqk41.f',
    'quadpack/dqagi.f',
    'quadpack/dqags.f',
    'quadpack/dqpsrt.f',
    'quadpack/dqk15.f',
    'quadpack/dqaws.f',
    'quadpack/dqc25f.f',
    'quadpack/dqwgtf.f',
    'quadpack/dqawfe.f',
    'quadpack/dqagse.f',
    'quadpack/dqawf.f',
    'quadpack/dqawse.f',
    'quadpack/dqawc.f',
    'quadpack/dqk61.f',
    'quadpack/dqawoe.f',
    'quadpack/dqagp.f',
    'quadpack/dqc25c.f',
    'quadpack/dqk15w.f',
    'quadpack/dqk31.f',
    'quadpack/dqmomo.f',
    'quadpack/dqawce.f',
    'quadpack/dqk21.f'
]
lsoda_src = ['odepack/blkdta000.f',
    'odepack/bnorm.f',
    'odepack/cfode.f',
    'odepack/ewset.f',
    'odepack/fnorm.f',
    'odepack/intdy.f',
    'odepack/lsoda.f',
    'odepack/prja.f',
    'odepack/solsy.f',
    'odepack/srcma.f',
    'odepack/stoda.f',
    'odepack/vmnorm.f',
    'odepack/xerrwv.f',
    'odepack/xsetf.f',
    'odepack/xsetun.f'
]
vode_src = ['odepack/vode.f', 'odepack/zvode.f']
dop_src = ['dop/dop853.f', 'dop/dopri5.f']
quadpack_test_src = ['tests/_test_multivariate.c']

mach_lib = static_library('mach_lib',
    mach_src,
    fortran_args: fortran_ignore_warnings)
quadpack_lib = static_library('quadpack_lib',
    quadpack_src,
    fortran_args: fortran_ignore_warnings)
lsoda_lib = static_library('lsoda_lib',
    lsoda_src,
    c_args: '-Wno-unused-variable',
    fortran_args: fortran_ignore_warnings)
vode_lib = static_library('vode_lib',
    vode_src,
    c_args: '-Wno-unused-variable',
    fortran_args: fortran_ignore_warnings)
dop_lib = static_library('dop_lib',
    dop_src,
    c_args: '-Wno-unused-variable',
    fortran_args: fortran_ignore_warnings)

py3.extension_module('_quadpack',
    ['__quadpack.h', '_quadpackmodule.c'] + quadpack_src + mach_src,
    c_args: numpy_nodepr_api,
    fortran_args: fortran_ignore_warnings,
    include_directories: [inc_np, '../_lib/src'],
    link_with: [quadpack_lib, mach_lib],
    dependencies: [py3_dep, lapack],
    install: true,
    subdir: 'scipy/integrate'
)

py3.extension_module('_odepack',
    lsoda_src + mach_src + ['_odepackmodule.c'],
    c_args: numpy_nodepr_api,
    fortran_args: fortran_ignore_warnings,
    link_with: [lsoda_lib, mach_lib],
    dependencies: [py3_dep, lapack],
    include_directories: [inc_np],
    install: true,
    subdir: 'scipy/integrate'
)

vode_module = custom_target('vode_module',
  output : ['vode-f2pywrappers.f', 'vodemodule.c'],
  input : 'vode.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

py3.extension_module('vode',
    [vode_module, fortranobject_c],
    link_with: [vode_lib],
    c_args: [numpy_nodepr_api, '-Wno-unused-variable'],
    include_directories: [inc_np, inc_f2py],
    dependencies: [py3_dep, lapack],
    install: true,
    subdir: 'scipy/integrate'
)

lsoda_module = custom_target('lsoda_module',
  output : ['lsoda-f2pywrappers.f', 'lsodamodule.c'],
  input : 'lsoda.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

py3.extension_module('lsoda',
    [lsoda_module, fortranobject_c],
    link_with: [lsoda_lib, mach_lib],
    c_args: [numpy_nodepr_api, '-Wno-unused-variable'],
    dependencies: [py3_dep, lapack],
    include_directories: [inc_np, inc_f2py],
    install: true,
    subdir: 'scipy/integrate'
)

_dop_module = custom_target('_dop_module',
  output : ['_dop-f2pywrappers.f', '_dopmodule.c'],
  input : 'dop.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)

py3.extension_module('_dop',
    [_dop_module, fortranobject_c],
    link_with: [dop_lib],
    c_args: [numpy_nodepr_api, '-Wno-unused-variable'],
    dependencies: [py3_dep],
    include_directories: [inc_np, inc_f2py],
    install: true,
    subdir: 'scipy/integrate'
)

py3.extension_module('_test_multivariate',
    [quadpack_test_src],
    dependencies: [py3_dep],
    install: true,
    subdir: 'scipy/integrate'
)

# FIXME: something is wrong with the signature file, subroutines are not used
# _test_odeint_banded
_test_odeint_banded_module = custom_target('_test_odeint_banded_module',
  output : ['_test_odeint_bandedmodule.c', '_test_odeint_banded-f2pywrappers.f'],
  input : 'tests/banded5x5.pyf',
  command : [py3, generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'],
)
py3.extension_module('_test_odeint_banded',
  ['tests/banded5x5.f', fortranobject_c, _test_odeint_banded_module],
  c_args: numpy_nodepr_api,
  link_with: [lsoda_lib, mach_lib],
  fortran_args: '-Wno-unused-dummy-argument',
  include_directories: [inc_np, inc_f2py],
  dependencies: [py3_dep, lapack],
  install: true,
  subdir: 'scipy/integrate'
)

subdir('_ivp')
subdir('tests')

py3.install_sources(['_quad_vec.py',
    '__init__.py',
    '_quadrature.py',
    '_ode.py',
    'quadpack.py',
    '_bvp.py',
    'setup.py',
    'odepack.py'],
    pure : false,
    subdir : 'scipy/integrate'
)
