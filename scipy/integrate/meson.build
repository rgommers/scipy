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
odeint_banded_test_src = ['tests/banded5x5.f']

mach = static_library('mach', 
    mach_src,
    fortran_args: fortran_ignore_warnings)
quadpack = static_library('quadpack', 
    quadpack_src,
    fortran_args: fortran_ignore_warnings)
lsoda = static_library('lsoda', 
    lsoda_src,
    fortran_args: fortran_ignore_warnings)
vode = static_library('vode', 
    vode_src,
    fortran_args: fortran_ignore_warnings)
dop = static_library('dop', 
    dop_src,
    fortran_args: fortran_ignore_warnings)

py3.extension_module('_quadpack',
    ['__quadpack.h', '_quadpackmodule.c'] + quadpack_src + mach_src,
    fortran_args: fortran_ignore_warnings,
    include_directories: [inc_np, '../_lib/src'],
    link_with: [quadpack, mach],
    dependencies: [py3_dep]
)