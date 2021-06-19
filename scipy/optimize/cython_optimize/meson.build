py3.install_sources(
    ['__init__.py',
     'c_zeros.pxd', 
     '_zeros.pxd'],
    pure: false,
    subdir: 'scipy/optimize'
)

# TODO: generated .pyx which has relative cimport in it doesn't work yet (see
#       note above `_decomp_update` target in linalg/meson.build.
_zeros_pyx = custom_target('_zeros_pyx',
   output : '_zeros.pyx',
   input : '_zeros.pyx.in',
   command : [py3, tempita, '@INPUT@', '-o', '@OUTDIR@'],
 )

_zeros = py3.extension_module('_zeros',
   _zeros_pyx,
   include_directories: ['.', inc_np],
   dependencies: [py3_dep],
   install: true,
   subdir: 'scipy/optimize')
