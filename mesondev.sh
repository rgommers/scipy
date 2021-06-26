# Stop on error
set -e

# Command Line Arguments
# $1 - First argument is an absolute/relative 
#      path to build directory.
# $2 - Second argument is an absolute path to
#      install directory.

# TODO: Default $1 and $2 arguments to standard paths
# once migration to meson build system is complete.

# Build SciPy - use a prefix, without it is broken
meson setup $1 --prefix=$2
ninja -C $1
pushd $1
meson install
popd
# Avoid being in a directory which has a scipy/ dir
pushd $2

export PYTHONPATH=$2/lib/python3.9/site-packages/
# python -c "from scipy import linalg as s; s.test()"  # relies on sparse
python -c "from scipy import constants as s; s.test()"
# python -c "from scipy import ndimage as s; s.test()"  # relies on special
python -c "from scipy import odr as s; s.test()"
python -c "from scipy.sparse import csgraph as s; s.test()"
# python -c "from scipy import sparse as s; s.test()"  # relies on sparse.linalg
# python -c "from scipy import fftpack as s; s.test()"  # relies on fft
# python -c "from scipy import fft as s; s.test()"  # relies on special
# python -c "from scipy import _lib as s; s.test()"  # relies on spatial
# python -c "from scipy import integrate as s; s.test()" # relies on special
# python -c "from scipy import optimize as s; s.test()" # relies on sparse.linalg
# python -c "from scipy import stats as s; s.test()" # relies on spatial
# python -c "from scipy import special as s; s.test()" # relies on spatial
popd

# Notes:
#
# In case of an issue with build options, use the introspect command:
#   meson introspect build/ -i --targets
