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
#python -c "from scipy import optimize as s; s.test()"
export OMP_NUM_THREADS=2
python -c "import scipy; scipy.test(parallel=2)"
popd

# Notes:
#
# In case of an issue with build options, use the introspect command:
#   meson introspect build/ -i --targets
