# Stop on error
set -e

# Build SciPy - use a prefix, without it is broken
meson setup build --prefix=/home/rgommers/code/bldscipy/installdir
ninja -C build
pushd build
meson install

export PYTHONPATH=~/code/bldscipy/installdir/lib/python3.9/site-packages/
python -c "from scipy import constants as s; s.test()"
# python -c "from scipy import ndimage as s; s.test()"  # relies on special
python -c "from scipy import odr as s; s.test()"
python -c "from scipy import fftpack as s; s.test()"
python -c "from scipy import fft as s; s.test()"
#python -c "from scipy import _lib as s; s.test()"  # relies on spatial
popd

# Notes:
#
# In case of an issue with build options, use the introspect command:
#   meson introspect build/ -i --targets
