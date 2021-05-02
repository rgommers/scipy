# Build SciPy - use a prefix, without it is broken
meson setup build --prefix=/home/rgommers/code/bldscipy/installdir
ninja -C build
pushd build
meson install

export PYTHONPATH=~/code/bldscipy/installdir/lib/python3.9/site-packages/
python -c "from scipy import _lib as s; s.test()"
popd
