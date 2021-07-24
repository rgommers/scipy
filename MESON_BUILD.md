# How to build SciPy with Meson

_note: these instructions are for Linux, if you are on another OS things likely
don't work yet! macOS support should be easy, please comment if we need to
prioritize adding that._

## quickstart from scratch

Clone the repo and check out the right branch:

```bash
git clone git@github.com:rgommers/scipy.git
git submodule update --init
git checkout meson
```

Create a conda development environment, build SciPy with Meson and run the test
suite:

```
conda env create -f environment.yml
conda activate scipy-dev
python -m pip install git+https://github.com/rgommers/meson.git@scipy
./mesondev.sh build --prefix=$PWD/install
```

## Full details and explanation

Now to build SciPy, we need the `meson` branch from `@rgommers`'s fork (this is
the integration branch until Meson support is merged into SciPy master):
```bash
git clone git@github.com:rgommers/scipy.git
git submodule update --init
git checkout meson
```

We will use conda here, because it's the easiest way to get a fully
reproducible environment. If you do not have a conda environment yet, the
recommended installer is
[Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (`mamba` is
basically a much faster `conda`).

To create a development environment:
```bash
conda env create -f environment.yml  # `mamba` works too for this command
conda activate scipy-dev
```

The one dependency that we now miss is `meson`. Support for Cython in Meson is
very new, and we expect to also need some bug fixes and new features in the
Meson master branch as well as fixes we may not even have submitted yet or are
pending review. So install Meson from the `scipy` branch of `@rgommers`'s fork:
```bash
python -m pip install git+https://github.com/rgommers/meson.git@scipy
```

Meson uses a configure and a build stage. To configure it for putting the build
artifacts in `build/` and a local install under `installdir/` and then build:
```bash
meson setup build --prefix=$PWD/installdir
ninja -C build
```
In the command above, `-C` is followed by the name of the build directory. You
can have multiple builds at the same time. Meson is fully out-of-place, so
those builds will not interfere with each other. You can for example have a GCC
build, a Clang build and a debug build in different directories.

To then install SciPy into the prefix (`installdir/` here):
```bash
meson install -C build
```
It will then install to `installdir/lib/python3.9/site-packages/scipy`, which
is not on your Python path, so to add it do:
```bash
export PYTHONPATH=$PWD/installdir/lib/python3.9/site-packages/
```

Now we should be able to import `scipy` and run the tests. Remembering that we
need to move out of the root of the repo to ensure we pick up the package and
not the local `scipy/` source directory:
```bash
cd doc
python -c "from scipy import constants as s; s.test()"
```
The above runs the tests for a single module, `constants`. Other ways of
running the tests should also work, for example:
```bash
pytest scipy
```

Current status (24 July) is that the full test suite passes on Linux with
OpenBLAS. And there is CI on this fork to keep it that way. Other platforms are
not tested yet.  The current status is already good enough to
work on both build related issues (e.g. build warnings, debugging some C/C++
extension) and on general SciPy development. It is already a much smoother than
working with the default `distutils`-based build one gets with
`python setup.py develop`.

The above configure-build-install-test docs are useful to understand and for
working on build improvements (you don't need to install for that to, for
example, see the build succeeds and a build warning has disappeared); if you
want the "all-in-one" command for the above, run:
```bash
./mesondev.sh build --prefix=$PWD/installdir
```

It's worth pointing out that Meson has [very good documentation](https://mesonbuild.com/);
it's worth reading and is often the best source of answers for "how to do X".

