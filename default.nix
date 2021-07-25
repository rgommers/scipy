# File is taken from nixos/nixpkgs and adapted.

{ lib
, fetchFromGitHub
, python
, buildPythonPackage
, gfortran
, nose
# Use a hook that implements the checkPhase calling pytest
# instead of us calling pytest directly.
, pytestCheckHook
, pytest-xdist
, numpy
, pybind11
, cython
, meson
, ninja
, pythran
, pkg-config
, boost175
}:

let

  excludeIndices = list: indices: let
    nitems = lib.length list;
    includeIndices = lib.subtractLists indices (lib.range 0 (nitems - 1));
  in map (idx: lib.elemAt list idx) includeIndices;

  meson_ = meson.overrideAttrs (oldAttrs: rec {
    version = "0.59.0.rc2";
    src = fetchFromGitHub {
    owner = "rgommers";
    repo = "meson";
    rev = "087870a2f15b31c56c4dc830ae856c3729f60776";
    sha256 = "sha256-7HJjxVnryaBkRDiwMHVkOo6mahfjrGAf1JijBIZ7Ay4=";
    };
    patches = excludeIndices oldAttrs.patches [ 2 ]; # remove gir fallback path patch
  });

in buildPythonPackage rec {
  pname = "scipy";
  version = "dev";
  format = "other";

  src = ./.;

  checkInputs = [ pytestCheckHook pytest-xdist ];
  nativeBuildInputs = [ cython gfortran meson_ ninja pythran pkg-config ];
  buildInputs = [ numpy.blas pybind11 boost175 ];
  propagatedBuildInputs = [ numpy ];

  # Remove tests because of broken wrapper
  prePatch = ''
    rm scipy/linalg/tests/test_lapack.py
  '';

  doCheck = true;

  preBuild = ''
    ln -s ${numpy.cfg} site.cfg
  '';

  passthru = {
    blas = numpy.blas;
    meson = meson_;
  };

  preCheck = ''
    pushd $out/${python.sitePackages}
    export PYTHONPATH=$out/${python.sitePackages}:$PYTHONPATH
  '';

  postCheck = ''
    popd
  '';

  SCIPY_USE_G77_ABI_WRAPPER = 1;

  meta = with lib; {
    description = "SciPy (pronounced 'Sigh Pie') is open-source software for mathematics, science, and engineering";
    homepage = "https://www.scipy.org/";
    license = licenses.bsd3;
    maintainers = [ maintainers.fridh ];
  };
}
