# File is taken from nixos/nixpkgs and adapted.

{ lib
, fetchFromGitHub
, python
, buildPythonPackage
, gfortran
, nose
, pytest
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
    owner = "mesonbuild";
    repo = "meson";
    rev = version;
    sha256 = "sha256-eC3aUTvUzMS34b2yz67HJ1jADnMiLWgwGsj5/+6njAI=";
    };
    patches = excludeIndices oldAttrs.patches [ 2 ]; # remove gir fallback path patch
  });



in buildPythonPackage rec {
  pname = "scipy";
  version = "dev";
  format = "other";

  src = ./.;

  checkInputs = [ nose pytest pytest-xdist ];
  nativeBuildInputs = [ cython gfortran meson_ ninja pythran pkg-config ];
  buildInputs = [ numpy.blas pybind11 boost175 ];
  propagatedBuildInputs = [ numpy ];

  # Remove tests because of broken wrapper
  prePatch = ''
    rm scipy/linalg/tests/test_lapack.py
  '';

  doCheck = false;

  preBuild = ''
    ln -s ${numpy.cfg} site.cfg
  '';

  passthru = {
    blas = numpy.blas;
    meson = meson_;
  };

  checkPhase = ''
    pytest
  '';

  SCIPY_USE_G77_ABI_WRAPPER = 1;

  meta = with lib; {
    description = "SciPy (pronounced 'Sigh Pie') is open-source software for mathematics, science, and engineering";
    homepage = "https://www.scipy.org/";
    license = licenses.bsd3;
    maintainers = [ maintainers.fridh ];
  };
}
