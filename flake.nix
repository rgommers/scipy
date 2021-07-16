{
  description = "Scipy built using meson";

  inputs.nixpkgs.url = "github:nixos/nixpkgs";
  inputs.utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, utils }: {
    # Provide an overlay that has an extension with Python packages.
    overlay = final: prev: nixpkgs.lib.composeManyExtensions [
      (final: prev: {

        pythonPackagesOverrides = (prev.pythonPackagesOverrides or []) ++ [
          (self: super: {
            # Replace Nixpkgs scipy with our version of scipy
            # We use meson from the top-level package set, and not from the
            # Python packages set because that one lacks the build hook.
            scipy = self.callPackage ./default.nix { inherit (prev) meson;};
          })
        ];
        # Remove when https://github.com/NixOS/nixpkgs/pull/91850 is fixed.
        python3 = let
          self = prev.python3.override {
            inherit self;
            packageOverrides = nixpkgs.lib.composeManyExtensions final.pythonPackagesOverrides;
          };
        in self;
      })
    ] final prev;
  } // (utils.lib.eachDefaultSystem (system: let

    # Our own overlay does not get applied to nixpkgs because that would lead to
    # an infinite recursion. Therefore, we need to import nixpkgs and apply it ourselves.
    pkgs = import nixpkgs {
      inherit system;
      overlays = [
          self.overlay
      ];
    };
    inherit (pkgs) lib;
    python = pkgs.python3;

  in rec {
    packages = {
      # Make the Python interpeter along with the package set that is now
      # using this version of scipy available.
      #
      # $ nix build .#packages.python.pkgs.scipy
      #
      # $ nix build .#packages.python.pkgs.pandas
      #
      inherit python;
    };
    # This is the default package to be built.
    # $ nix build
    defaultPackage = python.pkgs.scipy;
  }));
}
