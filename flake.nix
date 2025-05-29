{
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://devenv.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
    ];
  };
  inputs = {
    # nix-ml-ops.url = "github:Atry/nix-ml-ops";
    devenv-root = {
      url = "file+file:///dev/null";
      flake = false;
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-ml-ops = {
      url = "github:atry/nix-ml-ops";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.devenv-root.follows = "devenv-root";
    };
  };
  outputs =
    inputs@{ nix-ml-ops, ... }:
    nix-ml-ops.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.nix-ml-ops.flakeModules.systems
        inputs.nix-ml-ops.flakeModules.cuda
        inputs.nix-ml-ops.flakeModules.devcontainer
        inputs.nix-ml-ops.flakeModules.ldFallbackManylinux
        inputs.nix-ml-ops.flakeModules.nixpkgs
      ];
      perSystem =
        {
          pkgs,
          lib,
          system,
          ...
        }:
        {
          ml-ops = {
            common =
              common:
              {
                ldFallback.libraries = [
                  pkgs.hdf5
                  pkgs.sox.lib
                ];
              }
              // lib.optionalAttrs (system != "aarch64-darwin") {
                cuda =
                  let
                    cfg = common.config.cuda;
                  in
                  {
                    cudaPackages = pkgs.cudaPackages_12_8; # cuda 12.8
                    packages = with cfg.cudaPackages; [
                      cuda_nvcc
                      cudatoolkit
                      cuda_cudart
                      libcublas
                      cudnn
                    ];
                  };
              };
            devcontainer = {
              devenvShellModule =
                { config, ... }:
                {
                  packages = with pkgs; [
                    bashInteractive
                  ];
                  enterShell = '''';
                  env = {
                    UV_PYTHON = config.languages.python.package;
                  };
                  languages = {
                    python = {
                      enable = true;
                      package = pkgs.python313;
                      venv.enable = true;
                      uv = {
                        enable = true;
                        sync = {
                          enable = true;
                          arguments = [ "--frozen" ];
                        };
                      };
                    };
                  };
                };
            };
          };
        };
    };
}
