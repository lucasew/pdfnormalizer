{
  inputs = {
    nixpkgs = { url = "github:NixOS/nixpkgs/nixos-22.05"; };
  };
  outputs = {self, nixpkgs, ...}: let
      pkgs = import nixpkgs { inherit system; };
      system = "x86_64-linux";
    in {
      devShells.${system}.default = pkgs.mkShellNoCC {
        shellHook = ''
          PS1="(pdfnormalizer) $PS1"
          PYTHONPATH=$PYTHONPATH:$(pwd)
          PATH=$PATH:$(pwd)/pdfnormalizer
        '';
        buildInputs = with pkgs.python3Packages; [
          python
          matplotlib
          tqdm
        ];
      };
  };
}
