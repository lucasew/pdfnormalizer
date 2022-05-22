{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShellNoCC {
  shellHook = ''
    PYTHONPATH=$PYTHONPATH:$(pwd)
    PATH=$PATH:$(pwd)/pdfnormalizer
  '';
  buildInputs = with pkgs.python3Packages; [
    python
    matplotlib
    tqdm
  ];
}
