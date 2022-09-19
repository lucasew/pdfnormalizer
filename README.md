# pdfnormalizer

Extract text and images from PDF files using machine learning, or at least a cooler UI.

This project works by subdividing the rasterized version of a PDF file and classifying each bounding box as an action (horizontal or vertical subdivision) or as an entity (text, picture or thrash).

There is (so far) a naive multiclass classifier that tries to map a bounding box, plus some extra clues, to any of the previous cited classes.

This project is not made to generalize to every possible PDF shape or content pattern but can be used to train specialist models. The model architecture also is not the best cutting edge wonderful technology of unmatched precision, but works and converges.

## Components
- `app_boring_extract`: Extracts stuff from a PDF, but using a more boring and precise approach.
- `app_geracao_dataset`: Tool to ingest a PDF to build a training dataset.
- `app_pdf_extract`: Extracts stuff from a PDF by using an already trained model.
- `app_predicao`: Shows visually what the model sees in the page.
- `app_trainer`: Trains by using the provided dataset and generates the model file
- `db_clean_unnecessary.sql`: SQL query that must be run on the generated dataset file after using `app_geracao_dataset` and before training

The other programs are parallel proof of concepts used as basis to implement the final tools and are here for experimentation and exploration purposes.

## How can I use it?

- Use Linux (didn't tested with WSL2)
- Install [Nix](https://nixos.org)
- Enable flakes: `echo "experimental-features = nix-command flakes" | sudo tee -a /etc/nix/nix.conf`
- Go to this repository folder by using the `cd` command
- `nix develop` (or `direnv allow` if you use direnv)

If you use direnv you will need to do a `nix develop` before because pylsp seems to be broken without it. That's just development convenience because I use Neovim and I don't use a global LSP server. If you don't care about the provided LSP just ignore this line.
