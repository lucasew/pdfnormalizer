# Project Conventions

## General Guidelines

- **No silent failures:** Empty catch blocks and silent failures are strictly forbidden. Every `catch` block, every `.catch()`, every error callback that is not an expected/recoverable condition MUST call the centralized error-reporting function. "Out of scope" is not an excuse to swallow — at minimum, report and move on.
- **Error Handling:** All error handling must use the centralized reporting function. This function should be wired to Sentry.

## Operational Memory

- `pdfnormalizer/model.py` -> Core logic for image processing, bounding box calculations, and structural subdivision logic.
- `pdfnormalizer/utils.py` -> Shared utilities, logging, and base GUI classes (`GUI`, `GUIHandler`).
- `pdfnormalizer/app_*` -> Executable entrypoints (note: these often lack the `.py` extension but are Python scripts).
- `mise.toml` -> Manages tooling, tasks, and CI steps. Tasks with wildcard dependencies must include dummy tasks to prevent `mise` wildcard resolution errors.
- `.github/workflows/autorelease.yml` -> Standard CI/CD pipeline.
