# Contributing to DRL-Cox

Thank you for considering a contribution. This document explains how the
project is set up and what a change needs before it can be merged.

## Development setup

Prerequisites:

- Python 3.11, 3.12 or 3.13
- [Poetry](https://python-poetry.org/) 2.x
- Git

```bash
git clone https://github.com/<your-user>/drl-cox.git
cd drl-cox
poetry install --with docs
poetry run pre-commit install
```

`poetry install` creates a virtual environment with the package installed in
editable mode plus the development tools (pytest, ruff, mypy, pre-commit) and
the documentation toolchain.

## Branches

- `main` holds released code. Tags of the form `vX.Y.Z` are cut from `main`.
- `develop` is the integration branch. Open pull requests against `develop`.

## Running the checks

The same checks run in CI, so run them locally before pushing:

```bash
poetry run pytest                                   # tests (coverage is configured in pyproject.toml)
poetry run ruff check .                             # lint
poetry run ruff format --check .                    # formatting
poetry run mypy                                     # strict type checking of src/
poetry run mkdocs build --strict                    # documentation
poetry check --lock                                 # lock file matches pyproject.toml
```

`pre-commit run --all-files` runs ruff, mypy and the lock check in one go.

## Making changes

- Keep pull requests focused: one feature or fix per PR.
- Add or update tests for behaviour you change. The suite must stay green on all
  supported Python versions; CI runs Linux, macOS and Windows.
- Public functions and classes need NumPy-style docstrings; they are rendered in
  the API reference.
- Add an entry to the `Unreleased` section of `CHANGELOG.md`.
- Commit messages follow the Conventional Commits style used in the history
  (`feat:`, `fix:`, `docs:`, `chore(deps):`, ...).

### Dependencies

Runtime dependencies are declared in `pyproject.toml` with lower bounds only, so
downstream users are not forced onto the newest releases. The `poetry.lock` file
pins exact versions for development and CI and is kept current by Dependabot.
After changing `pyproject.toml`, run `poetry lock` and commit the lock file.

### Solvers

DRL-Cox is an exponential-cone program. The default solver is Clarabel, which is
installed with CVXPY on every supported Python version. Tests that need a
first-order solver use SCS. ECOS is optional (`drl-cox[ecos]`) and is not
available on Python 3.13.

## Releasing

1. Update the version in `pyproject.toml` and `CITATION.cff`, move the
   `Unreleased` notes in `CHANGELOG.md` under the new version, and merge to `main`.
2. Tag the merge commit: `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. The `Release` workflow builds the distribution, publishes it to PyPI through
   trusted publishing, and creates a GitHub release with generated notes.

## Reporting bugs and requesting features

Use the issue forms on GitHub. For security problems, follow `SECURITY.md`
instead of opening a public issue.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
