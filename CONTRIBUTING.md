# Contributing to DRL-Cox

Thank you for considering contributing to **DRL-Cox**! This document provides guidelines for contributing to the project.

---

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:

   ```bash
   git clone https://github.com/diogoribeiro7/drl-cox.git
   cd drl-cox
   ```
3. **Create a new branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:

   ```bash
   poetry install
   ```

---

## Development Setup

### Prerequisites

* Python **3.10**, **3.11**, or **3.12**
* [Poetry](https://python-poetry.org/) for dependency management
* (Optional) [pre-commit](https://pre-commit.com/) hooks

### Installation

```bash
poetry install
poetry run pre-commit install  # if using pre-commit hooks
```

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Linting
poetry run ruff check src

# Type checking
poetry run mypy src

# Format checking (without modifying files)
poetry run ruff format --check src
```

---

## Making Changes

* Write clear, descriptive commit messages.
* Add tests for new functionality.
* Update documentation as needed.
* Ensure all tests pass and code quality checks succeed.
* Keep changes focused — one feature/fix per pull request.

---

## Pull Request Process

* Update `README.md` with details of changes if applicable.
* Update `CHANGELOG.md` following the *Keep a Changelog* format.
* Ensure your code passes all CI checks.
* Request review from maintainers.
* Address any feedback from reviewers.

---

## Code Style

* Follow **PEP 8** style guide.
* Use **type hints** for function signatures.
* Write **docstrings** for public functions and classes.
* Keep functions focused and concise.
* Use descriptive variable names.

---

## Reporting Bugs

1. Check if the bug has already been reported in **Issues**.
2. If not, create a new issue with:

   * Clear title and description
   * Steps to reproduce
   * Expected vs actual behavior
   * Python version and environment details
   * Code samples or test cases if applicable

---

## Suggesting Enhancements

1. Check existing issues and pull requests first.
2. Create an issue describing:

   * The enhancement and its motivation
   * Potential implementation approach
   * Any relevant examples or references

---

## Questions?

Feel free to open an issue with the **question** label.

Thank you for contributing!
