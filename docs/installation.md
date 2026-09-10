# Installation Guide

## Requirements

- Python 3.11, 3.12 or 3.13
- pip, or Poetry 2 for development installs

## Installation Methods

### Using pip

```bash
pip install drl-cox
```

### Using Poetry

```bash
poetry add drl-cox
```

### From Source

```bash
git clone https://github.com/diogoribeiro7/drl-cox.git
cd drl-cox
poetry install
```

## Verifying Installation

```python
import drl_cox

print(drl_cox.__version__)
```

## Troubleshooting

### Solvers

DRL-Cox needs a solver that supports the exponential cone. Clarabel is installed
with CVXPY and is the default. SCS is also installed. The legacy ECOS solver can be
added on Python 3.11 and 3.12 with:

```bash
pip install "drl-cox[ecos]"
```

MOSEK works as well if you have a licence.

### macOS Apple Silicon

For M1/M2 Macs, ensure you're using native Python:

```bash
arch -arm64 poetry install
```
