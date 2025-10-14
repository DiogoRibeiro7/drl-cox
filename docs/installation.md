# Installation Guide

## Requirements

- Python 3.10, 3.11, or 3.12
- Poetry (recommended) or pip

## Installation Methods

### Using Poetry (Recommended)

```bash
poetry add drl-cox
```

### Using pip

```bash
pip install drl-cox
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

### CVXPY Installation Issues

If you encounter issues with CVXPY solvers:

```bash
poetry add cvxpy[CBC,GLPK]  # Additional solvers
```

### macOS Apple Silicon

For M1/M2 Macs, ensure you're using native Python:

```bash
arch -arm64 poetry install
```
