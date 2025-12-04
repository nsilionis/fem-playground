# FEMlib - Educational Finite Element Library

A modular, object-oriented finite element analysis library for
educational purposes, developed as part of an FEA course.

## Features

- 2D isoparametric elements (Q4)
- Linear elastic materials (plane stress/strain)
- Structured mesh generation
- Full and reduced integration schemes
- Modular architecture for extensibility

## Installation

Install in development mode:
```bash
pip install -e .
```

## Quick Start
```python
from femlib.mesh import create_rectangular_mesh
from femlib.materials import LinearElastic
from femlib.elements import Q4Element

# Create mesh
mesh = create_rectangular_mesh(Lx=10.0, Ly=1.0, nx=8, ny=2)

# Define material
steel = LinearElastic(E=210e9, nu=0.3, thickness=0.01)

# Analysis continues...
```

## Project Structure
```
src/femlib/          # Main library code
examples/            # Example scripts and verification
tests/               # Unit tests
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Matplotlib ≥ 3.3

## Development

Install with development dependencies:
```bash
pip install -e ".[dev]"
```