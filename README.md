# FEMlib - Educational Finite Element Library

A modular, object-oriented finite element analysis library for
educational purposes, developed as part of an FEA course.

## Features

- Elements
  - Four-node isoparametric quadrilateral (Q4), with full or reduced
    integration.
  - Two-node truss.
  - Two-node Euler-Bernoulli beam with Hermite cubic shape functions.
    Bending stiffness and mass per unit length may vary along the
    element, and are integrated by Gauss-Legendre quadrature.
- Materials and sections are separate objects. A material holds the
  elastic constants and, for vibration, the density. A section holds
  the thickness, area or second moment of area. Beam properties may
  be given as constants or as functions of position x.
- Assembly of the global stiffness matrix, dense or sparse, for any
  of the elements above. The consistent mass matrix is assembled for
  the beam element only.
- Dirichlet boundary conditions by elimination or by penalty,
  addressed by node and named degree of freedom.
- Linear static solution by Cholesky factorisation (dense) or by a
  sparse factorisation whose pivots are checked. On the sparse path,
  a zero or negative pivot is reported with its node and degree of
  freedom, and a large diagonal ratio gives a warning.
- Free vibration: natural frequencies in Hz and mass-normalised mode
  shapes, dense or sparse.
- Post-processing
  - Q4 stress and strain recovery, and contour plots of displacement,
    stress and strain.
  - Beam displacement, curvature and surface strain at any position
    along the beam, for a static solution or a set of modes.
  - Mode-shape plots for beams.
  - A common plot style for publication-quality figures.

## Installation

Install the library with its development tools and the packages the
examples need:
```bash
uv sync
```

With pip (version 25.1 or later):
```bash
pip install -e . --group dev --group examples
```

For the library alone:
```bash
pip install -e .
```

## Quick Start

The first four natural frequencies of a simply supported steel beam,
and the surface strain of its first mode:
```python
import numpy as np

from femlib.assembly import assemble_mass, assemble_stiffness
from femlib.elements import Beam
from femlib.materials import LinearElastic1D
from femlib.mesh import create_beam_mesh
from femlib.postprocess import surface_strain
from femlib.sections import BeamSection
from femlib.solvers import apply_dirichlet_bc, solve_eigen

# Steel beam, 10 m long, meshed with 32 elements
L = 10.0
material = LinearElastic1D(E=210e9, rho=7850.0)
section = BeamSection(I=8.0e-6, A=0.01)
mesh = create_beam_mesh(L, n_elements=32)

K = assemble_stiffness(mesh, material, section)
M = assemble_mass(mesh, material, section)

# Hold the deflection w at both ends; the rotations stay free
supports = {"constraints": [{"node": 0, "dof": "w", "value": 0.0},
                            {"node": 32, "dof": "w", "value": 0.0}]}
_, _, constrained, _ = apply_dirichlet_bc(
    K, np.zeros(K.shape[0]), supports, dof_names=Beam.dof_names
)

frequencies, modes = solve_eigen(K, M, constrained, n_modes=4,
                                 dof_names=Beam.dof_names)
print(frequencies)  # [Hz]

# Strain of mode 1 on the fibre 0.1 m above the neutral axis
x = np.array([2.5, 5.0, 7.5])
print(surface_strain(mesh, modes[:, 0], x, z=0.1))
```

The first frequency printed is 2.298 Hz. This agrees with the exact
value, (pi / 2 L^2) sqrt(EI / rho A).

## Conventions

- Units are not fixed by the library. Use a consistent set, such as
  SI throughout. Frequencies are always returned in Hz.
- Each node carries two degrees of freedom, numbered 2i and 2i + 1
  for node i. They are named `ux` and `uy` for the Q4 and truss
  elements, and `w` and `theta` for the beam. Functions that apply
  boundary conditions or loads take a `dof_names` argument; pass
  `Beam.dof_names` for beam meshes.
- For the beam, w is the transverse deflection and theta = dw/dx.
  The fibre distance z is positive in the direction of positive w,
  so the bending strain is epsilon = -z kappa, with kappa = d2w/dx2.
- Mode shapes satisfy phi^T M phi = 1. Their sign is fixed so that
  the first significant entry of each mode is positive.

## Verification and Tests

Run the tests with:
```bash
python -m pytest tests/
```

The tests compare elements, assembly, loads, boundary conditions,
solvers and beam response recovery against closed-form solutions.
A regression test also checks the Q4 and truss results against a
stored baseline.

The scripts in `examples/` write their figures to
`outputs/figures`:

- `verify_mesh.py`: node and element numbering of a structured mesh.
- `verify_cantilever.py`: Q4 cantilever against Euler-Bernoulli
  theory.
- `assignment_reference.py`: reference solution for the shear
  locking assignment.
- `verify_beam_frequencies.py`: convergence of the first four
  frequencies of simply supported and clamped beams under mesh
  refinement, against the exact values.
- `verify_beam_modes.py`: beam mode shapes, and convergence of the
  modal curvature in the energy norm.

The two beam scripts accept `--show` to display the figures as well
as saving them.

## Project Structure
```
src/femlib/
    elements/       # Q4, truss and beam elements
    materials/      # Linear elastic materials
    sections/       # Plane, truss and beam cross-sections
    mesh/           # Structured mesh generation
    assembly/       # Global stiffness, mass and load assembly
    solvers/        # Boundary conditions, static and eigen solvers
    postprocess/    # Stress recovery, beam response, plotting
examples/           # Example and verification scripts
tests/              # Unit and regression tests
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.10
- Matplotlib ≥ 3.3
- A LaTeX installation with the `libertinus`, `newtxmath` and
  `siunitx` packages, for figures. The plot style renders all text
  with LaTeX, so any script that calls `setup_plot_style` needs it.
