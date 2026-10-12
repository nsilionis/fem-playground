"""
Solver module for boundary conditions and system solution.

Provides utilities for applying boundary conditions and solving
the resulting linear systems and free-vibration eigenproblems.
"""

from .boundary_conditions import (
    identify_boundary_nodes,
    apply_dirichlet_bc
)
from .linear import (
    solve_system,
    compute_reaction_forces
)
from .eigen import solve_eigen

__all__ = [
    'identify_boundary_nodes',
    'apply_dirichlet_bc',
    'solve_system',
    'compute_reaction_forces',
    'solve_eigen',
]
