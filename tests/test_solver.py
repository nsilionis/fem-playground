"""Linear solver: accuracy, no determinant overflow, unconstrained systems."""

import warnings

import numpy as np
import pytest
from scipy.linalg import LinAlgWarning

from femlib.assembly import assemble_load_vector, assemble_stiffness
from femlib.materials import LinearElastic2D
from femlib.mesh import create_rectangular_mesh
from femlib.sections import PlaneSection
from femlib.solvers import (
    apply_dirichlet_bc,
    identify_boundary_nodes,
    solve_system,
)

RTOL = 1e-12


def cantilever(nx, ny, clamp=True):
    """Q4 cantilever with a top load; clamped left edge if clamp is True."""
    mesh = create_rectangular_mesh(4.0, 2.0, nx, ny)
    K = assemble_stiffness(
        mesh, LinearElastic2D(E=100e9, nu=0.3), PlaneSection(0.2), "full"
    )
    F = assemble_load_vector(
        mesh, {"distributed": [{"edge": "top", "load_density": [0.0, -1e3]}]}
    )
    if not clamp:
        return K, F
    bc = {
        "nodes": identify_boundary_nodes(mesh, "left"),
        "dofs": ["both"],
        "values": [0.0],
    }
    K_bc, F_bc, _, _ = apply_dirichlet_bc(K, F, bc)
    return K_bc, F_bc


def test_large_system_solves_without_warnings_to_rounding():
    K, F = cantilever(40, 20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        u = solve_system(K, F)
    residual = np.linalg.norm(K @ u - F)
    scale = np.linalg.norm(K, 2) * np.linalg.norm(u)
    assert residual <= RTOL * scale


def test_unconstrained_system_is_not_solved_silently():
    K, F = cantilever(4, 2, clamp=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        with pytest.raises((np.linalg.LinAlgError, LinAlgWarning)):
            solve_system(K, F)
