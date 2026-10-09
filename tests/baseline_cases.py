"""Small Q4 and truss cases recorded before the assembly is generalised.

Run from the repository root to write tests/data/baseline.npz:

    python -m tests.baseline_cases
"""

from pathlib import Path

import numpy as np

from femlib.assembly import assemble_load_vector, assemble_stiffness
from femlib.materials import LinearElastic2D, LinearElasticTruss
from femlib.mesh import create_rectangular_mesh
from femlib.solvers import apply_dirichlet_bc, identify_boundary_nodes

BASELINE_PATH = Path(__file__).parent / "data" / "baseline.npz"


def q4_stiffness():
    """Q4 global stiffness on a 4 x 2 mesh, full and reduced integration."""
    mesh = create_rectangular_mesh(4.0, 1.0, 4, 2)
    material = LinearElastic2D(E=210e9, nu=0.3, thickness=0.01)
    return {
        "q4_K_full": assemble_stiffness(mesh, material, "full"),
        "q4_K_reduced": assemble_stiffness(mesh, material, "reduced"),
    }


def truss_stiffness():
    """Truss global stiffness for a three-bar triangle."""
    mesh = {
        "nodes": np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        "elements": np.array([[0, 1], [1, 2], [2, 0]]),
        "element_type": "Truss",
    }
    material = LinearElasticTruss(E=200e9, A=1e-3)
    return {"truss_K": assemble_stiffness(mesh, material, "reduced")}


def q4_cantilever_system():
    """Q4 cantilever on a 10 x 1 mesh, clamped left edge, tip and top loads.

    Returns the load vector, the constrained DOF indices, and the
    stiffness and load after boundary conditions.
    """
    mesh = create_rectangular_mesh(10.0, 1.0, 10, 1)
    material = LinearElastic2D(E=210e9, nu=0.3, thickness=0.1)
    K = assemble_stiffness(mesh, material, "full")
    loads = {
        "concentrated": [{"location": (10.0, 1.0), "force": [0.0, -1.0e3]}],
        "distributed": [{"edge": "top", "load_density": [0.0, -5.0e2]}],
    }
    F = assemble_load_vector(mesh, loads)
    bc = {
        "nodes": identify_boundary_nodes(mesh, "left"),
        "dofs": ["both"],
        "values": [0.0],
    }
    K_bc, F_bc, c_dofs, _ = apply_dirichlet_bc(K, F, bc)
    return {
        "cantilever_F": F,
        "cantilever_constrained_dofs": c_dofs,
        "cantilever_K_bc": K_bc,
        "cantilever_F_bc": F_bc,
    }


def all_cases():
    """All recorded arrays, keyed by name."""
    return {**q4_stiffness(), **truss_stiffness(), **q4_cantilever_system()}


if __name__ == "__main__":
    BASELINE_PATH.parent.mkdir(exist_ok=True)
    np.savez(BASELINE_PATH, **all_cases())
